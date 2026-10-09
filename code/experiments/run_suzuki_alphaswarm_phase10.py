#!/usr/bin/env python3
"""Phase 10 AlphaSwarm-inspired controller for the finite-pool Suzuki B_dev study.

This driver is deliberately experiment-local. It reuses the retained
component-ARD GP and its ask/tell feedback path, but replaces direct candidate
argmax with a q=1 discrete projection of AlphaSwarm's personal/global/ML
particle update. Organizer candidate labels are never opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

EXPERIMENT_ROOT = Path(__file__).resolve().parent
CODE_ROOT = EXPERIMENT_ROOT.parent
DATASET_ROOT = CODE_ROOT.parents[1]
sys.path.insert(0, str(CODE_ROOT / "bgolearn_rxn_extension" / "src"))

from bgolearn_rxn.acquisition import stable_log_expected_improvement, upper_confidence_bound  # noqa: E402
from bgolearn_rxn.campaign import CampaignRunner  # noqa: E402
from bgolearn_rxn.config import load_run_config  # noqa: E402
from bgolearn_rxn.datasets import DatasetSpec, load_reaction_dataset  # noqa: E402
from bgolearn_rxn.metrics import compute_campaign_metrics  # noqa: E402
from bgolearn_rxn.oracle import OfflineTableOracle  # noqa: E402
from bgolearn_rxn.schemas import Recommendation  # noqa: E402

RUNNER_PATH = EXPERIMENT_ROOT / "run_suzuki_40round_landscape_zoo.py"
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("phase8_zoo", RUNNER_PATH)
_zoo = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(_zoo)

TRAIN_FILE = _zoo.TRAIN_FILE
CANDIDATE_FILE = _zoo.CANDIDATE_FILE
CONFIG_FILE = _zoo.CONFIG_FILE
BUDGET = _zoo.BUDGET


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values)
    replacement = float(np.min(values[finite]) - 10.0)
    clean = np.where(finite, values, replacement)
    sd = float(np.std(clean))
    return np.zeros_like(clean) if sd <= 1e-12 else (clean - float(np.mean(clean)))/sd


class DiscreteAlphaSwarm:
    """q=1 adapter preserving AlphaSwarm semantics on a finite candidate pool."""

    def __init__(self, X_all: np.ndarray, *, seed: int, n_particles: int = 24) -> None:
        self.rng = np.random.default_rng(int(seed))
        raw = np.asarray(X_all, dtype=float)
        lo = raw.min(axis=0)
        span = raw.max(axis=0) - lo
        span[span < 1e-12] = 1.0
        self.X = np.clip((raw - lo) / span, 0.0, 1.0)
        self.n_particles = int(min(n_particles, len(raw)))
        self.positions: np.ndarray | None = None
        self.velocities: np.ndarray | None = None
        self.pbest_positions: np.ndarray | None = None
        self.pbest_scores: np.ndarray | None = None
        self.gbest_position: np.ndarray | None = None
        self.gbest_score = -np.inf
        self.relocation_fraction = 0.20

    def _nearest_unseen(self, position: np.ndarray, available: np.ndarray, used: set[int]) -> int:
        if len(available) == 0:
            raise RuntimeError("No candidates remain for particle projection")
        pool = np.asarray([idx for idx in available if int(idx) not in used], dtype=int)
        if pool.size == 0:
            pool = np.asarray(available, dtype=int)
        distances = np.sum((self.X[pool] - position[None, :]) ** 2, axis=1)
        best = np.min(distances)
        ties = pool[np.flatnonzero(np.isclose(distances, best, rtol=0.0, atol=1e-14))]
        return int(np.min(ties))

    def update(self, acquisition: np.ndarray, available: np.ndarray, *, step: int) -> int:
        available = np.asarray(available, dtype=int)
        if available.size == 0:
            raise RuntimeError("Candidate pool exhausted")
        scores = np.asarray(acquisition, dtype=float)
        if scores.shape != (len(self.X),):
            raise ValueError("Acquisition must align with the full candidate pool")
        finite_available = available[np.isfinite(scores[available])]
        if finite_available.size == 0:
            finite_available = available
        guide_order = finite_available[np.argsort(-scores[finite_available], kind="stable")]
        guide_order = guide_order[: self.n_particles]
        if self.positions is None:
            chosen = guide_order.tolist()
            if len(chosen) < self.n_particles:
                chosen.extend([int(idx) for idx in available if int(idx) not in chosen])
            idx = np.asarray(chosen[: self.n_particles], dtype=int)
            self.positions = self.X[idx].copy()
            self.velocities = np.zeros_like(self.positions)
            self.pbest_positions = self.positions.copy()
            self.pbest_scores = scores[idx].copy()
            self.gbest_score = float(np.max(self.pbest_scores))
            self.gbest_position = self.pbest_positions[int(np.argmax(self.pbest_scores))].copy()
            return int(idx[int(np.argmax(self.pbest_scores))])

        # Re-project particles whose old candidate was removed by the q=1 query.
        available_set = set(int(idx) for idx in available)
        used: set[int] = set()
        current_indices = np.empty(self.n_particles, dtype=int)
        for particle in range(self.n_particles):
            current_indices[particle] = self._nearest_unseen(self.positions[particle], available, used)
            used.add(int(current_indices[particle]))
            self.positions[particle] = self.X[current_indices[particle]]

        order = np.argsort(self.pbest_scores)
        n_relocate = max(1, int(math.ceil(self.relocation_fraction * self.n_particles)))
        relocate_particles = set(int(i) for i in order[:n_relocate])
        guide_cursor = 0
        used = set()
        updated_indices = np.empty(self.n_particles, dtype=int)
        for particle in range(self.n_particles):
            if particle in relocate_particles and guide_cursor < len(guide_order):
                candidate = int(guide_order[guide_cursor])
                guide_cursor += 1
                if candidate in used:
                    candidate = self._nearest_unseen(self.X[candidate], available, used)
                updated_indices[particle] = candidate
                self.positions[particle] = self.X[candidate]
                self.velocities[particle] = self.rng.normal(0.0, 0.15, size=self.X.shape[1])
            else:
                phi_local = 1.5 * self.rng.random(self.X.shape[1])
                phi_social = 0.5 * self.rng.random(self.X.shape[1])
                phi_ml = 2.0 * self.rng.random(self.X.shape[1])
                self.velocities[particle] = (
                    0.7 * self.velocities[particle]
                    + phi_local * (self.pbest_positions[particle] - self.positions[particle])
                    + phi_social * (self.gbest_position - self.positions[particle])
                    + phi_ml * (self.X[int(guide_order[min(guide_cursor, len(guide_order)-1)])] - self.positions[particle])
                )
                proposed = np.clip(self.positions[particle] + self.velocities[particle], 0.0, 1.0)
                updated_indices[particle] = self._nearest_unseen(proposed, available, used)
                self.positions[particle] = self.X[updated_indices[particle]]
                guide_cursor = min(guide_cursor + 1, max(len(guide_order) - 1, 0))
            used.add(int(updated_indices[particle]))

        particle_scores = scores[updated_indices]
        improved = particle_scores > self.pbest_scores
        self.pbest_positions[improved] = self.positions[improved]
        self.pbest_scores[improved] = particle_scores[improved]
        best_particle = int(np.argmax(self.pbest_scores))
        if float(self.pbest_scores[best_particle]) >= self.gbest_score:
            self.gbest_score = float(self.pbest_scores[best_particle])
            self.gbest_position = self.pbest_positions[best_particle].copy()
        chosen = self._nearest_unseen(self.gbest_position, available, set())
        return int(chosen)


def _acquisition(runner: CampaignRunner, posterior, best_f: float, step: int) -> np.ndarray:
    if step == 1:
        return np.asarray(posterior.mean, dtype=float)
    if step >= runner.config.policy.final_start:
        return upper_confidence_bound(posterior.mean, posterior.std, beta=runner.config.policy.final_beta)
    return stable_log_expected_improvement(
        posterior.mean,
        posterior.std,
        best_f,
        xi=runner.policy._xi(step, runner.config.dataset.budget),
    )


def run_one(spec: dict, output: Path, *, variant: str, active_until: int = BUDGET) -> dict:
    config = load_run_config(CONFIG_FILE)
    config = config.__class__(
        dataset=config.dataset.__class__(**{**config.dataset.__dict__, "budget": BUDGET}),
        features=config.features,
        gp=config.gp,
        interaction_model=config.interaction_model,
        trees=config.trees,
        ensemble=config.ensemble.__class__(**{**config.ensemble.__dict__, "enabled_models": ("component_ard_gp",), "initial_weights": {"component_ard_gp": 1.0}}),
        policy=config.policy,
        runtime=config.runtime,
    )
    dataset = load_reaction_dataset(TRAIN_FILE, CANDIDATE_FILE, DatasetSpec.from_config(config.dataset))
    lookup, teacher = _zoo.build_landscape(spec)
    oracle = OfflineTableOracle(dataset=dataset, lookup=lookup)
    runner = CampaignRunner(dataset, config, seed=int(spec["campaign_seed"]), backend="native")
    session = runner.start_session(budget=BUDGET)
    swarm = DiscreteAlphaSwarm(runner.prepared.X_candidates, seed=int(spec["campaign_seed"]) + 991)
    started = time.perf_counter()
    while not session.is_complete:
        remaining_rows = session.remaining_rows
        X_remaining = runner.prepared.X_candidates[remaining_rows]
        posterior = runner.backend.fit_predict(session.X_observed, session.y_observed, X_remaining)
        best_before = dataset.target_incumbent(session.observed)
        local_scores = _acquisition(runner, posterior, best_before, session.step)
        full_scores = np.full(len(dataset.candidates), -np.inf, dtype=float)
        full_scores[remaining_rows] = local_scores
        if session.step <= active_until:
            selected_global_row = swarm.update(full_scores, remaining_rows, step=session.step)
        else:
            # Late rounds return to the retained stable finite-pool GP policy.
            # This isolates AlphaSwarm's intended early exploration benefit.
            selected_global_row = int(remaining_rows[int(np.argmax(local_scores))])
        selected_local = int(np.flatnonzero(remaining_rows == selected_global_row)[0])
        selected_row = dataset.candidates.iloc[selected_global_row]
        query_index = int(selected_row["query_index"])
        rec = Recommendation(
            local_index=selected_local,
            query_index=query_index,
            acquisition_name=("alpha_pso_reseed" if session.step > 1 else "alpha_pso_query1") if session.step <= active_until else "stable_log_ei_after_alpha_pso",
            acquisition_value=float(full_scores[selected_global_row]),
            predicted_mean=float(posterior.mean[selected_local]),
            predicted_std=float(posterior.std[selected_local]),
            component_means={name: float(values[selected_local]) for name, values in posterior.component_means.items()},
            component_stds={name: float(values[selected_local]) for name, values in posterior.component_stds.items()},
            model_weights=dict(posterior.weights),
            prior_weight=0.0,
        )
        session._pending = {
            "recommendation": rec,
            "selected_local": selected_local,
            "selected_global_row": selected_global_row,
            "selected_row": selected_row,
            "best_before": best_before,
        }
        session.observe(query_index, float(oracle.query(query_index)))
    elapsed = time.perf_counter() - started
    result = session.finalize(global_best=oracle.global_best)
    metrics = result.metrics.to_dict()
    payload = result.to_payload()
    payload["landscape_metrics"] = {
        "initial_round_found_best": metrics["initial_round_found_best"],
        "best_found": metrics["best_found"],
        "round_to_95_global_best": BUDGET + 1 if metrics["round_to_95_global_best"] is None else metrics["round_to_95_global_best"],
        "AUC_best_so_far": metrics["auc_best_so_far"],
    }
    payload["landscape_contract"] = {
        **spec,
        "variant": variant,
        "n_initial": len(dataset.initial),
        "n_candidates": len(dataset.candidates),
        "budget": BUDGET,
        "all_queries_unique": len({row["query_index"] for row in payload["trajectory"]}) == BUDGET,
        "all_queries_in_candidate_pool": True,
        "public_test_labels_used": False,
        "public_searchspace_labels_used": False,
        "production_default_changed": False,
        "teacher": teacher,
    }
    path = output / "trajectories" / variant / f"{spec['landscape']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {"dataset": "Suzuki", **spec, "variant": variant, **payload["landscape_metrics"], "elapsed_seconds": elapsed, "global_best": float(oracle.global_best), "query_ids_sha256": hashlib.sha256(json.dumps([r["query_index"] for r in payload["trajectory"]]).encode()).hexdigest(), "contract": payload["landscape_contract"]}


def mean_ci(values: np.ndarray) -> tuple[float, float, float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    half = float(t.ppf(0.975, len(values)-1) * sd / math.sqrt(len(values))) if len(values) > 1 else 0.0
    return mean, sd, mean-half, mean+half


def summarize(rows: pd.DataFrame, baseline: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    core = ["initial_round_found_best", "best_found", "round_to_95_global_best", "AUC_best_so_far", "elapsed_seconds"]
    abs_rows = []
    for variant, group in rows.groupby("variant", sort=True):
        for metric in core:
            mean, sd, low, high = mean_ci(group[metric].to_numpy())
            abs_rows.append({"variant": variant, "metric": metric, "n": len(group), "mean": mean, "sample_sd": sd, "ci95_low": low, "ci95_high": high})
    base = rows[rows.variant == baseline].set_index("landscape")
    paired = []
    for variant in sorted(set(rows.variant) - {baseline}):
        cand = rows[rows.variant == variant].set_index("landscape")
        common = base.index.intersection(cand.index)
        for metric in core:
            delta = cand.loc[common, metric].to_numpy() - base.loc[common, metric].to_numpy()
            mean, sd, low, high = mean_ci(delta)
            paired.append({"candidate": variant, "baseline": baseline, "metric": metric, "n_pairs": len(delta), "mean_delta": mean, "sample_sd_delta": sd, "ci95_low": low, "ci95_high": high, "wins": int((delta > 0).sum()) if metric not in {"round_to_95_global_best", "elapsed_seconds"} else int((delta < 0).sum()), "losses": int((delta < 0).sum()) if metric not in {"round_to_95_global_best", "elapsed_seconds"} else int((delta > 0).sum())})
    return pd.DataFrame(abs_rows), pd.DataFrame(paired)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("phase8_dev", "phase8_test"), default="phase8_dev")
    parser.add_argument("--episode-count", type=int, default=20)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    count = 2 if args.smoke else args.episode_count
    specs = _zoo.landscape_specs(args.split, count)
    variants = ["component_ard", "alpha_pso_20pct", "alpha_pso_early8"]
    rows = []
    for spec in specs:
        rows.append(run_one(spec, args.output_dir, variant="alpha_pso_20pct", active_until=BUDGET))
        rows.append(run_one(spec, args.output_dir, variant="alpha_pso_early8", active_until=8))
        # Baseline uses the unchanged campaign runner and is run in the same
        # process for exact paired landscapes without opening protected labels.
        config, dataset, _ = _zoo.build_problem("component_ard")
        lookup, teacher = _zoo.build_landscape(spec)
        oracle = OfflineTableOracle(dataset=dataset, lookup=lookup)
        baseline_runner = CampaignRunner(dataset, config, seed=int(spec["campaign_seed"]), backend="native")
        started = time.perf_counter()
        baseline_result = baseline_runner.run(oracle, budget=BUDGET)
        elapsed = time.perf_counter() - started
        raw = baseline_result.metrics.to_dict()
        rows.append({"dataset": "Suzuki", **spec, "variant": "component_ard", "initial_round_found_best": float(raw["initial_round_found_best"]), "best_found": float(raw["best_found"]), "round_to_95_global_best": BUDGET + 1 if raw["round_to_95_global_best"] is None else int(raw["round_to_95_global_best"]), "AUC_best_so_far": float(raw["auc_best_so_far"]), "elapsed_seconds": elapsed, "global_best": float(oracle.global_best), "contract": {"public_test_labels_used": False, "public_searchspace_labels_used": False, "teacher": teacher}})
    frame = pd.DataFrame(rows)
    abs_summary, paired = summarize(frame, "component_ard")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "landscape_metrics.csv", index=False)
    abs_summary.to_csv(args.output_dir / "absolute_summary.csv", index=False)
    paired.to_csv(args.output_dir / "paired_vs_component_ard.csv", index=False)
    (args.output_dir / "manifest.json").write_text(json.dumps({"phase": "phase10_alphaswarm", "split": args.split, "smoke": args.smoke, "variants": variants, "landscape_count": count, "official_geometry": {"n_initial": 29, "n_candidates": 5731, "budget": BUDGET}, "public_test_labels_used": False, "public_searchspace_labels_used": False}, indent=2, sort_keys=True), encoding="utf-8")
    print(abs_summary.to_string(index=False))
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
