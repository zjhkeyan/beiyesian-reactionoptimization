#!/usr/bin/env python3
"""Train-only Suzuki fixed/component-ARD prequential committee study."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


EXPERIMENT_ROOT = Path(__file__).resolve().parent
CODE_ROOT = EXPERIMENT_ROOT.parent
DATASET_ROOT = CODE_ROOT.parents[1]
EXTENSION_ROOT = CODE_ROOT / "bgolearn_rxn_extension"
sys.path.insert(0, str(EXTENSION_ROOT / "src"))
sys.path.insert(0, str(EXPERIMENT_ROOT))

import run_suzuki_holdout_proxy as suzuki_proxy  # noqa: E402
from bgolearn_rxn.acquisition import stable_log_expected_improvement  # noqa: E402
from bgolearn_rxn.campaign import CampaignRunner  # noqa: E402
from bgolearn_rxn.metrics import compute_campaign_metrics  # noqa: E402
from bgolearn_rxn.oracle import OfflineTableOracle  # noqa: E402
from bgolearn_rxn.schemas import Posterior, Recommendation  # noqa: E402
from bgolearn_rxn.utils import deterministic_argmax  # noqa: E402


FIXED = "fixed"
PURE_ARD = "pure_ard"
MIXTURE = "prequential_mixture"
GATED = "disagreement_gate"
SMOKE_VARIANTS = (FIXED, PURE_ARD, MIXTURE, GATED)
COMMITTEE_VARIANTS = (MIXTURE, GATED)
MODEL_FIXED = "gp"
MODEL_ARD = "component_ard_gp"
CORE_METRICS = (
    "initial_round_found_best",
    "best_found",
    "round_to_95_global_best",
    "AUC_best_so_far",
)
SUMMARY_METRICS = CORE_METRICS + ("elapsed_seconds",)
TRAIN_FILE = DATASET_ROOT / "Suzuki" / "suzuki_train.csv"
PILOT_SEED_START = 710_000
PILOT_EPISODES = 60
SMOKE_EPISODES = 4
PROTECTED_CONFIRMATION_START = 810_000
INITIAL_WEIGHTS = {MODEL_FIXED: 0.5, MODEL_ARD: 0.5}
PREQUENTIAL_LEARNING_RATE = 0.2
MINIMUM_WEIGHT = 0.08
PREDICTIVE_STD_FLOOR = 1.0
ARD_EVIDENCE_THRESHOLD = 2.0 / 3.0
PILOT_GATE = {
    "minimum_initial_delta_vs_fixed": -0.5,
    "minimum_best_delta_vs_fixed": -0.5,
    "maximum_t95_delta_vs_fixed": 0.0,
    "minimum_auc_delta_vs_fixed": 0.0,
    "require_lower_loss_frequency_than_pure_ard": True,
    "require_lower_loss_count_variance_than_pure_ard": True,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_pilot_episodes() -> list[dict]:
    frame = pd.read_csv(TRAIN_FILE)
    episodes = suzuki_proxy._phase_random_episodes(
        frame,
        phase="phase4_pilot",
        seed_start=PILOT_SEED_START,
        count=PILOT_EPISODES,
    )
    if len(episodes) != PILOT_EPISODES:
        raise RuntimeError("Phase-4 pilot episode count changed.")
    split_seeds = [int(item["split_seed"]) for item in episodes]
    if len(set(split_seeds)) != PILOT_EPISODES:
        raise RuntimeError("Phase-4 pilot split seeds are not unique.")
    if min(split_seeds) < PILOT_SEED_START:
        raise RuntimeError("Phase-4 pilot escaped its 710k seed family.")
    if max(split_seeds) >= PROTECTED_CONFIRMATION_START:
        raise RuntimeError("Phase-4 pilot entered the protected confirmation range.")
    return episodes


def variants_for_stage(stage: str, candidate: str | None) -> tuple[str, ...]:
    if stage == "smoke":
        if candidate is not None:
            raise ValueError("Smoke compares both committee candidates.")
        return SMOKE_VARIANTS
    if candidate not in COMMITTEE_VARIANTS:
        raise ValueError("Pilot requires exactly one frozen committee candidate.")
    return (FIXED, PURE_ARD, candidate)


def episodes_for_stage(stage: str) -> list[dict]:
    episodes = make_pilot_episodes()
    return episodes[:SMOKE_EPISODES] if stage == "smoke" else episodes


def build_problem(episode: dict, variant: str):
    config, dataset, lookup = suzuki_proxy.build_problem(episode, "gp_only")
    budget = int(episode["budget"])
    if variant == FIXED:
        enabled = (MODEL_FIXED,)
        weights = {MODEL_FIXED: 1.0}
    elif variant == PURE_ARD:
        enabled = (MODEL_ARD,)
        weights = {MODEL_ARD: 1.0}
    elif variant in COMMITTEE_VARIANTS:
        enabled = (MODEL_FIXED, MODEL_ARD)
        weights = INITIAL_WEIGHTS
    else:
        raise ValueError(f"Unknown variant: {variant}")
    config = replace(
        config,
        gp=replace(config.gp, target_transform="standard"),
        ensemble=replace(
            config.ensemble,
            enabled_models=enabled,
            initial_weights=weights,
            prequential_learning_rate=PREQUENTIAL_LEARNING_RATE,
            minimum_weight=MINIMUM_WEIGHT,
            predictive_std_floor=PREDICTIVE_STD_FLOOR,
        ),
        policy=replace(
            config.policy,
            first_query="posterior_mean",
            early_end=budget,
            final_start=budget + 1,
            challenge_every=0,
            novelty_disagreement_mode="off",
        ),
    )
    return config, dataset, lookup


def _acquisition_scores(
    mean: np.ndarray,
    std: np.ndarray,
    *,
    best_f: float,
    step: int,
    budget: int,
    xi_initial: float,
    xi_final: float,
) -> tuple[np.ndarray, str]:
    if step == 1:
        return np.asarray(mean, dtype=float).copy(), "posterior_mean"
    progress = (step - 1) / (budget - 1) if budget > 1 else 1.0
    xi = xi_initial + progress * (xi_final - xi_initial)
    return (
        stable_log_expected_improvement(mean, std, best_f, xi=xi),
        "stable_log_ei",
    )


class EvidenceDisagreementGate:
    """Use mixture scoring, anchored to fixed GP under weak disagreement evidence."""

    def __init__(self, config, *, seed: int) -> None:
        self.config = config
        self.seed = int(seed)
        self.last_decision: dict[str, object] | None = None

    def select(
        self,
        posterior: Posterior,
        candidate_ids: np.ndarray,
        *,
        best_f: float,
        step: int,
        budget: int,
        prior_scores: np.ndarray | None = None,
        novelty_scores: np.ndarray | None = None,
    ) -> Recommendation:
        del prior_scores, novelty_scores
        posterior.validate(len(candidate_ids))
        if set(posterior.component_means) != {MODEL_FIXED, MODEL_ARD}:
            raise ValueError("The disagreement gate requires fixed and ARD posteriors.")
        candidate_ids = np.asarray(candidate_ids, dtype=int)
        score_args = {
            "best_f": best_f,
            "step": step,
            "budget": budget,
            "xi_initial": float(self.config.xi_initial),
            "xi_final": float(self.config.xi_final),
        }
        mixture_scores, acquisition_name = _acquisition_scores(
            posterior.mean, posterior.std, **score_args
        )
        fixed_scores, _ = _acquisition_scores(
            posterior.component_means[MODEL_FIXED],
            posterior.component_stds[MODEL_FIXED],
            **score_args,
        )
        ard_scores, _ = _acquisition_scores(
            posterior.component_means[MODEL_ARD],
            posterior.component_stds[MODEL_ARD],
            **score_args,
        )
        fixed_index = deterministic_argmax(
            fixed_scores, candidate_ids, seed=self.seed + step
        )
        ard_index = deterministic_argmax(
            ard_scores, candidate_ids, seed=self.seed + step
        )
        mixture_index = deterministic_argmax(
            mixture_scores, candidate_ids, seed=self.seed + step
        )
        ard_weight = float(posterior.weights[MODEL_ARD])
        if fixed_index == ard_index:
            selected_index = fixed_index
            gate_reason = "expert_agreement"
        elif mixture_index == fixed_index:
            selected_index = mixture_index
            gate_reason = "mixture_matches_fixed"
        elif ard_weight >= ARD_EVIDENCE_THRESHOLD:
            selected_index = mixture_index
            gate_reason = "ard_prequential_evidence"
        else:
            selected_index = fixed_index
            gate_reason = "fixed_fallback"
        self.last_decision = {
            "step": step,
            "pre_observation_only": True,
            "gate_reason": gate_reason,
            "fixed_query_index": int(candidate_ids[fixed_index]),
            "ard_query_index": int(candidate_ids[ard_index]),
            "mixture_query_index": int(candidate_ids[mixture_index]),
            "selected_query_index": int(candidate_ids[selected_index]),
            "ard_weight_before_query": ard_weight,
            "ard_evidence_threshold": ARD_EVIDENCE_THRESHOLD,
            "selected_disagreement": float(posterior.disagreement[selected_index]),
        }
        component_means = {
            name: float(values[selected_index])
            for name, values in posterior.component_means.items()
        }
        component_stds = {
            name: float(values[selected_index])
            for name, values in posterior.component_stds.items()
        }
        return Recommendation(
            local_index=selected_index,
            query_index=int(candidate_ids[selected_index]),
            acquisition_name=f"{acquisition_name}_{gate_reason}",
            acquisition_value=float(mixture_scores[selected_index]),
            predicted_mean=float(posterior.mean[selected_index]),
            predicted_std=float(posterior.std[selected_index]),
            component_means=component_means,
            component_stds=component_stds,
            model_weights=dict(posterior.weights),
            prior_weight=0.0,
        )


def campaign_metrics(trajectory: list[dict], global_best: float, budget: int) -> dict:
    raw = compute_campaign_metrics(
        [float(record["observed_yield"]) for record in trajectory],
        global_best=global_best,
    ).to_dict()
    t95 = raw["round_to_95_global_best"]
    return {
        "initial_round_found_best": float(raw["initial_round_found_best"]),
        "best_found": float(raw["best_found"]),
        "round_to_95_global_best": budget + 1 if t95 is None else int(t95),
        "t95_reached": t95 is not None,
        "AUC_best_so_far": float(raw["auc_best_so_far"]),
    }


def run_one(job: tuple[str, dict, str, str]) -> dict:
    variant, episode, stage, output_root_string = job
    config, dataset, lookup = build_problem(episode, variant)
    seed = int(episode["campaign_seed"])
    oracle = OfflineTableOracle(dataset=dataset, lookup=lookup)
    runner = CampaignRunner(dataset, config, seed=seed, backend="native")
    if variant == GATED:
        runner.policy = EvidenceDisagreementGate(config.policy, seed=seed)
    session = runner.start_session(budget=int(episode["budget"]))
    started = time.perf_counter()
    gate_decisions: list[dict] = []
    while not session.is_complete:
        recommendation = session.recommend()
        decision = getattr(runner.policy, "last_decision", None)
        observed_yield = float(oracle.query(recommendation.query_index))
        record = session.observe(recommendation.query_index, observed_yield)
        if decision is not None:
            gate_decisions.append(dict(decision))
            record["gate_decision"] = dict(decision)
    result = session.finalize(global_best=oracle.global_best)
    elapsed = time.perf_counter() - started
    payload = result.to_payload()
    global_best = float(oracle.global_best)
    metrics = campaign_metrics(payload["trajectory"], global_best, session.budget)
    weight_trajectory = [
        {
            "step": int(record["step"]),
            "before": dict(record["model_weights_before_query"]),
            "after": dict(record["model_weights_after_query"]),
        }
        for record in payload["trajectory"]
    ]
    payload["committee_metrics"] = metrics
    payload["committee_contract"] = {
        "variant": variant,
        "stage": stage,
        "train_only": True,
        "public_test_labels_used": False,
        "input_file": str(TRAIN_FILE.relative_to(DATASET_ROOT)),
        "input_sha256": file_sha256(TRAIN_FILE),
        "candidate_information_identical": True,
        "target_transform": "standard",
        "acquisition": "posterior mean query 1, stable LogEI queries 2-5",
        "prequential_predictions_recorded_before_observation": True,
        "episode": episode,
    }
    output_path = (
        Path(output_root_string)
        / "trajectories"
        / stage
        / variant
        / f"{episode['episode']}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {
        "dataset": "Suzuki",
        "stage": stage,
        "phase": episode["phase"],
        "episode_family": episode["episode_family"],
        "episode": episode["episode"],
        "split_seed": episode["split_seed"],
        "campaign_seed": episode["campaign_seed"],
        "variant": variant,
        "budget": episode["budget"],
        "global_best": global_best,
        **metrics,
        "elapsed_seconds": elapsed,
        "weight_trajectory_json": json.dumps(weight_trajectory, sort_keys=True),
        "gate_decisions_json": (
            json.dumps(gate_decisions, sort_keys=True) if gate_decisions else None
        ),
    }


def mean_ci(values: pd.Series) -> tuple[float, float, float, float]:
    array = values.to_numpy(dtype=float)
    mean = float(array.mean())
    if len(array) == 1:
        return mean, 0.0, mean, mean
    sd = float(array.std(ddof=1))
    half = float(t.ppf(0.975, len(array) - 1) * sd / np.sqrt(len(array)))
    return mean, sd, mean - half, mean + half


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    for (stage, variant), group in rows.groupby(["stage", "variant"], sort=True):
        for metric in SUMMARY_METRICS:
            mean, sd, low, high = mean_ci(group[metric])
            records.append(
                {
                    "dataset": "Suzuki",
                    "stage": stage,
                    "variant": variant,
                    "metric": metric,
                    "n_episodes": len(group),
                    "mean": mean,
                    "sample_sd": sd,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    return pd.DataFrame(records)


def paired_summary(rows: pd.DataFrame, baseline_name: str) -> pd.DataFrame:
    records: list[dict] = []
    for stage, section in rows.groupby("stage", sort=True):
        baseline = section[section["variant"].eq(baseline_name)].set_index("episode")
        for variant, candidate in section.groupby("variant", sort=True):
            if variant == baseline_name:
                continue
            candidate = candidate.set_index("episode")
            common = baseline.index.intersection(candidate.index)
            for metric in SUMMARY_METRICS:
                delta = candidate.loc[common, metric] - baseline.loc[common, metric]
                mean, sd, low, high = mean_ci(delta)
                higher = metric not in {"round_to_95_global_best", "elapsed_seconds"}
                wins = int((delta > 1e-10).sum()) if higher else int((delta < -1e-10).sum())
                losses = int((delta < -1e-10).sum()) if higher else int((delta > 1e-10).sum())
                records.append(
                    {
                        "dataset": "Suzuki",
                        "stage": stage,
                        "candidate": variant,
                        "baseline": baseline_name,
                        "metric": metric,
                        "n_pairs": len(common),
                        "mean_delta": mean,
                        "sample_sd_delta": sd,
                        "ci95_low": low,
                        "ci95_high": high,
                        "wins": wins,
                        "ties": len(common) - wins - losses,
                        "losses": losses,
                    }
                )
    return pd.DataFrame(records)


def risk_summary(rows: pd.DataFrame, baseline_name: str = FIXED) -> pd.DataFrame:
    records: list[dict] = []
    baseline = rows[rows["variant"].eq(baseline_name)].set_index("episode")
    for variant, candidate in rows.groupby("variant", sort=True):
        if variant == baseline_name:
            continue
        candidate = candidate.set_index("episode")
        common = baseline.index.intersection(candidate.index)
        losses = pd.DataFrame(index=common)
        for metric in CORE_METRICS:
            delta = candidate.loc[common, metric] - baseline.loc[common, metric]
            losses[metric] = delta > 1e-10 if metric == "round_to_95_global_best" else delta < -1e-10
        loss_count = losses.sum(axis=1).astype(float)
        records.append(
            {
                "dataset": "Suzuki",
                "stage": str(rows["stage"].iloc[0]),
                "variant": variant,
                "baseline": baseline_name,
                "n_pairs": len(common),
                "n_metric_comparisons": len(common) * len(CORE_METRICS),
                "loss_frequency": float(losses.to_numpy().mean()),
                "any_loss_frequency": float((loss_count > 0).mean()),
                "mean_loss_count": float(loss_count.mean()),
                "sample_variance_loss_count": float(loss_count.var(ddof=1)),
            }
        )
    return pd.DataFrame(records)


def weight_rows(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    for row in rows.itertuples(index=False):
        for item in json.loads(row.weight_trajectory_json):
            for moment in ("before", "after"):
                for model, weight in item[moment].items():
                    records.append(
                        {
                            "dataset": "Suzuki",
                            "stage": row.stage,
                            "episode": row.episode,
                            "variant": row.variant,
                            "step": item["step"],
                            "moment": moment,
                            "model": model,
                            "weight": float(weight),
                        }
                    )
    return pd.DataFrame(records)


def weight_summary(weights: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    for keys, group in weights.groupby(
        ["stage", "variant", "step", "moment", "model"], sort=True
    ):
        mean, sd, low, high = mean_ci(group["weight"])
        records.append(
            {
                "dataset": "Suzuki",
                **dict(zip(("stage", "variant", "step", "moment", "model"), keys, strict=True)),
                "n_episodes": len(group),
                "mean": mean,
                "sample_sd": sd,
                "ci95_low": low,
                "ci95_high": high,
                "minimum": float(group["weight"].min()),
                "maximum": float(group["weight"].max()),
            }
        )
    return pd.DataFrame(records)


def gate_decision_rows(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    for row in rows.dropna(subset=["gate_decisions_json"]).itertuples(index=False):
        for decision in json.loads(row.gate_decisions_json):
            records.append(
                {
                    "dataset": "Suzuki",
                    "stage": row.stage,
                    "episode": row.episode,
                    "variant": row.variant,
                    **decision,
                }
            )
    return pd.DataFrame(records)


def select_smoke_candidate(rows: pd.DataFrame) -> dict[str, object]:
    means = rows.groupby("variant")["AUC_best_so_far"].mean()
    gates = gate_decision_rows(rows)
    active = bool(
        gates["gate_reason"].isin(
            ["fixed_fallback", "ard_prequential_evidence"]
        ).any()
    )
    weights = weight_rows(rows[rows["variant"].isin(COMMITTEE_VARIANTS)])
    before = weights[weights["moment"].eq("before")]
    after = weights[weights["moment"].eq("after")]
    merged = before.merge(
        after,
        on=["stage", "episode", "variant", "step", "model"],
        suffixes=("_before", "_after"),
    )
    weights_moved = bool(
        (merged["weight_after"] - merged["weight_before"]).abs().gt(1e-12).any()
    )
    selected = GATED if active and means[GATED] >= means[MIXTURE] else MIXTURE
    return {
        "selection_rule": (
            "Select disagreement_gate only if the gate activates and its smoke "
            "mean AUC is at least the plain prequential mixture; otherwise select "
            "prequential_mixture."
        ),
        "smoke_episodes_reused_from_pilot_prefix": True,
        "n_smoke_episodes": SMOKE_EPISODES,
        "mean_auc": {name: float(means[name]) for name in SMOKE_VARIANTS},
        "gate_activated": active,
        "weights_moved_after_prequential_observations": weights_moved,
        "selected_candidate": selected,
        "frozen_before_full_pilot": True,
    }


def pilot_gate(rows: pd.DataFrame, candidate_name: str) -> dict[str, object]:
    paired = paired_summary(rows, FIXED)
    candidate = paired[paired["candidate"].eq(candidate_name)].set_index("metric")
    deltas = {metric: float(candidate.loc[metric, "mean_delta"]) for metric in CORE_METRICS}
    risk = risk_summary(rows).set_index("variant")
    candidate_risk = risk.loc[candidate_name]
    ard_risk = risk.loc[PURE_ARD]
    checks = {
        "initial_nonmaterial": deltas["initial_round_found_best"]
        >= PILOT_GATE["minimum_initial_delta_vs_fixed"],
        "best_nonmaterial": deltas["best_found"]
        >= PILOT_GATE["minimum_best_delta_vs_fixed"],
        "t95_improves": deltas["round_to_95_global_best"]
        < PILOT_GATE["maximum_t95_delta_vs_fixed"] + 1e-12,
        "auc_improves": deltas["AUC_best_so_far"]
        > PILOT_GATE["minimum_auc_delta_vs_fixed"],
        "lower_loss_frequency_than_pure_ard": float(candidate_risk["loss_frequency"])
        < float(ard_risk["loss_frequency"]),
        "lower_loss_count_variance_than_pure_ard": float(
            candidate_risk["sample_variance_loss_count"]
        )
        < float(ard_risk["sample_variance_loss_count"]),
    }
    return {
        "gate": PILOT_GATE,
        "candidate": candidate_name,
        "mean_deltas_vs_fixed": deltas,
        "candidate_risk_vs_fixed": candidate_risk.to_dict(),
        "pure_ard_risk_vs_fixed": ard_risk.to_dict(),
        "checks": checks,
        "passed": bool(all(checks.values())),
        "confirmation_run": False,
        "production_default_changed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "pilot"), required=True)
    parser.add_argument("--candidate", choices=COMMITTEE_VARIANTS)
    parser.add_argument("--jobs", type=int, default=min(6, os.cpu_count() or 1))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.jobs <= 0:
        raise ValueError("--jobs must be positive.")
    variants = variants_for_stage(args.stage, args.candidate)
    episodes = episodes_for_stage(args.stage)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "stage": args.stage,
        "train_only": True,
        "public_test_labels_used": False,
        "input": {
            "path": str(TRAIN_FILE.relative_to(DATASET_ROOT)),
            "sha256": file_sha256(TRAIN_FILE),
        },
        "pilot_seed_start": PILOT_SEED_START,
        "episode_count": len(episodes),
        "protected_confirmation_seed_start": PROTECTED_CONFIRMATION_START,
        "protected_confirmation_run": False,
        "prior_610k_results_used": False,
        "variants": list(variants),
        "selected_candidate": args.candidate,
        "committee": {
            "models": [MODEL_FIXED, MODEL_ARD],
            "initial_weights": INITIAL_WEIGHTS,
            "prequential_learning_rate": PREQUENTIAL_LEARNING_RATE,
            "minimum_weight": MINIMUM_WEIGHT,
            "predictive_std_floor": PREDICTIVE_STD_FLOOR,
            "ard_evidence_threshold": ARD_EVIDENCE_THRESHOLD,
            "weights_use_pre_observation_predictions_only": True,
        },
        "pilot_gate": PILOT_GATE,
        "target_transform": "standard",
        "acquisition": "posterior mean query 1, stable LogEI queries 2-5",
        "production_default_changed": False,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    jobs = [
        (variant, episode, args.stage, str(args.output_dir))
        for episode in episodes
        for variant in variants
    ]
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(run_one, job): job for job in jobs}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"{row['stage']} {row['variant']} {row['episode']}: "
                f"AUC={row['AUC_best_so_far']:.3f}",
                flush=True,
            )
    detail = pd.DataFrame(rows).sort_values(["episode", "variant"])
    detail.to_csv(args.output_dir / "per_episode.csv", index=False)
    summarize(detail).to_csv(args.output_dir / "summary.csv", index=False)
    paired_summary(detail, FIXED).to_csv(
        args.output_dir / "paired_vs_fixed.csv", index=False
    )
    weights = weight_rows(detail)
    weights.to_csv(args.output_dir / "weight_trajectories.csv", index=False)
    weight_summary(weights).to_csv(
        args.output_dir / "weight_summary.csv", index=False
    )
    gates = gate_decision_rows(detail)
    if not gates.empty:
        gates.to_csv(args.output_dir / "gate_decisions.csv", index=False)
    if args.stage == "smoke":
        decision = select_smoke_candidate(detail)
        (args.output_dir / "smoke_selection.json").write_text(
            json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8"
        )
    else:
        paired_summary(detail, PURE_ARD).query(
            "candidate == @args.candidate"
        ).to_csv(
            args.output_dir / "paired_candidate_vs_pure_ard.csv", index=False
        )
        risk_summary(detail).to_csv(args.output_dir / "risk_summary.csv", index=False)
        decision = pilot_gate(detail, args.candidate)
        (args.output_dir / "pilot_gate.json").write_text(
            json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8"
        )
        first_ids = {f"random_phase4_pilot_{index:02d}" for index in range(20)}
        twenty = detail[detail["episode"].isin(first_ids)].copy()
        if len(twenty) != 20 * len(variants) or twenty["episode"].nunique() != 20:
            raise RuntimeError("The explicit first-20 pilot view is incomplete.")
        twenty.to_csv(args.output_dir / "twenty_run_per_episode.csv", index=False)
        summarize(twenty).to_csv(
            args.output_dir / "twenty_run_summary.csv", index=False
        )
        paired_summary(twenty, FIXED).to_csv(
            args.output_dir / "twenty_run_paired_vs_fixed.csv", index=False
        )
        paired_summary(twenty, PURE_ARD).query(
            "candidate == @args.candidate"
        ).to_csv(
            args.output_dir / "twenty_run_paired_candidate_vs_pure_ard.csv",
            index=False,
        )


if __name__ == "__main__":
    main()
