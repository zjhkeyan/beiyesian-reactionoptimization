#!/usr/bin/env python3
"""Learn Suzuki component amplitudes on 30 episodes and validate on 30 others."""

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
from bgolearn_rxn.datasets import DatasetSpec, ReactionDataset  # noqa: E402
from bgolearn_rxn.factory import prepare_problem  # noqa: E402
from bgolearn_rxn.metrics import compute_campaign_metrics  # noqa: E402
from bgolearn_rxn.oracle import OfflineTableOracle  # noqa: E402
from bgolearn_rxn.surrogates.anova_gp import StructuredANOVAGP  # noqa: E402
from bgolearn_rxn.surrogates.component_ard_gp import (  # noqa: E402
    ComponentARDStructuredGP,
)
from bgolearn_rxn.utils import deterministic_argmax  # noqa: E402


DEVELOPMENT_VARIANT = "development_ard"
VARIANTS = ("fixed", "pure_ard", "distilled")
BASELINE = "fixed"
PURE_ARD = "pure_ard"
CANDIDATE = "distilled"
CORE_METRICS = (
    "initial_round_found_best",
    "best_found",
    "round_to_95_global_best",
    "AUC_best_so_far",
)
SUMMARY_METRICS = CORE_METRICS + ("elapsed_seconds",)
TRAIN_FILE = DATASET_ROOT / "Suzuki" / "suzuki_train.csv"
CONFIG_FILE = EXTENSION_ROOT / "configs" / "suzuki.yaml"
SEED_FAMILY_START = 710_000
PROTECTED_PHASE3_START = 610_000
PROTECTED_CONFIRMATION_START = 810_000
TOTAL_EPISODES = 60
DEVELOPMENT_COUNT = 30
VALIDATION_COUNT = 30
PILOT_GATE = {
    "minimum_initial_delta_vs_fixed": 0.0,
    "minimum_best_delta_vs_fixed": 0.0,
    "minimum_auc_delta_vs_fixed": 0.0,
    "maximum_t95_delta_vs_fixed": 0.0,
    "require_lower_paired_auc_variance_than_pure_ard": True,
    "maximum_runtime_ratio_vs_fixed": 1.25,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_episode_partitions() -> tuple[list[dict], list[dict]]:
    frame = pd.read_csv(TRAIN_FILE)
    episodes = suzuki_proxy._phase_random_episodes(
        frame,
        phase="amplitude_distillation",
        seed_start=SEED_FAMILY_START,
        count=TOTAL_EPISODES,
    )
    development = episodes[:DEVELOPMENT_COUNT]
    validation = episodes[DEVELOPMENT_COUNT:]
    if len(development) != DEVELOPMENT_COUNT or len(validation) != VALIDATION_COUNT:
        raise RuntimeError("Amplitude-distillation partition sizes changed.")
    development_ids = {item["episode"] for item in development}
    validation_ids = {item["episode"] for item in validation}
    if development_ids.intersection(validation_ids):
        raise RuntimeError("Development and validation episode IDs overlap.")
    split_seeds = {int(item["split_seed"]) for item in episodes}
    if len(split_seeds) != TOTAL_EPISODES:
        raise RuntimeError("Amplitude-distillation split seeds are not unique.")
    if min(split_seeds) < SEED_FAMILY_START:
        raise RuntimeError("Episode generation escaped the 710k seed family.")
    if max(split_seeds) >= PROTECTED_CONFIRMATION_START:
        raise RuntimeError("Episode generation entered the protected 810k range.")
    return development, validation


def _copy_dataset(dataset: ReactionDataset) -> ReactionDataset:
    return ReactionDataset(
        spec=DatasetSpec(
            name=dataset.spec.name,
            feature_columns=dataset.spec.feature_columns,
            target_column=dataset.spec.target_column,
            task_column=dataset.spec.task_column,
            target_task_value=dataset.spec.target_task_value,
            condition_roles=dataset.spec.condition_roles,
            derived_roles=dict(dataset.spec.derived_roles),
            fixed_roles=dict(dataset.spec.fixed_roles),
            budget=dataset.spec.budget,
            seeds=dataset.spec.seeds,
        ),
        initial=dataset.initial.copy(),
        candidates=dataset.candidates.copy(),
    )


def build_problem(episode: dict):
    config, dataset, lookup = suzuki_proxy.build_problem(episode, "gp_only")
    budget = int(episode["budget"])
    config = replace(
        config,
        gp=replace(config.gp, target_transform="standard"),
        policy=replace(
            config.policy,
            early_end=budget,
            final_start=budget + 1,
            challenge_every=0,
            novelty_disagreement_mode="off",
        ),
    )
    return config, _copy_dataset(dataset), lookup.copy()


class FrozenAmplitudeStructuredGP(StructuredANOVAGP):
    """Structured GP with externally learned, immutable component amplitudes."""

    name = "distilled_gp"

    def __init__(self, schema, config, component_amplitudes: dict[str, float], *, seed=0):
        super().__init__(schema, config, seed=seed)
        expected = {"bias"}
        expected.update(f"role:{role}" for role in self.roles)
        expected.update(
            f"pair:{'|'.join(pair)}" for pair in self.config.interaction_pairs
        )
        if set(component_amplitudes) != expected:
            missing = sorted(expected - set(component_amplitudes))
            extra = sorted(set(component_amplitudes) - expected)
            raise ValueError(f"Frozen amplitude keys differ: missing={missing}, extra={extra}")
        values = {key: float(value) for key, value in component_amplitudes.items()}
        if not all(np.isfinite(value) and value > 0.0 for value in values.values()):
            raise ValueError("Frozen component amplitudes must be finite and positive.")
        total = float(sum(values.values()))
        self.component_amplitudes_ = values
        self.component_shares_ = {key: value / total for key, value in values.items()}

    def _initial_theta(self) -> np.ndarray:
        theta = super()._initial_theta()
        theta[self.layout.pair_scale_index] = 0.0
        return theta

    def _bounds(self) -> list[tuple[float, float]]:
        bounds = super()._bounds()
        bounds[self.layout.pair_scale_index] = (0.0, 0.0)
        return bounds

    def _kernel_from_distances(
        self, distances, same_task, task_similarity, theta
    ) -> np.ndarray:
        lengthscales, _, _, _, task_rho = self._unpack(theta)
        role_kernels = {
            role: np.exp(-0.5 * distances[role] / (lengthscales[role] ** 2))
            for role in self.roles
        }
        bias = self.component_amplitudes_["bias"]
        kernel = np.full_like(next(iter(role_kernels.values())), bias, dtype=float)
        normalizer = bias
        for role in self.roles:
            weight = self.component_amplitudes_[f"role:{role}"]
            kernel += weight * role_kernels[role]
            normalizer += weight
        for pair in self.config.interaction_pairs:
            weight = self.component_amplitudes_[f"pair:{'|'.join(pair)}"]
            kernel += weight * role_kernels[pair[0]] * role_kernels[pair[1]]
            normalizer += weight
        kernel /= normalizer
        if self.schema.task_role is not None:
            kernel *= task_rho * task_similarity + (1.0 - task_rho) * same_task
        return kernel


def make_model(variant: str, schema, config, seed: int, frozen: dict | None):
    if variant == BASELINE:
        return StructuredANOVAGP(schema, config.gp, seed=seed + 11)
    if variant in {PURE_ARD, DEVELOPMENT_VARIANT}:
        return ComponentARDStructuredGP(schema, config.gp, seed=seed + 17)
    if variant == CANDIDATE:
        if frozen is None:
            raise ValueError("Distilled validation requires frozen amplitudes.")
        return FrozenAmplitudeStructuredGP(
            schema, config.gp, frozen, seed=seed + 23
        )
    raise ValueError(f"Unknown variant: {variant}")


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


def run_one(job: tuple[str, dict, str, dict | None]) -> tuple[dict, list[dict]]:
    variant, episode, output_root_string, frozen = job
    config, dataset, lookup = build_problem(episode)
    prepared = prepare_problem(dataset, config)
    seed = int(episode["campaign_seed"])
    model = make_model(variant, prepared.encoder.schema, config, seed, frozen)
    oracle = OfflineTableOracle(dataset=dataset, lookup=lookup)
    observed = dataset.initial.copy()
    X_observed = prepared.X_initial.copy()
    y_observed = observed[dataset.spec.target_column].to_numpy(dtype=float)
    remaining = np.arange(len(dataset.candidates), dtype=int)
    trajectory: list[dict] = []
    fit_rows: list[dict] = []
    budget = int(episode["budget"])
    started = time.perf_counter()

    for step in range(1, budget + 1):
        X_remaining = prepared.X_candidates[remaining]
        candidate_rows = dataset.candidates.iloc[remaining]
        model.fit(X_observed, y_observed)
        amplitudes = dict(getattr(model, "component_amplitudes_", {}))
        shares = dict(getattr(model, "component_shares_", {}))
        if variant == DEVELOPMENT_VARIANT:
            for component, amplitude in amplitudes.items():
                fit_rows.append(
                    {
                        "episode": episode["episode"],
                        "split_seed": episode["split_seed"],
                        "campaign_seed": episode["campaign_seed"],
                        "step": step,
                        "component": component,
                        "amplitude": amplitude,
                        "share": shares[component],
                    }
                )
        posterior = model.predict(X_remaining)
        if step == 1:
            scores = posterior.mean.copy()
            acquisition_name = "posterior_mean"
        else:
            progress = (step - 1) / (budget - 1)
            scores = stable_log_expected_improvement(
                posterior.mean,
                posterior.std,
                dataset.target_incumbent(observed),
                xi=0.05 * (1.0 - progress),
            )
            acquisition_name = "stable_log_ei"
        candidate_ids = candidate_rows["query_index"].to_numpy(dtype=int)
        local_index = deterministic_argmax(scores, candidate_ids, seed=seed + step)
        query_index = int(candidate_ids[local_index])
        observed_yield = float(oracle.query(query_index))
        selected_global = int(remaining[local_index])
        selected_row = dataset.candidates.iloc[selected_global]
        trajectory.append(
            {
                "step": step,
                "query_index": query_index,
                "observed_yield": observed_yield,
                "predicted_yield": float(posterior.mean[local_index]),
                "predicted_std": float(posterior.std[local_index]),
                "model_family": model.name,
                "acquisition_name": acquisition_name,
                "acquisition_value": float(scores[local_index]),
                "component_amplitudes": amplitudes or None,
                "component_shares": shares or None,
            }
        )
        new_observation = selected_row.loc[
            list(dataset.spec.feature_columns)
        ].to_dict()
        new_observation[dataset.spec.target_column] = observed_yield
        observed = pd.concat(
            [observed, pd.DataFrame([new_observation])], ignore_index=True
        )
        X_observed = np.vstack([X_observed, prepared.X_candidates[selected_global]])
        y_observed = np.append(y_observed, observed_yield)
        remaining = np.delete(remaining, local_index)

    elapsed = time.perf_counter() - started
    global_best = float(oracle.global_best)
    metrics = campaign_metrics(trajectory, global_best, budget)
    payload = {
        "dataset": "Suzuki",
        "variant": variant,
        "episode": episode,
        "trajectory": trajectory,
        "metrics": metrics,
        "contract": {
            "train_only": True,
            "public_test_labels_used": False,
            "input_file": str(TRAIN_FILE.relative_to(DATASET_ROOT)),
            "input_sha256": file_sha256(TRAIN_FILE),
            "config_file": str(CONFIG_FILE.relative_to(DATASET_ROOT)),
            "config_sha256": file_sha256(CONFIG_FILE),
            "target_transform": "standard",
            "acquisition": "posterior mean query 1, stable LogEI queries 2-5",
            "component_amplitudes_frozen": variant == CANDIDATE,
            "episode": episode,
        },
    }
    output_path = (
        Path(output_root_string)
        / "trajectories"
        / variant
        / f"{episode['episode']}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return (
        {
            "dataset": "Suzuki",
            "phase": (
                "amplitude_development"
                if variant == DEVELOPMENT_VARIANT
                else "amplitude_validation"
            ),
            "episode_family": episode["episode_family"],
            "episode": episode["episode"],
            "split_seed": episode["split_seed"],
            "campaign_seed": episode["campaign_seed"],
            "variant": variant,
            "model": model.name,
            "budget": budget,
            "global_best": global_best,
            **metrics,
            "elapsed_seconds": elapsed,
            "final_component_amplitudes_json": (
                json.dumps(getattr(model, "component_amplitudes_", {}), sort_keys=True)
                or None
            ),
        },
        fit_rows,
    )


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
    for variant, group in rows.groupby("variant", sort=True):
        for metric in SUMMARY_METRICS:
            mean, sd, low, high = mean_ci(group[metric])
            records.append(
                {
                    "dataset": "Suzuki",
                    "phase": group["phase"].iloc[0],
                    "episode_family": "random_holdout",
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


def paired_summary(rows: pd.DataFrame, candidate_name: str, baseline_name: str) -> pd.DataFrame:
    candidate = rows[rows["variant"].eq(candidate_name)].set_index("episode")
    baseline = rows[rows["variant"].eq(baseline_name)].set_index("episode")
    common = candidate.index.intersection(baseline.index)
    records: list[dict] = []
    for metric in SUMMARY_METRICS:
        delta = candidate.loc[common, metric] - baseline.loc[common, metric]
        mean, sd, low, high = mean_ci(delta)
        higher = metric not in {"round_to_95_global_best", "elapsed_seconds"}
        wins = int((delta > 1e-10).sum()) if higher else int((delta < -1e-10).sum())
        losses = int((delta < -1e-10).sum()) if higher else int((delta > 1e-10).sum())
        records.append(
            {
                "dataset": "Suzuki",
                "phase": "amplitude_validation",
                "episode_family": "random_holdout",
                "candidate": candidate_name,
                "baseline": baseline_name,
                "metric": metric,
                "n_pairs": len(common),
                "mean_delta": mean,
                "sample_sd_delta": sd,
                "variance_delta": sd**2,
                "ci95_low": low,
                "ci95_high": high,
                "wins": wins,
                "ties": len(common) - wins - losses,
                "losses": losses,
            }
        )
    return pd.DataFrame(records)


def distill_weights(fit_rows: pd.DataFrame, development: list[dict]) -> dict:
    expected_ids = [item["episode"] for item in development]
    observed_ids = sorted(fit_rows["episode"].unique())
    if observed_ids != sorted(expected_ids):
        raise RuntimeError("Amplitude rows do not exactly match development episodes.")
    snapshots = fit_rows[["episode", "step"]].drop_duplicates()
    if len(snapshots) != DEVELOPMENT_COUNT * 5:
        raise RuntimeError("Expected five honest ARD fits per development episode.")
    counts = fit_rows.groupby(["episode", "step"])["component"].nunique()
    if counts.nunique() != 1:
        raise RuntimeError("Development fit snapshots have inconsistent components.")
    if not np.isfinite(fit_rows["amplitude"]).all() or not (fit_rows["amplitude"] > 0).all():
        raise RuntimeError("Development amplitudes must be finite and positive.")
    log_means = fit_rows.groupby("component")["amplitude"].apply(
        lambda values: float(np.log(values.to_numpy(dtype=float)).mean())
    )
    amplitudes = {key: float(np.exp(value)) for key, value in log_means.items()}
    total = float(sum(amplitudes.values()))
    shares = {key: value / total for key, value in amplitudes.items()}
    return {
        "method": "log-space mean of all honest sequential ARD fit amplitudes",
        "aggregation_predeclared": True,
        "n_development_episodes": DEVELOPMENT_COUNT,
        "fits_per_episode": 5,
        "n_fit_snapshots": len(snapshots),
        "n_component_observations": len(fit_rows),
        "development_episode_ids": expected_ids,
        "development_split_seeds": [int(item["split_seed"]) for item in development],
        "validation_episode_ids_used_for_learning": [],
        "source_seed_family_start": SEED_FAMILY_START,
        "protected_seed_families_not_used": [
            PROTECTED_PHASE3_START,
            PROTECTED_CONFIRMATION_START,
        ],
        "component_amplitudes": amplitudes,
        "component_shares": shares,
        "input": {
            "path": str(TRAIN_FILE.relative_to(DATASET_ROOT)),
            "sha256": file_sha256(TRAIN_FILE),
        },
        "config": {
            "path": str(CONFIG_FILE.relative_to(DATASET_ROOT)),
            "sha256": file_sha256(CONFIG_FILE),
        },
        "runner_sha256": file_sha256(Path(__file__)),
        "train_only": True,
        "public_test_labels_used": False,
        "production_default_changed": False,
    }


def validate_weight_provenance(weights: dict, development: list[dict]) -> None:
    expected_ids = [item["episode"] for item in development]
    if weights["development_episode_ids"] != expected_ids:
        raise ValueError("Frozen weights have unexpected development episode IDs.")
    if weights["development_split_seeds"] != [int(item["split_seed"]) for item in development]:
        raise ValueError("Frozen weights have unexpected development split seeds.")
    if weights["validation_episode_ids_used_for_learning"]:
        raise ValueError("Validation episodes were used for amplitude learning.")
    if weights["input"]["sha256"] != file_sha256(TRAIN_FILE):
        raise ValueError("Frozen weights use a different train table.")
    if weights["config"]["sha256"] != file_sha256(CONFIG_FILE):
        raise ValueError("Frozen weights use a different Suzuki config.")
    if weights["public_test_labels_used"] is not False:
        raise ValueError("Frozen weights provenance is not train-only.")


def pilot_decision(rows: pd.DataFrame) -> dict:
    distilled_fixed = paired_summary(rows, CANDIDATE, BASELINE).set_index("metric")
    ard_fixed = paired_summary(rows, PURE_ARD, BASELINE).set_index("metric")
    initial = float(distilled_fixed.loc["initial_round_found_best", "mean_delta"])
    best = float(distilled_fixed.loc["best_found", "mean_delta"])
    auc = float(distilled_fixed.loc["AUC_best_so_far", "mean_delta"])
    t95 = float(distilled_fixed.loc["round_to_95_global_best", "mean_delta"])
    distilled_variance = float(distilled_fixed.loc["AUC_best_so_far", "variance_delta"])
    ard_variance = float(ard_fixed.loc["AUC_best_so_far", "variance_delta"])
    runtime = rows.groupby("variant")["elapsed_seconds"].mean()
    runtime_ratio = float(runtime[CANDIDATE] / runtime[BASELINE])
    checks = {
        "initial_noninferior_vs_fixed": initial >= PILOT_GATE["minimum_initial_delta_vs_fixed"],
        "best_noninferior_vs_fixed": best >= PILOT_GATE["minimum_best_delta_vs_fixed"],
        "auc_noninferior_vs_fixed": auc >= PILOT_GATE["minimum_auc_delta_vs_fixed"],
        "t95_noninferior_vs_fixed": t95 <= PILOT_GATE["maximum_t95_delta_vs_fixed"],
        "paired_auc_variance_lower_than_pure_ard": distilled_variance < ard_variance,
        "runtime_within_1_25x_fixed": runtime_ratio <= PILOT_GATE["maximum_runtime_ratio_vs_fixed"],
    }
    return {
        "gate": PILOT_GATE,
        "checks": checks,
        "passed": bool(all(checks.values())),
        "decision": "pilot_pass" if all(checks.values()) else "reject",
        "distilled_vs_fixed_auc_delta_variance": distilled_variance,
        "pure_ard_vs_fixed_auc_delta_variance": ard_variance,
        "runtime_ratio_distilled_vs_fixed": runtime_ratio,
        "confirmation_executed": False,
        "production_default_changed": False,
        "public_test_labels_used": False,
    }


def explicit_twenty_validation(rows: pd.DataFrame) -> pd.DataFrame:
    episode_ids = {
        f"random_amplitude_distillation_{index:02d}" for index in range(30, 50)
    }
    view = rows[rows["episode"].isin(episode_ids)].copy()
    if len(view) != 20 * len(VARIANTS) or view["episode"].nunique() != 20:
        raise RuntimeError("The deterministic first-20 validation view is incomplete.")
    return view.sort_values(["episode", "variant"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("development", "validation"), required=True)
    parser.add_argument("--jobs", type=int, default=min(6, os.cpu_count() or 1))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weights-file", type=Path)
    return parser.parse_args()


def execute_jobs(
    variants: tuple[str, ...],
    episodes: list[dict],
    output_dir: Path,
    jobs_count: int,
    frozen: dict | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    jobs = [
        (variant, episode, str(output_dir), frozen)
        for episode in episodes
        for variant in variants
    ]
    rows: list[dict] = []
    fit_rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=jobs_count) as executor:
        futures = {executor.submit(run_one, job): job for job in jobs}
        for future in as_completed(futures):
            row, fitted = future.result()
            rows.append(row)
            fit_rows.extend(fitted)
            print(
                f"{row['phase']} {row['variant']} {row['episode']}: "
                f"AUC={row['AUC_best_so_far']:.3f}",
                flush=True,
            )
    detail = pd.DataFrame(rows).sort_values(["episode", "variant"])
    fits = pd.DataFrame(fit_rows)
    return detail, fits


def main() -> None:
    args = parse_args()
    if args.jobs <= 0:
        raise ValueError("--jobs must be positive.")
    development, validation = make_episode_partitions()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    common_manifest = {
        "stage": args.stage,
        "train_only": True,
        "public_test_labels_used": False,
        "input": {
            "path": str(TRAIN_FILE.relative_to(DATASET_ROOT)),
            "sha256": file_sha256(TRAIN_FILE),
        },
        "config": {
            "path": str(CONFIG_FILE.relative_to(DATASET_ROOT)),
            "sha256": file_sha256(CONFIG_FILE),
        },
        "runner_sha256": file_sha256(Path(__file__)),
        "development_episode_ids": [item["episode"] for item in development],
        "validation_episode_ids": [item["episode"] for item in validation],
        "protected_seed_families_not_used": [
            PROTECTED_PHASE3_START,
            PROTECTED_CONFIRMATION_START,
        ],
        "target_transform": "standard",
        "acquisition": "posterior mean query 1, stable LogEI queries 2-5",
        "parameter_sweep": False,
        "interim_selection": False,
        "confirmation_executed": False,
        "production_default_changed": False,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }

    if args.stage == "development":
        if args.weights_file is not None:
            raise ValueError("Development does not accept --weights-file.")
        manifest = {
            **common_manifest,
            "variants": [DEVELOPMENT_VARIANT],
            "episode_count": len(development),
            "split_seed_min": min(int(item["split_seed"]) for item in development),
            "split_seed_max": max(int(item["split_seed"]) for item in development),
            "aggregation": "log-space mean over all 150 sequential ARD fits",
        }
        (args.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        detail, fits = execute_jobs(
            (DEVELOPMENT_VARIANT,), development, args.output_dir, args.jobs, None
        )
        if len(detail) != DEVELOPMENT_COUNT or len(fits) == 0:
            raise RuntimeError("Development result count is incomplete.")
        detail.to_csv(args.output_dir / "per_episode.csv", index=False)
        fits.sort_values(["episode", "step", "component"]).to_csv(
            args.output_dir / "sequential_fit_amplitudes.csv", index=False
        )
        summarize(detail).to_csv(args.output_dir / "summary.csv", index=False)
        weights = distill_weights(fits, development)
        (args.output_dir / "learned_weights.json").write_text(
            json.dumps(weights, indent=2, sort_keys=True), encoding="utf-8"
        )
        return

    if args.weights_file is None:
        raise ValueError("Validation requires --weights-file.")
    weights = json.loads(args.weights_file.read_text(encoding="utf-8"))
    validate_weight_provenance(weights, development)
    frozen = {key: float(value) for key, value in weights["component_amplitudes"].items()}
    manifest = {
        **common_manifest,
        "variants": list(VARIANTS),
        "episode_count": len(validation),
        "split_seed_min": min(int(item["split_seed"]) for item in validation),
        "split_seed_max": max(int(item["split_seed"]) for item in validation),
        "weights_file": str(args.weights_file),
        "weights_sha256": file_sha256(args.weights_file),
        "weights_provenance_validated": True,
        "pilot_gate": PILOT_GATE,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    detail, fits = execute_jobs(VARIANTS, validation, args.output_dir, args.jobs, frozen)
    if len(detail) != VALIDATION_COUNT * len(VARIANTS) or len(fits) != 0:
        raise RuntimeError("Validation result count is incomplete.")
    if not detail.groupby("episode")["split_seed"].nunique().eq(1).all():
        raise RuntimeError("Validation variants do not share split seeds.")
    detail.to_csv(args.output_dir / "per_episode.csv", index=False)
    summarize(detail).to_csv(args.output_dir / "summary.csv", index=False)
    for candidate, baseline, filename in (
        (CANDIDATE, BASELINE, "paired_distilled_vs_fixed.csv"),
        (PURE_ARD, BASELINE, "paired_pure_ard_vs_fixed.csv"),
        (CANDIDATE, PURE_ARD, "paired_distilled_vs_pure_ard.csv"),
    ):
        paired_summary(detail, candidate, baseline).to_csv(
            args.output_dir / filename, index=False
        )
    decision = pilot_decision(detail)
    (args.output_dir / "pilot_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8"
    )
    twenty = explicit_twenty_validation(detail)
    twenty.to_csv(args.output_dir / "twenty_run_per_episode.csv", index=False)
    summarize(twenty).to_csv(args.output_dir / "twenty_run_summary.csv", index=False)
    for candidate, baseline, filename in (
        (CANDIDATE, BASELINE, "twenty_run_paired_distilled_vs_fixed.csv"),
        (PURE_ARD, BASELINE, "twenty_run_paired_pure_ard_vs_fixed.csv"),
        (CANDIDATE, PURE_ARD, "twenty_run_paired_distilled_vs_pure_ard.csv"),
    ):
        paired_summary(twenty, candidate, baseline).to_csv(
            args.output_dir / filename, index=False
        )


if __name__ == "__main__":
    main()
