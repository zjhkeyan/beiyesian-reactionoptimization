#!/usr/bin/env python3
"""Staged train-only study of observed-rank GP training precision."""

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
import run_transfer_proxy as transfer_proxy  # noqa: E402
from bgolearn_rxn.acquisition import stable_log_expected_improvement  # noqa: E402
from bgolearn_rxn.datasets import DatasetSpec, ReactionDataset  # noqa: E402
from bgolearn_rxn.factory import prepare_problem  # noqa: E402
from bgolearn_rxn.metrics import compute_campaign_metrics  # noqa: E402
from bgolearn_rxn.oracle import OfflineTableOracle  # noqa: E402
from bgolearn_rxn.surrogates.anova_gp import StructuredANOVAGP  # noqa: E402
from bgolearn_rxn.surrogates.component_ard_gp import (  # noqa: E402
    ComponentARDLowRankTaskGP,
)
from bgolearn_rxn.surrogates.rank_weighted_gp import (  # noqa: E402
    RankWeightedComponentARDLowRankTaskGP,
    RankWeightedStructuredGP,
)
from bgolearn_rxn.utils import deterministic_argmax  # noqa: E402


STRENGTHS = (0.0, 0.15, 0.30, 0.45)
BASELINE = "uniform"
CORE_METRICS = (
    "initial_round_found_best",
    "best_found",
    "round_to_95_global_best",
    "AUC_best_so_far",
)
SUMMARY_METRICS = CORE_METRICS + ("elapsed_seconds",)
OFFICIAL_BUCHWALD_PRODUCT = "N-(4-ethylphenyl)-4-methylaniline"
INPUT_FILES = {
    "Buchwald_sub4": DATASET_ROOT / "Buchwald_sub4" / "buchwald_sub4_train.csv",
    "Suzuki": DATASET_ROOT / "Suzuki" / "suzuki_train.csv",
}
GLOBAL_NOISE_BOUNDS = (1e-4, 0.75)
PILOT_GATE = {
    "minimum_combined_auc_delta_strict": 0.0,
    "minimum_per_dataset_auc_delta": 0.0,
    "minimum_initial_delta": -0.5,
    "minimum_best_delta": -0.5,
    "maximum_t95_delta": 0.25,
    "global_noise_bound_fraction_margin": 0.01,
}


def variant_name(strength: float) -> str:
    return BASELINE if strength == 0.0 else f"rank_{int(round(strength * 100)):03d}"


def variant_strength(name: str) -> float:
    mapping = {variant_name(value): value for value in STRENGTHS}
    if name not in mapping:
        raise ValueError(f"Unknown rank-weight variant: {name}")
    return mapping[name]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_auxiliary_buchwald_episodes() -> list[dict]:
    episodes: list[dict] = []
    for episode in transfer_proxy.make_buchwald_episodes(2):
        if episode["target_task"] == OFFICIAL_BUCHWALD_PRODUCT:
            continue
        item = dict(episode)
        item["phase"] = "auxiliary_selection"
        episodes.append(item)
    if len(episodes) != 8:
        raise RuntimeError("Expected eight auxiliary Buchwald selection episodes.")
    return episodes


def make_official_buchwald_episodes() -> list[dict]:
    episodes: list[dict] = []
    for episode in transfer_proxy.make_buchwald_exhaustive_pair_episodes(1):
        item = dict(episode)
        item["phase"] = "official_target_confirmation"
        if item["target_task"] != OFFICIAL_BUCHWALD_PRODUCT:
            raise ValueError("Sorted product index 1 is not the official target.")
        episodes.append(item)
    return episodes


def make_suzuki_episodes() -> list[dict]:
    frame = pd.read_csv(INPUT_FILES["Suzuki"])
    pilot = suzuki_proxy._phase_random_episodes(
        frame, phase="pilot", seed_start=310_000, count=20
    )
    confirmation = suzuki_proxy._phase_random_episodes(
        frame, phase="confirmation", seed_start=410_000, count=80
    )
    return pilot + confirmation


def episodes_for_stage(stage: str) -> list[tuple[str, dict]]:
    suzuki = make_suzuki_episodes()
    if stage == "smoke":
        return [
            ("Buchwald_sub4", make_auxiliary_buchwald_episodes()[0]),
            ("Suzuki", next(item for item in suzuki if item["phase"] == "pilot")),
        ]
    if stage == "pilot":
        output = [
            ("Buchwald_sub4", item)
            for item in make_auxiliary_buchwald_episodes()
        ]
        output.extend(
            ("Suzuki", item) for item in suzuki if item["phase"] == "pilot"
        )
        return output
    if stage == "confirmation":
        output = [
            ("Buchwald_sub4", item)
            for item in make_official_buchwald_episodes()
        ]
        output.extend(
            ("Suzuki", item)
            for item in suzuki
            if item["phase"] == "confirmation"
        )
        return output
    raise ValueError(f"Unknown stage: {stage}")


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


def build_problem(dataset_name: str, episode: dict):
    if dataset_name == "Buchwald_sub4":
        config, dataset, lookup = transfer_proxy.build_problem(
            dataset_name, "low_rank_task_gp", episode
        )
    elif dataset_name == "Suzuki":
        config, dataset, lookup = suzuki_proxy.build_problem(episode, "gp_only")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
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


def make_model(dataset_name: str, config, schema, strength: float, seed: int):
    if dataset_name == "Buchwald_sub4":
        if strength == 0.0:
            return ComponentARDLowRankTaskGP(schema, config.gp, seed=seed + 79)
        return RankWeightedComponentARDLowRankTaskGP(
            schema, config.gp, strength=strength, seed=seed + 79
        )
    if strength == 0.0:
        return StructuredANOVAGP(schema, config.gp, seed=seed + 11)
    return RankWeightedStructuredGP(
        schema, config.gp, strength=strength, seed=seed + 11
    )


def _metrics(trajectory: list[dict], global_best: float, budget: int) -> dict:
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


def _fit_diagnostics(model, y_observed: np.ndarray) -> dict[str, float | int]:
    global_noise = float(model._unpack(model.theta_)[2])
    scale = float(getattr(model.target_transform, "scale_", 1.0))
    if hasattr(model, "rank_weight_diagnostics"):
        diagnostics = dict(model.rank_weight_diagnostics())
    else:
        diagnostics = {
            "strength": 0.0,
            "n_observed": int(len(y_observed)),
            "precision_mean": 1.0,
            "precision_min": 1.0,
            "precision_max": 1.0,
            "effective_noise_min": global_noise * scale,
            "effective_noise_max": global_noise * scale,
        }
    diagnostics["global_noise"] = global_noise
    return diagnostics


def run_one(job: tuple[str, str, dict, str]) -> dict:
    dataset_name, variant, episode, output_root_string = job
    strength = variant_strength(variant)
    config, dataset, lookup = build_problem(dataset_name, episode)
    prepared = prepare_problem(dataset, config)
    seed = int(episode.get("seed", episode.get("campaign_seed")))
    model = make_model(
        dataset_name, config, prepared.encoder.schema, strength, seed
    )
    oracle = OfflineTableOracle(dataset=dataset, lookup=lookup)
    observed = dataset.initial.copy()
    X_observed = prepared.X_initial.copy()
    y_observed = observed[dataset.spec.target_column].to_numpy(dtype=float)
    remaining = np.arange(len(dataset.candidates), dtype=int)
    trajectory: list[dict] = []
    budget = int(episode["budget"])
    started = time.perf_counter()

    for step in range(1, budget + 1):
        X_remaining = prepared.X_candidates[remaining]
        candidate_rows = dataset.candidates.iloc[remaining]
        model.fit(X_observed, y_observed)
        diagnostics = _fit_diagnostics(model, y_observed)
        posterior = model.predict(X_remaining)
        if step == 1:
            scores = posterior.mean.copy()
            acquisition_name = "posterior_mean"
        else:
            progress = 0.0 if budget == 1 else (step - 1) / (budget - 1)
            scores = stable_log_expected_improvement(
                posterior.mean,
                posterior.std,
                dataset.target_incumbent(observed),
                xi=0.05 * (1.0 - progress),
            )
            acquisition_name = "stable_log_ei"
        candidate_ids = candidate_rows["query_index"].to_numpy(dtype=int)
        local_index = deterministic_argmax(
            scores, candidate_ids, seed=seed + step
        )
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
                "acquisition_name": acquisition_name,
                "acquisition_value": float(scores[local_index]),
                "training_diagnostics": diagnostics,
            }
        )
        new_observation = selected_row.loc[
            list(dataset.spec.feature_columns)
        ].to_dict()
        new_observation[dataset.spec.target_column] = observed_yield
        observed = pd.concat(
            [observed, pd.DataFrame([new_observation])], ignore_index=True
        )
        X_observed = np.vstack(
            [X_observed, prepared.X_candidates[selected_global]]
        )
        y_observed = np.append(y_observed, observed_yield)
        remaining = np.delete(remaining, local_index)

    elapsed = time.perf_counter() - started
    global_best = (
        float(oracle.global_best)
        if dataset_name == "Buchwald_sub4"
        else float(lookup[dataset.spec.target_column].max())
    )
    metrics = _metrics(trajectory, global_best, budget)
    final_diagnostics = trajectory[-1]["training_diagnostics"]
    payload = {
        "dataset": dataset_name,
        "variant": variant,
        "strength": strength,
        "episode": episode,
        "trajectory": trajectory,
        "metrics": metrics,
        "contract": {
            "train_only": True,
            "public_test_labels_used": False,
            "candidate_labels_used_in_weighting": False,
            "weight_inputs": "already-observed responses only",
            "input_file": str(INPUT_FILES[dataset_name].relative_to(DATASET_ROOT)),
            "input_sha256": file_sha256(INPUT_FILES[dataset_name]),
            "target_transform": "standard",
            "episode": episode,
        },
    }
    output_path = (
        Path(output_root_string)
        / "trajectories"
        / dataset_name
        / episode["phase"]
        / variant
        / f"{episode['episode']}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {
        "dataset": dataset_name,
        "phase": episode["phase"],
        "episode_family": episode["episode_family"],
        "episode": episode["episode"],
        "variant": variant,
        "strength": strength,
        "budget": budget,
        "global_best": global_best,
        **metrics,
        "elapsed_seconds": elapsed,
        **final_diagnostics,
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
    for (dataset, phase, family, variant), group in rows.groupby(
        ["dataset", "phase", "episode_family", "variant"], sort=True
    ):
        for metric in SUMMARY_METRICS:
            mean, sd, low, high = mean_ci(group[metric])
            records.append(
                {
                    "dataset": dataset,
                    "phase": phase,
                    "episode_family": family,
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


def paired_summary(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    for (dataset, phase, family), section in rows.groupby(
        ["dataset", "phase", "episode_family"], sort=True
    ):
        baseline = section[section["variant"].eq(BASELINE)].set_index("episode")
        for variant, candidate in section.groupby("variant", sort=True):
            if variant == BASELINE:
                continue
            candidate = candidate.set_index("episode")
            common = baseline.index.intersection(candidate.index)
            for metric in SUMMARY_METRICS:
                delta = candidate.loc[common, metric] - baseline.loc[common, metric]
                mean, sd, low, high = mean_ci(delta)
                higher = metric not in {
                    "round_to_95_global_best",
                    "elapsed_seconds",
                }
                wins = int((delta > 1e-10).sum()) if higher else int((delta < -1e-10).sum())
                losses = int((delta < -1e-10).sum()) if higher else int((delta > 1e-10).sum())
                records.append(
                    {
                        "dataset": dataset,
                        "phase": phase,
                        "episode_family": family,
                        "candidate": variant,
                        "baseline": BASELINE,
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


def diagnostic_summary(rows: pd.DataFrame) -> pd.DataFrame:
    columns = (
        "global_noise",
        "precision_min",
        "precision_max",
        "effective_noise_min",
        "effective_noise_max",
    )
    records: list[dict] = []
    for (dataset, phase, variant), group in rows.groupby(
        ["dataset", "phase", "variant"], sort=True
    ):
        for metric in columns:
            mean, sd, low, high = mean_ci(group[metric])
            records.append(
                {
                    "dataset": dataset,
                    "phase": phase,
                    "variant": variant,
                    "diagnostic": metric,
                    "n_episodes": len(group),
                    "mean": mean,
                    "sample_sd": sd,
                    "ci95_low": low,
                    "ci95_high": high,
                    "min": float(group[metric].min()),
                    "max": float(group[metric].max()),
                }
            )
    return pd.DataFrame(records)


def select_strength(rows: pd.DataFrame, paired: pd.DataFrame) -> dict[str, object]:
    candidates: list[dict] = []
    lower, upper = GLOBAL_NOISE_BOUNDS
    margin = PILOT_GATE["global_noise_bound_fraction_margin"]
    for strength in STRENGTHS[1:]:
        variant = variant_name(strength)
        section = paired[paired["candidate"].eq(variant)]
        deltas: dict[str, dict[str, float]] = {}
        checks: dict[str, bool] = {}
        for dataset in ("Buchwald_sub4", "Suzuki"):
            dataset_rows = section[section["dataset"].eq(dataset)]
            metric_delta = {
                row.metric: float(row.mean_delta)
                for row in dataset_rows.itertuples()
            }
            deltas[dataset] = metric_delta
            checks[f"{dataset}_auc"] = (
                metric_delta["AUC_best_so_far"]
                >= PILOT_GATE["minimum_per_dataset_auc_delta"]
            )
            checks[f"{dataset}_initial"] = (
                metric_delta["initial_round_found_best"]
                >= PILOT_GATE["minimum_initial_delta"]
            )
            checks[f"{dataset}_best"] = (
                metric_delta["best_found"] >= PILOT_GATE["minimum_best_delta"]
            )
            checks[f"{dataset}_t95"] = (
                metric_delta["round_to_95_global_best"]
                <= PILOT_GATE["maximum_t95_delta"]
            )
        combined_auc = float(
            np.mean([deltas[name]["AUC_best_so_far"] for name in deltas])
        )
        checks["combined_auc_strict"] = (
            combined_auc > PILOT_GATE["minimum_combined_auc_delta_strict"]
        )
        model_rows = rows[rows["variant"].eq(variant)]
        checks["global_noise_interior"] = bool(
            (model_rows["global_noise"] > lower + margin * (upper - lower)).all()
            and (model_rows["global_noise"] < upper - margin * (upper - lower)).all()
        )
        candidates.append(
            {
                "variant": variant,
                "strength": strength,
                "combined_auc_delta": combined_auc,
                "per_dataset_deltas": deltas,
                "global_noise_min": float(model_rows["global_noise"].min()),
                "global_noise_max": float(model_rows["global_noise"].max()),
                "checks": checks,
                "passed": bool(all(checks.values())),
            }
        )
    eligible = [item for item in candidates if item["passed"]]
    selected = (
        sorted(eligible, key=lambda item: (-item["combined_auc_delta"], item["strength"]))[0]
        if eligible
        else None
    )
    return {
        "gate": PILOT_GATE,
        "strength_grid": list(STRENGTHS[1:]),
        "candidates": candidates,
        "passed": selected is not None,
        "selected_variant": None if selected is None else selected["variant"],
        "selected_strength": None if selected is None else selected["strength"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("smoke", "pilot", "confirmation"), required=True
    )
    parser.add_argument("--strength", type=float, default=None)
    parser.add_argument("--jobs", type=int, default=min(6, os.cpu_count() or 1))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.jobs <= 0:
        raise ValueError("--jobs must be positive.")
    episodes = episodes_for_stage(args.stage)
    if args.stage == "confirmation":
        if args.strength not in STRENGTHS[1:]:
            raise ValueError("Confirmation requires one predeclared nonzero strength.")
        strengths = (0.0, float(args.strength))
    else:
        if args.strength is not None:
            raise ValueError("Strength is selected by the pilot, not supplied to it.")
        strengths = STRENGTHS
    variants = tuple(variant_name(value) for value in strengths)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "stage": args.stage,
        "train_only": True,
        "public_test_labels_used": False,
        "candidate_labels_used_in_weighting": False,
        "inputs": {
            name: {
                "path": str(path.relative_to(DATASET_ROOT)),
                "sha256": file_sha256(path),
            }
            for name, path in INPUT_FILES.items()
        },
        "strengths": list(strengths),
        "variants": list(variants),
        "precision_parameterization": "exp(strength * centered empirical percentile rank), arithmetic mean normalized to one",
        "pilot_gate": PILOT_GATE,
        "global_noise_bounds": list(GLOBAL_NOISE_BOUNDS),
        "acquisition": "posterior mean query 1, stable LogEI thereafter",
        "target_transform": "standard",
        "episode_count": len(episodes),
        "official_target_excluded_from_selection": args.stage != "confirmation",
        "production_default_changed": False,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    jobs = [
        (dataset, variant, episode, str(args.output_dir))
        for dataset, episode in episodes
        for variant in variants
    ]
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(run_one, job): job for job in jobs}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"{row['dataset']} {row['phase']} {row['variant']} "
                f"{row['episode']}: AUC={row['AUC_best_so_far']:.3f}",
                flush=True,
            )
    detail = pd.DataFrame(rows).sort_values(
        ["dataset", "phase", "episode", "variant"]
    )
    summary = summarize(detail)
    paired = paired_summary(detail)
    detail.to_csv(args.output_dir / "per_episode.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    paired.to_csv(args.output_dir / "paired_vs_uniform.csv", index=False)
    diagnostic_summary(detail).to_csv(
        args.output_dir / "weight_noise_diagnostics.csv", index=False
    )
    if args.stage == "pilot":
        decision = select_strength(detail, paired)
        (args.output_dir / "selection_decision.json").write_text(
            json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8"
        )
        twenty = detail[
            detail["dataset"].eq("Suzuki") & detail["phase"].eq("pilot")
        ].copy()
        if len(twenty) != 20 * len(variants) or twenty["episode"].nunique() != 20:
            raise RuntimeError("The explicit 20-run pilot slice is incomplete.")
        twenty.to_csv(args.output_dir / "twenty_run_per_episode.csv", index=False)
        summarize(twenty).to_csv(
            args.output_dir / "twenty_run_summary.csv", index=False
        )
        paired_summary(twenty).to_csv(
            args.output_dir / "twenty_run_paired_vs_uniform.csv", index=False
        )
    if args.stage == "confirmation":
        twenty_ids = {f"random_confirmation_{index:02d}" for index in range(20)}
        twenty = detail[
            detail["dataset"].eq("Suzuki")
            & detail["episode"].isin(twenty_ids)
        ].copy()
        if len(twenty) != 40 or set(twenty["episode"]) != twenty_ids:
            raise RuntimeError("The explicit 20-run confirmation slice is incomplete.")
        twenty.to_csv(args.output_dir / "twenty_run_per_episode.csv", index=False)
        summarize(twenty).to_csv(
            args.output_dir / "twenty_run_summary.csv", index=False
        )
        paired_summary(twenty).to_csv(
            args.output_dir / "twenty_run_paired_vs_uniform.csv", index=False
        )


if __name__ == "__main__":
    main()
