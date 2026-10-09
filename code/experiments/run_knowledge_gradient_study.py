#!/usr/bin/env python3
"""Staged train-only finite-pool knowledge-gradient study."""

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
from bgolearn_rxn.factory import build_ensemble, prepare_problem  # noqa: E402
from bgolearn_rxn.knowledge_gradient import (  # noqa: E402
    finite_pool_knowledge_gradient,
)
from bgolearn_rxn.metrics import compute_campaign_metrics  # noqa: E402
from bgolearn_rxn.oracle import OfflineTableOracle  # noqa: E402
from bgolearn_rxn.utils import deterministic_argmax  # noqa: E402


POLICIES = ("mean_then_logei", "mean_then_kg")
BASELINE = "mean_then_logei"
CORE_METRICS = (
    "initial_round_found_best",
    "best_found",
    "round_to_95_global_best",
    "AUC_best_so_far",
)
QUADRATURE_ORDER = 32
OFFICIAL_BUCHWALD_PRODUCT = "N-(4-ethylphenyl)-4-methylaniline"
INPUT_FILES = {
    "Buchwald_sub4": DATASET_ROOT / "Buchwald_sub4" / "buchwald_sub4_train.csv",
    "Suzuki": DATASET_ROOT / "Suzuki" / "suzuki_train.csv",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_buchwald_episodes() -> list[dict]:
    episodes: list[dict] = []
    for episode in transfer_proxy.make_buchwald_exhaustive_pair_episodes(1):
        item = dict(episode)
        item["phase"] = "official_target_exhaustive"
        if item["target_task"] != OFFICIAL_BUCHWALD_PRODUCT:
            raise ValueError("Sorted product index 1 is not the official target.")
        episodes.append(item)
    return episodes


def make_suzuki_episodes() -> list[dict]:
    frame = pd.read_csv(INPUT_FILES["Suzuki"])
    pilot = suzuki_proxy._phase_random_episodes(
        frame,
        phase="pilot",
        seed_start=310_000,
        count=20,
    )
    confirmation = suzuki_proxy._phase_random_episodes(
        frame,
        phase="confirmation",
        seed_start=410_000,
        count=80,
    )
    return pilot + confirmation


def make_episodes() -> dict[str, list[dict]]:
    return {
        "Buchwald_sub4": make_buchwald_episodes(),
        "Suzuki": make_suzuki_episodes(),
    }


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
        model_name = "component_ard_low_rank_task_gp"
    elif dataset_name == "Suzuki":
        config, dataset, lookup = suzuki_proxy.build_problem(episode, "gp_only")
        model_name = "gp"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    budget = int(episode["budget"])
    config = replace(
        config,
        gp=replace(config.gp, target_transform="standard"),
        ensemble=replace(
            config.ensemble,
            enabled_models=(model_name,),
            initial_weights={model_name: 1.0},
        ),
        policy=replace(
            config.policy,
            first_query=model_name,
            early_end=budget,
            final_start=budget + 1,
            challenge_every=0,
            novelty_disagreement_mode="off",
        ),
    )
    return config, _copy_dataset(dataset), lookup.copy(), model_name


def _observation_variance(model) -> float:
    if model.theta_ is None:
        raise RuntimeError("Observation noise requires a fitted GP.")
    noise = float(model._unpack(model.theta_)[2])
    scale = float(getattr(model.target_transform, "scale_", 1.0))
    return (noise * scale) ** 2


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


def run_one(job: tuple[str, str, dict, str]) -> dict:
    dataset_name, policy, episode, output_root_string = job
    config, dataset, lookup, model_name = build_problem(dataset_name, episode)
    prepared = prepare_problem(dataset, config)
    ensemble = build_ensemble(
        config,
        prepared,
        seed=int(episode.get("seed", episode.get("campaign_seed"))),
    )
    model = ensemble.models[model_name]
    oracle = OfflineTableOracle(dataset=dataset, lookup=lookup)
    observed = dataset.initial.copy()
    X_observed = prepared.X_initial.copy()
    y_observed = observed[dataset.spec.target_column].to_numpy(dtype=float)
    remaining = np.arange(len(dataset.candidates), dtype=int)
    trajectory: list[dict] = []
    budget = int(episode["budget"])
    campaign_seed = int(episode.get("seed", episode.get("campaign_seed")))
    started = time.perf_counter()
    kg_seconds = 0.0

    for step in range(1, budget + 1):
        X_remaining = prepared.X_candidates[remaining]
        candidate_rows = dataset.candidates.iloc[remaining]
        model.fit(X_observed, y_observed)
        posterior = model.predict(X_remaining)
        best_f = dataset.target_incumbent(observed)
        if step == 1:
            scores = posterior.mean.copy()
            acquisition_name = "posterior_mean"
        elif policy == "mean_then_logei":
            progress = 0.0 if budget == 1 else (step - 1) / (budget - 1)
            scores = stable_log_expected_improvement(
                posterior.mean,
                posterior.std,
                best_f,
                xi=0.05 * (1.0 - progress),
            )
            acquisition_name = "stable_log_ei"
        else:
            kg_started = time.perf_counter()
            covariance = model.posterior_covariance(X_remaining)
            scores = finite_pool_knowledge_gradient(
                posterior.mean,
                covariance,
                observation_variance=_observation_variance(model),
                best_f=best_f,
                quadrature_order=QUADRATURE_ORDER,
            )
            kg_seconds += time.perf_counter() - kg_started
            acquisition_name = "finite_pool_one_step_kg"

        candidate_ids = candidate_rows["query_index"].to_numpy(dtype=int)
        local_index = deterministic_argmax(
            scores,
            candidate_ids,
            seed=campaign_seed + step,
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
    payload = {
        "dataset": dataset_name,
        "policy": policy,
        "episode": episode,
        "trajectory": trajectory,
        "metrics": metrics,
        "contract": {
            "train_only": True,
            "public_test_labels_used": False,
            "input_file": str(INPUT_FILES[dataset_name].relative_to(DATASET_ROOT)),
            "input_sha256": file_sha256(INPUT_FILES[dataset_name]),
            "model": model_name,
            "target_transform": config.gp.target_transform,
            "quadrature_order": QUADRATURE_ORDER if policy == "mean_then_kg" else None,
            "observation_noise_included": policy == "mean_then_kg",
            "episode": episode,
        },
    }
    output_path = (
        Path(output_root_string)
        / "trajectories"
        / dataset_name
        / episode["phase"]
        / policy
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
        "policy": policy,
        "model": model_name,
        "budget": budget,
        "n_initial": len(dataset.initial),
        "n_candidates": len(dataset.candidates),
        "global_best": global_best,
        **metrics,
        "elapsed_seconds": elapsed,
        "kg_scoring_seconds": kg_seconds,
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
    for (dataset, phase, family, policy), group in rows.groupby(
        ["dataset", "phase", "episode_family", "policy"], sort=True
    ):
        for metric in CORE_METRICS + ("elapsed_seconds", "kg_scoring_seconds"):
            mean, sd, low, high = mean_ci(group[metric])
            records.append(
                {
                    "dataset": dataset,
                    "phase": phase,
                    "episode_family": family,
                    "policy": policy,
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
        baseline = section[section["policy"].eq(BASELINE)].set_index("episode")
        candidate = section[section["policy"].eq("mean_then_kg")].set_index(
            "episode"
        )
        common = baseline.index.intersection(candidate.index)
        for metric in CORE_METRICS + ("elapsed_seconds",):
            delta = candidate.loc[common, metric] - baseline.loc[common, metric]
            mean, sd, low, high = mean_ci(delta)
            higher = metric not in {"round_to_95_global_best", "elapsed_seconds"}
            wins = int((delta > 1e-10).sum()) if higher else int((delta < -1e-10).sum())
            losses = int((delta < -1e-10).sum()) if higher else int((delta > 1e-10).sum())
            records.append(
                {
                    "dataset": dataset,
                    "phase": phase,
                    "episode_family": family,
                    "policy": "mean_then_kg",
                    "baseline": BASELINE,
                    "metric": metric,
                    "n_pairs": len(common),
                    "mean_delta_kg_minus_logei": mean,
                    "sample_sd_delta": sd,
                    "ci95_low": low,
                    "ci95_high": high,
                    "wins": wins,
                    "ties": len(common) - wins - losses,
                    "losses": losses,
                }
            )
    return pd.DataFrame(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=min(6, os.cpu_count() or 1))
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("buchwald", "pilot", "confirmation"),
        default=["buchwald", "pilot", "confirmation"],
    )
    parser.add_argument("--limit-episodes", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            DATASET_ROOT
            / "submission"
            / "results"
            / "experimental_investigations"
            / "1-finite-pool-kg-study"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.jobs <= 0:
        raise ValueError("--jobs must be positive.")
    episodes_by_dataset = make_episodes()
    selected: list[tuple[str, dict]] = []
    if "buchwald" in args.stages:
        selected.extend(("Buchwald_sub4", item) for item in episodes_by_dataset["Buchwald_sub4"])
    for stage in ("pilot", "confirmation"):
        if stage in args.stages:
            selected.extend(
                ("Suzuki", item)
                for item in episodes_by_dataset["Suzuki"]
                if item["phase"] == stage
            )
    if args.limit_episodes is not None:
        if args.limit_episodes <= 0:
            raise ValueError("--limit-episodes must be positive.")
        selected = selected[: args.limit_episodes]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "train_only": True,
        "public_test_labels_used": False,
        "inputs": {
            name: {
                "path": str(path.relative_to(DATASET_ROOT)),
                "sha256": file_sha256(path),
            }
            for name, path in INPUT_FILES.items()
        },
        "models": {
            "Buchwald_sub4": "component_ard_low_rank_task_gp",
            "Suzuki": "gp",
        },
        "policies": list(POLICIES),
        "first_query": "model posterior mean",
        "quadrature_order": QUADRATURE_ORDER,
        "stages": args.stages,
        "episode_counts": {
            "Buchwald_sub4": sum(name == "Buchwald_sub4" for name, _ in selected),
            "Suzuki_pilot": sum(name == "Suzuki" and item["phase"] == "pilot" for name, item in selected),
            "Suzuki_confirmation": sum(name == "Suzuki" and item["phase"] == "confirmation" for name, item in selected),
        },
        "twenty_run_contract": {
            "dataset": "Suzuki",
            "phase": "confirmation",
            "episode_ids": [f"random_confirmation_{index:02d}" for index in range(20)],
            "reporting": "per-run rows plus mean, sample SD, and 95% t confidence intervals",
        },
        "target_transform": "standard",
        "t95_miss_penalty": "episode budget + 1",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    jobs = [
        (dataset, policy, episode, str(args.output_dir))
        for dataset, episode in selected
        for policy in POLICIES
    ]
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(run_one, job): job for job in jobs}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"{row['dataset']} {row['phase']} {row['policy']} "
                f"{row['episode']}: AUC={row['AUC_best_so_far']:.3f}",
                flush=True,
            )
    detail = pd.DataFrame(rows).sort_values(
        ["dataset", "phase", "episode", "policy"]
    )
    detail.to_csv(args.output_dir / "per_episode.csv", index=False)
    summarize(detail).to_csv(args.output_dir / "summary.csv", index=False)
    paired_summary(detail).to_csv(
        args.output_dir / "paired_kg_vs_logei.csv", index=False
    )
    twenty_ids = {f"random_confirmation_{index:02d}" for index in range(20)}
    twenty = detail[
        detail["dataset"].eq("Suzuki")
        & detail["phase"].eq("confirmation")
        & detail["episode"].isin(twenty_ids)
    ].copy()
    if not twenty.empty:
        if set(twenty["episode"]) != twenty_ids or len(twenty) != 40:
            raise RuntimeError("The explicit 20-run confirmation slice is incomplete.")
        twenty.to_csv(
            args.output_dir / "twenty_run_per_episode.csv", index=False
        )
        summarize(twenty).to_csv(
            args.output_dir / "twenty_run_summary.csv", index=False
        )
        paired_summary(twenty).to_csv(
            args.output_dir / "twenty_run_paired_kg_vs_logei.csv", index=False
        )


if __name__ == "__main__":
    main()
