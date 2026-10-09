#!/usr/bin/env python3
"""Leakage-safe acquisition study using a fixed incumbent structured GP."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from acquisition_study_helpers import (  # noqa: E402
    BASELINE_POLICY,
    POLICIES,
    acquisition_scores,
    select_from_screening,
)
from bgolearn_rxn.factory import prepare_problem  # noqa: E402
from bgolearn_rxn.metrics import compute_campaign_metrics  # noqa: E402
from bgolearn_rxn.oracle import OfflineTableOracle  # noqa: E402
from bgolearn_rxn.surrogates.anova_gp import StructuredANOVAGP  # noqa: E402
from bgolearn_rxn.utils import deterministic_argmax  # noqa: E402


INPUT_FILES = {
    "Buchwald_sub4": DATASET_ROOT / "Buchwald_sub4" / "buchwald_sub4_train.csv",
    "Suzuki": DATASET_ROOT / "Suzuki" / "suzuki_train.csv",
}
CORE_METRICS = (
    "initial_round_found_best",
    "best_found",
    "round_to_95_global_best",
    "AUC_best_so_far",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_buchwald_episodes() -> list[dict]:
    episodes = transfer_proxy.make_buchwald_episodes(8)
    output = []
    for episode in episodes:
        repeat = int(episode["episode"].rsplit("_", 1)[1])
        item = dict(episode)
        item["phase"] = "screening" if repeat < 4 else "confirmation"
        output.append(item)
    return output


def make_suzuki_episodes() -> list[dict]:
    return suzuki_proxy.make_random_episodes() + suzuki_proxy.make_block_episodes()


def build_problem(dataset_name: str, episode: dict):
    if dataset_name == "Buchwald_sub4":
        return transfer_proxy.build_problem(
            dataset_name, "pooled_gp_scalar_task", episode
        )
    if dataset_name == "Suzuki":
        return suzuki_proxy.build_problem(episode, "gp_only")
    raise ValueError(f"Unknown dataset: {dataset_name}")


def run_one(job: tuple[str, str, dict, str]) -> dict:
    dataset_name, policy, episode, output_root_string = job
    config, dataset, lookup = build_problem(dataset_name, episode)
    prepared = prepare_problem(dataset, config)
    model = StructuredANOVAGP(
        schema=prepared.encoder.schema,
        config=config.gp,
        seed=int(episode["seed"] if dataset_name == "Buchwald_sub4" else episode["campaign_seed"]) + 11,
    )
    oracle = OfflineTableOracle(dataset=dataset, lookup=lookup)
    observed = dataset.initial.copy()
    X_observed = prepared.X_initial.copy()
    y_observed = observed[dataset.spec.target_column].to_numpy(dtype=float)
    remaining = np.arange(len(dataset.candidates), dtype=int)
    trajectory: list[dict] = []
    budget = int(episode["budget"])
    campaign_seed = int(
        episode["seed"] if dataset_name == "Buchwald_sub4" else episode["campaign_seed"]
    )
    started = time.perf_counter()

    for step in range(1, budget + 1):
        X_remaining = prepared.X_candidates[remaining]
        candidate_rows = dataset.candidates.iloc[remaining]
        model.fit(X_observed, y_observed)
        posterior = model.predict(X_remaining)
        covariance = (
            model.posterior_covariance(X_remaining)
            if policy == "mean_then_correlated_thompson"
            else None
        )
        best_f = dataset.target_incumbent(observed)
        scores = acquisition_scores(
            policy,
            posterior.mean,
            posterior.std,
            best_f=best_f,
            step=step,
            budget=budget,
            covariance=covariance,
            seed=campaign_seed + 100_000 * step + POLICIES.index(policy),
        )
        candidate_ids = candidate_rows["query_index"].to_numpy(dtype=int)
        local_index = deterministic_argmax(
            scores, candidate_ids, seed=campaign_seed + step
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
                "acquisition_name": policy,
                "acquisition_value": float(scores[local_index]),
            }
        )
        new_observation = selected_row.loc[list(dataset.spec.feature_columns)].to_dict()
        new_observation[dataset.spec.target_column] = observed_yield
        observed = pd.concat([observed, pd.DataFrame([new_observation])], ignore_index=True)
        X_observed = np.vstack([X_observed, prepared.X_candidates[selected_global]])
        y_observed = np.append(y_observed, observed_yield)
        remaining = np.delete(remaining, local_index)

    elapsed = time.perf_counter() - started
    candidate_best = float(lookup[dataset.spec.target_column].max())
    raw_metrics = compute_campaign_metrics(
        [record["observed_yield"] for record in trajectory],
        global_best=candidate_best,
    ).to_dict()
    t95 = raw_metrics["round_to_95_global_best"]
    metrics = {
        "initial_round_found_best": raw_metrics["initial_round_found_best"],
        "best_found": raw_metrics["best_found"],
        "round_to_95_global_best": budget + 1 if t95 is None else int(t95),
        "t95_reached": t95 is not None,
        "AUC_best_so_far": raw_metrics["auc_best_so_far"],
    }
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
            "held_candidate_global_best": candidate_best,
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
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "dataset": dataset_name,
        "phase": episode["phase"],
        "episode_family": episode["episode_family"],
        "episode": episode["episode"],
        "policy": policy,
        "budget": budget,
        "n_initial": len(dataset.initial),
        "n_candidates": len(dataset.candidates),
        "candidate_global_best": candidate_best,
        **metrics,
        "elapsed_seconds": elapsed,
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
    records = []
    for (dataset, phase, family, policy), group in rows.groupby(
        ["dataset", "phase", "episode_family", "policy"], sort=True
    ):
        for metric in CORE_METRICS:
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
    records = []
    for (dataset, phase, family), section in rows.groupby(
        ["dataset", "phase", "episode_family"], sort=True
    ):
        baseline = section[section["policy"].eq(BASELINE_POLICY)].set_index("episode")
        for policy, candidate in section.groupby("policy", sort=True):
            if policy == BASELINE_POLICY:
                continue
            candidate = candidate.set_index("episode")
            common = baseline.index.intersection(candidate.index)
            for metric in CORE_METRICS:
                delta = candidate.loc[common, metric] - baseline.loc[common, metric]
                mean, sd, low, high = mean_ci(delta)
                higher = metric != "round_to_95_global_best"
                wins = int((delta > 1e-10).sum()) if higher else int((delta < -1e-10).sum())
                losses = int((delta < -1e-10).sum()) if higher else int((delta > 1e-10).sum())
                records.append(
                    {
                        "dataset": dataset,
                        "phase": phase,
                        "episode_family": family,
                        "policy": policy,
                        "baseline": BASELINE_POLICY,
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


def execute_jobs(jobs: list[tuple], workers: int) -> list[dict]:
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_one, job): job for job in jobs}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"{row['dataset']} {row['phase']} {row['policy']} {row['episode']}: "
                f"AUC={row['AUC_best_so_far']:.3f}",
                flush=True,
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=min(6, os.cpu_count() or 1))
    parser.add_argument("--limit-screen", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            DATASET_ROOT
            / "submission"
            / "results"
            / "experimental_investigations"
            / "3.1-acquisition-study"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.jobs <= 0:
        raise ValueError("--jobs must be positive.")
    episodes = {
        "Buchwald_sub4": make_buchwald_episodes(),
        "Suzuki": make_suzuki_episodes(),
    }
    screening = {
        dataset: [episode for episode in values if episode["phase"] == "screening"]
        for dataset, values in episodes.items()
    }
    if args.limit_screen is not None:
        if args.limit_screen <= 0:
            raise ValueError("--limit-screen must be positive.")
        screening = {
            dataset: values[: args.limit_screen] for dataset, values in screening.items()
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "train_only": True,
        "public_test_labels_used": False,
        "inputs": {
            dataset: {
                "path": str(path.relative_to(DATASET_ROOT)),
                "sha256": file_sha256(path),
            }
            for dataset, path in INPUT_FILES.items()
        },
        "policies": list(POLICIES),
        "baseline_policy": BASELINE_POLICY,
        "selection_rule": {
            "max_t95_increase": 0.25,
            "max_best_found_decrease": 0.5,
            "max_initial_round_decrease": 0.5,
            "auc_tie_band": 0.1,
            "tie_break": ["lower_t95", "higher_best_found", "fixed_policy_order"],
        },
        "first_query_contract": (
            "Every policy uses GP posterior mean at query 1; the named "
            "acquisition begins at query 2."
        ),
        "screening_episode_counts": {
            dataset: len(values) for dataset, values in screening.items()
        },
        "python": platform.python_version(),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    screen_jobs = [
        (dataset, policy, episode, str(args.output_dir))
        for dataset, values in screening.items()
        for policy in POLICIES
        for episode in values
    ]
    screen_rows = pd.DataFrame(execute_jobs(screen_jobs, args.jobs)).sort_values(
        ["dataset", "episode", "policy"]
    )
    screen_summary = summarize(screen_rows)
    screen_rows.to_csv(args.output_dir / "screening_per_episode.csv", index=False)
    screen_summary.to_csv(args.output_dir / "screening_summary.csv", index=False)
    paired_summary(screen_rows).to_csv(
        args.output_dir / "screening_paired_vs_logei.csv", index=False
    )
    selections = {
        dataset: select_from_screening(screen_summary, dataset)
        for dataset in screening
    }
    (args.output_dir / "selection.json").write_text(
        json.dumps(selections, indent=2, sort_keys=True), encoding="utf-8"
    )

    confirm_jobs = []
    for dataset, values in episodes.items():
        selected = str(selections[dataset]["selected_policy"])
        policies = tuple(dict.fromkeys((BASELINE_POLICY, selected)))
        confirmation = [e for e in values if e["phase"] == "confirmation"]
        confirm_jobs.extend(
            (dataset, policy, episode, str(args.output_dir))
            for policy in policies
            for episode in confirmation
        )
    confirm_rows = pd.DataFrame(execute_jobs(confirm_jobs, args.jobs)).sort_values(
        ["dataset", "episode", "policy"]
    )
    confirm_rows.to_csv(args.output_dir / "confirmation_per_episode.csv", index=False)
    summarize(confirm_rows).to_csv(args.output_dir / "confirmation_summary.csv", index=False)
    paired_summary(confirm_rows).to_csv(
        args.output_dir / "confirmation_paired_vs_logei.csv", index=False
    )

    blocks = [e for e in episodes["Suzuki"] if e["phase"] == "block_diagnostic"]
    selected_suzuki = str(selections["Suzuki"]["selected_policy"])
    block_policies = tuple(dict.fromkeys((BASELINE_POLICY, selected_suzuki)))
    block_jobs = [
        ("Suzuki", policy, episode, str(args.output_dir))
        for policy in block_policies
        for episode in blocks
    ]
    block_rows = pd.DataFrame(execute_jobs(block_jobs, args.jobs)).sort_values(
        ["episode", "policy"]
    )
    block_rows.to_csv(args.output_dir / "block_per_episode.csv", index=False)
    summarize(block_rows).to_csv(args.output_dir / "block_summary.csv", index=False)
    paired_summary(block_rows).to_csv(
        args.output_dir / "block_paired_vs_logei.csv", index=False
    )


if __name__ == "__main__":
    main()
