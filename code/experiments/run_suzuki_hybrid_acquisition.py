#!/usr/bin/env python3
"""Predeclared Suzuki LogEI/Thompson hybrid on untouched train holdouts."""

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
from acquisition_study_helpers import acquisition_scores  # noqa: E402
from bgolearn_rxn.factory import prepare_problem  # noqa: E402
from bgolearn_rxn.metrics import compute_campaign_metrics  # noqa: E402
from bgolearn_rxn.oracle import OfflineTableOracle  # noqa: E402
from bgolearn_rxn.surrogates.anova_gp import StructuredANOVAGP  # noqa: E402
from bgolearn_rxn.utils import deterministic_argmax  # noqa: E402


TRAIN_FILE = DATASET_ROOT / "Suzuki" / "suzuki_train.csv"
BASELINE_POLICY = "mean_then_logei"
THOMPSON_POLICY = "mean_then_correlated_thompson"
HYBRID_POLICY = "mean_logei_ts_alternating"
POLICIES = (BASELINE_POLICY, THOMPSON_POLICY, HYBRID_POLICY)
CORE_METRICS = (
    "initial_round_found_best",
    "best_found",
    "round_to_95_global_best",
    "AUC_best_so_far",
)
SPLIT_SEED_START = 310_000
N_EPISODES = 100
BUDGET = 5
BEST_FOUND_SAFETY_MARGIN = -0.5
THOMPSON_STREAM_ID = 5


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_episodes() -> list[dict]:
    """Build the frozen full-level-coverage confirmation split set."""
    frame = pd.read_csv(TRAIN_FILE)
    episodes = suzuki_proxy._phase_random_episodes(
        frame,
        phase="untouched_confirmation",
        seed_start=SPLIT_SEED_START,
        count=N_EPISODES,
    )
    for episode in episodes:
        episode["budget"] = BUDGET
    return episodes


def policy_component(policy: str, step: int) -> str:
    """Resolve the predeclared acquisition used at a campaign step."""
    if policy not in POLICIES:
        raise ValueError(f"Unknown policy: {policy}")
    if step < 1 or step > BUDGET:
        raise ValueError(f"Step must lie in [1, {BUDGET}].")
    if step == 1:
        return "posterior_mean"
    if policy == HYBRID_POLICY:
        return BASELINE_POLICY if step % 2 == 0 else THOMPSON_POLICY
    return policy


def scheduled_scores(
    policy: str,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    best_f: float,
    step: int,
    covariance: np.ndarray | None,
    campaign_seed: int,
) -> tuple[np.ndarray, str]:
    component = policy_component(policy, step)
    scores = acquisition_scores(
        component,
        mean,
        std,
        best_f=best_f,
        step=step,
        budget=BUDGET,
        covariance=covariance,
        seed=campaign_seed + 100_000 * step + THOMPSON_STREAM_ID,
    )
    return scores, component


def run_one(job: tuple[str, dict, str]) -> dict:
    policy, episode, output_root_string = job
    config, dataset, lookup = suzuki_proxy.build_problem(episode, "gp_only")
    prepared = prepare_problem(dataset, config)
    campaign_seed = int(episode["campaign_seed"])
    model = StructuredANOVAGP(
        schema=prepared.encoder.schema,
        config=config.gp,
        seed=campaign_seed + 11,
    )
    oracle = OfflineTableOracle(dataset=dataset, lookup=lookup)
    observed = dataset.initial.copy()
    X_observed = prepared.X_initial.copy()
    y_observed = observed[dataset.spec.target_column].to_numpy(dtype=float)
    remaining = np.arange(len(dataset.candidates), dtype=int)
    trajectory: list[dict] = []
    started = time.perf_counter()

    for step in range(1, BUDGET + 1):
        X_remaining = prepared.X_candidates[remaining]
        candidate_rows = dataset.candidates.iloc[remaining]
        model.fit(X_observed, y_observed)
        posterior = model.predict(X_remaining)
        component = policy_component(policy, step)
        covariance = (
            model.posterior_covariance(X_remaining)
            if component == THOMPSON_POLICY
            else None
        )
        scores, component = scheduled_scores(
            policy,
            posterior.mean,
            posterior.std,
            best_f=dataset.target_incumbent(observed),
            step=step,
            covariance=covariance,
            campaign_seed=campaign_seed,
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
                "policy": policy,
                "acquisition_component": component,
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
    candidate_best = float(lookup[dataset.spec.target_column].max())
    raw = compute_campaign_metrics(
        [record["observed_yield"] for record in trajectory],
        global_best=candidate_best,
    ).to_dict()
    t95 = raw["round_to_95_global_best"]
    metrics = {
        "initial_round_found_best": raw["initial_round_found_best"],
        "best_found": raw["best_found"],
        "round_to_95_global_best": BUDGET + 1 if t95 is None else int(t95),
        "t95_reached": t95 is not None,
        "AUC_best_so_far": raw["auc_best_so_far"],
    }
    payload = {
        "dataset": "Suzuki",
        "policy": policy,
        "episode": episode,
        "trajectory": trajectory,
        "metrics": metrics,
        "contract": {
            "train_only": True,
            "public_test_labels_used": False,
            "input_file": str(TRAIN_FILE.relative_to(DATASET_ROOT)),
            "input_sha256": file_sha256(TRAIN_FILE),
            "held_candidate_global_best": candidate_best,
        },
    }
    output_path = (
        Path(output_root_string)
        / "trajectories"
        / policy
        / f"{episode['episode']}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {
        "dataset": "Suzuki",
        "episode": episode["episode"],
        "episode_family": episode["episode_family"],
        "split_seed": episode["split_seed"],
        "campaign_seed": campaign_seed,
        "policy": policy,
        "budget": BUDGET,
        "n_initial": len(dataset.initial),
        "n_candidates": len(dataset.candidates),
        "candidate_global_best": candidate_best,
        "first_query_index": trajectory[0]["query_index"],
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
    for policy, group in rows.groupby("policy", sort=True):
        for metric in CORE_METRICS:
            mean, sd, low, high = mean_ci(group[metric])
            records.append(
                {
                    "dataset": "Suzuki",
                    "episode_family": "random_holdout_full_level_coverage",
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
    baseline = rows[rows["policy"].eq(BASELINE_POLICY)].set_index("episode")
    records = []
    for policy, candidate in rows.groupby("policy", sort=True):
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
                    "dataset": "Suzuki",
                    "episode_family": "random_holdout_full_level_coverage",
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


def validate_first_query_contract(rows: pd.DataFrame) -> None:
    for _, group in rows.groupby("episode"):
        if group["first_query_index"].nunique() != 1:
            raise RuntimeError("Policies did not select an identical first query.")
        if group["initial_round_found_best"].nunique() != 1:
            raise RuntimeError("Policies did not have identical first-round yield.")


def execute_jobs(jobs: list[tuple], workers: int) -> list[dict]:
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_one, job): job for job in jobs}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"{row['policy']} {row['episode']}: "
                f"AUC={row['AUC_best_so_far']:.3f}",
                flush=True,
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=min(6, os.cpu_count() or 1))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            DATASET_ROOT
            / "submission"
            / "results"
            / "experimental_investigations"
            / "3.2-suzuki-hybrid-acquisition"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.jobs <= 0:
        raise ValueError("--jobs must be positive.")
    episodes = make_episodes()
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive.")
        episodes = episodes[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "train_only": True,
        "public_test_labels_used": False,
        "input": {
            "path": str(TRAIN_FILE.relative_to(DATASET_ROOT)),
            "sha256": file_sha256(TRAIN_FILE),
        },
        "predeclared_before_evaluation": True,
        "episode_contract": {
            "episode_family": "random_holdout_full_level_coverage",
            "split_seed_start": SPLIT_SEED_START,
            "accepted_split_seed_min": min(e["split_seed"] for e in episodes),
            "accepted_split_seed_max": max(e["split_seed"] for e in episodes),
            "n_episodes": len(episodes),
            "n_initial": 20,
            "budget": BUDGET,
        },
        "policies": list(POLICIES),
        "schedule": {
            BASELINE_POLICY: ["mean", "LogEI", "LogEI", "LogEI", "LogEI"],
            THOMPSON_POLICY: ["mean", "TS", "TS", "TS", "TS"],
            HYBRID_POLICY: ["mean", "LogEI", "TS", "LogEI", "TS"],
        },
        "shared_thompson_stream": True,
        "first_query_requirement": "identical query index and yield for all policies",
        "best_found_safety_rule": {
            "comparison": f"{HYBRID_POLICY} - {BASELINE_POLICY}",
            "minimum_acceptable_mean_delta": BEST_FOUND_SAFETY_MARGIN,
        },
        "no_parameter_search_on_episode_range": True,
        "python": platform.python_version(),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    jobs = [
        (policy, episode, str(args.output_dir))
        for policy in POLICIES
        for episode in episodes
    ]
    rows = pd.DataFrame(execute_jobs(jobs, args.jobs)).sort_values(
        ["episode", "policy"]
    )
    validate_first_query_contract(rows)
    aggregate = summarize(rows)
    paired = paired_summary(rows)
    rows.to_csv(args.output_dir / "per_episode.csv", index=False)
    aggregate.to_csv(args.output_dir / "aggregate.csv", index=False)
    paired.to_csv(args.output_dir / "paired_vs_logei.csv", index=False)

    hybrid_best = paired[
        paired["policy"].eq(HYBRID_POLICY)
        & paired["metric"].eq("best_found")
    ].iloc[0]
    decision = {
        "policy": HYBRID_POLICY,
        "first_query_contract_passed": True,
        "best_found_mean_delta": float(hybrid_best["mean_delta"]),
        "minimum_acceptable_best_found_mean_delta": BEST_FOUND_SAFETY_MARGIN,
        "no_material_best_found_loss": bool(
            hybrid_best["mean_delta"] >= BEST_FOUND_SAFETY_MARGIN
        ),
    }
    (args.output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
