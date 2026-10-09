#!/usr/bin/env python3
"""Staged train-only Bgolearn noise-aware acquisition study."""

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
from bgolearn_rxn.datasets import DatasetSpec, ReactionDataset  # noqa: E402
from bgolearn_rxn.factory import build_ensemble, prepare_problem  # noqa: E402
from bgolearn_rxn.metrics import compute_campaign_metrics  # noqa: E402
from bgolearn_rxn.oracle import OfflineTableOracle  # noqa: E402
from bgolearn_rxn.utils import deterministic_argmax  # noqa: E402
from noise_aware_acquisition import (  # noqa: E402
    BASELINE_POLICY,
    POLICIES,
    noise_aware_scores,
    observation_noise_std,
    score_diagnostics,
)


CORE_METRICS = (
    "initial_round_found_best",
    "best_found",
    "round_to_95_global_best",
    "AUC_best_so_far",
)
OFFICIAL_BUCHWALD_PRODUCT = "N-(4-ethylphenyl)-4-methylaniline"
INPUT_FILES = {
    "Buchwald_sub4": DATASET_ROOT / "Buchwald_sub4" / "buchwald_sub4_train.csv",
    "Suzuki": DATASET_ROOT / "Suzuki" / "suzuki_train.csv",
}
GATE = {
    "min_auc_delta": 0.0,
    "strict_candidate_auc_win": True,
    "max_initial_decrease": 0.5,
    "max_best_decrease": 0.5,
    "max_t95_increase": 0.25,
    "require_clean_numerics": True,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_buchwald_pilot_episodes() -> list[dict]:
    episodes = []
    for source in transfer_proxy.make_buchwald_episodes(5):
        if source["target_task"] == OFFICIAL_BUCHWALD_PRODUCT:
            continue
        item = dict(source)
        item["phase"] = "pilot"
        episodes.append(item)
    if len(episodes) != 20:
        raise RuntimeError("Buchwald pilot must contain 20 auxiliary-target episodes.")
    return episodes


def make_buchwald_confirmation_episodes() -> list[dict]:
    episodes = []
    for source in transfer_proxy.make_buchwald_exhaustive_pair_episodes(1):
        if source["target_task"] != OFFICIAL_BUCHWALD_PRODUCT:
            raise RuntimeError("Buchwald confirmation target does not match the official task.")
        item = dict(source)
        item["phase"] = "confirmation"
        episodes.append(item)
    return episodes


def make_suzuki_episodes(phase: str) -> list[dict]:
    frame = pd.read_csv(INPUT_FILES["Suzuki"])
    if phase == "pilot":
        return suzuki_proxy._phase_random_episodes(
            frame, phase="pilot", seed_start=310_000, count=20
        )
    if phase == "confirmation":
        return suzuki_proxy._phase_random_episodes(
            frame, phase="confirmation", seed_start=410_000, count=80
        )
    raise ValueError(f"Unknown phase: {phase}")


def make_episodes(dataset: str, phase: str) -> list[dict]:
    if dataset == "Buchwald_sub4":
        return (
            make_buchwald_pilot_episodes()
            if phase == "pilot"
            else make_buchwald_confirmation_episodes()
        )
    if dataset == "Suzuki":
        return make_suzuki_episodes(phase)
    raise ValueError(f"Unknown dataset: {dataset}")


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


def _reinterpolated_posterior(model, X_observed: np.ndarray, smoothed_y: np.ndarray, X: np.ndarray, step: int):
    refit = type(model)(
        schema=model.schema,
        config=model.config,
        seed=int(model.seed) + 1_000_003 + step,
    )
    refit.fit(X_observed, smoothed_y)
    return refit.predict(X)


def run_one(job: tuple[str, str, dict, str]) -> dict:
    dataset_name, policy, episode, output_root_string = job
    config, dataset, lookup, model_name = build_problem(dataset_name, episode)
    prepared = prepare_problem(dataset, config)
    campaign_seed = int(episode.get("seed", episode.get("campaign_seed")))
    model = build_ensemble(config, prepared, seed=campaign_seed).models[model_name]
    oracle = OfflineTableOracle(dataset=dataset, lookup=lookup)
    observed = dataset.initial.copy()
    X_observed = prepared.X_initial.copy()
    y_observed = observed[dataset.spec.target_column].to_numpy(dtype=float)
    remaining = np.arange(len(dataset.candidates), dtype=int)
    trajectory: list[dict] = []
    budget = int(episode["budget"])
    started = time.perf_counter()
    acquisition_seconds = 0.0

    for step in range(1, budget + 1):
        X_remaining = prepared.X_candidates[remaining]
        candidate_rows = dataset.candidates.iloc[remaining]
        model.fit(X_observed, y_observed)
        posterior = model.predict(X_remaining)
        train_posterior = model.predict(X_observed)
        target_mask = dataset.spec.target_mask(observed)
        re_mean = re_std = None
        scoring_started = time.perf_counter()
        if policy == "mean_then_reinterpolation_ei" and step > 1:
            re_posterior = _reinterpolated_posterior(
                model, X_observed, train_posterior.mean, X_remaining, step
            )
            re_mean, re_std = re_posterior.mean, re_posterior.std
        scores, formula = noise_aware_scores(
            policy,
            posterior.mean,
            posterior.std,
            observed_best=dataset.target_incumbent(observed),
            train_mean=train_posterior.mean,
            train_std=train_posterior.std,
            target_mask=target_mask,
            observation_noise=observation_noise_std(model),
            step=step,
            budget=budget,
            reinterpolated_mean=re_mean,
            reinterpolated_std=re_std,
        )
        acquisition_seconds += time.perf_counter() - scoring_started
        diagnostics = score_diagnostics(scores)
        candidate_ids = candidate_rows["query_index"].to_numpy(dtype=int)
        local_index = deterministic_argmax(scores, candidate_ids, seed=campaign_seed + step)
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
                "formula": formula,
                "score_diagnostics": diagnostics,
            }
        )
        new_observation = selected_row.loc[list(dataset.spec.feature_columns)].to_dict()
        new_observation[dataset.spec.target_column] = observed_yield
        observed = pd.concat([observed, pd.DataFrame([new_observation])], ignore_index=True)
        X_observed = np.vstack([X_observed, prepared.X_candidates[selected_global]])
        y_observed = np.append(y_observed, observed_yield)
        remaining = np.delete(remaining, local_index)

    elapsed = time.perf_counter() - started
    # The benchmark optimum includes target-task starting observations. Using
    # only the held candidate lookup would make t95 episode-dependent and would
    # under-penalize a miss whenever the optimum was in the initial pair.
    global_best = float(oracle.global_best)
    metrics = _metrics(trajectory, global_best, budget)
    noninitial = trajectory[1:]
    numerics_clean = all(
        np.isfinite(record["acquisition_value"])
        and not record["score_diagnostics"]["all_finite_scores_tied"]
        for record in noninitial
    )
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
            "target_transform": "standard",
            "first_query": "posterior_mean",
            "policy_parameters": {
                "aei_alpha": 1.0,
                "aei_noise": "fitted GP observation standard deviation",
                "eqi_beta": 0.75,
                "eqi_noise": "fitted GP observation standard deviation",
                "logei_xi": "0.05 linearly decayed to 0",
            },
        },
    }
    output_path = (
        Path(output_root_string)
        / "trajectories"
        / dataset_name
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
        "model": model_name,
        "budget": budget,
        "n_initial": len(dataset.initial),
        "n_candidates": len(dataset.candidates),
        "global_best": global_best,
        **metrics,
        "elapsed_seconds": elapsed,
        "acquisition_seconds": acquisition_seconds,
        "numerics_clean": numerics_clean,
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
        for metric in CORE_METRICS + ("elapsed_seconds", "acquisition_seconds"):
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


def select_pilot_winner(rows: pd.DataFrame, dataset: str) -> dict:
    section = rows[rows["dataset"].eq(dataset)].copy()
    table = section.groupby("policy", sort=False)[list(CORE_METRICS)].mean()
    baseline = table.loc[BASELINE_POLICY]
    candidates = []
    for order, policy in enumerate(POLICIES[1:], start=1):
        current = table.loc[policy]
        delta = current - baseline
        clean = bool(section.loc[section["policy"].eq(policy), "numerics_clean"].all())
        eligible = bool(
            delta["AUC_best_so_far"] >= GATE["min_auc_delta"]
            and delta["initial_round_found_best"] >= -GATE["max_initial_decrease"]
            and delta["best_found"] >= -GATE["max_best_decrease"]
            and delta["round_to_95_global_best"] <= GATE["max_t95_increase"]
            and clean
        )
        candidates.append(
            {
                "policy": policy,
                "policy_order": order,
                "eligible": eligible,
                "numerics_clean": clean,
                "mean_deltas": {name: float(delta[name]) for name in CORE_METRICS},
            }
        )
    eligible = [item for item in candidates if item["eligible"]]
    if eligible:
        winner = sorted(
            eligible,
            key=lambda item: (
                -item["mean_deltas"]["AUC_best_so_far"],
                item["mean_deltas"]["round_to_95_global_best"],
                -item["mean_deltas"]["best_found"],
                item["policy_order"],
            ),
        )[0]
        selected = winner["policy"]
        passed = winner["mean_deltas"]["AUC_best_so_far"] > 1e-10
    else:
        selected = BASELINE_POLICY
        passed = False
    return {
        "dataset": dataset,
        "passed": passed,
        "selected_policy": selected,
        "baseline_policy": BASELINE_POLICY,
        "gate": GATE,
        "candidates": candidates,
    }


def confirmation_policies(selection: dict, dataset: str) -> tuple[str, ...]:
    decision = selection[dataset]
    if not decision["passed"]:
        return ()
    selected = str(decision["selected_policy"])
    if selected not in POLICIES[1:]:
        raise ValueError("A passing pilot must freeze a non-baseline policy.")
    return BASELINE_POLICY, selected


def confirmation_decision(rows: pd.DataFrame, selection: dict) -> dict:
    """Apply the frozen pilot safety rule to each available confirmation."""

    output = {}
    for dataset in INPUT_FILES:
        section = rows[rows["dataset"].eq(dataset)]
        if section.empty:
            output[dataset] = {
                "confirmed": False,
                "reason": "not_run_after_failed_pilot_gate",
                "selected_policy": selection[dataset]["selected_policy"],
            }
            continue
        policy = str(selection[dataset]["selected_policy"])
        baseline = section[section["policy"].eq(BASELINE_POLICY)].set_index("episode")
        candidate = section[section["policy"].eq(policy)].set_index("episode")
        common = baseline.index.intersection(candidate.index)
        deltas = {
            metric: float((candidate.loc[common, metric] - baseline.loc[common, metric]).mean())
            for metric in CORE_METRICS
        }
        clean = bool(candidate.loc[common, "numerics_clean"].all())
        confirmed = bool(
            deltas["AUC_best_so_far"] > 1e-10
            and deltas["initial_round_found_best"] >= -GATE["max_initial_decrease"]
            and deltas["best_found"] >= -GATE["max_best_decrease"]
            and deltas["round_to_95_global_best"] <= GATE["max_t95_increase"]
            and clean
        )
        output[dataset] = {
            "confirmed": confirmed,
            "reason": "confirmed" if confirmed else "pilot_direction_did_not_confirm",
            "selected_policy": policy,
            "baseline_policy": BASELINE_POLICY,
            "n_pairs": len(common),
            "mean_deltas": deltas,
            "numerics_clean": clean,
            "gate": GATE,
        }
    return output


def execute(jobs: list[tuple], workers: int) -> pd.DataFrame:
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
    return pd.DataFrame(rows).sort_values(["dataset", "episode", "policy"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "pilot", "confirmation"), required=True)
    parser.add_argument("--jobs", type=int, default=min(6, os.cpu_count() or 1))
    parser.add_argument("--selection-file", type=Path)
    parser.add_argument("--limit-episodes", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.jobs <= 0:
        raise ValueError("--jobs must be positive.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    phase = "pilot" if args.stage in {"smoke", "pilot"} else "confirmation"
    episodes = {name: make_episodes(name, phase) for name in INPUT_FILES}
    if args.stage == "smoke":
        episodes = {name: values[:1] for name, values in episodes.items()}
        policies = {name: POLICIES for name in INPUT_FILES}
        selection_sha256 = None
    elif args.stage == "pilot":
        policies = {name: POLICIES for name in INPUT_FILES}
        selection_sha256 = None
    else:
        if args.selection_file is None or not args.selection_file.is_file():
            raise ValueError("Confirmation requires an existing --selection-file.")
        selection = json.loads(args.selection_file.read_text(encoding="utf-8"))
        policies = {name: confirmation_policies(selection, name) for name in INPUT_FILES}
        selection_sha256 = file_sha256(args.selection_file)
    if args.limit_episodes is not None:
        if args.limit_episodes <= 0:
            raise ValueError("--limit-episodes must be positive.")
        episodes = {name: values[: args.limit_episodes] for name, values in episodes.items()}

    manifest = {
        "stage": args.stage,
        "train_only": True,
        "public_test_labels_used": False,
        "inputs": {
            name: {"path": str(path.relative_to(DATASET_ROOT)), "sha256": file_sha256(path)}
            for name, path in INPUT_FILES.items()
        },
        "models": {"Buchwald_sub4": "component_ard_low_rank_task_gp", "Suzuki": "gp"},
        "policies": {name: list(value) for name, value in policies.items()},
        "available_episode_counts": {name: len(value) for name, value in episodes.items()},
        "executed_episode_counts": {
            name: len(episodes[name]) if policies[name] else 0 for name in INPUT_FILES
        },
        "selection_file_sha256": selection_sha256,
        "target_transform": "standard",
        "t95_miss_penalty": "episode budget + 1",
        "gate": GATE,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    jobs = [
        (dataset, policy, episode, str(args.output_dir))
        for dataset, dataset_episodes in episodes.items()
        for episode in dataset_episodes
        for policy in policies[dataset]
    ]
    rows = execute(jobs, args.jobs)
    rows.to_csv(args.output_dir / "per_episode.csv", index=False)
    summarize(rows).to_csv(args.output_dir / "summary.csv", index=False)
    paired_summary(rows).to_csv(args.output_dir / "paired_vs_logei.csv", index=False)

    if args.stage == "pilot":
        selection = {name: select_pilot_winner(rows, name) for name in INPUT_FILES}
        selection["contract"] = {
            "pilot_per_episode_sha256": file_sha256(args.output_dir / "per_episode.csv"),
            "confirmation_isolation": "only baseline plus a passing frozen winner",
        }
        (args.output_dir / "selection.json").write_text(
            json.dumps(selection, indent=2, sort_keys=True), encoding="utf-8"
        )

    if args.stage == "confirmation":
        decision = confirmation_decision(rows, selection)
        (args.output_dir / "confirmation_decision.json").write_text(
            json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8"
        )
        twenty_parts = []
        for dataset in INPUT_FILES:
            section = rows[rows["dataset"].eq(dataset)]
            if section.empty:
                continue
            episode_ids = sorted(section["episode"].unique())[:20]
            part = section[section["episode"].isin(episode_ids)].copy()
            expected = 20 * len(policies[dataset])
            if len(episode_ids) != 20 or len(part) != expected:
                raise RuntimeError(f"The explicit {dataset} 20-run slice is incomplete.")
            twenty_parts.append(part)
        if twenty_parts:
            twenty = pd.concat(twenty_parts, ignore_index=True)
            twenty.to_csv(args.output_dir / "twenty_run_per_episode.csv", index=False)
            summarize(twenty).to_csv(args.output_dir / "twenty_run_summary.csv", index=False)
            paired_summary(twenty).to_csv(
                args.output_dir / "twenty_run_paired_vs_logei.csv", index=False
            )


if __name__ == "__main__":
    main()
