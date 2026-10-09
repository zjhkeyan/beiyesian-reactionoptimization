#!/usr/bin/env python3
"""Paired 1010-series component-ARD finite-pool KG discovery study."""

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
from scipy.stats import norm, t


EXPERIMENT_ROOT = Path(__file__).resolve().parent
CODE_ROOT = EXPERIMENT_ROOT.parent
DATASET_ROOT = CODE_ROOT.parents[1]
EXTENSION_ROOT = CODE_ROOT / "bgolearn_rxn_extension"
sys.path.insert(0, str(EXTENSION_ROOT / "src"))
sys.path.insert(0, str(EXPERIMENT_ROOT))

import run_suzuki_holdout_proxy as suzuki_proxy  # noqa: E402
from bgolearn_rxn.acquisition import stable_log_expected_improvement  # noqa: E402
from bgolearn_rxn.datasets import ReactionDataset  # noqa: E402
from bgolearn_rxn.factory import build_ensemble, prepare_problem  # noqa: E402
from bgolearn_rxn.knowledge_gradient import finite_pool_knowledge_gradient  # noqa: E402
from bgolearn_rxn.metrics import compute_campaign_metrics  # noqa: E402
from bgolearn_rxn.oracle import OfflineTableOracle  # noqa: E402
from bgolearn_rxn.utils import deterministic_argmax  # noqa: E402


PHASE = "phase6_deadline_discovery"
BASELINE = "component_ard_logei"
CANDIDATE = "component_ard_kg"
VARIANTS = (BASELINE, CANDIDATE)
MODEL_NAME = "component_ard_gp"
QUADRATURE_ORDER = 32
CAMPAIGN_METRICS = (
    "initial_round_found_best",
    "best_found",
    "round_to_95_global_best",
    "AUC_best_so_far",
)
CALIBRATION_LEVELS = (0.50, 0.80, 0.90, 0.95)
CALIBRATION_METRICS = ("queried_nll",) + tuple(
    f"coverage_{int(level * 100)}" for level in CALIBRATION_LEVELS
)
SUMMARY_METRICS = CAMPAIGN_METRICS + CALIBRATION_METRICS + ("elapsed_seconds",)
TRAIN_FILE = DATASET_ROOT / "Suzuki/suzuki_train.csv"
CONFIG_FILE = EXTENSION_ROOT / "configs/suzuki.yaml"
MODEL_SOURCE = EXTENSION_ROOT / "src/bgolearn_rxn/surrogates/component_ard_gp.py"
KG_SOURCE = EXTENSION_ROOT / "src/bgolearn_rxn/knowledge_gradient.py"
DEFAULT_OUTPUT = DATASET_ROOT / "submission/results/experimental_investigations/phase6-node6-component-ard-kg"
SPLIT_SEED_START = 1_010_000
SPLIT_SEED_STOP = 1_020_000
PROTECTED_SEED_STARTS = (1_020_000, 1_030_000)
EPISODE_COUNT = 60
INITIAL_ROWS = 20
CANDIDATE_ROWS = 9
BUDGET = 5
TAIL_FRACTION = 0.10
RUNTIME_RATIO_MAX = 2.0


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_episodes() -> list[dict]:
    source = pd.read_csv(TRAIN_FILE)
    episodes = suzuki_proxy._phase_random_episodes(
        source, phase=PHASE, seed_start=SPLIT_SEED_START, count=EPISODE_COUNT
    )
    seeds = [int(item["split_seed"]) for item in episodes]
    checks = {
        "episode_count": len(episodes) == EPISODE_COUNT,
        "unique_seeds": len(set(seeds)) == EPISODE_COUNT,
        "seed_family": min(seeds) >= SPLIT_SEED_START and max(seeds) < SPLIT_SEED_STOP,
        "initial_rows": all(len(item["initial_indices"]) == INITIAL_ROWS for item in episodes),
        "candidate_rows": all(len(item["candidate_indices"]) == CANDIDATE_ROWS for item in episodes),
        "budget": all(int(item["budget"]) == BUDGET for item in episodes),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Component-ARD KG episode contract failed: {checks}")
    return episodes


def build_problem(episode: dict):
    config, dataset, lookup = suzuki_proxy.build_problem(episode, "gp_only")
    config = replace(
        config,
        gp=replace(config.gp, target_transform="standard"),
        ensemble=replace(
            config.ensemble,
            enabled_models=(MODEL_NAME,),
            initial_weights={MODEL_NAME: 1.0},
        ),
        policy=replace(
            config.policy,
            first_query=MODEL_NAME,
            early_end=BUDGET,
            final_start=BUDGET + 1,
            challenge_every=0,
            novelty_disagreement_mode="off",
        ),
    )
    return config, dataset, lookup


def observation_variance(model) -> float:
    if model.theta_ is None:
        raise RuntimeError("Observation noise requires a fitted component-ARD GP.")
    noise = float(model._unpack(model.theta_)[2])
    scale = float(getattr(model.target_transform, "scale_", 1.0))
    return (noise * scale) ** 2


def policy_scores(
    variant: str,
    step: int,
    budget: int,
    posterior,
    model,
    X_remaining: np.ndarray,
    best_f: float,
) -> tuple[np.ndarray, str, dict[str, object]]:
    if step == 1:
        return posterior.mean.copy(), "posterior_mean", {
            "kg_scoring_seconds": 0.0,
            "quadrature_order": None,
        }
    if variant == BASELINE:
        progress = 0.0 if budget == 1 else (step - 1) / (budget - 1)
        scores = stable_log_expected_improvement(
            posterior.mean,
            posterior.std,
            best_f,
            xi=0.05 * (1.0 - progress),
        )
        return scores, "stable_log_ei", {
            "kg_scoring_seconds": 0.0,
            "quadrature_order": None,
        }
    if variant != CANDIDATE:
        raise ValueError(f"Unknown variant: {variant}")
    started = time.perf_counter()
    covariance = model.posterior_covariance(X_remaining)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues = np.linalg.eigvalsh(covariance)
    noise_variance = observation_variance(model)
    scores = finite_pool_knowledge_gradient(
        posterior.mean,
        covariance,
        observation_variance=noise_variance,
        best_f=best_f,
        quadrature_order=QUADRATURE_ORDER,
    )
    elapsed = time.perf_counter() - started
    return scores, "finite_pool_one_step_kg", {
        "kg_scoring_seconds": elapsed,
        "quadrature_order": QUADRATURE_ORDER,
        "observation_variance": noise_variance,
        "covariance_symmetry_error": float(np.max(np.abs(covariance - covariance.T))),
        "covariance_min_eigenvalue": float(eigenvalues.min()),
        "covariance_max_eigenvalue": float(eigenvalues.max()),
        "covariance_diagonal_min": float(np.diag(covariance).min()),
        "covariance_diagonal_max": float(np.diag(covariance).max()),
        "kg_score_min": float(scores.min()),
        "kg_score_max": float(scores.max()),
        "kg_score_mean": float(scores.mean()),
        "kg_zero_count": int(np.sum(scores <= 1e-15)),
        "candidate_count": len(scores),
    }


def campaign_metrics(trajectory: list[dict], candidate_best: float) -> dict[str, object]:
    raw = compute_campaign_metrics(
        [float(record["observed_yield"]) for record in trajectory],
        global_best=candidate_best,
    ).to_dict()
    t95 = raw["round_to_95_global_best"]
    return {
        "initial_round_found_best": float(raw["initial_round_found_best"]),
        "best_found": float(raw["best_found"]),
        "round_to_95_global_best": BUDGET + 1 if t95 is None else int(t95),
        "t95_reached": t95 is not None,
        "AUC_best_so_far": float(raw["auc_best_so_far"]),
    }


def calibration_record(item: dict, episode: dict, variant: str) -> dict:
    residual = float(item["observed_yield"] - item["predicted_yield"])
    std = max(float(item["predicted_std"]), 1e-12)
    row = {
        "episode": episode["episode"],
        "split_seed": int(episode["split_seed"]),
        "variant": variant,
        "step": int(item["step"]),
        "query_index": int(item["query_index"]),
        "predicted_yield": float(item["predicted_yield"]),
        "predicted_std": std,
        "observed_yield": float(item["observed_yield"]),
        "residual": residual,
        "standardized_residual": residual / std,
        "queried_nll": float(0.5 * (np.log(2.0 * np.pi) + 2.0 * np.log(std) + (residual / std) ** 2)),
    }
    for level in CALIBRATION_LEVELS:
        z_value = float(norm.ppf((1.0 + level) / 2.0))
        row[f"covered_{int(level * 100)}"] = bool(abs(residual) <= z_value * std)
    return row


def run_one(job: tuple[str, dict, str]) -> dict:
    variant, episode, output_root_string = job
    config, dataset, lookup = build_problem(episode)
    prepared = prepare_problem(dataset, config)
    ensemble = build_ensemble(config, prepared, seed=int(episode["campaign_seed"]))
    model = ensemble.models[MODEL_NAME]
    oracle = OfflineTableOracle(dataset=dataset, lookup=lookup)
    observed = dataset.initial.copy()
    X_observed = prepared.X_initial.copy()
    y_observed = observed[dataset.spec.target_column].to_numpy(dtype=float)
    remaining = np.arange(len(dataset.candidates), dtype=int)
    trajectory = []
    diagnostics = []
    started = time.perf_counter()
    for step in range(1, BUDGET + 1):
        X_remaining = prepared.X_candidates[remaining]
        candidate_rows = dataset.candidates.iloc[remaining]
        model.fit(X_observed, y_observed)
        posterior = model.predict(X_remaining)
        best_f = dataset.target_incumbent(observed)
        scores, acquisition_name, diagnostic = policy_scores(
            variant, step, BUDGET, posterior, model, X_remaining, best_f
        )
        candidate_ids = candidate_rows["query_index"].to_numpy(dtype=int)
        local_index = deterministic_argmax(
            scores, candidate_ids, seed=int(episode["campaign_seed"]) + step
        )
        query_index = int(candidate_ids[local_index])
        observed_yield = float(oracle.query(query_index))
        selected_global = int(remaining[local_index])
        selected_row = dataset.candidates.iloc[selected_global]
        item = {
            "step": step,
            "query_index": query_index,
            "observed_yield": observed_yield,
            "predicted_yield": float(posterior.mean[local_index]),
            "predicted_std": float(posterior.std[local_index]),
            "acquisition_name": acquisition_name,
            "acquisition_value": float(scores[local_index]),
        }
        trajectory.append(item)
        diagnostic.update(
            {
                "episode": episode["episode"],
                "split_seed": int(episode["split_seed"]),
                "variant": variant,
                "step": step,
                "selected_query_index": query_index,
                "selected_score": float(scores[local_index]),
                "selected_score_rank": int(np.sum(scores > scores[local_index] + 1e-15) + 1),
            }
        )
        diagnostics.append(diagnostic)
        new_observation = selected_row.loc[list(dataset.spec.feature_columns)].to_dict()
        new_observation[dataset.spec.target_column] = observed_yield
        observed = pd.concat([observed, pd.DataFrame([new_observation])], ignore_index=True)
        X_observed = np.vstack([X_observed, prepared.X_candidates[selected_global]])
        y_observed = np.append(y_observed, observed_yield)
        remaining = np.delete(remaining, local_index)
    elapsed = time.perf_counter() - started
    candidate_best = float(lookup[dataset.spec.target_column].max())
    metrics = campaign_metrics(trajectory, candidate_best)
    calibration = [calibration_record(item, episode, variant) for item in trajectory]
    metrics["queried_nll"] = float(np.mean([row["queried_nll"] for row in calibration]))
    for level in CALIBRATION_LEVELS:
        metrics[f"coverage_{int(level * 100)}"] = float(
            np.mean([row[f"covered_{int(level * 100)}"] for row in calibration])
        )
    payload = {
        "dataset": "Suzuki",
        "variant": variant,
        "episode": episode,
        "trajectory": trajectory,
        "metrics": metrics,
        "diagnostics": diagnostics,
        "contract": {
            "phase": PHASE,
            "train_only": True,
            "public_test_labels_used": False,
            "historical_confirmation_outcomes_used": False,
            "protected_1020_or_1030_episodes_opened": False,
            "model": MODEL_NAME,
            "target_transform": "standard",
            "query_1": "posterior_mean",
            "queries_2_to_5": "stable LogEI" if variant == BASELINE else "finite-pool one-step KG",
            "quadrature_order": QUADRATURE_ORDER if variant == CANDIDATE else None,
            "observation_noise_included": variant == CANDIDATE,
            "observed_incumbent_terminal_alternative": variant == CANDIDATE,
            "held_candidate_global_best": candidate_best,
        },
    }
    path = Path(output_root_string) / "trajectories" / variant / f"{episode['episode']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "metrics": {
            "dataset": "Suzuki",
            "phase": PHASE,
            "episode": episode["episode"],
            "split_seed": int(episode["split_seed"]),
            "campaign_seed": int(episode["campaign_seed"]),
            "variant": variant,
            "model": MODEL_NAME,
            "budget": BUDGET,
            "candidate_global_best": candidate_best,
            **metrics,
            "elapsed_seconds": elapsed,
            "kg_scoring_seconds": float(sum(row["kg_scoring_seconds"] for row in diagnostics)),
            "predicted_std_min": float(min(row["predicted_std"] for row in calibration)),
            "predicted_std_max": float(max(row["predicted_std"] for row in calibration)),
        },
        "calibration": calibration,
        "diagnostics": diagnostics,
    }


def mean_ci(values) -> tuple[float, float, float, float]:
    array = np.asarray(values, dtype=float).reshape(-1)
    mean = float(np.mean(array))
    if len(array) == 1:
        return mean, 0.0, mean, mean
    sd = float(np.std(array, ddof=1))
    half = float(t.ppf(0.975, len(array) - 1) * sd / np.sqrt(len(array)))
    return mean, sd, mean - half, mean + half


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for variant, group in rows.groupby("variant", sort=True):
        for metric in SUMMARY_METRICS + ("kg_scoring_seconds",):
            mean, sd, low, high = mean_ci(group[metric])
            records.append({"variant": variant, "metric": metric, "n_episodes": len(group), "mean": mean, "sample_sd": sd, "ci95_low": low, "ci95_high": high})
    return pd.DataFrame(records)


def paired_summary(rows: pd.DataFrame) -> pd.DataFrame:
    baseline = rows[rows["variant"].eq(BASELINE)].set_index("episode")
    candidate = rows[rows["variant"].eq(CANDIDATE)].set_index("episode")
    common = baseline.index.intersection(candidate.index)
    lower_better = {"round_to_95_global_best", "queried_nll", "elapsed_seconds"}
    records = []
    for metric in SUMMARY_METRICS:
        delta = candidate.loc[common, metric] - baseline.loc[common, metric]
        mean, sd, low, high = mean_ci(delta)
        wins = int((delta < -1e-10).sum()) if metric in lower_better else int((delta > 1e-10).sum())
        losses = int((delta > 1e-10).sum()) if metric in lower_better else int((delta < -1e-10).sum())
        records.append({"candidate": CANDIDATE, "baseline": BASELINE, "metric": metric, "n_pairs": len(common), "mean_delta": mean, "sample_sd_delta": sd, "ci95_low": low, "ci95_high": high, "wins": wins, "ties": len(common)-wins-losses, "losses": losses})
    return pd.DataFrame(records)


def tail_summary(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    count = max(1, int(np.ceil(TAIL_FRACTION * rows["episode"].nunique())))
    for variant, group in rows.groupby("variant", sort=True):
        values = np.sort(group["AUC_best_so_far"].to_numpy(dtype=float))
        records.append({"variant": variant, "n_episodes": len(values), "tail_fraction": TAIL_FRACTION, "tail_count": count, "auc_minimum": float(values[0]), "auc_decile_quantile": float(np.quantile(values, TAIL_FRACTION)), "auc_cvar10": float(np.mean(values[:count]))})
    return pd.DataFrame(records)


def calibration_summary(rounds: pd.DataFrame) -> pd.DataFrame:
    records = []
    for variant, group in rounds.groupby("variant", sort=True):
        record = {"variant": variant, "n_predictions": len(group), "queried_nll": float(group["queried_nll"].mean()), "standardized_residual_sd": float(group["standardized_residual"].std(ddof=1))}
        for level in CALIBRATION_LEVELS:
            record[f"coverage_{int(level * 100)}"] = float(group[f"covered_{int(level * 100)}"].mean())
        records.append(record)
    return pd.DataFrame(records)


def gate_decision(paired: pd.DataFrame, tails: pd.DataFrame, rows: pd.DataFrame, rounds: pd.DataFrame) -> dict:
    metrics = paired.set_index("metric")["mean_delta"]
    tail = tails.set_index("variant")["auc_cvar10"]
    candidate = rows[rows["variant"].eq(CANDIDATE)]
    baseline = rows[rows["variant"].eq(BASELINE)]
    query_one = rounds[rounds["step"].eq(1)].pivot(index="episode", columns="variant", values="query_index")
    runtime_ratio = float(candidate["elapsed_seconds"].mean() / baseline["elapsed_seconds"].mean())
    checks = {
        "exact_initial_ties": bool((query_one[CANDIDATE] == query_one[BASELINE]).all()) and float(metrics["initial_round_found_best"]) == 0.0,
        "auc_strictly_positive": float(metrics["AUC_best_so_far"]) > 0.0,
        "best_nonnegative": float(metrics["best_found"]) >= 0.0,
        "t95_nonpositive": float(metrics["round_to_95_global_best"]) <= 0.0,
        "cvar10_nonnegative": float(tail[CANDIDATE] - tail[BASELINE]) >= 0.0,
        "finite_noncollapsed_uncertainty": bool(np.isfinite(candidate[["predicted_std_min", "predicted_std_max"]].to_numpy()).all() and (candidate["predicted_std_min"] > 0.0).all()),
        "runtime_ratio_below_two": runtime_ratio < RUNTIME_RATIO_MAX,
    }
    return {"checks": checks, "passed": bool(all(checks.values())), "decision": "eligible_for_1020_validation" if all(checks.values()) else "stop_before_1020", "auc_cvar10_delta": float(tail[CANDIDATE]-tail[BASELINE]), "runtime_ratio": runtime_ratio, "runtime_ratio_limit_exclusive": RUNTIME_RATIO_MAX, "public_test_labels_used": False, "historical_confirmation_outcomes_used": False, "protected_1020_or_1030_episodes_opened": False, "production_default_changed": False}


def audit_results(rows: pd.DataFrame, rounds: pd.DataFrame, diagnostics: pd.DataFrame, output: Path, expected: int) -> dict:
    seeds = rows["split_seed"].astype(int)
    candidate = rows[rows["variant"].eq(CANDIDATE)]
    kg = diagnostics[(diagnostics["variant"].eq(CANDIDATE)) & (diagnostics["step"].gt(1))]
    query_one = rounds[rounds["step"].eq(1)].pivot(index="episode", columns="variant", values="query_index")
    checks = {
        "metric_rows": len(rows) == expected * 2,
        "episodes": rows["episode"].nunique() == expected,
        "paired_splits": bool(rows.groupby("episode")["split_seed"].nunique().eq(1).all()),
        "paired_candidate_best": bool(rows.groupby("episode")["candidate_global_best"].nunique().eq(1).all()),
        "round_rows": len(rounds) == expected * 2 * BUDGET,
        "trajectory_count": len(list((output / "trajectories").glob("*/*.json"))) == expected * 2,
        "only_1010_series": int(seeds.min()) >= SPLIT_SEED_START and int(seeds.max()) < SPLIT_SEED_STOP,
        "finite_metrics": bool(np.isfinite(rows[list(SUMMARY_METRICS)].to_numpy(dtype=float)).all()),
        "query_one_identical": bool((query_one[CANDIDATE] == query_one[BASELINE]).all()),
        "kg_rows_complete": len(kg) == expected * (BUDGET - 1),
        "quadrature_fixed_32": bool(kg["quadrature_order"].eq(QUADRATURE_ORDER).all()),
        "covariance_symmetric": bool((kg["covariance_symmetry_error"] <= 1e-10).all()),
        "covariance_psd": bool((kg["covariance_min_eigenvalue"] >= -1e-7).all()),
        "kg_scores_finite_nonnegative": bool(np.isfinite(kg[["kg_score_min", "kg_score_max", "selected_score"]].to_numpy()).all() and (kg["kg_score_min"] >= 0.0).all()),
        "finite_uncertainty": bool(np.isfinite(candidate[["predicted_std_min", "predicted_std_max"]].to_numpy()).all() and (candidate["predicted_std_min"] > 0.0).all()),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Component-ARD KG audit failed: {checks}")
    return {"passed": True, "checks": checks, "n_episodes": expected, "split_seed_min": int(seeds.min()), "split_seed_max": int(seeds.max()), "budget": BUDGET, "initial_rows": INITIAL_ROWS, "candidate_rows": CANDIDATE_ROWS, "public_test_labels_used": False, "historical_confirmation_outcomes_used": False, "protected_1020_or_1030_episodes_opened": False}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit-episodes", type=int, default=None, help="Smoke only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.jobs <= 0:
        raise ValueError("--jobs must be positive.")
    episodes = make_episodes()
    if args.limit_episodes is not None:
        if args.limit_episodes <= 0 or args.limit_episodes > EPISODE_COUNT:
            raise ValueError(f"--limit-episodes must be in [1, {EPISODE_COUNT}].")
        episodes = episodes[:args.limit_episodes]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(TRAIN_FILE)
    split_seeds = [int(item["split_seed"]) for item in episodes]
    manifest = {"stage": PHASE, "smoke": len(episodes) != EPISODE_COUNT, "train_only": True, "public_test_labels_used": False, "historical_confirmation_outcomes_used": False, "protected_1020_or_1030_episodes_opened": False, "input": {"path": str(TRAIN_FILE.relative_to(DATASET_ROOT)), "sha256": file_sha256(TRAIN_FILE), "n_rows": len(source)}, "implementations": {"driver": {"path": str(Path(__file__).relative_to(DATASET_ROOT)), "sha256": file_sha256(Path(__file__))}, "model": {"path": str(MODEL_SOURCE.relative_to(DATASET_ROOT)), "sha256": file_sha256(MODEL_SOURCE)}, "kg": {"path": str(KG_SOURCE.relative_to(DATASET_ROOT)), "sha256": file_sha256(KG_SOURCE)}, "config": {"path": str(CONFIG_FILE.relative_to(DATASET_ROOT)), "sha256": file_sha256(CONFIG_FILE)}}, "variants": list(VARIANTS), "model": MODEL_NAME, "episode_count": len(episodes), "full_episode_count": EPISODE_COUNT, "initial_rows": INITIAL_ROWS, "candidate_rows": CANDIDATE_ROWS, "budget": BUDGET, "split_seed_start": SPLIT_SEED_START, "split_seed_min": min(split_seeds), "split_seed_max": max(split_seeds), "protected_seed_starts": list(PROTECTED_SEED_STARTS), "target_transform": "standard", "baseline_acquisition": "posterior mean query 1, stable LogEI queries 2-5", "candidate_acquisition": "posterior mean query 1, exact finite-pool one-step KG queries 2-5", "quadrature_order": QUADRATURE_ORDER, "observation_noise_included": True, "observed_incumbent_terminal_alternative": True, "kg_mathematics_modified": False, "parameter_sweep": False, "production_default_changed": False, "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    jobs = [(variant, episode, str(args.output_dir)) for episode in episodes for variant in VARIANTS]
    results = []
    wall_started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(run_one, job): job for job in jobs}
        for future in as_completed(futures):
            result = future.result(); results.append(result); row = result["metrics"]
            print(f"{row['variant']} {row['episode']}: AUC={row['AUC_best_so_far']:.3f} kg_s={row['kg_scoring_seconds']:.4f}", flush=True)
    wall_seconds = time.perf_counter() - wall_started
    rows = pd.DataFrame([item["metrics"] for item in results]).sort_values(["episode", "variant"])
    rounds = pd.DataFrame([row for item in results for row in item["calibration"]]).sort_values(["episode", "variant", "step"])
    diagnostics = pd.DataFrame([row for item in results for row in item["diagnostics"]]).sort_values(["episode", "variant", "step"])
    summary = summarize(rows); paired = paired_summary(rows); tails = tail_summary(rows); calibration = calibration_summary(rounds)
    gate = gate_decision(paired, tails, rows, rounds); audit = audit_results(rows, rounds, diagnostics, args.output_dir, len(episodes))
    audit.update({"wall_seconds": wall_seconds, "aggregate_campaign_seconds": float(rows["elapsed_seconds"].sum()), "effective_parallelism": float(rows["elapsed_seconds"].sum()/max(wall_seconds,1e-12))})
    rows.to_csv(args.output_dir/"per_episode.csv", index=False); rounds.to_csv(args.output_dir/"calibration_predictions.csv", index=False); diagnostics.to_csv(args.output_dir/"acquisition_diagnostics.csv", index=False); summary.to_csv(args.output_dir/"summary.csv", index=False); paired.to_csv(args.output_dir/"paired_summary.csv", index=False); tails.to_csv(args.output_dir/"auc_tail_summary.csv", index=False); calibration.to_csv(args.output_dir/"calibration_summary.csv", index=False)
    first20_ids = {item["episode"] for item in episodes[:20]}; first20 = rows[rows["episode"].isin(first20_ids)]; first20.to_csv(args.output_dir/"first20_per_episode.csv", index=False); summarize(first20).to_csv(args.output_dir/"first20_summary.csv", index=False); paired_summary(first20).to_csv(args.output_dir/"first20_paired_summary.csv", index=False)
    (args.output_dir/"gate_decision.json").write_text(json.dumps(gate, indent=2, sort_keys=True), encoding="utf-8"); (args.output_dir/"audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    score = float(summary[(summary["variant"].eq(CANDIDATE)) & (summary["metric"].eq("AUC_best_so_far"))]["mean"].iloc[0]); print(f"absolute mean AUC_best_so_far={score:.8f}"); print(json.dumps(gate, sort_keys=True))


if __name__ == "__main__":
    main()
