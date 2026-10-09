#!/usr/bin/env python3
"""Paired 1010-series Suzuki discovery panel for Bayesian CP tensor learning."""

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
from bgolearn_rxn.campaign import CampaignRunner  # noqa: E402
from bgolearn_rxn.metrics import compute_campaign_metrics  # noqa: E402
from bgolearn_rxn.oracle import OfflineTableOracle  # noqa: E402
from bgolearn_rxn.surrogates.bayesian_cp_tensor import BayesianCPTensorSurrogate  # noqa: E402


PHASE = "phase6_deadline_discovery"
BASELINE = "component_ard"
CANDIDATE = "bayesian_cp"
VARIANTS = (BASELINE, CANDIDATE)
MODEL_NAMES = {BASELINE: "component_ard_gp", CANDIDATE: "bayesian_cp_tensor"}
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
TRAIN_FILE = DATASET_ROOT / "Suzuki" / "suzuki_train.csv"
CONFIG_FILE = EXTENSION_ROOT / "configs" / "suzuki.yaml"
BASELINE_SOURCE = EXTENSION_ROOT / "src/bgolearn_rxn/surrogates/component_ard_gp.py"
CANDIDATE_SOURCE = EXTENSION_ROOT / "src/bgolearn_rxn/surrogates/bayesian_cp_tensor.py"
FACTORY_SOURCE = EXTENSION_ROOT / "src/bgolearn_rxn/factory.py"
DEFAULT_OUTPUT = DATASET_ROOT / "submission/results/experimental_investigations/phase6-node3-bayesian-cp"
SPLIT_SEED_START = 1_010_000
SPLIT_SEED_STOP = 1_020_000
PROTECTED_SEED_STARTS = (1_020_000, 1_030_000)
EPISODE_COUNT = 60
INITIAL_ROWS = 20
CANDIDATE_ROWS = 9
BUDGET = 5
TAIL_FRACTION = 0.10
RUNTIME_RATIO_MAX = 3.0
RUNTIME_SECONDS_MAX = 15.0


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
        raise RuntimeError(f"Bayesian CP episode contract failed: {checks}")
    return episodes


def build_problem(episode: dict, variant: str):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    config, dataset, lookup = suzuki_proxy.build_problem(episode, "gp_only")
    model_name = MODEL_NAMES[variant]
    config = replace(
        config,
        gp=replace(config.gp, target_transform="standard"),
        interaction_model=replace(config.interaction_model, target_transform="standard"),
        ensemble=replace(
            config.ensemble,
            enabled_models=(model_name,),
            initial_weights={model_name: 1.0},
        ),
        policy=replace(
            config.policy,
            first_query=model_name,
            early_end=BUDGET,
            final_start=BUDGET + 1,
            challenge_every=0,
            novelty_disagreement_mode="off",
        ),
    )
    if tuple(config.ensemble.enabled_models) != (model_name,):
        raise RuntimeError("Discovery enabled an unexpected ensemble model.")
    if config.policy.first_query != model_name:
        raise RuntimeError("Discovery first-query model changed.")
    return config, dataset, lookup, model_name


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


def calibration_rows(trajectory: list[dict], episode: dict, variant: str) -> list[dict]:
    rows = []
    for item in trajectory:
        residual = float(item["observed_yield"] - item["predicted_yield"])
        std = max(float(item["predicted_std"]), 1e-12)
        row = {
            "episode": episode["episode"],
            "split_seed": int(episode["split_seed"]),
            "variant": variant,
            "step": int(item["step"]),
            "predicted_yield": float(item["predicted_yield"]),
            "predicted_std": std,
            "observed_yield": float(item["observed_yield"]),
            "residual": residual,
            "standardized_residual": residual / std,
            "queried_nll": float(
                0.5 * (np.log(2.0 * np.pi) + 2.0 * np.log(std) + (residual / std) ** 2)
            ),
        }
        for level in CALIBRATION_LEVELS:
            z_value = float(norm.ppf((1.0 + level) / 2.0))
            row[f"covered_{int(level * 100)}"] = bool(abs(residual) <= z_value * std)
        rows.append(row)
    return rows


def run_one(job: tuple[str, dict, str]) -> dict:
    variant, episode, output_root_string = job
    config, dataset, lookup, model_name = build_problem(episode, variant)
    candidate_best = float(lookup[config.dataset.target_column].max())
    runner = CampaignRunner(
        dataset, config, seed=int(episode["campaign_seed"]), backend="native"
    )
    started = time.perf_counter()
    result = runner.run(
        OfflineTableOracle(dataset=dataset, lookup=lookup), budget=BUDGET
    )
    elapsed = time.perf_counter() - started
    payload = result.to_payload()
    trajectory = payload["trajectory"]
    metrics = campaign_metrics(trajectory, candidate_best)
    calibration = calibration_rows(trajectory, episode, variant)
    metrics["queried_nll"] = float(np.mean([row["queried_nll"] for row in calibration]))
    for level in CALIBRATION_LEVELS:
        key = f"coverage_{int(level * 100)}"
        metrics[key] = float(
            np.mean([row[f"covered_{int(level * 100)}"] for row in calibration])
        )
    model = runner.ensemble.models[model_name]
    fit_history = list(getattr(model, "fit_history_", []))
    final_prediction = dict(getattr(model, "last_prediction_diagnostics_", {}))
    payload["bayesian_cp_contract"] = {
        "phase": PHASE,
        "variant": variant,
        "model": model_name,
        "train_only": True,
        "public_test_labels_used": False,
        "historical_confirmation_outcomes_used": False,
        "protected_1020_or_1030_episodes_opened": False,
        "held_candidate_global_best": candidate_best,
        "target_transform": "standard",
        "acquisition": "posterior mean query 1, stable LogEI queries 2-5",
        "raw_roles": list(BayesianCPTensorSurrogate.roles),
        "rank": getattr(model, "rank", None),
        "ensemble_size": getattr(model, "ensemble_size", None),
        "fit_history": fit_history,
        "final_prediction_diagnostics": final_prediction,
        "episode": episode,
    }
    path = Path(output_root_string) / "trajectories" / variant / f"{episode['episode']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    diagnostics = []
    for record in fit_history:
        diagnostics.append(
            {
                "episode": episode["episode"],
                "split_seed": int(episode["split_seed"]),
                "variant": variant,
                **record,
            }
        )
    if not diagnostics:
        diagnostics.append(
            {
                "episode": episode["episode"],
                "split_seed": int(episode["split_seed"]),
                "variant": variant,
                "fit_call": int(model.optimization_result_["fit_calls"]),
                "n_rows": INITIAL_ROWS + BUDGET - 1,
                "rank": None,
                "ensemble_size": 1,
                "factor_rms": None,
                "noise_std_min": None,
                "noise_std_max": None,
            }
        )
    return {
        "metrics": {
            "dataset": "Suzuki",
            "phase": PHASE,
            "episode_family": "covered_random_holdout",
            "episode": episode["episode"],
            "split_seed": int(episode["split_seed"]),
            "campaign_seed": int(episode["campaign_seed"]),
            "variant": variant,
            "model": model_name,
            "budget": BUDGET,
            "candidate_global_best": candidate_best,
            **metrics,
            "elapsed_seconds": elapsed,
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
        for metric in SUMMARY_METRICS:
            mean, sd, low, high = mean_ci(group[metric])
            records.append(
                {"variant": variant, "metric": metric, "n_episodes": len(group), "mean": mean, "sample_sd": sd, "ci95_low": low, "ci95_high": high}
            )
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
        records.append(
            {"candidate": CANDIDATE, "baseline": BASELINE, "metric": metric, "n_pairs": len(common), "mean_delta": mean, "sample_sd_delta": sd, "ci95_low": low, "ci95_high": high, "wins": wins, "ties": len(common) - wins - losses, "losses": losses}
        )
    return pd.DataFrame(records)


def tail_summary(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    count = max(1, int(np.ceil(TAIL_FRACTION * rows["episode"].nunique())))
    for variant, group in rows.groupby("variant", sort=True):
        values = np.sort(group["AUC_best_so_far"].to_numpy(dtype=float))
        records.append(
            {"variant": variant, "n_episodes": len(values), "tail_fraction": TAIL_FRACTION, "tail_count": count, "auc_minimum": float(values[0]), "auc_decile_quantile": float(np.quantile(values, TAIL_FRACTION)), "auc_cvar10": float(np.mean(values[:count]))}
        )
    return pd.DataFrame(records)


def calibration_summary(rounds: pd.DataFrame) -> pd.DataFrame:
    records = []
    for variant, group in rounds.groupby("variant", sort=True):
        record = {
            "variant": variant,
            "n_predictions": len(group),
            "queried_nll": float(group["queried_nll"].mean()),
            "standardized_residual_sd": float(group["standardized_residual"].std(ddof=1)),
        }
        for level in CALIBRATION_LEVELS:
            record[f"coverage_{int(level * 100)}"] = float(
                group[f"covered_{int(level * 100)}"].mean()
            )
        records.append(record)
    return pd.DataFrame(records)


def gate_decision(paired: pd.DataFrame, tails: pd.DataFrame, rows: pd.DataFrame) -> dict:
    metrics = paired.set_index("metric")["mean_delta"]
    tail = tails.set_index("variant")["auc_cvar10"]
    candidate = rows[rows["variant"].eq(CANDIDATE)]
    baseline = rows[rows["variant"].eq(BASELINE)]
    runtime_ratio = float(candidate["elapsed_seconds"].mean() / baseline["elapsed_seconds"].mean())
    checks = {
        "initial_nonnegative": float(metrics["initial_round_found_best"]) >= 0.0,
        "best_nonnegative": float(metrics["best_found"]) >= 0.0,
        "t95_nonpositive": float(metrics["round_to_95_global_best"]) <= 0.0,
        "auc_strictly_positive": float(metrics["AUC_best_so_far"]) > 0.0,
        "cvar10_nonnegative": float(tail[CANDIDATE] - tail[BASELINE]) >= 0.0,
        "finite_noncollapsed_uncertainty": bool(
            np.isfinite(candidate[["predicted_std_min", "predicted_std_max"]].to_numpy()).all()
            and (candidate["predicted_std_min"] > 0.0).all()
        ),
        "practical_runtime": runtime_ratio <= RUNTIME_RATIO_MAX
        and float(candidate["elapsed_seconds"].mean()) <= RUNTIME_SECONDS_MAX,
    }
    return {
        "checks": checks,
        "passed": bool(all(checks.values())),
        "decision": "eligible_for_1020_validation" if all(checks.values()) else "stop_before_1020",
        "auc_cvar10_delta": float(tail[CANDIDATE] - tail[BASELINE]),
        "runtime_ratio": runtime_ratio,
        "runtime_limits": {"ratio_max": RUNTIME_RATIO_MAX, "candidate_mean_seconds_max": RUNTIME_SECONDS_MAX},
        "public_test_labels_used": False,
        "historical_confirmation_outcomes_used": False,
        "protected_1020_or_1030_episodes_opened": False,
        "production_default_changed": False,
    }


def audit_results(rows: pd.DataFrame, rounds: pd.DataFrame, diagnostics: pd.DataFrame, output: Path, expected: int) -> dict:
    seeds = rows["split_seed"].astype(int)
    candidate = rows[rows["variant"].eq(CANDIDATE)]
    candidate_diagnostics = diagnostics[diagnostics["variant"].eq(CANDIDATE)]
    checks = {
        "metric_rows": len(rows) == expected * len(VARIANTS),
        "episodes": rows["episode"].nunique() == expected,
        "paired_splits": bool(rows.groupby("episode")["split_seed"].nunique().eq(1).all()),
        "paired_candidate_best": bool(rows.groupby("episode")["candidate_global_best"].nunique().eq(1).all()),
        "round_rows": len(rounds) == expected * len(VARIANTS) * BUDGET,
        "trajectory_count": len(list((output / "trajectories").glob("*/*.json"))) == expected * len(VARIANTS),
        "only_1010_series": int(seeds.min()) >= SPLIT_SEED_START and int(seeds.max()) < SPLIT_SEED_STOP,
        "finite_metrics": bool(np.isfinite(rows[list(SUMMARY_METRICS)].to_numpy(dtype=float)).all()),
        "finite_noncollapsed_uncertainty": bool(np.isfinite(candidate[["predicted_std_min", "predicted_std_max"]].to_numpy()).all() and (candidate["predicted_std_min"] > 0.0).all()),
        "cp_diagnostics_complete": len(candidate_diagnostics) == expected * BUDGET,
        "rank_fixed_three": bool(candidate_diagnostics["rank"].eq(3).all()),
        "posterior_ensemble_noncollapsed": bool(candidate_diagnostics["ensemble_size"].ge(2).all()),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Bayesian CP result audit failed: {checks}")
    return {
        "passed": True,
        "checks": checks,
        "n_episodes": expected,
        "split_seed_min": int(seeds.min()),
        "split_seed_max": int(seeds.max()),
        "budget": BUDGET,
        "initial_rows": INITIAL_ROWS,
        "candidate_rows": CANDIDATE_ROWS,
        "public_test_labels_used": False,
        "historical_confirmation_outcomes_used": False,
        "protected_1020_or_1030_episodes_opened": False,
    }


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
        episodes = episodes[: args.limit_episodes]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(TRAIN_FILE)
    split_seeds = [int(item["split_seed"]) for item in episodes]
    manifest = {
        "stage": PHASE,
        "smoke": len(episodes) != EPISODE_COUNT,
        "train_only": True,
        "covered_random_holdout": True,
        "public_test_labels_used": False,
        "historical_confirmation_outcomes_used": False,
        "protected_1020_or_1030_episodes_opened": False,
        "input": {"path": str(TRAIN_FILE.relative_to(DATASET_ROOT)), "sha256": file_sha256(TRAIN_FILE), "n_rows": len(source)},
        "implementations": {
            "driver": {"path": str(Path(__file__).relative_to(DATASET_ROOT)), "sha256": file_sha256(Path(__file__))},
            "candidate": {"path": str(CANDIDATE_SOURCE.relative_to(DATASET_ROOT)), "sha256": file_sha256(CANDIDATE_SOURCE)},
            "baseline": {"path": str(BASELINE_SOURCE.relative_to(DATASET_ROOT)), "sha256": file_sha256(BASELINE_SOURCE)},
            "factory": {"path": str(FACTORY_SOURCE.relative_to(DATASET_ROOT)), "sha256": file_sha256(FACTORY_SOURCE)},
            "config": {"path": str(CONFIG_FILE.relative_to(DATASET_ROOT)), "sha256": file_sha256(CONFIG_FILE)},
        },
        "variants": list(VARIANTS),
        "models": MODEL_NAMES,
        "episode_count": len(episodes),
        "full_episode_count": EPISODE_COUNT,
        "initial_rows": INITIAL_ROWS,
        "candidate_rows": CANDIDATE_ROWS,
        "budget": BUDGET,
        "split_seed_start": SPLIT_SEED_START,
        "split_seed_min": min(split_seeds),
        "split_seed_max": max(split_seeds),
        "protected_seed_starts": list(PROTECTED_SEED_STARTS),
        "target_transform": "standard",
        "acquisition": "posterior mean query 1, stable LogEI queries 2-5",
        "bayesian_cp": {"raw_roles": list(BayesianCPTensorSurrogate.roles), "rank": 3, "ensemble_size": BayesianCPTensorSurrogate.ensemble_size, "sweeps": BayesianCPTensorSurrogate.sweeps},
        "parameter_sweep": False,
        "interim_selection": False,
        "production_default_changed": False,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    jobs = [(variant, episode, str(args.output_dir)) for episode in episodes for variant in VARIANTS]
    results = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(run_one, job): job for job in jobs}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            row = result["metrics"]
            print(f"{row['variant']} {row['episode']}: AUC={row['AUC_best_so_far']:.3f} std_min={row['predicted_std_min']:.3f}", flush=True)
    wall_seconds = time.perf_counter() - started
    rows = pd.DataFrame([item["metrics"] for item in results]).sort_values(["episode", "variant"])
    rounds = pd.DataFrame([row for item in results for row in item["calibration"]]).sort_values(["episode", "variant", "step"])
    diagnostics = pd.DataFrame([row for item in results for row in item["diagnostics"]]).sort_values(["episode", "variant", "fit_call"])
    summary = summarize(rows)
    paired = paired_summary(rows)
    tails = tail_summary(rows)
    calibration = calibration_summary(rounds)
    gate = gate_decision(paired, tails, rows)
    audit = audit_results(rows, rounds, diagnostics, args.output_dir, len(episodes))
    audit["wall_seconds"] = wall_seconds
    audit["aggregate_campaign_seconds"] = float(rows["elapsed_seconds"].sum())
    audit["effective_parallelism"] = float(rows["elapsed_seconds"].sum() / max(wall_seconds, 1e-12))
    rows.to_csv(args.output_dir / "per_episode.csv", index=False)
    rounds.to_csv(args.output_dir / "calibration_predictions.csv", index=False)
    diagnostics.to_csv(args.output_dir / "cp_fit_diagnostics.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    paired.to_csv(args.output_dir / "paired_summary.csv", index=False)
    tails.to_csv(args.output_dir / "auc_tail_summary.csv", index=False)
    calibration.to_csv(args.output_dir / "calibration_summary.csv", index=False)
    first20_ids = {item["episode"] for item in episodes[:20]}
    first20 = rows[rows["episode"].isin(first20_ids)]
    first20.to_csv(args.output_dir / "first20_per_episode.csv", index=False)
    summarize(first20).to_csv(args.output_dir / "first20_summary.csv", index=False)
    paired_summary(first20).to_csv(args.output_dir / "first20_paired_summary.csv", index=False)
    (args.output_dir / "gate_decision.json").write_text(json.dumps(gate, indent=2, sort_keys=True), encoding="utf-8")
    (args.output_dir / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    score = float(summary[(summary["variant"].eq(CANDIDATE)) & (summary["metric"].eq("AUC_best_so_far"))]["mean"].iloc[0])
    print(f"absolute mean AUC_best_so_far={score:.8f}")
    print(json.dumps(gate, sort_keys=True))


if __name__ == "__main__":
    main()
