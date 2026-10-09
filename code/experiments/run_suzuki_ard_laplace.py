#!/usr/bin/env python3
"""Paired 910-series Suzuki pilot for local-Laplace component ARD."""

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


EXPERIMENT_ROOT = Path(__file__).resolve().parent
CODE_ROOT = EXPERIMENT_ROOT.parent
DATASET_ROOT = CODE_ROOT.parents[1]
EXTENSION_ROOT = CODE_ROOT / "bgolearn_rxn_extension"
sys.path.insert(0, str(EXTENSION_ROOT / "src"))
sys.path.insert(0, str(EXPERIMENT_ROOT))

import run_suzuki_ard_mode_integration as shared  # noqa: E402
import run_suzuki_holdout_proxy as suzuki_proxy  # noqa: E402
from bgolearn_rxn.campaign import CampaignRunner  # noqa: E402
from bgolearn_rxn.metrics import compute_campaign_metrics  # noqa: E402
from bgolearn_rxn.oracle import OfflineTableOracle  # noqa: E402
from bgolearn_rxn.surrogates.local_laplace_component_ard_gp import (  # noqa: E402
    LocalLaplaceComponentARDStructuredGP,
)


PHASE = "phase5_ard_laplace"
BASELINE = "component_ard"
CANDIDATE = "local_laplace_component_ard"
VARIANTS = (BASELINE, CANDIDATE)
MODEL_NAMES = {
    BASELINE: "component_ard_gp",
    CANDIDATE: "local_laplace_component_ard_gp",
}
CAMPAIGN_METRICS = shared.CAMPAIGN_METRICS
CALIBRATION_LEVELS = shared.CALIBRATION_LEVELS
CALIBRATION_METRICS = shared.CALIBRATION_METRICS
SUMMARY_METRICS = shared.SUMMARY_METRICS
TRAIN_FILE = DATASET_ROOT / "Suzuki" / "suzuki_train.csv"
CONFIG_FILE = EXTENSION_ROOT / "configs" / "suzuki.yaml"
BASELINE_SOURCE = (
    EXTENSION_ROOT / "src" / "bgolearn_rxn" / "surrogates" / "component_ard_gp.py"
)
CANDIDATE_SOURCE = (
    EXTENSION_ROOT
    / "src"
    / "bgolearn_rxn"
    / "surrogates"
    / "local_laplace_component_ard_gp.py"
)
FACTORY_SOURCE = EXTENSION_ROOT / "src" / "bgolearn_rxn" / "factory.py"
DEFAULT_OUTPUT = (
    DATASET_ROOT
    / "submission"
    / "results"
    / "experimental_investigations"
    / "phase5-node5-ard-laplace"
)
SPLIT_SEED_START = 910_000
SPLIT_SEED_STOP = 920_000
EPISODE_COUNT = 40
BUDGET = 5
TAIL_FRACTION = 0.10
PROMOTION_GATE = {
    "calibration_error_improvement_minimum": 0.01,
    "queried_nll_delta_maximum": 0.0,
    "auc_mean_delta_minimum": 0.0,
    "auc_cvar10_delta_minimum": -0.50,
    "best_found_delta_minimum": -0.25,
    "t95_delta_maximum": 0.10,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_episodes() -> list[dict]:
    frame = pd.read_csv(TRAIN_FILE)
    episodes = suzuki_proxy._phase_random_episodes(
        frame, phase=PHASE, seed_start=SPLIT_SEED_START, count=EPISODE_COUNT
    )
    seeds = [int(item["split_seed"]) for item in episodes]
    if len(episodes) != EPISODE_COUNT or len(set(seeds)) != EPISODE_COUNT:
        raise RuntimeError("Local-Laplace episode contract is incomplete.")
    if min(seeds) < SPLIT_SEED_START or max(seeds) >= SPLIT_SEED_STOP:
        raise RuntimeError("Local-Laplace pilot escaped the 910-series family.")
    if any(int(item["budget"]) != BUDGET for item in episodes):
        raise RuntimeError("Local-Laplace pilot budget changed.")
    return episodes


def build_problem(episode: dict, variant: str):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    config, dataset, lookup = suzuki_proxy.build_problem(episode, "gp_only")
    model_name = MODEL_NAMES[variant]
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
            early_end=BUDGET,
            final_start=BUDGET + 1,
            challenge_every=0,
            novelty_disagreement_mode="off",
        ),
    )
    if tuple(config.ensemble.enabled_models) != (model_name,):
        raise RuntimeError("Pilot enabled an unexpected ensemble model.")
    if config.policy.first_query != model_name:
        raise RuntimeError("Pilot first-query model changed.")
    if config.policy.early_end != BUDGET or config.policy.final_start != BUDGET + 1:
        raise RuntimeError("Pilot is not stable LogEI through query 5.")
    return config, dataset, lookup, model_name


def campaign_metrics(trajectory: list[dict], global_best: float) -> dict[str, object]:
    raw = compute_campaign_metrics(
        [float(record["observed_yield"]) for record in trajectory],
        global_best=global_best,
    ).to_dict()
    t95 = raw["round_to_95_global_best"]
    return {
        "initial_round_found_best": float(raw["initial_round_found_best"]),
        "best_found": float(raw["best_found"]),
        "round_to_95_global_best": BUDGET + 1 if t95 is None else int(t95),
        "t95_reached": t95 is not None,
        "AUC_best_so_far": float(raw["auc_best_so_far"]),
    }


def run_one(job: tuple[str, dict, str, float]) -> dict:
    variant, episode, output_root_string, global_best = job
    config, dataset, lookup, model_name = build_problem(episode, variant)
    runner = CampaignRunner(
        dataset, config, seed=int(episode["campaign_seed"]), backend="native"
    )
    started = time.perf_counter()
    result = runner.run(OfflineTableOracle(dataset=dataset, lookup=lookup), budget=BUDGET)
    elapsed = time.perf_counter() - started
    payload = result.to_payload()
    trajectory = payload["trajectory"]
    metrics = campaign_metrics(trajectory, global_best)
    calibration = shared.calibration_rows(trajectory, episode, variant)
    for level in CALIBRATION_LEVELS:
        key = f"coverage_{int(level * 100)}"
        metrics[key] = float(
            np.mean([row[f"covered_{int(level * 100)}"] for row in calibration])
        )
    metrics["queried_nll"] = float(np.mean([row["queried_nll"] for row in calibration]))
    model = runner.ensemble.models[model_name]
    history = list(getattr(model, "laplace_history_", []))
    final = history[-1] if history else {
        "fit_call": int(model.optimization_result_["fit_calls"]),
        "center_objective": float(model.optimization_result_["objective"]),
        "parameter_count": len(model.theta_),
        "curvature_diagonal": [],
        "curvature_steps": [],
        "normalized_posterior_std": [],
        "selected_axes": [],
        "attempts": [],
        "accepted_point_count": 1,
        "weights": [1.0],
        "effective_sample_count": 1.0,
        "fallback_to_map": False,
    }
    contract = {
        "phase": PHASE,
        "variant": variant,
        "model": model_name,
        "train_only": True,
        "covered_episode": True,
        "public_test_labels_used": False,
        "public_searchspace_labels_used": False,
        "retired_810_outcomes_used": False,
        "protected_920_or_930_episodes_opened": False,
        "target_transform": "standard",
        "acquisition": "posterior mean query 1, stable LogEI queries 2-5",
        "laplace_settings": {
            "curvature_step_fraction": getattr(model, "curvature_step_fraction", None),
            "curvature_floor_normalized": getattr(model, "curvature_floor_normalized", None),
            "max_sigma_axes": getattr(model, "max_sigma_axes", 0),
            "max_sigma_bound_fraction": getattr(model, "max_sigma_bound_fraction", None),
            "sigma_objective_window": getattr(model, "sigma_objective_window", None),
        },
        "laplace_history": history,
        "episode": episode,
    }
    payload["ard_laplace_metrics"] = metrics
    payload["ard_laplace_contract"] = contract
    output_path = (
        Path(output_root_string) / "trajectories" / variant / f"{episode['episode']}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    diagnostic_rows = []
    for record in history:
        diagnostic_rows.append(
            {
                "episode": episode["episode"],
                "split_seed": int(episode["split_seed"]),
                "variant": variant,
                "fit_call": int(record["fit_call"]),
                "center_objective": float(record["center_objective"]),
                "parameter_count": int(record["parameter_count"]),
                "accepted_point_count": int(record["accepted_point_count"]),
                "effective_sample_count": float(record["effective_sample_count"]),
                "fallback_to_map": bool(record["fallback_to_map"]),
                "selected_axes_json": json.dumps(record["selected_axes"]),
                "curvature_diagonal_json": json.dumps(record["curvature_diagonal"]),
                "normalized_posterior_std_json": json.dumps(record["normalized_posterior_std"]),
                "weights_json": json.dumps(record["weights"]),
                "attempts_json": json.dumps(record["attempts"]),
            }
        )
    if not diagnostic_rows:
        diagnostic_rows.append(
            {
                "episode": episode["episode"],
                "split_seed": int(episode["split_seed"]),
                "variant": variant,
                "fit_call": int(final["fit_call"]),
                "center_objective": float(final["center_objective"]),
                "parameter_count": int(final["parameter_count"]),
                "accepted_point_count": 1,
                "effective_sample_count": 1.0,
                "fallback_to_map": False,
                "selected_axes_json": "[]",
                "curvature_diagonal_json": "[]",
                "normalized_posterior_std_json": "[]",
                "weights_json": "[1.0]",
                "attempts_json": "[]",
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
            "full_global_best": global_best,
            **metrics,
            "elapsed_seconds": elapsed,
            "final_accepted_point_count": int(final["accepted_point_count"]),
            "final_effective_sample_count": float(final["effective_sample_count"]),
            "final_fallback_to_map": bool(final["fallback_to_map"]),
        },
        "calibration": calibration,
        "diagnostics": diagnostic_rows,
    }


def paired_summary(rows: pd.DataFrame) -> pd.DataFrame:
    baseline = rows[rows["variant"].eq(BASELINE)].set_index("episode")
    candidate = rows[rows["variant"].eq(CANDIDATE)].set_index("episode")
    common = baseline.index.intersection(candidate.index)
    records = []
    lower_better = {"round_to_95_global_best", "queried_nll", "elapsed_seconds"}
    for metric in SUMMARY_METRICS:
        delta = candidate.loc[common, metric] - baseline.loc[common, metric]
        mean, sd, low, high = shared.mean_ci(delta)
        wins = int((delta < -1e-10).sum()) if metric in lower_better else int((delta > 1e-10).sum())
        losses = int((delta > 1e-10).sum()) if metric in lower_better else int((delta < -1e-10).sum())
        records.append(
            {
                "candidate": CANDIDATE,
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


def gate_decision(
    paired: pd.DataFrame, tails: pd.DataFrame, calibration: pd.DataFrame
) -> dict:
    metrics = paired.set_index("metric")["mean_delta"]
    tail = tails.set_index("variant")["auc_cvar10"]
    cal = calibration.set_index("variant")
    coverage_improvement = float(
        cal.loc[BASELINE, "mean_absolute_coverage_error"]
        - cal.loc[CANDIDATE, "mean_absolute_coverage_error"]
    )
    checks = {
        "coverage_error_materially_improved": coverage_improvement
        >= PROMOTION_GATE["calibration_error_improvement_minimum"],
        "queried_nll_noninferior": float(metrics["queried_nll"])
        <= PROMOTION_GATE["queried_nll_delta_maximum"],
        "auc_mean_noninferior": float(metrics["AUC_best_so_far"])
        >= PROMOTION_GATE["auc_mean_delta_minimum"],
        "auc_cvar10_noninferior": float(tail[CANDIDATE] - tail[BASELINE])
        >= PROMOTION_GATE["auc_cvar10_delta_minimum"],
        "best_found_noninferior": float(metrics["best_found"])
        >= PROMOTION_GATE["best_found_delta_minimum"],
        "t95_noninferior": float(metrics["round_to_95_global_best"])
        <= PROMOTION_GATE["t95_delta_maximum"],
    }
    return {
        "gate": PROMOTION_GATE,
        "checks": checks,
        "coverage_error_improvement": coverage_improvement,
        "auc_cvar10_delta": float(tail[CANDIDATE] - tail[BASELINE]),
        "passed": bool(all(checks.values())),
        "decision": "eligible_for_920_validation" if all(checks.values()) else "stop_before_920",
        "public_test_labels_used": False,
    }


def audit_results(
    rows: pd.DataFrame,
    rounds: pd.DataFrame,
    diagnostics: pd.DataFrame,
    output: Path,
    expected: int,
) -> dict:
    seeds = rows["split_seed"].astype(int)
    candidate_diagnostics = diagnostics[diagnostics["variant"].eq(CANDIDATE)]
    checks = {
        "metric_rows": len(rows) == expected * len(VARIANTS),
        "episodes": rows["episode"].nunique() == expected,
        "paired_splits": bool(rows.groupby("episode")["split_seed"].nunique().eq(1).all()),
        "round_rows": len(rounds) == expected * len(VARIANTS) * BUDGET,
        "diagnostic_rows_present": len(diagnostics) >= expected * len(VARIANTS),
        "trajectory_count": len(list((output / "trajectories").glob("*/*.json"))) == expected * len(VARIANTS),
        "only_910_series": int(seeds.min()) >= SPLIT_SEED_START and int(seeds.max()) < SPLIT_SEED_STOP,
        "finite_metrics": bool(np.isfinite(rows[list(SUMMARY_METRICS)].to_numpy(dtype=float)).all()),
        "candidate_points_noncollapsed": bool(
            len(candidate_diagnostics)
            and candidate_diagnostics["accepted_point_count"].min() >= 2
            and not candidate_diagnostics["fallback_to_map"].any()
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Local-Laplace result audit failed: {checks}")
    return {
        "passed": True,
        "checks": checks,
        "n_episodes": expected,
        "split_seed_min": int(seeds.min()),
        "split_seed_max": int(seeds.max()),
        "budget": BUDGET,
        "public_test_labels_used": False,
        "public_searchspace_labels_used": False,
        "retired_810_outcomes_used": False,
        "protected_920_or_930_episodes_opened": False,
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
    global_best = float(source["Yield"].max())
    split_seeds = [int(item["split_seed"]) for item in episodes]
    laplace_policy = {
        "curvature_step_fraction": LocalLaplaceComponentARDStructuredGP.curvature_step_fraction,
        "curvature_floor_normalized": LocalLaplaceComponentARDStructuredGP.curvature_floor_normalized,
        "max_sigma_axes": LocalLaplaceComponentARDStructuredGP.max_sigma_axes,
        "max_sigma_bound_fraction": LocalLaplaceComponentARDStructuredGP.max_sigma_bound_fraction,
        "sigma_objective_window": LocalLaplaceComponentARDStructuredGP.sigma_objective_window,
    }
    manifest = {
        "stage": PHASE,
        "smoke": len(episodes) != EPISODE_COUNT,
        "train_only": True,
        "covered_random_holdout": True,
        "public_test_labels_used": False,
        "public_searchspace_labels_used": False,
        "retired_810_outcomes_used": False,
        "protected_920_or_930_episodes_opened": False,
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
        "split_seed_start": SPLIT_SEED_START,
        "split_seed_min": min(split_seeds),
        "split_seed_max": max(split_seeds),
        "target_transform": "standard",
        "acquisition": "posterior mean query 1, stable LogEI queries 2-5",
        "local_laplace_policy": laplace_policy,
        "parameter_sweep": False,
        "interim_selection": False,
        "promotion_gate": PROMOTION_GATE,
        "production_default_changed": False,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    jobs = [
        (variant, episode, str(args.output_dir), global_best)
        for episode in episodes
        for variant in VARIANTS
    ]
    results = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(run_one, job): job for job in jobs}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            row = result["metrics"]
            print(
                f"{row['variant']} {row['episode']}: AUC={row['AUC_best_so_far']:.3f} points={row['final_accepted_point_count']}",
                flush=True,
            )
    wall_seconds = time.perf_counter() - started

    rows = pd.DataFrame([item["metrics"] for item in results]).sort_values(["episode", "variant"])
    rounds = pd.DataFrame([row for item in results for row in item["calibration"]]).sort_values(["episode", "variant", "step"])
    diagnostics = pd.DataFrame([row for item in results for row in item["diagnostics"]]).sort_values(["episode", "variant", "fit_call"])
    summary = shared.summarize(rows)
    paired = paired_summary(rows)
    tails = shared.tail_summary(rows)
    calibration = shared.calibration_summary(rounds)
    decision = gate_decision(paired, tails, calibration)
    audit = audit_results(rows, rounds, diagnostics, args.output_dir, len(episodes))
    audit["wall_seconds"] = wall_seconds
    audit["aggregate_campaign_seconds"] = float(rows["elapsed_seconds"].sum())
    audit["effective_parallelism"] = float(rows["elapsed_seconds"].sum() / max(wall_seconds, 1e-12))
    rows.to_csv(args.output_dir / "per_episode.csv", index=False)
    rounds.to_csv(args.output_dir / "round_calibration.csv", index=False)
    diagnostics.to_csv(args.output_dir / "laplace_diagnostics.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    paired.to_csv(args.output_dir / "paired_laplace_vs_component_ard.csv", index=False)
    tails.to_csv(args.output_dir / "tail_summary.csv", index=False)
    calibration.to_csv(args.output_dir / "calibration_summary.csv", index=False)
    first20_names = {item["episode"] for item in episodes[:20]}
    first20 = rows[rows["episode"].isin(first20_names)]
    first20.to_csv(args.output_dir / "first20_per_episode.csv", index=False)
    shared.summarize(first20).to_csv(args.output_dir / "first20_summary.csv", index=False)
    paired_summary(first20).to_csv(args.output_dir / "first20_paired.csv", index=False)
    (args.output_dir / "gate_decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")
    (args.output_dir / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    score = float(summary[(summary["variant"].eq(CANDIDATE)) & (summary["metric"].eq("AUC_best_so_far"))]["mean"].iloc[0])
    print(f"absolute mean AUC_best_so_far={score:.8f}")
    print(json.dumps(decision, sort_keys=True))


if __name__ == "__main__":
    main()
