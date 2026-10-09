#!/usr/bin/env python3
"""Paired phase-6 Suzuki discovery panel for graph-diffusion product GP."""

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

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

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


PHASE = "phase6_deadline_discovery"
BASELINE = "component_ard"
CANDIDATE = "graph_diffusion"
VARIANTS = (BASELINE, CANDIDATE)
MODEL_NAMES = {
    BASELINE: "component_ard_gp",
    CANDIDATE: "graph_diffusion_product_gp",
}
RAW_ROLES = ("Electrophile", "Nucleophile", "Ligand", "Base", "Solvent")
CORE_METRICS = (
    "initial_round_found_best",
    "best_found",
    "round_to_95_global_best",
    "AUC_best_so_far",
)
SUMMARY_METRICS = CORE_METRICS + ("elapsed_seconds",)
TRAIN_FILE = DATASET_ROOT / "Suzuki" / "suzuki_train.csv"
CONFIG_FILE = EXTENSION_ROOT / "configs" / "suzuki.yaml"
CANDIDATE_SOURCE = (
    EXTENSION_ROOT / "src" / "bgolearn_rxn" / "surrogates" / "graph_diffusion_gp.py"
)
BASELINE_SOURCE = (
    EXTENSION_ROOT / "src" / "bgolearn_rxn" / "surrogates" / "component_ard_gp.py"
)
FACTORY_SOURCE = EXTENSION_ROOT / "src" / "bgolearn_rxn" / "factory.py"
DEFAULT_OUTPUT = (
    DATASET_ROOT
    / "submission"
    / "results"
    / "experimental_investigations"
    / "phase6-node1-graph-diffusion"
)
SPLIT_SEED_START = 1_010_000
SPLIT_SEED_STOP = 1_020_000
EPISODE_COUNT = 60
INITIAL_COUNT = 20
CANDIDATE_COUNT = 9
BUDGET = 5
TAIL_FRACTION = 0.10
PRACTICAL_RUNTIME_RATIO_MAX = 4.0
PRACTICAL_TOTAL_SECONDS_MAX = 1_500.0


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def make_episodes() -> list[dict]:
    frame = pd.read_csv(TRAIN_FILE)
    episodes = suzuki_proxy._phase_random_episodes(
        frame, phase=PHASE, seed_start=SPLIT_SEED_START, count=EPISODE_COUNT
    )
    seeds = [int(item["split_seed"]) for item in episodes]
    checks = {
        "episode_count": len(episodes) == EPISODE_COUNT,
        "unique_split_seeds": len(set(seeds)) == EPISODE_COUNT,
        "only_1010_series": min(seeds) >= SPLIT_SEED_START
        and max(seeds) < SPLIT_SEED_STOP,
        "twenty_initial": all(
            len(item["initial_indices"]) == INITIAL_COUNT for item in episodes
        ),
        "nine_candidates": all(
            len(item["candidate_indices"]) == CANDIDATE_COUNT for item in episodes
        ),
        "budget_five": all(int(item["budget"]) == BUDGET for item in episodes),
        "phase": all(item["phase"] == PHASE for item in episodes),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Phase-6 episode contract failed: {checks}")
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
        raise RuntimeError("Unexpected model enabled in phase-6 graph panel.")
    if config.policy.first_query != model_name:
        raise RuntimeError("Posterior-mean first-query model changed.")
    return config, dataset, lookup, model_name


def campaign_metrics(
    trajectory: list[dict], candidate_global_best: float
) -> dict[str, float | int | bool]:
    raw = compute_campaign_metrics(
        [float(item["observed_yield"]) for item in trajectory],
        global_best=candidate_global_best,
    ).to_dict()
    t95 = raw["round_to_95_global_best"]
    return {
        "initial_round_found_best": float(raw["initial_round_found_best"]),
        "best_found": float(raw["best_found"]),
        "round_to_95_global_best": BUDGET + 1 if t95 is None else int(t95),
        "t95_reached": t95 is not None,
        "AUC_best_so_far": float(raw["auc_best_so_far"]),
    }


def calibration_rows(
    trajectory: list[dict], *, episode: dict, variant: str
) -> list[dict]:
    z95 = float(norm.ppf(0.975))
    rows = []
    for item in trajectory:
        observed = float(item["observed_yield"])
        predicted = float(item["predicted_yield"])
        std = float(item["predicted_std"])
        residual = observed - predicted
        rows.append(
            {
                "episode": episode["episode"],
                "split_seed": int(episode["split_seed"]),
                "campaign_seed": int(episode["campaign_seed"]),
                "variant": variant,
                "step": int(item["step"]),
                "query_index": int(item["query_index"]),
                "predicted_yield": predicted,
                "predicted_std": std,
                "observed_yield": observed,
                "residual": residual,
                "absolute_residual": abs(residual),
                "standardized_residual": residual / max(std, 1e-12),
                "covered_95": bool(abs(residual) <= z95 * max(std, 1e-12)),
            }
        )
    return rows


def run_one(job: tuple[str, dict, str]) -> dict:
    variant, episode, output_root_string = job
    config, dataset, lookup, model_name = build_problem(episode, variant)
    oracle = OfflineTableOracle(dataset=dataset, lookup=lookup)
    started = time.perf_counter()
    runner = CampaignRunner(
        dataset, config, seed=int(episode["campaign_seed"]), backend="native"
    )
    result = runner.run(oracle, budget=BUDGET)
    elapsed = time.perf_counter() - started
    payload = result.to_payload()
    candidate_global_best = float(lookup[config.dataset.target_column].max())
    metrics = campaign_metrics(payload["trajectory"], candidate_global_best)
    calibration = calibration_rows(
        payload["trajectory"], episode=episode, variant=variant
    )
    model = runner.ensemble.models[model_name]
    diffusion = dict(getattr(model, "diffusion_scales_", {}))
    heat_times = dict(getattr(model, "heat_times_", {}))
    off_diagonal = dict(getattr(model, "off_diagonal_correlations_", {}))
    theta = getattr(model, "theta_", None)
    output_scale = None if theta is None else float(model._unpack(theta)[1])
    noise = None if theta is None else float(model._unpack(theta)[2])
    candidate_indices = set(int(value) for value in episode["candidate_indices"])
    queried = [int(item["query_index"]) for item in payload["trajectory"]]
    contract = {
        "phase": PHASE,
        "variant": variant,
        "model": model_name,
        "train_only": True,
        "input_file": str(TRAIN_FILE.relative_to(DATASET_ROOT)),
        "public_test_labels_used": False,
        "public_searchspace_labels_used": False,
        "historical_810_910_920_930_outcomes_used": False,
        "protected_1020_or_1030_outcomes_opened": False,
        "target_transform": "standard",
        "acquisition": "posterior mean query 1, stable LogEI queries 2-5",
        "held_candidate_global_best": candidate_global_best,
        "t95_miss_penalty": BUDGET + 1,
        "raw_roles": list(getattr(model, "roles", ())),
        "diffusion_scales": diffusion or None,
        "heat_times": heat_times or None,
        "off_diagonal_correlations": off_diagonal or None,
        "output_scale": output_scale,
        "noise": noise,
        "episode": episode,
        "query_audit": {
            "all_queries_in_candidate_pool": set(queried).issubset(candidate_indices),
            "queries_unique": len(queried) == len(set(queried)) == BUDGET,
            "finite_positive_uncertainty": all(
                np.isfinite(item["predicted_std"]) and item["predicted_std"] > 0.0
                for item in calibration
            ),
        },
    }
    payload["graph_diffusion_metrics"] = metrics
    payload["graph_diffusion_contract"] = contract
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
            "n_initial": len(episode["initial_indices"]),
            "n_candidates": len(episode["candidate_indices"]),
            "budget": BUDGET,
            "candidate_global_best": candidate_global_best,
            **metrics,
            "elapsed_seconds": elapsed,
            "diffusion_scales_json": json.dumps(diffusion, sort_keys=True),
            "heat_times_json": json.dumps(heat_times, sort_keys=True),
            "off_diagonal_correlations_json": json.dumps(
                off_diagonal, sort_keys=True
            ),
            "output_scale": output_scale,
            "noise": noise,
        },
        "calibration": calibration,
        "contract": contract,
    }


def mean_ci(values: pd.Series | np.ndarray) -> tuple[float, float, float, float]:
    array = np.asarray(values, dtype=float).reshape(-1)
    mean = float(np.mean(array))
    if len(array) == 1:
        return mean, 0.0, mean, mean
    sd = float(np.std(array, ddof=1))
    half = float(t.ppf(0.975, len(array) - 1) * sd / np.sqrt(len(array)))
    return mean, sd, mean - half, mean + half


def absolute_summary(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for variant, group in rows.groupby("variant", sort=True):
        for metric in SUMMARY_METRICS:
            mean, sd, low, high = mean_ci(group[metric])
            records.append(
                {
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
    baseline = rows[rows["variant"].eq(BASELINE)].set_index("episode")
    candidate = rows[rows["variant"].eq(CANDIDATE)].set_index("episode")
    common = baseline.index.intersection(candidate.index)
    records = []
    for metric in SUMMARY_METRICS:
        delta = candidate.loc[common, metric] - baseline.loc[common, metric]
        mean, sd, low, high = mean_ci(delta)
        higher = metric not in {"round_to_95_global_best", "elapsed_seconds"}
        wins = int((delta > 1e-10).sum()) if higher else int((delta < -1e-10).sum())
        losses = int((delta < -1e-10).sum()) if higher else int((delta > 1e-10).sum())
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


def tail_summary(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    tail_count = max(1, int(np.ceil(rows["episode"].nunique() * TAIL_FRACTION)))
    for variant, group in rows.groupby("variant", sort=True):
        values = np.sort(group["AUC_best_so_far"].to_numpy(dtype=float))
        tail = values[:tail_count]
        records.append(
            {
                "variant": variant,
                "n_episodes": len(values),
                "tail_fraction": TAIL_FRACTION,
                "tail_count": tail_count,
                "auc_minimum": float(values[0]),
                "auc_decile_quantile": float(np.quantile(values, TAIL_FRACTION)),
                "auc_cvar10": float(np.mean(tail)),
                "auc_tail_sample_sd": float(np.std(tail, ddof=1))
                if len(tail) > 1
                else 0.0,
            }
        )
    return pd.DataFrame(records)


def calibration_summary(rounds: pd.DataFrame) -> pd.DataFrame:
    records = []
    for variant, group in rounds.groupby("variant", sort=True):
        records.append(
            {
                "variant": variant,
                "n_predictions": len(group),
                "finite_positive_uncertainty": bool(
                    np.isfinite(group["predicted_std"]).all()
                    and (group["predicted_std"] > 0.0).all()
                ),
                "coverage_95": float(group["covered_95"].mean()),
                "mean_residual": float(group["residual"].mean()),
                "mean_absolute_error": float(group["absolute_residual"].mean()),
                "standardized_residual_sd": float(
                    group["standardized_residual"].std(ddof=1)
                ),
                "mean_predicted_std": float(group["predicted_std"].mean()),
                "minimum_predicted_std": float(group["predicted_std"].min()),
            }
        )
    return pd.DataFrame(records)


def diffusion_diagnostics(rows: pd.DataFrame) -> pd.DataFrame:
    candidate = rows[rows["variant"].eq(CANDIDATE)]
    records = []
    for value_type, column in (
        ("spectral_diffusion", "diffusion_scales_json"),
        ("heat_time", "heat_times_json"),
        ("off_diagonal_correlation", "off_diagonal_correlations_json"),
    ):
        parsed = [json.loads(value) for value in candidate[column]]
        for role in RAW_ROLES:
            values = np.asarray([item[role] for item in parsed], dtype=float)
            mean, sd, low, high = mean_ci(values)
            records.append(
                {
                    "value_type": value_type,
                    "role": role,
                    "n_episodes": len(values),
                    "mean": mean,
                    "sample_sd": sd,
                    "ci95_low": low,
                    "ci95_high": high,
                    "minimum": float(values.min()),
                    "maximum": float(values.max()),
                }
            )
    return pd.DataFrame(records)


def gate_decision(
    paired: pd.DataFrame,
    tails: pd.DataFrame,
    calibration: pd.DataFrame,
    rows: pd.DataFrame,
) -> dict[str, object]:
    metrics = paired.set_index("metric")
    tail_values = tails.set_index("variant")["auc_cvar10"]
    calibration_values = calibration.set_index("variant")
    baseline_runtime = float(
        rows.loc[rows["variant"].eq(BASELINE), "elapsed_seconds"].sum()
    )
    candidate_runtime = float(
        rows.loc[rows["variant"].eq(CANDIDATE), "elapsed_seconds"].sum()
    )
    checks = {
        "initial_mean_delta_nonnegative": float(
            metrics.loc["initial_round_found_best", "mean_delta"]
        )
        >= 0.0,
        "best_mean_delta_nonnegative": float(
            metrics.loc["best_found", "mean_delta"]
        )
        >= 0.0,
        "t95_mean_delta_nonpositive": float(
            metrics.loc["round_to_95_global_best", "mean_delta"]
        )
        <= 0.0,
        "auc_mean_delta_strictly_positive": float(
            metrics.loc["AUC_best_so_far", "mean_delta"]
        )
        > 0.0,
        "cvar10_nonnegative": float(tail_values[CANDIDATE])
        >= float(tail_values[BASELINE]),
        "finite_positive_uncertainty": bool(
            calibration_values.loc[CANDIDATE, "finite_positive_uncertainty"]
        ),
        "practical_runtime": candidate_runtime <= PRACTICAL_TOTAL_SECONDS_MAX
        and candidate_runtime
        <= PRACTICAL_RUNTIME_RATIO_MAX * max(baseline_runtime, 1e-12),
    }
    return {
        "checks": checks,
        "passed": bool(all(checks.values())),
        "decision": "eligible_for_1020_validation"
        if all(checks.values())
        else "stop_before_1020",
        "auc_cvar10": {key: float(value) for key, value in tail_values.items()},
        "runtime": {
            "baseline_total_seconds": baseline_runtime,
            "candidate_total_seconds": candidate_runtime,
            "candidate_to_baseline_ratio": candidate_runtime
            / max(baseline_runtime, 1e-12),
            "ratio_limit": PRACTICAL_RUNTIME_RATIO_MAX,
            "candidate_total_seconds_limit": PRACTICAL_TOTAL_SECONDS_MAX,
        },
        "public_test_labels_used": False,
        "protected_1020_or_1030_outcomes_opened": False,
        "production_default_changed": False,
    }


def audit_results(
    rows: pd.DataFrame,
    rounds: pd.DataFrame,
    contracts: list[dict],
    output_root: Path,
    expected: int,
) -> dict[str, object]:
    seeds = rows["split_seed"].astype(int)
    trajectory_count = len(list((output_root / "trajectories").glob("*/*.json")))
    query_audits = [item["query_audit"] for item in contracts]
    candidate_roles = [
        tuple(item["raw_roles"])
        for item in contracts
        if item["variant"] == CANDIDATE
    ]
    checks = {
        "metric_rows": len(rows) == expected * len(VARIANTS),
        "episodes": rows["episode"].nunique() == expected,
        "paired_split_seeds": bool(
            rows.groupby("episode")["split_seed"].nunique().eq(1).all()
        ),
        "paired_campaign_seeds": bool(
            rows.groupby("episode")["campaign_seed"].nunique().eq(1).all()
        ),
        "paired_candidate_global_best": bool(
            rows.groupby("episode")["candidate_global_best"].nunique().eq(1).all()
        ),
        "twenty_initial": bool(rows["n_initial"].eq(INITIAL_COUNT).all()),
        "nine_candidates": bool(rows["n_candidates"].eq(CANDIDATE_COUNT).all()),
        "five_rounds": len(rounds) == expected * len(VARIANTS) * BUDGET,
        "trajectory_count": trajectory_count == expected * len(VARIANTS),
        "only_1010_series": int(seeds.min()) >= SPLIT_SEED_START
        and int(seeds.max()) < SPLIT_SEED_STOP,
        "all_queries_in_candidate_pool": all(
            item["all_queries_in_candidate_pool"] for item in query_audits
        ),
        "queries_unique": all(item["queries_unique"] for item in query_audits),
        "finite_positive_uncertainty": all(
            item["finite_positive_uncertainty"] for item in query_audits
        ),
        "candidate_uses_exact_five_raw_roles": all(
            roles == RAW_ROLES for roles in candidate_roles
        )
        and len(candidate_roles) == expected,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Graph-diffusion result audit failed: {checks}")
    return {
        "passed": True,
        "checks": checks,
        "n_episodes": expected,
        "n_metric_rows": len(rows),
        "n_round_rows": len(rounds),
        "n_trajectories": trajectory_count,
        "split_seed_min": int(seeds.min()),
        "split_seed_max": int(seeds.max()),
        "budget": BUDGET,
        "public_test_labels_used": False,
        "public_searchspace_labels_used": False,
        "historical_810_910_920_930_outcomes_used": False,
        "protected_1020_or_1030_outcomes_opened": False,
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
    implementations = {
        "driver": {
            "path": str(Path(__file__).relative_to(DATASET_ROOT)),
            "sha256": file_sha256(Path(__file__)),
        },
        "candidate": {
            "path": str(CANDIDATE_SOURCE.relative_to(DATASET_ROOT)),
            "sha256": file_sha256(CANDIDATE_SOURCE),
        },
        "baseline": {
            "path": str(BASELINE_SOURCE.relative_to(DATASET_ROOT)),
            "sha256": file_sha256(BASELINE_SOURCE),
        },
        "factory": {
            "path": str(FACTORY_SOURCE.relative_to(DATASET_ROOT)),
            "sha256": file_sha256(FACTORY_SOURCE),
        },
        "config": {
            "path": str(CONFIG_FILE.relative_to(DATASET_ROOT)),
            "sha256": file_sha256(CONFIG_FILE),
        },
    }
    split_seeds = [int(item["split_seed"]) for item in episodes]
    manifest = {
        "stage": PHASE,
        "smoke": len(episodes) != EPISODE_COUNT,
        "train_only": True,
        "covered_random_holdout": True,
        "public_test_labels_used": False,
        "public_searchspace_labels_used": False,
        "historical_810_910_920_930_outcomes_used": False,
        "protected_1020_or_1030_outcomes_opened": False,
        "input": {
            "path": str(TRAIN_FILE.relative_to(DATASET_ROOT)),
            "sha256": file_sha256(TRAIN_FILE),
            "n_rows": len(source),
        },
        "implementations": implementations,
        "variants": list(VARIANTS),
        "models": MODEL_NAMES,
        "episode_count": len(episodes),
        "episode_contract_sha256": payload_sha256(episodes),
        "split_seed_start": SPLIT_SEED_START,
        "split_seed_min": min(split_seeds),
        "split_seed_max": max(split_seeds),
        "n_initial": INITIAL_COUNT,
        "n_candidates": CANDIDATE_COUNT,
        "budget": BUDGET,
        "raw_roles": list(RAW_ROLES),
        "target_transform": "standard",
        "acquisition": "posterior mean query 1, stable LogEI queries 2-5",
        "global_best_scope": "held candidate pool",
        "t95_miss_penalty": BUDGET + 1,
        "parameter_sweep": False,
        "interim_selection": False,
        "runtime_gate": {
            "candidate_to_baseline_ratio_max": PRACTICAL_RUNTIME_RATIO_MAX,
            "candidate_total_seconds_max": PRACTICAL_TOTAL_SECONDS_MAX,
        },
        "production_default_changed": False,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    jobs = [
        (variant, episode, str(args.output_dir))
        for episode in episodes
        for variant in VARIANTS
    ]
    results = []
    wall_started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(run_one, job): job for job in jobs}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            row = result["metrics"]
            print(
                f"{row['variant']} {row['episode']}: "
                f"AUC={row['AUC_best_so_far']:.3f}",
                flush=True,
            )
    wall_seconds = time.perf_counter() - wall_started

    rows = pd.DataFrame([item["metrics"] for item in results]).sort_values(
        ["episode", "variant"]
    )
    rounds = pd.DataFrame(
        [record for item in results for record in item["calibration"]]
    ).sort_values(["episode", "variant", "step"])
    contracts = [item["contract"] for item in results]
    rows.to_csv(args.output_dir / "per_episode.csv", index=False)
    rounds.to_csv(args.output_dir / "round_calibration.csv", index=False)
    summary = absolute_summary(rows)
    paired = paired_summary(rows)
    tails = tail_summary(rows)
    calibration = calibration_summary(rounds)
    diffusion = diffusion_diagnostics(rows)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    paired.to_csv(
        args.output_dir / "paired_graph_diffusion_vs_component_ard.csv", index=False
    )
    tails.to_csv(args.output_dir / "tail_summary.csv", index=False)
    calibration.to_csv(args.output_dir / "calibration_summary.csv", index=False)
    diffusion.to_csv(args.output_dir / "diffusion_diagnostics.csv", index=False)
    first_count = min(20, len(episodes))
    first_ids = [f"random_{PHASE}_{index:02d}" for index in range(first_count)]
    first = rows[rows["episode"].isin(first_ids)].sort_values(["episode", "variant"])
    if len(first) != first_count * len(VARIANTS):
        raise RuntimeError("Deterministic first slice is incomplete.")
    first.to_csv(args.output_dir / "first20_per_episode.csv", index=False)
    absolute_summary(first).to_csv(args.output_dir / "first20_summary.csv", index=False)
    paired_summary(first).to_csv(
        args.output_dir / "first20_paired_graph_diffusion_vs_component_ard.csv",
        index=False,
    )
    decision = gate_decision(paired, tails, calibration, rows)
    decision["wall_seconds"] = wall_seconds
    (args.output_dir / "gate_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8"
    )
    audit = audit_results(rows, rounds, contracts, args.output_dir, len(episodes))
    audit["wall_seconds"] = wall_seconds
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
