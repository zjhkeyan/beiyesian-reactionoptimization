#!/usr/bin/env python3
"""Paired 910-series Suzuki pilot for global-local strong-heredity ARD."""

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


PHASE = "phase5_strong_heredity_ard"
VARIANTS = ("component_ard", "strong_heredity_ard")
BASELINE = "component_ard"
CANDIDATE = "strong_heredity_ard"
MODEL_NAMES = {
    BASELINE: "component_ard_gp",
    CANDIDATE: "strong_heredity_component_ard_gp",
}
CORE_METRICS = (
    "initial_round_found_best",
    "best_found",
    "round_to_95_global_best",
    "AUC_best_so_far",
)
SUMMARY_METRICS = CORE_METRICS + ("coverage_95", "elapsed_seconds")
TRAIN_FILE = DATASET_ROOT / "Suzuki" / "suzuki_train.csv"
CONFIG_FILE = EXTENSION_ROOT / "configs" / "suzuki.yaml"
CANDIDATE_SOURCE = (
    EXTENSION_ROOT
    / "src"
    / "bgolearn_rxn"
    / "surrogates"
    / "strong_heredity_component_ard_gp.py"
)
BASELINE_SOURCE = (
    EXTENSION_ROOT
    / "src"
    / "bgolearn_rxn"
    / "surrogates"
    / "component_ard_gp.py"
)
SPLIT_SEED_START = 910_000
SPLIT_SEED_STOP = 920_000
EPISODE_COUNT = 60
BUDGET = 5
TAIL_FRACTION = 0.10
PROMOTION_GATE = {
    "auc_mean_delta_strictly_positive": True,
    "initial_mean_delta_nonnegative": True,
    "best_mean_delta_nonnegative": True,
    "t95_mean_delta_nonpositive": True,
    "cvar10_nonnegative": True,
    "finite_noncollapsed_amplitudes": True,
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
        frame,
        phase=PHASE,
        seed_start=SPLIT_SEED_START,
        count=EPISODE_COUNT,
    )
    seeds = [int(item["split_seed"]) for item in episodes]
    if len(episodes) != EPISODE_COUNT or len(set(seeds)) != EPISODE_COUNT:
        raise RuntimeError("Strong-heredity episode contract is incomplete.")
    if min(seeds) < SPLIT_SEED_START or max(seeds) >= SPLIT_SEED_STOP:
        raise RuntimeError("Strong-heredity pilot escaped the 910-series family.")
    if any(int(item["budget"]) != BUDGET for item in episodes):
        raise RuntimeError("Strong-heredity pilot budget changed.")
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


def campaign_metrics(
    trajectory: list[dict], global_best: float, budget: int
) -> dict[str, float | int | bool]:
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


def _calibration_rows(
    trajectory: list[dict], *, episode: dict, variant: str
) -> list[dict]:
    records = []
    z95 = float(norm.ppf(0.975))
    for item in trajectory:
        residual = float(item["observed_yield"] - item["predicted_yield"])
        std = max(float(item["predicted_std"]), 1e-12)
        records.append(
            {
                "episode": episode["episode"],
                "split_seed": int(episode["split_seed"]),
                "variant": variant,
                "step": int(item["step"]),
                "predicted_yield": float(item["predicted_yield"]),
                "predicted_std": std,
                "observed_yield": float(item["observed_yield"]),
                "residual": residual,
                "absolute_residual": abs(residual),
                "standardized_residual": residual / std,
                "covered_95": bool(abs(residual) <= z95 * std),
            }
        )
    return records


def run_one(job: tuple[str, dict, str, float]) -> dict:
    variant, episode, output_root_string, full_global_best = job
    config, dataset, lookup, model_name = build_problem(episode, variant)
    oracle = OfflineTableOracle(dataset=dataset, lookup=lookup)
    runner = CampaignRunner(
        dataset,
        config,
        seed=int(episode["campaign_seed"]),
        backend="native",
    )
    started = time.perf_counter()
    result = runner.run(oracle, budget=BUDGET)
    elapsed = time.perf_counter() - started
    payload = result.to_payload()
    metrics = campaign_metrics(payload["trajectory"], full_global_best, BUDGET)
    calibration = _calibration_rows(
        payload["trajectory"], episode=episode, variant=variant
    )
    model = runner.ensemble.models[model_name]
    amplitudes = dict(getattr(model, "component_amplitudes_", {}))
    shares = dict(getattr(model, "component_shares_", {}))
    local = dict(getattr(model, "local_interaction_multipliers_", {}))
    parents = dict(getattr(model, "parent_geometric_means_", {}))
    global_multiplier = getattr(model, "global_interaction_multiplier_", None)
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
        "full_train_global_best_t95": True,
        "t95_miss_penalty": BUDGET + 1,
        "episode": episode,
        "component_amplitudes": amplitudes,
        "component_shares": shares,
        "global_interaction_multiplier": global_multiplier,
        "local_interaction_multipliers": local,
        "parent_geometric_means": parents,
    }
    payload["strong_heredity_metrics"] = metrics
    payload["strong_heredity_contract"] = contract
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
            "budget": BUDGET,
            "full_global_best": full_global_best,
            **metrics,
            "coverage_95": float(np.mean([item["covered_95"] for item in calibration])),
            "elapsed_seconds": elapsed,
            "component_amplitudes_json": json.dumps(amplitudes, sort_keys=True),
            "component_shares_json": json.dumps(shares, sort_keys=True),
            "global_interaction_multiplier": global_multiplier,
            "local_interaction_multipliers_json": (
                json.dumps(local, sort_keys=True) if local else None
            ),
            "parent_geometric_means_json": (
                json.dumps(parents, sort_keys=True) if parents else None
            ),
        },
        "calibration": calibration,
    }


def mean_ci(values: pd.Series | np.ndarray) -> tuple[float, float, float, float]:
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
                {
                    "dataset": "Suzuki",
                    "phase": PHASE,
                    "episode_family": "covered_random_holdout",
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
                "dataset": "Suzuki",
                "phase": PHASE,
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
                "auc_tail_sample_sd": float(np.std(tail, ddof=1)) if len(tail) > 1 else 0.0,
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
                "coverage_95": float(group["covered_95"].mean()),
                "mean_residual": float(group["residual"].mean()),
                "mean_absolute_error": float(group["absolute_residual"].mean()),
                "standardized_residual_sd": float(
                    group["standardized_residual"].std(ddof=1)
                ),
            }
        )
    return pd.DataFrame(records)


def amplitude_diagnostics(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for variant, group in rows.groupby("variant", sort=True):
        for value_type, column in (
            ("amplitude", "component_amplitudes_json"),
            ("share", "component_shares_json"),
        ):
            parsed = [json.loads(value) for value in group[column]]
            for component in sorted(set().union(*(item.keys() for item in parsed))):
                values = np.asarray([item[component] for item in parsed], dtype=float)
                mean, sd, low, high = mean_ci(values)
                records.append(
                    {
                        "variant": variant,
                        "value_type": value_type,
                        "component": component,
                        "n_episodes": len(values),
                        "mean": mean,
                        "sample_sd": sd,
                        "ci95_low": low,
                        "ci95_high": high,
                        "minimum": float(values.min()),
                        "maximum": float(values.max()),
                    }
                )
    candidate = rows[rows["variant"].eq(CANDIDATE)]
    globals_ = candidate["global_interaction_multiplier"].to_numpy(dtype=float)
    mean, sd, low, high = mean_ci(globals_)
    records.append(
        {
            "variant": CANDIDATE,
            "value_type": "global_multiplier",
            "component": "all_pairs",
            "n_episodes": len(globals_),
            "mean": mean,
            "sample_sd": sd,
            "ci95_low": low,
            "ci95_high": high,
            "minimum": float(globals_.min()),
            "maximum": float(globals_.max()),
        }
    )
    for value_type, column in (
        ("local_multiplier", "local_interaction_multipliers_json"),
        ("parent_geometric_mean", "parent_geometric_means_json"),
    ):
        parsed = [json.loads(value) for value in candidate[column]]
        for component in sorted(set().union(*(item.keys() for item in parsed))):
            values = np.asarray([item[component] for item in parsed], dtype=float)
            mean, sd, low, high = mean_ci(values)
            records.append(
                {
                    "variant": CANDIDATE,
                    "value_type": value_type,
                    "component": component,
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
    paired: pd.DataFrame, tails: pd.DataFrame, amplitudes: pd.DataFrame
) -> dict[str, object]:
    metrics = paired.set_index("metric")
    tail_values = tails.set_index("variant")["auc_cvar10"]
    candidate_amplitudes = amplitudes[
        (amplitudes["variant"].eq(CANDIDATE))
        & (amplitudes["value_type"].isin(["amplitude", "share"]))
    ]
    diagnostics = candidate_amplitudes[
        ["mean", "sample_sd", "minimum", "maximum"]
    ].to_numpy(dtype=float)
    checks = {
        "auc_mean_delta_strictly_positive": float(
            metrics.loc["AUC_best_so_far", "mean_delta"]
        )
        > 0.0,
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
        "cvar10_nonnegative": float(tail_values[CANDIDATE])
        >= float(tail_values[BASELINE]),
        "finite_noncollapsed_amplitudes": bool(
            np.isfinite(diagnostics).all()
            and (candidate_amplitudes["minimum"] > 0.0).all()
            and (
                candidate_amplitudes[candidate_amplitudes["value_type"].eq("share")][
                    "maximum"
                ]
                < 0.95
            ).all()
        ),
    }
    return {
        "gate": PROMOTION_GATE,
        "checks": checks,
        "passed": bool(all(checks.values())),
        "decision": "eligible_for_920_validation" if all(checks.values()) else "stop_before_920",
        "auc_cvar10": {key: float(value) for key, value in tail_values.items()},
        "public_test_labels_used": False,
        "protected_920_or_930_episodes_opened": False,
        "production_default_changed": False,
    }


def audit_results(
    rows: pd.DataFrame, rounds: pd.DataFrame, output_root: Path, expected: int
) -> dict[str, object]:
    seeds = rows["split_seed"].astype(int)
    trajectory_count = len(list((output_root / "trajectories").glob("*/*.json")))
    checks = {
        "metric_rows": len(rows) == expected * len(VARIANTS),
        "episodes": rows["episode"].nunique() == expected,
        "paired_splits": bool(rows.groupby("episode")["split_seed"].nunique().eq(1).all()),
        "round_rows": len(rounds) == expected * len(VARIANTS) * BUDGET,
        "trajectory_count": trajectory_count == expected * len(VARIANTS),
        "only_910_series": int(seeds.min()) >= SPLIT_SEED_START
        and int(seeds.max()) < SPLIT_SEED_STOP,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Strong-heredity result audit failed: {checks}")
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
        "retired_810_outcomes_used": False,
        "protected_920_or_930_episodes_opened": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--output-dir", type=Path, required=True)
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
    implementations = {
        "driver": {"path": str(Path(__file__).relative_to(DATASET_ROOT)), "sha256": file_sha256(Path(__file__))},
        "candidate": {"path": str(CANDIDATE_SOURCE.relative_to(DATASET_ROOT)), "sha256": file_sha256(CANDIDATE_SOURCE)},
        "baseline": {"path": str(BASELINE_SOURCE.relative_to(DATASET_ROOT)), "sha256": file_sha256(BASELINE_SOURCE)},
        "config": {"path": str(CONFIG_FILE.relative_to(DATASET_ROOT)), "sha256": file_sha256(CONFIG_FILE)},
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
        "implementations": implementations,
        "variants": list(VARIANTS),
        "models": MODEL_NAMES,
        "episode_count": len(episodes),
        "split_seed_start": SPLIT_SEED_START,
        "split_seed_min": min(split_seeds),
        "split_seed_max": max(split_seeds),
        "target_transform": "standard",
        "acquisition": "posterior mean query 1, stable LogEI queries 2-5",
        "full_train_global_best": global_best,
        "t95_miss_penalty": BUDGET + 1,
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
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(run_one, job): job for job in jobs}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            row = result["metrics"]
            print(
                f"{row['variant']} {row['episode']}: AUC={row['AUC_best_so_far']:.3f}",
                flush=True,
            )

    rows = pd.DataFrame([item["metrics"] for item in results]).sort_values(
        ["episode", "variant"]
    )
    rounds = pd.DataFrame(
        [record for item in results for record in item["calibration"]]
    ).sort_values(["episode", "variant", "step"])
    rows.to_csv(args.output_dir / "per_episode.csv", index=False)
    rounds.to_csv(args.output_dir / "round_calibration.csv", index=False)
    summary = summarize(rows)
    paired = paired_summary(rows)
    tails = tail_summary(rows)
    calibration = calibration_summary(rounds)
    amplitudes = amplitude_diagnostics(rows)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    paired.to_csv(args.output_dir / "paired_strong_heredity_vs_component_ard.csv", index=False)
    tails.to_csv(args.output_dir / "tail_summary.csv", index=False)
    calibration.to_csv(args.output_dir / "calibration_summary.csv", index=False)
    amplitudes.to_csv(args.output_dir / "amplitude_diagnostics.csv", index=False)
    decision = gate_decision(paired, tails, amplitudes)
    (args.output_dir / "gate_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8"
    )
    first_count = min(20, len(episodes))
    first_ids = [f"random_{PHASE}_{index:02d}" for index in range(first_count)]
    first = rows[rows["episode"].isin(first_ids)].sort_values(["episode", "variant"])
    if len(first) != first_count * len(VARIANTS):
        raise RuntimeError("Deterministic first slice is incomplete.")
    first.to_csv(args.output_dir / "first20_per_episode.csv", index=False)
    summarize(first).to_csv(args.output_dir / "first20_summary.csv", index=False)
    paired_summary(first).to_csv(
        args.output_dir / "first20_paired_strong_heredity_vs_component_ard.csv",
        index=False,
    )
    audit = audit_results(rows, rounds, args.output_dir, len(episodes))
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
