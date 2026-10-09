#!/usr/bin/env python3
"""Fresh-seed high-power Suzuki fixed-GP versus component-ARD replication."""

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
from bgolearn_rxn.campaign import CampaignRunner  # noqa: E402
from bgolearn_rxn.metrics import compute_campaign_metrics  # noqa: E402
from bgolearn_rxn.oracle import OfflineTableOracle  # noqa: E402


VARIANTS = ("fixed", "component_ard")
BASELINE = "fixed"
CANDIDATE = "component_ard"
MODEL_NAMES = {"fixed": "gp", "component_ard": "component_ard_gp"}
CORE_METRICS = (
    "initial_round_found_best",
    "best_found",
    "round_to_95_global_best",
    "AUC_best_so_far",
)
SUMMARY_METRICS = CORE_METRICS + ("elapsed_seconds",)
TRAIN_FILE = DATASET_ROOT / "Suzuki" / "suzuki_train.csv"
SPLIT_SEED_START = 610_000
EPISODE_COUNT = 200
PRIOR_SEED_STARTS = (110_000, 210_000, 310_000, 410_000)
PROMOTION_GATE = {
    "minimum_initial_mean_delta": 0.0,
    "minimum_best_mean_delta": 0.0,
    "minimum_auc_mean_delta": 0.0,
    "minimum_auc_ci95_low": 0.0,
    "maximum_t95_mean_delta": 0.0,
    "maximum_t95_ci95_high": 0.0,
    "strict_auc_and_t95_direction": True,
    "require_finite_noncollapsed_amplitudes": True,
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
        phase="fresh_replication",
        seed_start=SPLIT_SEED_START,
        count=EPISODE_COUNT,
    )
    if len(episodes) != EPISODE_COUNT:
        raise RuntimeError("Fresh replication episode count changed.")
    split_seeds = {int(item["split_seed"]) for item in episodes}
    if len(split_seeds) != EPISODE_COUNT:
        raise RuntimeError("Fresh replication split seeds are not unique.")
    if min(split_seeds) < SPLIT_SEED_START:
        raise RuntimeError("Fresh replication escaped its frozen seed range.")
    if any(start in split_seeds for start in PRIOR_SEED_STARTS):
        raise RuntimeError("Fresh replication overlaps a prior seed start.")
    return episodes


def build_problem(episode: dict, variant: str):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    config, dataset, lookup = suzuki_proxy.build_problem(episode, "gp_only")
    model_name = MODEL_NAMES[variant]
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
    return config, dataset, lookup, model_name


def campaign_metrics(trajectory: list[dict], global_best: float, budget: int) -> dict:
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


def run_one(job: tuple[str, dict, str]) -> dict:
    variant, episode, output_root_string = job
    config, dataset, lookup, model_name = build_problem(episode, variant)
    oracle = OfflineTableOracle(dataset=dataset, lookup=lookup)
    seed = int(episode["campaign_seed"])
    started = time.perf_counter()
    runner = CampaignRunner(dataset, config, seed=seed, backend="native")
    result = runner.run(oracle, budget=int(episode["budget"]))
    elapsed = time.perf_counter() - started
    payload = result.to_payload()
    global_best = float(oracle.global_best)
    metrics = campaign_metrics(payload["trajectory"], global_best, int(episode["budget"]))
    model = runner.ensemble.models[model_name]
    amplitudes = dict(getattr(model, "component_amplitudes_", {}))
    shares = dict(getattr(model, "component_shares_", {}))
    payload["replication_metrics"] = metrics
    payload["replication_contract"] = {
        "variant": variant,
        "model": model_name,
        "target_transform": "standard",
        "acquisition": "posterior mean query 1, stable LogEI queries 2-5",
        "train_only": True,
        "public_test_labels_used": False,
        "input_file": str(TRAIN_FILE.relative_to(DATASET_ROOT)),
        "input_sha256": file_sha256(TRAIN_FILE),
        "component_amplitudes": amplitudes or None,
        "component_shares": shares or None,
        "episode": episode,
    }
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
        "dataset": "Suzuki",
        "phase": episode["phase"],
        "episode_family": episode["episode_family"],
        "episode": episode["episode"],
        "split_seed": episode["split_seed"],
        "campaign_seed": episode["campaign_seed"],
        "variant": variant,
        "model": model_name,
        "budget": episode["budget"],
        "global_best": global_best,
        **metrics,
        "elapsed_seconds": elapsed,
        "component_amplitudes_json": (
            json.dumps(amplitudes, sort_keys=True) if amplitudes else None
        ),
        "component_shares_json": (
            json.dumps(shares, sort_keys=True) if shares else None
        ),
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
    for variant, group in rows.groupby("variant", sort=True):
        for metric in SUMMARY_METRICS:
            mean, sd, low, high = mean_ci(group[metric])
            records.append(
                {
                    "dataset": "Suzuki",
                    "phase": "fresh_replication",
                    "episode_family": "random_holdout",
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
                "phase": "fresh_replication",
                "episode_family": "random_holdout",
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


def amplitude_summary(rows: pd.DataFrame) -> pd.DataFrame:
    ard = rows[rows["variant"].eq(CANDIDATE)]
    parsed = [json.loads(value) for value in ard["component_shares_json"]]
    records = []
    for component in sorted(set().union(*(item.keys() for item in parsed))):
        values = pd.Series([item[component] for item in parsed], dtype=float)
        mean, sd, low, high = mean_ci(values)
        records.append(
            {
                "dataset": "Suzuki",
                "phase": "fresh_replication",
                "variant": CANDIDATE,
                "component": component,
                "n_episodes": len(values),
                "mean_share": mean,
                "sample_sd_share": sd,
                "ci95_low": low,
                "ci95_high": high,
                "min_share": float(values.min()),
                "max_share": float(values.max()),
            }
        )
    return pd.DataFrame(records)


def promotion_decision(rows: pd.DataFrame, amplitudes: pd.DataFrame) -> dict:
    paired = paired_summary(rows).set_index("metric")
    initial = paired.loc["initial_round_found_best"]
    best = paired.loc["best_found"]
    auc = paired.loc["AUC_best_so_far"]
    t95 = paired.loc["round_to_95_global_best"]
    amplitude_values = amplitudes[
        ["mean_share", "sample_sd_share", "min_share", "max_share"]
    ].to_numpy(dtype=float)
    amplitude_stable = bool(
        np.isfinite(amplitude_values).all()
        and (amplitudes["min_share"] > 0.0).all()
        and (amplitudes["max_share"] < 0.95).all()
    )
    checks = {
        "initial_point_noninferior": float(initial["mean_delta"])
        >= PROMOTION_GATE["minimum_initial_mean_delta"],
        "best_point_noninferior": float(best["mean_delta"])
        >= PROMOTION_GATE["minimum_best_mean_delta"],
        "auc_positive": float(auc["mean_delta"])
        > PROMOTION_GATE["minimum_auc_mean_delta"],
        "auc_ci_excludes_zero": float(auc["ci95_low"])
        > PROMOTION_GATE["minimum_auc_ci95_low"],
        "t95_improves": float(t95["mean_delta"])
        < PROMOTION_GATE["maximum_t95_mean_delta"],
        "t95_ci_excludes_zero": float(t95["ci95_high"])
        < PROMOTION_GATE["maximum_t95_ci95_high"],
        "amplitudes_stable": amplitude_stable,
    }
    return {
        "gate": PROMOTION_GATE,
        "checks": checks,
        "passed": bool(all(checks.values())),
        "decision": "promote_component_ard" if all(checks.values()) else "retain_fixed",
        "paired_metrics": {
            metric: {
                key: (
                    int(paired.loc[metric, key])
                    if key in {"n_pairs", "wins", "ties", "losses"}
                    else float(paired.loc[metric, key])
                )
                for key in (
                    "n_pairs",
                    "mean_delta",
                    "sample_sd_delta",
                    "ci95_low",
                    "ci95_high",
                    "wins",
                    "ties",
                    "losses",
                )
            }
            for metric in SUMMARY_METRICS
        },
        "production_default_changed": False,
        "public_test_labels_used": False,
    }


def explicit_twenty_run(rows: pd.DataFrame) -> pd.DataFrame:
    episode_ids = [f"random_fresh_replication_{index:02d}" for index in range(20)]
    view = rows[rows["episode"].isin(episode_ids)].copy()
    if len(view) != 40 or set(view["episode"]) != set(episode_ids):
        raise RuntimeError("The deterministic first-20 view is incomplete.")
    return view.sort_values(["episode", "variant"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.jobs <= 0:
        raise ValueError("--jobs must be positive.")
    episodes = make_episodes()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_seeds = [int(item["split_seed"]) for item in episodes]
    manifest = {
        "stage": "fresh_replication",
        "train_only": True,
        "public_test_labels_used": False,
        "input": {
            "path": str(TRAIN_FILE.relative_to(DATASET_ROOT)),
            "sha256": file_sha256(TRAIN_FILE),
        },
        "variants": list(VARIANTS),
        "models": MODEL_NAMES,
        "target_transform": "standard",
        "acquisition": "posterior mean query 1, stable LogEI queries 2-5",
        "episode_count": len(episodes),
        "split_seed_start": SPLIT_SEED_START,
        "split_seed_min": min(split_seeds),
        "split_seed_max": max(split_seeds),
        "prior_seed_starts": list(PRIOR_SEED_STARTS),
        "parameter_sweep": False,
        "interim_selection": False,
        "promotion_gate": PROMOTION_GATE,
        "t95_global_best": "full train partition via OfflineTableOracle.global_best",
        "t95_miss_penalty": "episode budget + 1",
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
    rows = []
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(run_one, job): job for job in jobs}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"{row['variant']} {row['episode']}: "
                f"AUC={row['AUC_best_so_far']:.3f}",
                flush=True,
            )
    detail = pd.DataFrame(rows).sort_values(["episode", "variant"])
    if len(detail) != EPISODE_COUNT * len(VARIANTS):
        raise RuntimeError("Replication result count is incomplete.")
    if not detail.groupby("episode")["split_seed"].nunique().eq(1).all():
        raise RuntimeError("Paired variants do not share split seeds.")
    detail.to_csv(args.output_dir / "per_episode.csv", index=False)
    summary = summarize(detail)
    paired = paired_summary(detail)
    amplitudes = amplitude_summary(detail)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    paired.to_csv(args.output_dir / "paired_ard_vs_fixed.csv", index=False)
    amplitudes.to_csv(args.output_dir / "amplitude_stability.csv", index=False)
    decision = promotion_decision(detail, amplitudes)
    (args.output_dir / "promotion_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8"
    )
    twenty = explicit_twenty_run(detail)
    twenty.to_csv(args.output_dir / "twenty_run_per_episode.csv", index=False)
    summarize(twenty).to_csv(args.output_dir / "twenty_run_summary.csv", index=False)
    paired_summary(twenty).to_csv(
        args.output_dir / "twenty_run_paired_ard_vs_fixed.csv", index=False
    )


if __name__ == "__main__":
    main()
