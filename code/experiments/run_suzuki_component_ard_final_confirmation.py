#!/usr/bin/env python3
"""Sealed 810k confirmation of unchanged Suzuki fixed and component-ARD GPs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENT_ROOT = Path(__file__).resolve().parent
CODE_ROOT = EXPERIMENT_ROOT.parent
DATASET_ROOT = CODE_ROOT.parents[1]
EXTENSION_ROOT = CODE_ROOT / "bgolearn_rxn_extension"
sys.path.insert(0, str(EXTENSION_ROOT / "src"))
sys.path.insert(0, str(EXPERIMENT_ROOT))

import run_suzuki_component_ard_replication as frozen  # noqa: E402
import run_suzuki_holdout_proxy as suzuki_proxy  # noqa: E402
from bgolearn_rxn.surrogates.component_ard_gp import (  # noqa: E402
    ComponentARDStructuredGP,
)


VARIANTS = frozen.VARIANTS
BASELINE = frozen.BASELINE
CANDIDATE = frozen.CANDIDATE
MODEL_NAMES = frozen.MODEL_NAMES
CORE_METRICS = frozen.CORE_METRICS
SUMMARY_METRICS = frozen.SUMMARY_METRICS
TRAIN_FILE = frozen.TRAIN_FILE
CONFIG_FILE = suzuki_proxy.CONFIG_FILE
SPLIT_SEED_START = 810_000
SPLIT_SEED_STOP = 910_000
EPISODE_COUNT = 200
BUDGET = 5
PHASE = "phase4_final_ard_confirmation"
RUNTIME_RATIO_LIMIT = 2.5
PROMOTION_GATE = {
    "minimum_initial_mean_delta": 0.0,
    "minimum_best_mean_delta": 0.0,
    "minimum_auc_mean_delta_exclusive": 0.0,
    "minimum_auc_ci95_low_exclusive": 0.0,
    "maximum_t95_mean_delta_exclusive": 0.0,
    "maximum_t95_ci95_high_exclusive": 0.0,
    "maximum_runtime_ratio": RUNTIME_RATIO_LIMIT,
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
        phase=PHASE,
        seed_start=SPLIT_SEED_START,
        count=EPISODE_COUNT,
    )
    seeds = [int(item["split_seed"]) for item in episodes]
    if len(episodes) != EPISODE_COUNT or len(set(seeds)) != EPISODE_COUNT:
        raise RuntimeError("Final confirmation episode contract is incomplete.")
    if min(seeds) < SPLIT_SEED_START or max(seeds) >= SPLIT_SEED_STOP:
        raise RuntimeError("Final confirmation escaped the reserved 810k family.")
    if any(int(item["budget"]) != BUDGET for item in episodes):
        raise RuntimeError("Final confirmation budget changed.")
    return episodes


def build_problem(episode: dict, variant: str):
    config, dataset, lookup, model_name = frozen.build_problem(episode, variant)
    if tuple(config.ensemble.enabled_models) != (model_name,):
        raise RuntimeError("Confirmation enabled an unexpected ensemble model.")
    if config.policy.first_query != model_name:
        raise RuntimeError("Confirmation first-query policy changed.")
    if config.policy.early_end != BUDGET or config.policy.final_start != BUDGET + 1:
        raise RuntimeError("Confirmation is not stable LogEI through query 5.")
    if config.gp.target_transform != "standard":
        raise RuntimeError("Confirmation target transform changed.")
    if ComponentARDStructuredGP.amplitude_prior_scale != 0.60:
        raise RuntimeError("Default component-ARD prior scale changed.")
    return config, dataset, lookup, model_name


def run_one(job: tuple[str, dict, str]) -> dict:
    variant, episode, output_root_string = job
    # Frozen run_one calls frozen.build_problem; both configurations are audited
    # above once per job before any oracle query.
    build_problem(episode, variant)
    row = frozen.run_one(job)
    output_path = (
        Path(output_root_string)
        / "trajectories"
        / variant
        / f"{episode['episode']}.json"
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    payload["final_confirmation_contract"] = {
        "sealed_before_scoring": True,
        "train_only": True,
        "public_test_labels_used": False,
        "prior_outcome_artifacts_read_during_scoring": False,
        "parameter_tuning": False,
        "variant": variant,
        "model": MODEL_NAMES[variant],
        "amplitude_prior_scale": 0.60 if variant == CANDIDATE else None,
        "target_transform": "standard",
        "acquisition": "posterior mean query 1, stable LogEI queries 2-5",
        "full_global_best_t95": True,
        "t95_miss_penalty": BUDGET + 1,
    }
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return row


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    result = frozen.summarize(rows)
    result["phase"] = PHASE
    return result


def paired_summary(rows: pd.DataFrame) -> pd.DataFrame:
    result = frozen.paired_summary(rows)
    result["phase"] = PHASE
    return result


def amplitude_diagnostics(rows: pd.DataFrame) -> pd.DataFrame:
    ard = rows[rows["variant"].eq(CANDIDATE)]
    raw = [json.loads(value) for value in ard["component_amplitudes_json"]]
    shares = [json.loads(value) for value in ard["component_shares_json"]]
    records: list[dict] = []
    for value_type, parsed in (("amplitude", raw), ("share", shares)):
        components = sorted(set().union(*(item.keys() for item in parsed)))
        for component in components:
            values = pd.Series([item[component] for item in parsed], dtype=float)
            mean, sd, low, high = frozen.mean_ci(values)
            records.append(
                {
                    "dataset": "Suzuki",
                    "phase": PHASE,
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


def promotion_decision(
    rows: pd.DataFrame, diagnostics: pd.DataFrame
) -> dict[str, object]:
    paired = paired_summary(rows).set_index("metric")
    initial = paired.loc["initial_round_found_best"]
    best = paired.loc["best_found"]
    auc = paired.loc["AUC_best_so_far"]
    t95 = paired.loc["round_to_95_global_best"]
    summary = summarize(rows)
    runtime = summary[summary["metric"].eq("elapsed_seconds")].set_index("variant")
    runtime_ratio = float(
        runtime.loc[CANDIDATE, "mean"] / runtime.loc[BASELINE, "mean"]
    )
    values = diagnostics[["mean", "sample_sd", "minimum", "maximum"]].to_numpy(
        dtype=float
    )
    shares = diagnostics[diagnostics["value_type"].eq("share")]
    amplitudes_stable = bool(
        np.isfinite(values).all()
        and (diagnostics["minimum"] > 0.0).all()
        and (shares["maximum"] < 0.95).all()
    )
    checks = {
        "initial_point_noninferior": float(initial["mean_delta"]) >= 0.0,
        "best_point_noninferior": float(best["mean_delta"]) >= 0.0,
        "auc_positive": float(auc["mean_delta"]) > 0.0,
        "auc_ci95_low_positive": float(auc["ci95_low"]) > 0.0,
        "t95_negative": float(t95["mean_delta"]) < 0.0,
        "t95_ci95_high_negative": float(t95["ci95_high"]) < 0.0,
        "runtime_ratio_within_limit": runtime_ratio <= RUNTIME_RATIO_LIMIT,
        "finite_noncollapsed_amplitudes": amplitudes_stable,
    }
    return {
        "gate": PROMOTION_GATE,
        "checks": checks,
        "passed": bool(all(checks.values())),
        "decision": (
            "promote_component_ard" if all(checks.values()) else "retain_fixed"
        ),
        "runtime_ratio": runtime_ratio,
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
        "prior_outcome_artifacts_read_during_scoring": False,
    }


def explicit_twenty_run(rows: pd.DataFrame) -> pd.DataFrame:
    episode_ids = [f"random_{PHASE}_{index:02d}" for index in range(20)]
    view = rows[rows["episode"].isin(episode_ids)].copy()
    if len(view) != 20 * len(VARIANTS) or set(view["episode"]) != set(episode_ids):
        raise RuntimeError("Deterministic first-20 confirmation view is incomplete.")
    return view.sort_values(["episode", "variant"])


def audit_results(rows: pd.DataFrame, output_root: Path) -> dict[str, object]:
    expected_rows = EPISODE_COUNT * len(VARIANTS)
    if len(rows) != expected_rows or rows["episode"].nunique() != EPISODE_COUNT:
        raise RuntimeError("Final confirmation result count is incomplete.")
    if not rows.groupby("episode")["split_seed"].nunique().eq(1).all():
        raise RuntimeError("Paired confirmation variants use different splits.")
    seeds = rows["split_seed"].astype(int)
    if seeds.min() < SPLIT_SEED_START or seeds.max() >= SPLIT_SEED_STOP:
        raise RuntimeError("Scored row escaped the reserved 810k family.")
    trajectory_files = list((output_root / "trajectories").glob("*/*.json"))
    if len(trajectory_files) != expected_rows:
        raise RuntimeError("Final confirmation trajectory count is incomplete.")
    for path in trajectory_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        contract = payload["final_confirmation_contract"]
        if contract["public_test_labels_used"]:
            raise RuntimeError("A confirmation trajectory reports public-label use.")
        if contract["prior_outcome_artifacts_read_during_scoring"]:
            raise RuntimeError("A confirmation trajectory reports prior-outcome use.")
        if len(payload["trajectory"]) != BUDGET:
            raise RuntimeError("A confirmation trajectory has the wrong budget.")
    return {
        "passed": True,
        "n_rows": len(rows),
        "n_episodes": rows["episode"].nunique(),
        "n_trajectories": len(trajectory_files),
        "split_seed_min": int(seeds.min()),
        "split_seed_max": int(seeds.max()),
        "budget": BUDGET,
        "variants": list(VARIANTS),
        "public_test_labels_used": False,
        "prior_outcome_artifacts_read_during_scoring": False,
        "parameter_tuning": False,
    }


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
        "stage": PHASE,
        "sealed_before_scoring": True,
        "train_only": True,
        "public_test_labels_used": False,
        "prior_outcome_artifacts_read_during_scoring": False,
        "parameter_sweep": False,
        "interim_selection": False,
        "input": {
            "path": str(TRAIN_FILE.relative_to(DATASET_ROOT)),
            "sha256": file_sha256(TRAIN_FILE),
        },
        "implementations": {
            "confirmation_runner": {
                "path": str(Path(__file__).resolve().relative_to(DATASET_ROOT)),
                "sha256": file_sha256(Path(__file__).resolve()),
            },
            "frozen_replication_runner": {
                "path": str(Path(frozen.__file__).resolve().relative_to(DATASET_ROOT)),
                "sha256": file_sha256(Path(frozen.__file__).resolve()),
            },
            "source_config": {
                "path": str(CONFIG_FILE.relative_to(DATASET_ROOT)),
                "sha256": file_sha256(CONFIG_FILE),
            },
        },
        "variants": list(VARIANTS),
        "models": MODEL_NAMES,
        "component_ard_amplitude_prior_scale": 0.60,
        "target_transform": "standard",
        "acquisition": "posterior mean query 1, stable LogEI queries 2-5",
        "episode_count": EPISODE_COUNT,
        "budget": BUDGET,
        "split_seed_start": SPLIT_SEED_START,
        "split_seed_stop_exclusive": SPLIT_SEED_STOP,
        "split_seed_min": min(split_seeds),
        "split_seed_max": max(split_seeds),
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
    rows: list[dict] = []
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
    audit = audit_results(detail, args.output_dir)
    summary = summarize(detail)
    paired = paired_summary(detail)
    diagnostics = amplitude_diagnostics(detail)
    decision = promotion_decision(detail, diagnostics)
    detail.to_csv(args.output_dir / "per_episode.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    paired.to_csv(args.output_dir / "paired_ard_vs_fixed.csv", index=False)
    diagnostics.to_csv(args.output_dir / "amplitude_diagnostics.csv", index=False)
    (args.output_dir / "promotion_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    twenty = explicit_twenty_run(detail)
    twenty.to_csv(args.output_dir / "twenty_run_per_episode.csv", index=False)
    summarize(twenty).to_csv(
        args.output_dir / "twenty_run_summary.csv", index=False
    )
    paired_summary(twenty).to_csv(
        args.output_dir / "twenty_run_paired_ard_vs_fixed.csv", index=False
    )


if __name__ == "__main__":
    main()
