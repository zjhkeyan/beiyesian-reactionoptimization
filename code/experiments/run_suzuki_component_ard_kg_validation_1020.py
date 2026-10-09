#!/usr/bin/env python3
"""Frozen authorized 1020 validation of component-ARD finite-pool KG."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

import run_suzuki_component_ard_kg as discovery


PHASE = "phase6_deadline_validation"
SPLIT_SEED_START = 1_020_000
SPLIT_SEED_STOP = 1_030_000
EPISODE_COUNT = 80
DEFAULT_OUTPUT = (
    discovery.DATASET_ROOT
    / "submission/results/experimental_investigations/phase6-node6-component-ard-kg-validation-1020"
)


def make_episodes() -> list[dict]:
    source = pd.read_csv(discovery.TRAIN_FILE)
    episodes = discovery.suzuki_proxy._phase_random_episodes(
        source, phase=PHASE, seed_start=SPLIT_SEED_START, count=EPISODE_COUNT
    )
    seeds = [int(item["split_seed"]) for item in episodes]
    checks = {
        "episode_count": len(episodes) == EPISODE_COUNT,
        "unique_seeds": len(set(seeds)) == EPISODE_COUNT,
        "seed_family": min(seeds) >= SPLIT_SEED_START and max(seeds) < SPLIT_SEED_STOP,
        "initial_rows": all(
            len(item["initial_indices"]) == discovery.INITIAL_ROWS for item in episodes
        ),
        "candidate_rows": all(
            len(item["candidate_indices"]) == discovery.CANDIDATE_ROWS
            for item in episodes
        ),
        "budget": all(int(item["budget"]) == discovery.BUDGET for item in episodes),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Frozen 1020 validation episode contract failed: {checks}")
    return episodes


def run_one(job: tuple[str, dict, str]) -> dict:
    discovery.PHASE = PHASE
    result = discovery.run_one(job)
    variant, episode, output_root_string = job
    path = (
        Path(output_root_string)
        / "trajectories"
        / variant
        / f"{episode['episode']}.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["contract"].update(
        {
            "phase": PHASE,
            "validation_1020_authorized": True,
            "validation_1020_opened": True,
            "sealed_1030_opened": False,
            "public_test_labels_used": False,
            "historical_confirmation_outcomes_used": False,
        }
    )
    payload["contract"].pop("protected_1020_or_1030_episodes_opened", None)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    result["metrics"]["phase"] = PHASE
    return result


def validation_gate(
    paired: pd.DataFrame,
    tails: pd.DataFrame,
    rows: pd.DataFrame,
    rounds: pd.DataFrame,
) -> dict:
    gate = discovery.gate_decision(paired, tails, rows, rounds)
    gate.update(
        {
            "decision": (
                "eligible_for_1030_confirmation"
                if gate["passed"]
                else "stop_before_1030"
            ),
            "validation_1020_authorized": True,
            "validation_1020_opened": True,
            "sealed_1030_opened": False,
            "public_test_labels_used": False,
            "historical_confirmation_outcomes_used": False,
        }
    )
    gate.pop("protected_1020_or_1030_episodes_opened", None)
    return gate


def audit_results(
    rows: pd.DataFrame,
    rounds: pd.DataFrame,
    diagnostics: pd.DataFrame,
    output: Path,
) -> dict:
    seeds = rows["split_seed"].astype(int)
    candidate = rows[rows["variant"].eq(discovery.CANDIDATE)]
    kg = diagnostics[
        diagnostics["variant"].eq(discovery.CANDIDATE)
        & diagnostics["step"].gt(1)
    ]
    query_one = rounds[rounds["step"].eq(1)].pivot(
        index="episode", columns="variant", values="query_index"
    )
    checks = {
        "metric_rows": len(rows) == EPISODE_COUNT * 2,
        "episodes": rows["episode"].nunique() == EPISODE_COUNT,
        "paired_splits": bool(
            rows.groupby("episode")["split_seed"].nunique().eq(1).all()
        ),
        "paired_candidate_best": bool(
            rows.groupby("episode")["candidate_global_best"].nunique().eq(1).all()
        ),
        "round_rows": len(rounds) == EPISODE_COUNT * 2 * discovery.BUDGET,
        "trajectory_count": len(list((output / "trajectories").glob("*/*.json")))
        == EPISODE_COUNT * 2,
        "only_authorized_1020_series": int(seeds.min()) >= SPLIT_SEED_START
        and int(seeds.max()) < SPLIT_SEED_STOP,
        "finite_metrics": bool(
            np.isfinite(
                rows[list(discovery.SUMMARY_METRICS)].to_numpy(dtype=float)
            ).all()
        ),
        "query_one_identical": bool(
            (query_one[discovery.CANDIDATE] == query_one[discovery.BASELINE]).all()
        ),
        "kg_rows_complete": len(kg) == EPISODE_COUNT * (discovery.BUDGET - 1),
        "quadrature_fixed_32": bool(
            kg["quadrature_order"].eq(discovery.QUADRATURE_ORDER).all()
        ),
        "covariance_symmetric": bool(
            (kg["covariance_symmetry_error"] <= 1e-10).all()
        ),
        "covariance_psd": bool(
            (kg["covariance_min_eigenvalue"] >= -1e-7).all()
        ),
        "kg_scores_finite_nonnegative": bool(
            np.isfinite(
                kg[["kg_score_min", "kg_score_max", "selected_score"]].to_numpy()
            ).all()
            and (kg["kg_score_min"] >= 0.0).all()
        ),
        "finite_uncertainty": bool(
            np.isfinite(
                candidate[["predicted_std_min", "predicted_std_max"]].to_numpy()
            ).all()
            and (candidate["predicted_std_min"] > 0.0).all()
        ),
        "validation_1020_authorized": True,
        "sealed_1030_not_opened": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Frozen 1020 validation audit failed: {checks}")
    return {
        "passed": True,
        "checks": checks,
        "n_episodes": EPISODE_COUNT,
        "split_seed_min": int(seeds.min()),
        "split_seed_max": int(seeds.max()),
        "budget": discovery.BUDGET,
        "initial_rows": discovery.INITIAL_ROWS,
        "candidate_rows": discovery.CANDIDATE_ROWS,
        "validation_1020_authorized": True,
        "validation_1020_opened": True,
        "sealed_1030_opened": False,
        "public_test_labels_used": False,
        "historical_confirmation_outcomes_used": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.jobs <= 0:
        raise ValueError("--jobs must be positive.")
    episodes = make_episodes()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(discovery.TRAIN_FILE)
    split_seeds = [int(item["split_seed"]) for item in episodes]
    manifest = {
        "stage": PHASE,
        "frozen_validation": True,
        "validation_1020_authorized": True,
        "validation_1020_opened": True,
        "sealed_1030_opened": False,
        "public_test_labels_used": False,
        "historical_confirmation_outcomes_used": False,
        "train_only": True,
        "input": {
            "path": str(discovery.TRAIN_FILE.relative_to(discovery.DATASET_ROOT)),
            "sha256": discovery.file_sha256(discovery.TRAIN_FILE),
            "n_rows": len(source),
        },
        "implementations": {
            "validation_driver": {
                "path": str(Path(__file__).relative_to(discovery.DATASET_ROOT)),
                "sha256": discovery.file_sha256(Path(__file__)),
            },
            "audited_discovery_driver": {
                "path": str(
                    Path(discovery.__file__).resolve().relative_to(discovery.DATASET_ROOT)
                ),
                "sha256": discovery.file_sha256(Path(discovery.__file__)),
            },
            "model": {
                "path": str(discovery.MODEL_SOURCE.relative_to(discovery.DATASET_ROOT)),
                "sha256": discovery.file_sha256(discovery.MODEL_SOURCE),
            },
            "kg": {
                "path": str(discovery.KG_SOURCE.relative_to(discovery.DATASET_ROOT)),
                "sha256": discovery.file_sha256(discovery.KG_SOURCE),
            },
            "config": {
                "path": str(discovery.CONFIG_FILE.relative_to(discovery.DATASET_ROOT)),
                "sha256": discovery.file_sha256(discovery.CONFIG_FILE),
            },
        },
        "variants": list(discovery.VARIANTS),
        "model": discovery.MODEL_NAME,
        "episode_count": EPISODE_COUNT,
        "initial_rows": discovery.INITIAL_ROWS,
        "candidate_rows": discovery.CANDIDATE_ROWS,
        "budget": discovery.BUDGET,
        "split_seed_start": SPLIT_SEED_START,
        "split_seed_min": min(split_seeds),
        "split_seed_max": max(split_seeds),
        "sealed_seed_start": SPLIT_SEED_STOP,
        "target_transform": "standard",
        "baseline_acquisition": "posterior mean query 1, stable LogEI queries 2-5",
        "candidate_acquisition": "posterior mean query 1, exact finite-pool one-step KG queries 2-5",
        "quadrature_order": discovery.QUADRATURE_ORDER,
        "observation_noise_included": True,
        "observed_incumbent_terminal_alternative": True,
        "model_or_acquisition_changed_after_discovery": False,
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
        for variant in discovery.VARIANTS
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
                f"AUC={row['AUC_best_so_far']:.3f} "
                f"kg_s={row['kg_scoring_seconds']:.4f}",
                flush=True,
            )
    wall_seconds = time.perf_counter() - wall_started
    rows = pd.DataFrame([item["metrics"] for item in results]).sort_values(
        ["episode", "variant"]
    )
    rounds = pd.DataFrame(
        [row for item in results for row in item["calibration"]]
    ).sort_values(["episode", "variant", "step"])
    diagnostics = pd.DataFrame(
        [row for item in results for row in item["diagnostics"]]
    ).sort_values(["episode", "variant", "step"])
    summary = discovery.summarize(rows)
    paired = discovery.paired_summary(rows)
    tails = discovery.tail_summary(rows)
    calibration = discovery.calibration_summary(rounds)
    gate = validation_gate(paired, tails, rows, rounds)
    audit = audit_results(rows, rounds, diagnostics, args.output_dir)
    audit.update(
        {
            "wall_seconds": wall_seconds,
            "aggregate_campaign_seconds": float(rows["elapsed_seconds"].sum()),
            "effective_parallelism": float(
                rows["elapsed_seconds"].sum() / max(wall_seconds, 1e-12)
            ),
        }
    )
    rows.to_csv(args.output_dir / "per_episode.csv", index=False)
    rounds.to_csv(args.output_dir / "calibration_predictions.csv", index=False)
    diagnostics.to_csv(args.output_dir / "acquisition_diagnostics.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    paired.to_csv(args.output_dir / "paired_summary.csv", index=False)
    tails.to_csv(args.output_dir / "auc_tail_summary.csv", index=False)
    calibration.to_csv(args.output_dir / "calibration_summary.csv", index=False)
    first20_ids = {item["episode"] for item in episodes[:20]}
    first20 = rows[rows["episode"].isin(first20_ids)]
    first20.to_csv(args.output_dir / "first20_per_episode.csv", index=False)
    discovery.summarize(first20).to_csv(
        args.output_dir / "first20_summary.csv", index=False
    )
    discovery.paired_summary(first20).to_csv(
        args.output_dir / "first20_paired_summary.csv", index=False
    )
    (args.output_dir / "validation_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    score = float(
        summary[
            summary["variant"].eq(discovery.CANDIDATE)
            & summary["metric"].eq("AUC_best_so_far")
        ]["mean"].iloc[0]
    )
    print(f"absolute mean AUC_best_so_far={score:.8f}")
    print(json.dumps(gate, sort_keys=True))


if __name__ == "__main__":
    main()
