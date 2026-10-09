#!/usr/bin/env python3
"""Authorized frozen 1020 validation for stage-split delete-fold ARD."""

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


EXPERIMENT_ROOT = Path(__file__).resolve().parent
CODE_ROOT = EXPERIMENT_ROOT.parent
DATASET_ROOT = CODE_ROOT.parents[1]
EXTENSION_ROOT = CODE_ROOT / "bgolearn_rxn_extension"
sys.path.insert(0, str(EXTENSION_ROOT / "src"))
sys.path.insert(0, str(EXPERIMENT_ROOT))

import run_suzuki_stage_split_delete_fold as discovery  # noqa: E402


PHASE = "phase6_stage_split_validation"
SPLIT_SEED_START = 1_020_000
SPLIT_SEED_STOP = 1_030_000
EPISODE_COUNT = 80
BASELINE = discovery.BASELINE
CANDIDATE = discovery.CANDIDATE
VARIANTS = discovery.VARIANTS
MODEL_NAMES = discovery.MODEL_NAMES
INITIAL_COUNT = discovery.INITIAL_COUNT
CANDIDATE_COUNT = discovery.CANDIDATE_COUNT
BUDGET = discovery.BUDGET
MEMBER_COUNT = discovery.MEMBER_COUNT
RUNTIME_RATIO_MAX = discovery.RUNTIME_RATIO_MAX
TRAIN_FILE = discovery.TRAIN_FILE
CONFIG_FILE = discovery.CONFIG_FILE
DEFAULT_OUTPUT = (
    DATASET_ROOT
    / "submission"
    / "results"
    / "experimental_investigations"
    / "phase6-node5b-stage-split-delete-fold-validation-1020"
)
VALIDATION_AUTHORIZED = True
VALIDATION_1020_OPENED = True
SEALED_1030_OPENED = False

for module in (discovery, discovery.base, discovery.base.common):
    module.PHASE = PHASE
    module.SPLIT_SEED_START = SPLIT_SEED_START
    module.SPLIT_SEED_STOP = SPLIT_SEED_STOP
    module.EPISODE_COUNT = EPISODE_COUNT


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_episodes() -> list[dict]:
    frame = pd.read_csv(TRAIN_FILE)
    episodes = discovery.base.common.suzuki_proxy._phase_random_episodes(
        frame,
        phase=PHASE,
        seed_start=SPLIT_SEED_START,
        count=EPISODE_COUNT,
    )
    seeds = [int(item["split_seed"]) for item in episodes]
    checks = {
        "episode_count": len(episodes) == EPISODE_COUNT,
        "unique_split_seeds": len(set(seeds)) == EPISODE_COUNT,
        "only_1020_series": min(seeds) >= SPLIT_SEED_START
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
        raise RuntimeError(f"Authorized 1020 validation contract failed: {checks}")
    return episodes


def run_one(job: tuple[str, dict, str]) -> dict:
    result = discovery.run_one(job)
    variant, episode, output_root_string = job
    output_path = (
        Path(output_root_string)
        / "trajectories"
        / variant
        / f"{episode['episode']}.json"
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    contract = payload["stage_split_contract"]
    contract["validation_1020_authorized"] = VALIDATION_AUTHORIZED
    contract["validation_1020_opened"] = VALIDATION_1020_OPENED
    contract["sealed_1030_opened"] = SEALED_1030_OPENED
    contract["protected_1020_or_1030_outcomes_opened"] = True
    payload["stage_split_contract"] = contract
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    result["contract"] = contract
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit-episodes", type=int, default=None, help="Plumbing check only.")
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
        "validation_driver": {
            "path": str(Path(__file__).relative_to(DATASET_ROOT)),
            "sha256": file_sha256(Path(__file__)),
        },
        "discovery_driver": {
            "path": str(Path(discovery.__file__).relative_to(DATASET_ROOT)),
            "sha256": file_sha256(Path(discovery.__file__)),
        },
        "stage_split": {
            "path": str(discovery.CANDIDATE_SOURCE.relative_to(DATASET_ROOT)),
            "sha256": file_sha256(discovery.CANDIDATE_SOURCE),
        },
        "delete_fold": {
            "path": str(discovery.DELETE_FOLD_SOURCE.relative_to(DATASET_ROOT)),
            "sha256": file_sha256(discovery.DELETE_FOLD_SOURCE),
        },
        "baseline": {
            "path": str(discovery.BASELINE_SOURCE.relative_to(DATASET_ROOT)),
            "sha256": file_sha256(discovery.BASELINE_SOURCE),
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
        "validation_1020_authorized": VALIDATION_AUTHORIZED,
        "validation_1020_opened": VALIDATION_1020_OPENED,
        "sealed_1030_opened": SEALED_1030_OPENED,
        "public_test_labels_used": False,
        "public_searchspace_labels_used": False,
        "historical_810_910_920_930_outcomes_used": False,
        "input": {
            "path": str(TRAIN_FILE.relative_to(DATASET_ROOT)),
            "sha256": file_sha256(TRAIN_FILE),
            "n_rows": len(source),
        },
        "implementations": implementations,
        "candidate_unchanged_from_discovery": True,
        "variants": list(VARIANTS),
        "models": MODEL_NAMES,
        "episode_count": len(episodes),
        "episode_contract_sha256": discovery.base.common.payload_sha256(episodes),
        "split_seed_start": SPLIT_SEED_START,
        "split_seed_min": min(split_seeds),
        "split_seed_max": max(split_seeds),
        "n_initial": INITIAL_COUNT,
        "n_candidates": CANDIDATE_COUNT,
        "budget": BUDGET,
        "fixed_schedule": discovery.STAGE_HISTORY,
        "member_count": MEMBER_COUNT,
        "member_weights": "equal",
        "variance_rule": "mean within variance plus population variance of means",
        "target_transform": "standard",
        "incumbent_scope": "full currently observed feedback",
        "global_best_scope": "held candidate pool",
        "t95_miss_penalty": BUDGET + 1,
        "schedule_sweep": False,
        "parameter_sweep": False,
        "interim_selection": False,
        "runtime_ratio_gate": RUNTIME_RATIO_MAX,
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
                f"{row['variant']} {row['episode']}: AUC={row['AUC_best_so_far']:.3f}",
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
    summary = discovery.base.common.absolute_summary(rows)
    paired = discovery.base.common.paired_summary(rows)
    tails = discovery.base.common.tail_summary(rows)
    calibration = discovery.base.common.calibration_summary(rounds)
    diversity = discovery.base.diversity_summary(rows)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    paired.to_csv(
        args.output_dir / "paired_stage_split_vs_component_ard.csv", index=False
    )
    tails.to_csv(args.output_dir / "tail_summary.csv", index=False)
    calibration.to_csv(args.output_dir / "calibration_summary.csv", index=False)
    diversity.to_csv(args.output_dir / "diversity_summary.csv", index=False)
    first_count = min(20, len(episodes))
    first_ids = [f"random_{PHASE}_{index:02d}" for index in range(first_count)]
    first = rows[rows["episode"].isin(first_ids)].sort_values(["episode", "variant"])
    if len(first) != first_count * len(VARIANTS):
        raise RuntimeError("Deterministic validation first slice is incomplete.")
    first.to_csv(args.output_dir / "first20_per_episode.csv", index=False)
    discovery.base.common.absolute_summary(first).to_csv(
        args.output_dir / "first20_summary.csv", index=False
    )
    discovery.base.common.paired_summary(first).to_csv(
        args.output_dir / "first20_paired_stage_split_vs_component_ard.csv",
        index=False,
    )
    decision = discovery.gate_decision(rows, paired, tails, calibration)
    decision["decision"] = (
        "report_validation_gate_to_coordinator"
        if decision["passed"]
        else "stop_before_1030"
    )
    decision["validation_1020_authorized"] = VALIDATION_AUTHORIZED
    decision["validation_1020_opened"] = VALIDATION_1020_OPENED
    decision["validation_launched"] = True
    decision["sealed_1030_opened"] = SEALED_1030_OPENED
    decision["protected_1020_or_1030_outcomes_opened"] = True
    decision["confirmation_1030_launched"] = False
    decision["wall_seconds"] = wall_seconds
    (args.output_dir / "gate_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8"
    )
    audit = discovery.audit_results(
        rows, rounds, contracts, args.output_dir, len(episodes)
    )
    audit["checks"]["only_1020_series"] = audit["checks"].pop(
        "only_1010_series"
    )
    audit["checks"]["validation_1020_authorized_and_opened"] = bool(
        VALIDATION_AUTHORIZED
        and VALIDATION_1020_OPENED
        and all(item["validation_1020_opened"] for item in contracts)
    )
    audit["checks"]["sealed_1030_unopened"] = bool(
        not SEALED_1030_OPENED
        and all(not item["sealed_1030_opened"] for item in contracts)
    )
    audit["validation_1020_authorized"] = VALIDATION_AUTHORIZED
    audit["validation_1020_opened"] = VALIDATION_1020_OPENED
    audit["sealed_1030_opened"] = SEALED_1030_OPENED
    audit["protected_1020_or_1030_outcomes_opened"] = True
    audit["wall_seconds"] = wall_seconds
    audit["artifact_sha256"] = {
        "per_episode.csv": file_sha256(args.output_dir / "per_episode.csv"),
        "round_calibration.csv": file_sha256(
            args.output_dir / "round_calibration.csv"
        ),
    }
    if not all(audit["checks"].values()):
        raise RuntimeError(f"Authorized 1020 validation audit failed: {audit['checks']}")
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    score = float(
        summary[
            summary["variant"].eq(CANDIDATE)
            & summary["metric"].eq("AUC_best_so_far")
        ]["mean"].iloc[0]
    )
    print(
        f"absolute_mean_AUC_best_so_far={score:.8f} wall_seconds={wall_seconds:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
