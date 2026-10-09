#!/usr/bin/env python3
"""Phase 6 paired Suzuki discovery for ARD query 1 then delete-fold LogEI."""

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

import run_suzuki_delete_fold_ard as base  # noqa: E402
from bgolearn_rxn.campaign import CampaignRunner  # noqa: E402
from bgolearn_rxn.oracle import OfflineTableOracle  # noqa: E402


PHASE = base.PHASE
BASELINE = base.BASELINE
CANDIDATE = "stage_split_delete_fold"
VARIANTS = (BASELINE, CANDIDATE)
MODEL_NAMES = {
    BASELINE: "component_ard_gp",
    CANDIDATE: "stage_split_delete_fold_ard_gp",
}
TRAIN_FILE = base.TRAIN_FILE
CONFIG_FILE = base.CONFIG_FILE
BASELINE_SOURCE = base.BASELINE_SOURCE
CANDIDATE_SOURCE = (
    EXTENSION_ROOT
    / "src"
    / "bgolearn_rxn"
    / "surrogates"
    / "stage_split_delete_fold_ard.py"
)
DELETE_FOLD_SOURCE = (
    EXTENSION_ROOT
    / "src"
    / "bgolearn_rxn"
    / "surrogates"
    / "delete_fold_component_ard_gp.py"
)
FACTORY_SOURCE = EXTENSION_ROOT / "src" / "bgolearn_rxn" / "factory.py"
DEFAULT_OUTPUT = (
    DATASET_ROOT
    / "submission"
    / "results"
    / "experimental_investigations"
    / "phase6-node5b-stage-split-delete-fold"
)
SPLIT_SEED_START = base.SPLIT_SEED_START
SPLIT_SEED_STOP = base.SPLIT_SEED_STOP
EPISODE_COUNT = base.EPISODE_COUNT
INITIAL_COUNT = base.INITIAL_COUNT
CANDIDATE_COUNT = base.CANDIDATE_COUNT
BUDGET = base.BUDGET
MEMBER_COUNT = base.MEMBER_COUNT
RUNTIME_RATIO_MAX = base.RUNTIME_RATIO_MAX
STAGE_HISTORY = [
    "component_ard_query_1",
    "delete_fold_logei_queries_2_5",
    "delete_fold_logei_queries_2_5",
    "delete_fold_logei_queries_2_5",
    "delete_fold_logei_queries_2_5",
]

base.CANDIDATE = CANDIDATE
base.VARIANTS = VARIANTS
base.MODEL_NAMES = MODEL_NAMES
base.common.CANDIDATE = CANDIDATE
base.common.VARIANTS = VARIANTS
base.common.MODEL_NAMES = MODEL_NAMES


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_episodes() -> list[dict]:
    return base.make_episodes()


def build_problem(episode: dict, variant: str):
    return base.build_problem(episode, variant)


def run_one(job: tuple[str, dict, str]) -> dict:
    variant, episode, output_root_string = job
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
    candidate_global_best = float(lookup[config.dataset.target_column].max())
    metrics = base.common.campaign_metrics(payload["trajectory"], candidate_global_best)
    calibration = base.common.calibration_rows(
        payload["trajectory"], episode=episode, variant=variant
    )
    model = runner.ensemble.models[model_name]
    diversity = base._diversity_diagnostics(model, variant)
    trajectory = payload["trajectory"]
    first = trajectory[0]
    stage_diagnostics = {
        "fit_calls": int(model.fit_calls_) if variant == CANDIDATE else None,
        "initial_observed_count": (
            int(model.initial_observed_count_) if variant == CANDIDATE else None
        ),
        "transition_observed_count": (
            int(model.transition_observed_count_) if variant == CANDIDATE else None
        ),
        "stage_history_json": (
            json.dumps(model.stage_history_) if variant == CANDIDATE else None
        ),
    }
    candidate_indices = set(int(value) for value in episode["candidate_indices"])
    queried = [int(item["query_index"]) for item in trajectory]
    query_audit = {
        "all_queries_in_candidate_pool": set(queried).issubset(candidate_indices),
        "queries_unique": len(queried) == len(set(queried)) == BUDGET,
        "finite_positive_uncertainty": all(
            np.isfinite(item["predicted_std"]) and item["predicted_std"] > 0.0
            for item in calibration
        ),
    }
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
        "stage_schedule": (
            "component ARD posterior mean query 1; delete-fold ARD stable LogEI queries 2-5"
            if variant == CANDIDATE
            else "component ARD posterior mean query 1; component ARD stable LogEI queries 2-5"
        ),
        "incumbent_scope": "full currently observed feedback",
        "held_candidate_global_best": candidate_global_best,
        "t95_miss_penalty": BUDGET + 1,
        "diversity": diversity,
        "stage_diagnostics": stage_diagnostics,
        "episode": episode,
        "query_audit": query_audit,
    }
    payload["stage_split_metrics"] = metrics
    payload["stage_split_contract"] = contract
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
            "query_1_index": int(first["query_index"]),
            "query_1_predicted_yield": float(first["predicted_yield"]),
            "query_1_predicted_std": float(first["predicted_std"]),
            "query_1_observed_yield": float(first["observed_yield"]),
            **diversity,
            **stage_diagnostics,
        },
        "calibration": calibration,
        "contract": contract,
    }


def _query_one_exact(rows: pd.DataFrame) -> bool:
    baseline = rows[rows["variant"].eq(BASELINE)].set_index("episode").sort_index()
    candidate = rows[rows["variant"].eq(CANDIDATE)].set_index("episode").sort_index()
    columns = [
        "query_1_index",
        "query_1_predicted_yield",
        "query_1_predicted_std",
        "query_1_observed_yield",
        "initial_round_found_best",
    ]
    return all(
        np.array_equal(
            candidate[column].to_numpy(), baseline[column].to_numpy()
        )
        for column in columns
    )


def gate_decision(
    rows: pd.DataFrame,
    paired: pd.DataFrame,
    tails: pd.DataFrame,
    calibration: pd.DataFrame,
) -> dict[str, object]:
    metrics = paired.set_index("metric")
    tail_values = tails.set_index("variant")["auc_cvar10"]
    calibration_values = calibration.set_index("variant")
    runtime = rows.groupby("variant")["elapsed_seconds"].sum()
    runtime_ratio = float(runtime[CANDIDATE] / runtime[BASELINE])
    candidate_rows = rows[rows["variant"].eq(CANDIDATE)]
    noncollapsed = bool(
        np.isfinite(candidate_rows["mean_total_variance"]).all()
        and candidate_rows["mean_total_variance"].gt(1e-12).all()
        and candidate_rows["maximum_between_variance"].gt(1e-12).all()
    )
    checks = {
        "query_1_bit_identical_and_initial_exact_tie": bool(
            _query_one_exact(rows)
            and float(metrics.loc["initial_round_found_best", "mean_delta"]) == 0.0
        ),
        "best_mean_delta_nonnegative": float(metrics.loc["best_found", "mean_delta"])
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
        "finite_noncollapsed_uncertainty": bool(
            calibration_values.loc[CANDIDATE, "finite_positive_uncertainty"]
            and noncollapsed
        ),
        "runtime_ratio_at_most_six": runtime_ratio <= RUNTIME_RATIO_MAX,
    }
    return {
        "checks": checks,
        "passed": bool(all(checks.values())),
        "decision": "report_gate_to_coordinator" if all(checks.values()) else "stop_before_1020",
        "validation_launched": False,
        "auc_cvar10": {key: float(value) for key, value in tail_values.items()},
        "auc_cvar10_delta": float(tail_values[CANDIDATE] - tail_values[BASELINE]),
        "runtime": {
            "baseline_total_seconds": float(runtime[BASELINE]),
            "candidate_total_seconds": float(runtime[CANDIDATE]),
            "candidate_to_baseline_ratio": runtime_ratio,
            "ratio_limit": RUNTIME_RATIO_MAX,
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
    candidate = rows[rows["variant"].eq(CANDIDATE)]
    query_audits = [item["query_audit"] for item in contracts]
    checks = {
        "metric_rows": len(rows) == expected * len(VARIANTS),
        "episodes": rows["episode"].nunique() == expected,
        "paired_split_campaign_and_best": bool(
            rows.groupby("episode")[["split_seed", "campaign_seed", "candidate_global_best"]]
            .nunique().eq(1).all().all()
        ),
        "query_1_bit_identical": _query_one_exact(rows),
        "twenty_initial": bool(rows["n_initial"].eq(INITIAL_COUNT).all()),
        "nine_candidates": bool(rows["n_candidates"].eq(CANDIDATE_COUNT).all()),
        "five_rounds": len(rounds) == expected * len(VARIANTS) * BUDGET,
        "trajectory_count": trajectory_count == expected * len(VARIANTS),
        "only_1010_series": int(seeds.min()) >= SPLIT_SEED_START and int(seeds.max()) < SPLIT_SEED_STOP,
        "all_queries_in_candidate_pool": all(item["all_queries_in_candidate_pool"] for item in query_audits),
        "queries_unique": all(item["queries_unique"] for item in query_audits),
        "finite_positive_uncertainty": all(item["finite_positive_uncertainty"] for item in query_audits),
        "exact_stage_history": all(json.loads(value) == STAGE_HISTORY for value in candidate["stage_history_json"]),
        "exact_fit_calls": bool(candidate["fit_calls"].eq(BUDGET).all()),
        "initial_count_20_transition_count_21": bool(
            candidate["initial_observed_count"].eq(INITIAL_COUNT).all()
            and candidate["transition_observed_count"].eq(INITIAL_COUNT + 1).all()
        ),
        "exact_five_members": bool(candidate["member_count"].eq(MEMBER_COUNT).all()),
        "balanced_final_folds": all(sorted(json.loads(value)) == [4, 5, 5, 5, 5] for value in candidate["fold_sizes_json"]),
        "finite_noncollapsed_diversity": bool(
            np.isfinite(candidate["mean_between_variance"]).all()
            and candidate["maximum_between_variance"].gt(1e-12).all()
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Stage-split delete-fold audit failed: {checks}")
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
        "driver": {"path": str(Path(__file__).relative_to(DATASET_ROOT)), "sha256": file_sha256(Path(__file__))},
        "stage_split": {"path": str(CANDIDATE_SOURCE.relative_to(DATASET_ROOT)), "sha256": file_sha256(CANDIDATE_SOURCE)},
        "delete_fold": {"path": str(DELETE_FOLD_SOURCE.relative_to(DATASET_ROOT)), "sha256": file_sha256(DELETE_FOLD_SOURCE)},
        "baseline": {"path": str(BASELINE_SOURCE.relative_to(DATASET_ROOT)), "sha256": file_sha256(BASELINE_SOURCE)},
        "factory": {"path": str(FACTORY_SOURCE.relative_to(DATASET_ROOT)), "sha256": file_sha256(FACTORY_SOURCE)},
        "config": {"path": str(CONFIG_FILE.relative_to(DATASET_ROOT)), "sha256": file_sha256(CONFIG_FILE)},
    }
    split_seeds = [int(item["split_seed"]) for item in episodes]
    manifest = {
        "stage": PHASE,
        "smoke": len(episodes) != EPISODE_COUNT,
        "train_only": True,
        "public_test_labels_used": False,
        "public_searchspace_labels_used": False,
        "historical_810_910_920_930_outcomes_used": False,
        "protected_1020_or_1030_outcomes_opened": False,
        "input": {"path": str(TRAIN_FILE.relative_to(DATASET_ROOT)), "sha256": file_sha256(TRAIN_FILE), "n_rows": len(source)},
        "implementations": implementations,
        "variants": list(VARIANTS),
        "models": MODEL_NAMES,
        "episode_count": len(episodes),
        "episode_contract_sha256": base.common.payload_sha256(episodes),
        "split_seed_start": SPLIT_SEED_START,
        "split_seed_min": min(split_seeds),
        "split_seed_max": max(split_seeds),
        "n_initial": INITIAL_COUNT,
        "n_candidates": CANDIDATE_COUNT,
        "budget": BUDGET,
        "fixed_schedule": STAGE_HISTORY,
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
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    jobs = [(variant, episode, str(args.output_dir)) for episode in episodes for variant in VARIANTS]
    results = []
    wall_started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(run_one, job): job for job in jobs}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            row = result["metrics"]
            print(f"{row['variant']} {row['episode']}: AUC={row['AUC_best_so_far']:.3f}", flush=True)
    wall_seconds = time.perf_counter() - wall_started
    rows = pd.DataFrame([item["metrics"] for item in results]).sort_values(["episode", "variant"])
    rounds = pd.DataFrame([record for item in results for record in item["calibration"]]).sort_values(["episode", "variant", "step"])
    contracts = [item["contract"] for item in results]
    rows.to_csv(args.output_dir / "per_episode.csv", index=False)
    rounds.to_csv(args.output_dir / "round_calibration.csv", index=False)
    summary = base.common.absolute_summary(rows)
    paired = base.common.paired_summary(rows)
    tails = base.common.tail_summary(rows)
    calibration = base.common.calibration_summary(rounds)
    diversity = base.diversity_summary(rows)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    paired.to_csv(args.output_dir / "paired_stage_split_vs_component_ard.csv", index=False)
    tails.to_csv(args.output_dir / "tail_summary.csv", index=False)
    calibration.to_csv(args.output_dir / "calibration_summary.csv", index=False)
    diversity.to_csv(args.output_dir / "diversity_summary.csv", index=False)
    first_count = min(20, len(episodes))
    first_ids = [f"random_{PHASE}_{index:02d}" for index in range(first_count)]
    first = rows[rows["episode"].isin(first_ids)].sort_values(["episode", "variant"])
    if len(first) != first_count * len(VARIANTS):
        raise RuntimeError("Deterministic first slice is incomplete.")
    first.to_csv(args.output_dir / "first20_per_episode.csv", index=False)
    base.common.absolute_summary(first).to_csv(args.output_dir / "first20_summary.csv", index=False)
    base.common.paired_summary(first).to_csv(args.output_dir / "first20_paired_stage_split_vs_component_ard.csv", index=False)
    decision = gate_decision(rows, paired, tails, calibration)
    decision["wall_seconds"] = wall_seconds
    (args.output_dir / "gate_decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")
    audit = audit_results(rows, rounds, contracts, args.output_dir, len(episodes))
    audit["wall_seconds"] = wall_seconds
    audit["artifact_sha256"] = {
        "per_episode.csv": file_sha256(args.output_dir / "per_episode.csv"),
        "round_calibration.csv": file_sha256(args.output_dir / "round_calibration.csv"),
    }
    (args.output_dir / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    score = float(summary[summary["variant"].eq(CANDIDATE) & summary["metric"].eq("AUC_best_so_far")]["mean"].iloc[0])
    print(f"absolute_mean_AUC_best_so_far={score:.8f} wall_seconds={wall_seconds:.3f}", flush=True)


if __name__ == "__main__":
    main()
