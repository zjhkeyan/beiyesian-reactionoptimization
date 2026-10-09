#!/usr/bin/env python3
"""Phase 6 paired Suzuki discovery for five-member delete-fold component ARD."""

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

import run_suzuki_graph_diffusion_gp as common  # noqa: E402
from bgolearn_rxn.campaign import CampaignRunner  # noqa: E402
from bgolearn_rxn.oracle import OfflineTableOracle  # noqa: E402


PHASE = "phase6_deadline_discovery"
BASELINE = "component_ard"
CANDIDATE = "delete_fold_ard"
VARIANTS = (BASELINE, CANDIDATE)
MODEL_NAMES = {
    BASELINE: "component_ard_gp",
    CANDIDATE: "delete_fold_component_ard_gp",
}
CORE_METRICS = common.CORE_METRICS
SUMMARY_METRICS = common.SUMMARY_METRICS
TRAIN_FILE = common.TRAIN_FILE
CONFIG_FILE = common.CONFIG_FILE
CANDIDATE_SOURCE = (
    EXTENSION_ROOT
    / "src"
    / "bgolearn_rxn"
    / "surrogates"
    / "delete_fold_component_ard_gp.py"
)
BASELINE_SOURCE = common.BASELINE_SOURCE
FACTORY_SOURCE = EXTENSION_ROOT / "src" / "bgolearn_rxn" / "factory.py"
DEFAULT_OUTPUT = (
    DATASET_ROOT
    / "submission"
    / "results"
    / "experimental_investigations"
    / "phase6-node5-delete-fold-ard"
)
SPLIT_SEED_START = common.SPLIT_SEED_START
SPLIT_SEED_STOP = common.SPLIT_SEED_STOP
EPISODE_COUNT = common.EPISODE_COUNT
INITIAL_COUNT = common.INITIAL_COUNT
CANDIDATE_COUNT = common.CANDIDATE_COUNT
BUDGET = common.BUDGET
MEMBER_COUNT = 5
RUNTIME_RATIO_MAX = 6.0

# Reuse the frozen common-panel construction and statistical summaries with
# this node's candidate identity. No graph-model behavior is reused.
common.CANDIDATE = CANDIDATE
common.VARIANTS = VARIANTS
common.MODEL_NAMES = MODEL_NAMES


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_episodes() -> list[dict]:
    return common.make_episodes()


def build_problem(episode: dict, variant: str):
    return common.build_problem(episode, variant)


def _diversity_diagnostics(model, variant: str) -> dict[str, object]:
    if variant != CANDIDATE:
        return {
            "member_count": None,
            "observed_count": None,
            "fold_sizes_json": None,
            "member_training_sizes_json": None,
            "mean_within_variance": None,
            "mean_between_variance": None,
            "maximum_between_variance": None,
            "mean_total_variance": None,
            "mean_disagreement_std": None,
            "between_variance_fraction": None,
        }
    within = np.asarray(model.within_variance_, dtype=float)
    between = np.asarray(model.between_variance_, dtype=float)
    total = np.asarray(model.total_variance_, dtype=float)
    return {
        "member_count": len(model.members),
        "observed_count": int(model.observed_count_),
        "fold_sizes_json": json.dumps(list(model.fold_sizes_)),
        "member_training_sizes_json": json.dumps(list(model.member_training_sizes_)),
        "mean_within_variance": float(np.mean(within)),
        "mean_between_variance": float(np.mean(between)),
        "maximum_between_variance": float(np.max(between)),
        "mean_total_variance": float(np.mean(total)),
        "mean_disagreement_std": float(np.mean(np.sqrt(np.maximum(between, 0.0)))),
        "between_variance_fraction": float(
            np.sum(between) / max(float(np.sum(total)), 1e-12)
        ),
    }


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
    metrics = common.campaign_metrics(payload["trajectory"], candidate_global_best)
    calibration = common.calibration_rows(
        payload["trajectory"], episode=episode, variant=variant
    )
    model = runner.ensemble.models[model_name]
    diversity = _diversity_diagnostics(model, variant)
    candidate_indices = set(int(value) for value in episode["candidate_indices"])
    queried = [int(item["query_index"]) for item in payload["trajectory"]]
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
        "acquisition": "posterior mean query 1, stable LogEI queries 2-5",
        "incumbent_scope": "full currently observed feedback",
        "held_candidate_global_best": candidate_global_best,
        "t95_miss_penalty": BUDGET + 1,
        "member_count": MEMBER_COUNT if variant == CANDIDATE else None,
        "fold_rule": (
            "sort current observations by (yield,index), assign sorted_position modulo 5"
            if variant == CANDIDATE
            else None
        ),
        "member_weights": "equal" if variant == CANDIDATE else None,
        "variance_rule": (
            "mean within-member variance plus population variance of member means"
            if variant == CANDIDATE
            else None
        ),
        "diversity": diversity,
        "episode": episode,
        "query_audit": query_audit,
    }
    payload["delete_fold_metrics"] = metrics
    payload["delete_fold_contract"] = contract
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
            **diversity,
        },
        "calibration": calibration,
        "contract": contract,
    }


def diversity_summary(rows: pd.DataFrame) -> pd.DataFrame:
    candidate = rows[rows["variant"].eq(CANDIDATE)]
    records = []
    for metric in (
        "mean_within_variance",
        "mean_between_variance",
        "maximum_between_variance",
        "mean_total_variance",
        "mean_disagreement_std",
        "between_variance_fraction",
    ):
        mean, sd, low, high = common.mean_ci(candidate[metric])
        records.append(
            {
                "metric": metric,
                "n_episodes": len(candidate),
                "mean": mean,
                "sample_sd": sd,
                "ci95_low": low,
                "ci95_high": high,
                "minimum": float(candidate[metric].min()),
                "maximum": float(candidate[metric].max()),
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
    runtime = rows.groupby("variant")["elapsed_seconds"].sum()
    runtime_ratio = float(runtime[CANDIDATE] / runtime[BASELINE])
    candidate_rows = rows[rows["variant"].eq(CANDIDATE)]
    noncollapsed = bool(
        np.isfinite(candidate_rows["mean_total_variance"]).all()
        and candidate_rows["mean_total_variance"].gt(1e-12).all()
        and candidate_rows["maximum_between_variance"].gt(1e-12).all()
    )
    checks = {
        "initial_mean_delta_nonnegative": float(
            metrics.loc["initial_round_found_best", "mean_delta"]
        )
        >= 0.0,
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
        "decision": "eligible_for_1020_validation" if all(checks.values()) else "stop_before_1020",
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
    query_audits = [item["query_audit"] for item in contracts]
    candidate = rows[rows["variant"].eq(CANDIDATE)]
    fold_sizes = [json.loads(value) for value in candidate["fold_sizes_json"]]
    training_sizes = [
        json.loads(value) for value in candidate["member_training_sizes_json"]
    ]
    checks = {
        "metric_rows": len(rows) == expected * len(VARIANTS),
        "episodes": rows["episode"].nunique() == expected,
        "paired_split_campaign_and_best": bool(
            rows.groupby("episode")[["split_seed", "campaign_seed", "candidate_global_best"]]
            .nunique()
            .eq(1)
            .all()
            .all()
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
        "exact_five_members": bool(candidate["member_count"].eq(MEMBER_COUNT).all()),
        "observed_only_final_fit_count": bool(
            candidate["observed_count"].eq(INITIAL_COUNT + BUDGET - 1).all()
        ),
        "balanced_final_fold_sizes": all(
            sorted(values) == [4, 5, 5, 5, 5] for values in fold_sizes
        ),
        "exact_member_training_sizes": all(
            sorted(values) == [19, 19, 19, 19, 20] for values in training_sizes
        ),
        "finite_noncollapsed_diversity": bool(
            np.isfinite(candidate["mean_between_variance"]).all()
            and candidate["maximum_between_variance"].gt(1e-12).all()
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Delete-fold ARD result audit failed: {checks}")
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
        "candidate": {"path": str(CANDIDATE_SOURCE.relative_to(DATASET_ROOT)), "sha256": file_sha256(CANDIDATE_SOURCE)},
        "baseline": {"path": str(BASELINE_SOURCE.relative_to(DATASET_ROOT)), "sha256": file_sha256(BASELINE_SOURCE)},
        "factory": {"path": str(FACTORY_SOURCE.relative_to(DATASET_ROOT)), "sha256": file_sha256(FACTORY_SOURCE)},
        "config": {"path": str(CONFIG_FILE.relative_to(DATASET_ROOT)), "sha256": file_sha256(CONFIG_FILE)},
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
        "input": {"path": str(TRAIN_FILE.relative_to(DATASET_ROOT)), "sha256": file_sha256(TRAIN_FILE), "n_rows": len(source)},
        "implementations": implementations,
        "variants": list(VARIANTS),
        "models": MODEL_NAMES,
        "episode_count": len(episodes),
        "episode_contract_sha256": common.payload_sha256(episodes),
        "split_seed_start": SPLIT_SEED_START,
        "split_seed_min": min(split_seeds),
        "split_seed_max": max(split_seeds),
        "n_initial": INITIAL_COUNT,
        "n_candidates": CANDIDATE_COUNT,
        "budget": BUDGET,
        "member_count": MEMBER_COUNT,
        "fold_rule": "yield/index lexicographic sort then round-robin modulo five",
        "member_weights": "equal",
        "variance_rule": "mean within variance plus population variance of means",
        "target_transform": "standard",
        "acquisition": "posterior mean query 1, stable LogEI queries 2-5",
        "incumbent_scope": "full currently observed feedback",
        "global_best_scope": "held candidate pool",
        "t95_miss_penalty": BUDGET + 1,
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
    summary = common.absolute_summary(rows)
    paired = common.paired_summary(rows)
    tails = common.tail_summary(rows)
    calibration = common.calibration_summary(rounds)
    diversity = diversity_summary(rows)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    paired.to_csv(args.output_dir / "paired_delete_fold_ard_vs_component_ard.csv", index=False)
    tails.to_csv(args.output_dir / "tail_summary.csv", index=False)
    calibration.to_csv(args.output_dir / "calibration_summary.csv", index=False)
    diversity.to_csv(args.output_dir / "diversity_summary.csv", index=False)
    first_count = min(20, len(episodes))
    first_ids = [f"random_{PHASE}_{index:02d}" for index in range(first_count)]
    first = rows[rows["episode"].isin(first_ids)].sort_values(["episode", "variant"])
    if len(first) != first_count * len(VARIANTS):
        raise RuntimeError("Deterministic first slice is incomplete.")
    first.to_csv(args.output_dir / "first20_per_episode.csv", index=False)
    common.absolute_summary(first).to_csv(args.output_dir / "first20_summary.csv", index=False)
    common.paired_summary(first).to_csv(args.output_dir / "first20_paired_delete_fold_ard_vs_component_ard.csv", index=False)
    decision = gate_decision(paired, tails, calibration, rows)
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
