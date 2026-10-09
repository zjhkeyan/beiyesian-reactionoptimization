#!/usr/bin/env python3
"""Paired phase-6 Suzuki discovery panel for top-three ARD/graph consensus."""

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


EXPERIMENT_ROOT = Path(__file__).resolve().parent
CODE_ROOT = EXPERIMENT_ROOT.parent
DATASET_ROOT = CODE_ROOT.parents[1]
EXTENSION_ROOT = CODE_ROOT / "bgolearn_rxn_extension"
sys.path.insert(0, str(EXTENSION_ROOT / "src"))
sys.path.insert(0, str(EXPERIMENT_ROOT))

import run_suzuki_graph_diffusion_gp as common  # noqa: E402
import run_suzuki_holdout_proxy as suzuki_proxy  # noqa: E402
from bgolearn_rxn.acquisition import stable_log_expected_improvement  # noqa: E402
from bgolearn_rxn.campaign import CampaignRunner  # noqa: E402
from bgolearn_rxn.factory import prepare_problem  # noqa: E402
from bgolearn_rxn.oracle import OfflineTableOracle  # noqa: E402
from bgolearn_rxn.utils import deterministic_argmax  # noqa: E402


PHASE = "phase6_deadline_discovery"
BASELINE = "component_ard"
CANDIDATE = "consensus_portfolio"
VARIANTS = (BASELINE, CANDIDATE)
ARD_MODEL = "component_ard_gp"
GRAPH_MODEL = "graph_diffusion_product_gp"
TOP_K = 3
CORE_METRICS = common.CORE_METRICS
SUMMARY_METRICS = common.SUMMARY_METRICS
TRAIN_FILE = common.TRAIN_FILE
CONFIG_FILE = common.CONFIG_FILE
ARD_SOURCE = common.BASELINE_SOURCE
GRAPH_SOURCE = common.CANDIDATE_SOURCE
FACTORY_SOURCE = common.FACTORY_SOURCE
COMMON_DRIVER_SOURCE = Path(common.__file__).resolve()
DEFAULT_OUTPUT = (
    DATASET_ROOT
    / "submission"
    / "results"
    / "experimental_investigations"
    / "phase6-node4-consensus-portfolio"
)
EPISODE_COUNT = common.EPISODE_COUNT
INITIAL_COUNT = common.INITIAL_COUNT
CANDIDATE_COUNT = common.CANDIDATE_COUNT
BUDGET = common.BUDGET
SPLIT_SEED_START = common.SPLIT_SEED_START
SPLIT_SEED_STOP = common.SPLIT_SEED_STOP
TAIL_FRACTION = common.TAIL_FRACTION
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


def deterministic_top_k(
    scores: np.ndarray,
    candidate_ids: np.ndarray,
    *,
    k: int,
    seed: int,
) -> np.ndarray:
    """Rank top candidates by repeated repository-standard deterministic argmax."""

    values = np.asarray(scores, dtype=float).reshape(-1)
    ids = np.asarray(candidate_ids, dtype=int).reshape(-1)
    if values.shape != ids.shape:
        raise ValueError("scores and candidate_ids must align.")
    if k <= 0 or k > len(ids):
        raise ValueError("k must be positive and no larger than the candidate pool.")
    remaining = np.arange(len(ids), dtype=int)
    ranking = []
    for rank in range(k):
        local = deterministic_argmax(
            values[remaining], ids[remaining], seed=int(seed) + rank
        )
        ranking.append(int(remaining[local]))
        remaining = np.delete(remaining, local)
    return np.asarray(ranking, dtype=int)


def arbitrate_top_three(
    ard_scores: np.ndarray,
    graph_scores: np.ndarray,
    candidate_ids: np.ndarray,
    *,
    seed: int,
) -> dict[str, object]:
    """Apply the predeclared candidate-level top-three consensus gate."""

    ids = np.asarray(candidate_ids, dtype=int).reshape(-1)
    ard_ranking = deterministic_top_k(
        ard_scores, ids, k=len(ids), seed=seed
    )
    graph_top = int(
        deterministic_argmax(graph_scores, ids, seed=seed)
    )
    ard_top = int(ard_ranking[0])
    ard_top_three = ard_ranking[: min(TOP_K, len(ids))]
    graph_rank = int(np.flatnonzero(ard_ranking == graph_top)[0]) + 1
    gate_accepted = bool(graph_top in set(ard_top_three.tolist()))
    selected = graph_top if gate_accepted else ard_top
    return {
        "selected_local_index": selected,
        "selected_query_index": int(ids[selected]),
        "selected_expert": "graph" if gate_accepted else "ard",
        "ard_top_local_index": ard_top,
        "ard_top_query_index": int(ids[ard_top]),
        "ard_top_three_query_indices": [int(ids[index]) for index in ard_top_three],
        "graph_top_local_index": graph_top,
        "graph_top_query_index": int(ids[graph_top]),
        "graph_top_ard_rank": graph_rank,
        "gate_accepted": gate_accepted,
        "selection_changed_from_ard": int(ids[selected]) != int(ids[ard_top]),
    }


def make_episodes() -> list[dict]:
    episodes = common.make_episodes()
    if payload_sha256(episodes) != "0f14b7731f84bd1d69b654e2438a5f9c5cbe8bb13f805648e908ab866726bd28":
        raise RuntimeError("Consensus portfolio episode contract changed.")
    return episodes


def build_problem(episode: dict):
    config, dataset, lookup = suzuki_proxy.build_problem(episode, "gp_only")
    config = replace(
        config,
        gp=replace(config.gp, target_transform="standard"),
        policy=replace(
            config.policy,
            early_end=BUDGET,
            final_start=BUDGET + 1,
            challenge_every=0,
            novelty_disagreement_mode="off",
        ),
    )
    prepared = prepare_problem(dataset, config)
    return config, dataset, lookup, prepared


def model_config(config, model_name: str):
    return replace(
        config,
        ensemble=replace(
            config.ensemble,
            enabled_models=(model_name,),
            initial_weights={model_name: 1.0},
        ),
        policy=replace(config.policy, first_query=model_name),
    )


def acquisition_scores(posterior, observed: pd.DataFrame, dataset, step: int):
    if step == 1:
        return posterior.mean.copy(), "posterior_mean"
    progress = (step - 1) / (BUDGET - 1)
    xi = 0.05 * (1.0 - progress)
    scores = stable_log_expected_improvement(
        posterior.mean,
        posterior.std,
        dataset.target_incumbent(observed),
        xi=xi,
    )
    return scores, "stable_log_ei"


def run_one(job: tuple[str, dict, str]) -> dict:
    variant, episode, output_root_string = job
    config, dataset, lookup, prepared = build_problem(episode)
    seed = int(episode["campaign_seed"])
    ard_runner = CampaignRunner(
        dataset,
        model_config(config, ARD_MODEL),
        seed=seed,
        backend="native",
        prepared=prepared,
    )
    graph_runner = None
    if variant == CANDIDATE:
        graph_runner = CampaignRunner(
            dataset,
            model_config(config, GRAPH_MODEL),
            seed=seed,
            backend="native",
            prepared=prepared,
        )
    elif variant != BASELINE:
        raise ValueError(f"Unknown variant: {variant}")

    oracle = OfflineTableOracle(dataset=dataset, lookup=lookup)
    observed = dataset.initial.copy()
    X_observed = prepared.X_initial.copy()
    y_observed = observed[dataset.spec.target_column].to_numpy(dtype=float)
    remaining = np.arange(len(dataset.candidates), dtype=int)
    trajectory: list[dict] = []
    arbitration: list[dict] = []
    started = time.perf_counter()

    for step in range(1, BUDGET + 1):
        X_remaining = prepared.X_candidates[remaining]
        candidate_rows = dataset.candidates.iloc[remaining]
        candidate_ids = candidate_rows["query_index"].to_numpy(dtype=int)
        ard_posterior = ard_runner.backend.fit_predict(
            X_observed, y_observed, X_remaining
        )
        ard_scores, acquisition_name = acquisition_scores(
            ard_posterior, observed, dataset, step
        )
        ard_top = int(
            deterministic_argmax(ard_scores, candidate_ids, seed=seed + step)
        )

        graph_posterior = None
        graph_scores = None
        if variant == CANDIDATE and step > 1:
            graph_posterior = graph_runner.backend.fit_predict(
                X_observed, y_observed, X_remaining
            )
            graph_scores, graph_acquisition = acquisition_scores(
                graph_posterior, observed, dataset, step
            )
            if graph_acquisition != "stable_log_ei":
                raise RuntimeError("Graph expert acquisition schedule changed.")
            decision = arbitrate_top_three(
                ard_scores, graph_scores, candidate_ids, seed=seed + step
            )
            selected_local = int(decision["selected_local_index"])
            selected_expert = str(decision["selected_expert"])
        else:
            selected_local = ard_top
            selected_expert = "ard"
            decision = {
                "selected_local_index": selected_local,
                "selected_query_index": int(candidate_ids[selected_local]),
                "selected_expert": selected_expert,
                "ard_top_local_index": ard_top,
                "ard_top_query_index": int(candidate_ids[ard_top]),
                "ard_top_three_query_indices": None,
                "graph_top_local_index": None,
                "graph_top_query_index": None,
                "graph_top_ard_rank": None,
                "gate_accepted": False,
                "selection_changed_from_ard": False,
            }

        selected_posterior = (
            graph_posterior if selected_expert == "graph" else ard_posterior
        )
        selected_scores = graph_scores if selected_expert == "graph" else ard_scores
        query_index = int(candidate_ids[selected_local])
        observed_yield = float(oracle.query(query_index))
        selected_global = int(remaining[selected_local])
        selected_row = dataset.candidates.iloc[selected_global]
        graph_mean = (
            None if graph_posterior is None else float(graph_posterior.mean[selected_local])
        )
        graph_std = (
            None if graph_posterior is None else float(graph_posterior.std[selected_local])
        )
        trajectory.append(
            {
                "step": step,
                "query_index": query_index,
                "observed_yield": observed_yield,
                "predicted_yield": float(selected_posterior.mean[selected_local]),
                "predicted_std": float(selected_posterior.std[selected_local]),
                "acquisition_name": acquisition_name,
                "acquisition_value": float(selected_scores[selected_local]),
                "selected_expert": selected_expert,
                "ard_predicted_yield": float(ard_posterior.mean[selected_local]),
                "ard_predicted_std": float(ard_posterior.std[selected_local]),
                "graph_predicted_yield": graph_mean,
                "graph_predicted_std": graph_std,
                **decision,
            }
        )
        arbitration.append(
            {
                "episode": episode["episode"],
                "split_seed": int(episode["split_seed"]),
                "campaign_seed": seed,
                "variant": variant,
                "step": step,
                "gate_evaluated": variant == CANDIDATE and step > 1,
                "top_k": TOP_K,
                "remaining_candidates": len(candidate_ids),
                "ard_top_score": float(ard_scores[ard_top]),
                "graph_top_score": None
                if graph_scores is None
                else float(graph_scores[int(decision["graph_top_local_index"])]),
                **decision,
            }
        )
        new_observation = selected_row.loc[
            list(dataset.spec.feature_columns)
        ].to_dict()
        new_observation[dataset.spec.target_column] = observed_yield
        observed = pd.concat(
            [observed, pd.DataFrame([new_observation])], ignore_index=True
        )
        X_observed = np.vstack([X_observed, prepared.X_candidates[selected_global]])
        y_observed = np.append(y_observed, observed_yield)
        remaining = np.delete(remaining, selected_local)

    elapsed = time.perf_counter() - started
    candidate_global_best = float(lookup[config.dataset.target_column].max())
    metrics = common.campaign_metrics(trajectory, candidate_global_best)
    calibration = common.calibration_rows(
        trajectory, episode=episode, variant=variant
    )
    queried = [int(item["query_index"]) for item in trajectory]
    candidate_pool = set(int(value) for value in episode["candidate_indices"])
    contract = {
        "phase": PHASE,
        "variant": variant,
        "train_only": True,
        "input_file": str(TRAIN_FILE.relative_to(DATASET_ROOT)),
        "public_test_labels_used": False,
        "public_searchspace_labels_used": False,
        "historical_810_910_920_930_outcomes_used": False,
        "protected_1020_or_1030_outcomes_opened": False,
        "baseline_model": ARD_MODEL,
        "graph_model": GRAPH_MODEL if variant == CANDIDATE else None,
        "query_1": "component ARD posterior mean only",
        "queries_2_to_5": "graph top LogEI iff graph top is in ARD top three; otherwise ARD top LogEI",
        "top_k": TOP_K,
        "threshold_tuned": False,
        "target_transform": "standard",
        "held_candidate_global_best": candidate_global_best,
        "t95_miss_penalty": BUDGET + 1,
        "episode": episode,
        "query_audit": {
            "all_queries_in_candidate_pool": set(queried).issubset(candidate_pool),
            "queries_unique": len(queried) == len(set(queried)) == BUDGET,
            "finite_positive_uncertainty": all(
                np.isfinite(item["predicted_std"]) and item["predicted_std"] > 0.0
                for item in calibration
            ),
            "query_1_uses_ard": trajectory[0]["selected_expert"] == "ard",
            "same_history_for_experts": True,
        },
    }
    payload = {
        "dataset": "Suzuki",
        "variant": variant,
        "episode": episode,
        "trajectory": trajectory,
        "arbitration": arbitration,
        "metrics": metrics,
        "contract": contract,
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
        "metrics": {
            "dataset": "Suzuki",
            "phase": PHASE,
            "episode_family": "covered_random_holdout",
            "episode": episode["episode"],
            "split_seed": int(episode["split_seed"]),
            "campaign_seed": seed,
            "variant": variant,
            "model": ARD_MODEL if variant == BASELINE else "ard_graph_top3_consensus",
            "n_initial": len(episode["initial_indices"]),
            "n_candidates": len(episode["candidate_indices"]),
            "budget": BUDGET,
            "candidate_global_best": candidate_global_best,
            **metrics,
            "elapsed_seconds": elapsed,
        },
        "calibration": calibration,
        "arbitration": arbitration,
        "contract": contract,
    }


def paired_summary(rows: pd.DataFrame) -> pd.DataFrame:
    baseline = rows[rows["variant"].eq(BASELINE)].set_index("episode")
    candidate = rows[rows["variant"].eq(CANDIDATE)].set_index("episode")
    common_ids = baseline.index.intersection(candidate.index)
    records = []
    for metric in SUMMARY_METRICS:
        delta = candidate.loc[common_ids, metric] - baseline.loc[common_ids, metric]
        mean, sd, low, high = common.mean_ci(delta)
        higher = metric not in {"round_to_95_global_best", "elapsed_seconds"}
        wins = int((delta > 1e-10).sum()) if higher else int((delta < -1e-10).sum())
        losses = int((delta < -1e-10).sum()) if higher else int((delta > 1e-10).sum())
        records.append(
            {
                "candidate": CANDIDATE,
                "baseline": BASELINE,
                "metric": metric,
                "n_pairs": len(common_ids),
                "mean_delta": mean,
                "sample_sd_delta": sd,
                "ci95_low": low,
                "ci95_high": high,
                "wins": wins,
                "ties": len(common_ids) - wins - losses,
                "losses": losses,
            }
        )
    return pd.DataFrame(records)


def arbitration_summary(rows: pd.DataFrame) -> pd.DataFrame:
    candidate = rows[rows["variant"].eq(CANDIDATE)].copy()
    records = []
    for step, group in candidate.groupby("step", sort=True):
        evaluated = group[group["gate_evaluated"]]
        records.append(
            {
                "step": int(step),
                "n_episodes": len(group),
                "gate_evaluated": len(evaluated),
                "graph_gate_accepted": int(evaluated["gate_accepted"].sum()),
                "graph_gate_acceptance_rate": float(evaluated["gate_accepted"].mean())
                if len(evaluated)
                else 0.0,
                "selection_changed_from_ard": int(
                    evaluated["selection_changed_from_ard"].sum()
                ),
                "selection_change_rate": float(
                    evaluated["selection_changed_from_ard"].mean()
                )
                if len(evaluated)
                else 0.0,
                "graph_top_equals_ard_top": int(
                    (evaluated["graph_top_ard_rank"] == 1).sum()
                ),
                "graph_top_ard_rank_mean": float(
                    evaluated["graph_top_ard_rank"].mean()
                )
                if len(evaluated)
                else None,
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
        "initial_exact_tie": abs(
            float(metrics.loc["initial_round_found_best", "mean_delta"])
        )
        <= 1e-12,
        "auc_mean_delta_strictly_positive": float(
            metrics.loc["AUC_best_so_far", "mean_delta"]
        )
        > 0.0,
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
        "top_k": TOP_K,
        "threshold_tuned": False,
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
    arbitration: pd.DataFrame,
    contracts: list[dict],
    output_root: Path,
    expected: int,
) -> dict[str, object]:
    seeds = rows["split_seed"].astype(int)
    trajectory_count = len(list((output_root / "trajectories").glob("*/*.json")))
    query_audits = [item["query_audit"] for item in contracts]
    candidate_arb = arbitration[arbitration["variant"].eq(CANDIDATE)]
    evaluated = candidate_arb[candidate_arb["gate_evaluated"]]
    initial = rounds[rounds["step"].eq(1)].pivot(
        index="episode",
        columns="variant",
        values=["query_index", "observed_yield", "predicted_yield", "predicted_std"],
    )
    exact_initial = all(
        np.array_equal(
            initial[(field, BASELINE)].to_numpy(),
            initial[(field, CANDIDATE)].to_numpy(),
        )
        for field in ("query_index", "observed_yield", "predicted_yield", "predicted_std")
    )
    logic_ok = bool(
        (
            evaluated["gate_accepted"]
            == (evaluated["graph_top_ard_rank"].astype(int) <= TOP_K)
        ).all()
        and (
            evaluated["selected_query_index"]
            == np.where(
                evaluated["gate_accepted"],
                evaluated["graph_top_query_index"],
                evaluated["ard_top_query_index"],
            )
        ).all()
    )
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
        "query_1_bit_identical": exact_initial,
        "twenty_initial": bool(rows["n_initial"].eq(INITIAL_COUNT).all()),
        "nine_candidates": bool(rows["n_candidates"].eq(CANDIDATE_COUNT).all()),
        "five_rounds": len(rounds) == expected * len(VARIANTS) * BUDGET,
        "candidate_arbitration_rows": len(candidate_arb) == expected * BUDGET,
        "candidate_gate_evaluations": len(evaluated) == expected * (BUDGET - 1),
        "fixed_top_three": bool(candidate_arb["top_k"].eq(TOP_K).all()),
        "arbitration_logic": logic_ok,
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
        "query_1_uses_ard": all(item["query_1_uses_ard"] for item in query_audits),
        "same_history_for_experts": all(
            item["same_history_for_experts"] for item in query_audits
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Consensus portfolio result audit failed: {checks}")
    return {
        "passed": True,
        "checks": checks,
        "n_episodes": expected,
        "n_metric_rows": len(rows),
        "n_round_rows": len(rounds),
        "n_arbitration_rows": len(arbitration),
        "n_candidate_gate_evaluations": len(evaluated),
        "n_trajectories": trajectory_count,
        "split_seed_min": int(seeds.min()),
        "split_seed_max": int(seeds.max()),
        "top_k": TOP_K,
        "threshold_tuned": False,
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
        "common_discovery_driver": {"path": str(COMMON_DRIVER_SOURCE.relative_to(DATASET_ROOT)), "sha256": file_sha256(COMMON_DRIVER_SOURCE)},
        "ard": {"path": str(ARD_SOURCE.relative_to(DATASET_ROOT)), "sha256": file_sha256(ARD_SOURCE)},
        "graph": {"path": str(GRAPH_SOURCE.relative_to(DATASET_ROOT)), "sha256": file_sha256(GRAPH_SOURCE)},
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
        "episode_count": len(episodes),
        "episode_contract_sha256": payload_sha256(episodes),
        "split_seed_start": SPLIT_SEED_START,
        "split_seed_min": min(split_seeds),
        "split_seed_max": max(split_seeds),
        "n_initial": INITIAL_COUNT,
        "n_candidates": CANDIDATE_COUNT,
        "budget": BUDGET,
        "portfolio_contract": {
            "top_k": TOP_K,
            "threshold_tuned": False,
            "query_1": "component ARD posterior mean only",
            "queries_2_to_5": "graph top LogEI iff in ARD top three; else ARD top LogEI",
            "posterior_averaging": False,
            "prequential_weighting": False,
            "outcome_selector_fitting": False,
        },
        "target_transform": "standard",
        "global_best_scope": "held candidate pool",
        "t95_miss_penalty": BUDGET + 1,
        "runtime_gate": {"candidate_to_baseline_ratio_max": PRACTICAL_RUNTIME_RATIO_MAX, "candidate_total_seconds_max": PRACTICAL_TOTAL_SECONDS_MAX},
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
            print(f"{row['variant']} {row['episode']}: AUC={row['AUC_best_so_far']:.3f}", flush=True)
    wall_seconds = time.perf_counter() - wall_started

    rows = pd.DataFrame([item["metrics"] for item in results]).sort_values(["episode", "variant"])
    rounds = pd.DataFrame([record for item in results for record in item["calibration"]]).sort_values(["episode", "variant", "step"])
    arbitration = pd.DataFrame([record for item in results for record in item["arbitration"]]).sort_values(["episode", "variant", "step"])
    contracts = [item["contract"] for item in results]
    rows.to_csv(args.output_dir / "per_episode.csv", index=False)
    rounds.to_csv(args.output_dir / "round_calibration.csv", index=False)
    arbitration.to_csv(args.output_dir / "arbitration.csv", index=False)
    summary = common.absolute_summary(rows)
    paired = paired_summary(rows)
    tails = common.tail_summary(rows)
    calibration = common.calibration_summary(rounds)
    arb_summary = arbitration_summary(arbitration)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    paired.to_csv(args.output_dir / "paired_consensus_vs_component_ard.csv", index=False)
    tails.to_csv(args.output_dir / "tail_summary.csv", index=False)
    calibration.to_csv(args.output_dir / "calibration_summary.csv", index=False)
    arb_summary.to_csv(args.output_dir / "arbitration_summary.csv", index=False)
    first_count = min(20, len(episodes))
    first_ids = [f"random_{PHASE}_{index:02d}" for index in range(first_count)]
    first = rows[rows["episode"].isin(first_ids)].sort_values(["episode", "variant"])
    if len(first) != first_count * len(VARIANTS):
        raise RuntimeError("Deterministic first slice is incomplete.")
    first.to_csv(args.output_dir / "first20_per_episode.csv", index=False)
    common.absolute_summary(first).to_csv(args.output_dir / "first20_summary.csv", index=False)
    paired_summary(first).to_csv(args.output_dir / "first20_paired_consensus_vs_component_ard.csv", index=False)
    decision = gate_decision(paired, tails, calibration, rows)
    decision["wall_seconds"] = wall_seconds
    (args.output_dir / "gate_decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")
    audit = audit_results(rows, rounds, arbitration, contracts, args.output_dir, len(episodes))
    audit["wall_seconds"] = wall_seconds
    (args.output_dir / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
