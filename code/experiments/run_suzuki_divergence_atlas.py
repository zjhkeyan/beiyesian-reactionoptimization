#!/usr/bin/env python3
"""Fresh 910-series policy-divergence atlas for Suzuki fixed and ARD GPs.

This is an opt-in, train-only diagnostic. It deliberately leaves all promoted
configuration files untouched and never reads the public test/search-space
tables. The experiment records honest pre-query predictions and fitted kernel
geometry so differences between the two otherwise identical policies can be
localized to the first recommendation on which they disagree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm, t


EXPERIMENT_ROOT = Path(__file__).resolve().parent
CODE_ROOT = EXPERIMENT_ROOT.parent
DATASET_ROOT = CODE_ROOT.parents[1]
EXTENSION_ROOT = CODE_ROOT / "bgolearn_rxn_extension"
sys.path.insert(0, str(EXTENSION_ROOT / "src"))
sys.path.insert(0, str(EXPERIMENT_ROOT))

import run_suzuki_component_ard_replication as replication  # noqa: E402
import run_suzuki_holdout_proxy as suzuki_proxy  # noqa: E402
from bgolearn_rxn.acquisition import stable_log_expected_improvement  # noqa: E402
from bgolearn_rxn.campaign import CampaignRunner  # noqa: E402
from bgolearn_rxn.oracle import OfflineTableOracle  # noqa: E402


VARIANTS = replication.VARIANTS
BASELINE = replication.BASELINE
CANDIDATE = replication.CANDIDATE
MODEL_NAMES = replication.MODEL_NAMES
CORE_METRICS = replication.CORE_METRICS
SUMMARY_METRICS = replication.SUMMARY_METRICS
TRAIN_FILE = replication.TRAIN_FILE
CONFIG_FILE = suzuki_proxy.CONFIG_FILE
PHASE = "phase5_divergence_atlas"
SPLIT_SEED_START = 910_000
SPLIT_SEED_STOP = 920_000
PROTECTED_SEED_STARTS = (810_000, 920_000, 930_000)
EPISODE_COUNT = 100
BUDGET = 5
LARGE_WIN_AUC_DELTA = 5.0
CALIBRATION_LEVELS = (0.50, 0.80, 0.90, 0.95)
PAIR_COLUMNS = ("Electrophile", "Nucleophile")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean_ci(values: pd.Series | np.ndarray) -> tuple[float, float, float, float]:
    array = np.asarray(values, dtype=float).reshape(-1)
    mean = float(np.mean(array))
    if len(array) == 1:
        return mean, 0.0, mean, mean
    sd = float(np.std(array, ddof=1))
    half = float(t.ppf(0.975, len(array) - 1) * sd / np.sqrt(len(array)))
    return mean, sd, mean - half, mean + half


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
        raise RuntimeError("The 910-series atlas episode contract is incomplete.")
    if min(seeds) < SPLIT_SEED_START or max(seeds) >= SPLIT_SEED_STOP:
        raise RuntimeError("Atlas episode construction escaped the 910-series family.")
    if any(int(item["budget"]) != BUDGET for item in episodes):
        raise RuntimeError("Atlas episode budget changed.")
    return episodes


def build_problem(episode: dict, variant: str):
    config, dataset, lookup, model_name = replication.build_problem(episode, variant)
    if tuple(config.ensemble.enabled_models) != (model_name,):
        raise RuntimeError("Atlas enabled an unexpected ensemble model.")
    if config.policy.first_query != model_name:
        raise RuntimeError("Atlas first-query policy changed.")
    if config.policy.early_end != BUDGET or config.policy.final_start != BUDGET + 1:
        raise RuntimeError("Atlas policy is not stable LogEI through query 5.")
    if config.gp.target_transform != "standard":
        raise RuntimeError("Atlas target transform changed.")
    return config, dataset, lookup, model_name


def _jsonable(value):
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _component_geometry(model) -> dict[str, object]:
    theta = np.asarray(model.theta_, dtype=float)
    lengthscales, output_scale, noise, pair_scale, task_rho = model._unpack(theta)
    amplitudes = dict(getattr(model, "component_amplitudes_", {}))
    shares = dict(getattr(model, "component_shares_", {}))
    if not amplitudes:
        amplitudes = {"bias": 1.0}
        amplitudes.update(
            {f"role:{role}": float(max(model._role_weight(role), 0.0)) for role in model.roles}
        )
        amplitudes.update(
            {
                f"pair:{'|'.join(pair)}": float(
                    pair_scale * max(model._interaction_weight(pair), 0.0)
                )
                for pair in model.config.interaction_pairs
            }
        )
        total = float(sum(amplitudes.values()))
        shares = {name: value / total for name, value in amplitudes.items()}
    return {
        "theta": theta.tolist(),
        "lengthscales": lengthscales,
        "output_scale": float(output_scale),
        "noise": float(noise),
        "pair_scale": float(pair_scale),
        "task_rho": float(task_rho),
        "component_amplitudes": amplitudes,
        "component_shares": shares,
        "optimization": _jsonable(model.optimization_result_),
    }


def _acquisition_scores(runner: CampaignRunner, posterior, best_f: float, step: int) -> np.ndarray:
    if step == 1:
        return np.asarray(posterior.mean, dtype=float).copy()
    return stable_log_expected_improvement(
        posterior.mean,
        posterior.std,
        best_f,
        xi=runner.policy._xi(step, BUDGET),
    )


def _selected_margin(values: np.ndarray, selected: int) -> tuple[float, float, int]:
    values = np.asarray(values, dtype=float)
    competitors = np.delete(values, selected)
    if len(competitors) == 0:
        return math.inf, float("nan"), -1
    alternate_local = int(np.argmax(np.where(np.arange(len(values)) == selected, -np.inf, values)))
    return (
        float(values[selected] - values[alternate_local]),
        float(values[alternate_local]),
        alternate_local,
    )


def _coverage_context(source: pd.DataFrame, episode: dict) -> dict[str, object]:
    initial = source.loc[np.asarray(episode["initial_indices"], dtype=int)]
    candidate = source.loc[np.asarray(episode["candidate_indices"], dtype=int)]
    initial_pairs = set(map(tuple, initial.loc[:, PAIR_COLUMNS].astype(str).to_numpy()))
    candidate_pairs = list(map(tuple, candidate.loc[:, PAIR_COLUMNS].astype(str).to_numpy()))
    unseen = [pair not in initial_pairs for pair in candidate_pairs]
    return {
        "n_initial_unique_pairs": len(initial_pairs),
        "n_candidate_unique_pairs": len(set(candidate_pairs)),
        "n_candidate_rows_unseen_pair": int(sum(unseen)),
        "candidate_pair_coverage_rate": float(1.0 - np.mean(unseen)),
    }


def _query_context(
    source: pd.DataFrame,
    episode: dict,
    query_index: int,
) -> dict[str, object]:
    initial = source.loc[np.asarray(episode["initial_indices"], dtype=int)]
    query = source.loc[int(query_index)]
    electrophile = str(query[PAIR_COLUMNS[0]])
    nucleophile = str(query[PAIR_COLUMNS[1]])
    same_pair = initial[PAIR_COLUMNS[0]].astype(str).eq(electrophile) & initial[
        PAIR_COLUMNS[1]
    ].astype(str).eq(nucleophile)
    return {
        "electrophile": electrophile,
        "nucleophile": nucleophile,
        "substrate_pair": f"{electrophile} || {nucleophile}",
        "pair_seen_in_initial": bool(same_pair.any()),
        "initial_pair_count": int(same_pair.sum()),
        "initial_electrophile_count": int(
            initial[PAIR_COLUMNS[0]].astype(str).eq(electrophile).sum()
        ),
        "initial_nucleophile_count": int(
            initial[PAIR_COLUMNS[1]].astype(str).eq(nucleophile).sum()
        ),
    }


def _calibration_record(observed: float, predicted: float, std: float) -> dict[str, object]:
    safe_std = max(float(std), 1e-12)
    residual = float(observed - predicted)
    standardized = residual / safe_std
    record: dict[str, object] = {
        "residual": residual,
        "absolute_residual": abs(residual),
        "standardized_residual": standardized,
        "squared_standardized_residual": standardized**2,
        "gaussian_nll": float(
            0.5 * standardized**2 + math.log(safe_std) + 0.5 * math.log(2.0 * math.pi)
        ),
    }
    for level in CALIBRATION_LEVELS:
        z_value = float(norm.ppf(0.5 + level / 2.0))
        record[f"covered_{int(level * 100)}"] = bool(abs(residual) <= z_value * safe_std)
    return record


def run_one(job: tuple[str, dict, str]) -> dict:
    variant, episode, output_root_string = job
    source = pd.read_csv(TRAIN_FILE).reset_index(drop=True)
    config, dataset, lookup, model_name = build_problem(episode, variant)
    oracle = OfflineTableOracle(dataset=dataset, lookup=lookup)
    runner = CampaignRunner(
        dataset,
        config,
        seed=int(episode["campaign_seed"]),
        backend="native",
    )
    session = runner.start_session(budget=BUDGET)
    round_diagnostics: list[dict] = []
    candidate_scores: list[dict] = []
    started = time.perf_counter()
    while not session.is_complete:
        step = session.step
        recommendation = session.recommend()
        remaining_frame = runner.dataset.candidates.iloc[session.remaining_rows]
        remaining_X = runner.prepared.X_candidates[session.remaining_rows]
        # The native backend has just fitted the persistent single-model
        # ensemble. Predicting again is label-free and reproduces the posterior
        # used by the committed recommendation without another optimization.
        posterior = runner.ensemble.predict(remaining_X)
        best_before = float(session._pending["best_before"])
        scores = _acquisition_scores(runner, posterior, best_before, step)
        selected = int(recommendation.local_index)
        acquisition_margin, alternate_score, alternate_local = _selected_margin(scores, selected)
        mean_margin, alternate_mean, mean_alternate_local = _selected_margin(
            posterior.mean, selected
        )
        candidate_ids = remaining_frame["query_index"].to_numpy(dtype=int)
        for local_index, query_id in enumerate(candidate_ids):
            candidate_scores.append(
                {
                    "episode": episode["episode"],
                    "split_seed": int(episode["split_seed"]),
                    "variant": variant,
                    "step": step,
                    "query_index": int(query_id),
                    "selected": bool(local_index == selected),
                    "posterior_mean": float(posterior.mean[local_index]),
                    "posterior_std": float(posterior.std[local_index]),
                    "acquisition_score": float(scores[local_index]),
                }
            )
        observed = float(oracle.query(recommendation.query_index))
        context = _query_context(source, episode, recommendation.query_index)
        calibration = _calibration_record(
            observed, recommendation.predicted_mean, recommendation.predicted_std
        )
        model = runner.ensemble.models[model_name]
        geometry = _component_geometry(model)
        record = session.observe(recommendation.query_index, observed)
        round_diagnostics.append(
            {
                "episode": episode["episode"],
                "split_seed": int(episode["split_seed"]),
                "campaign_seed": int(episode["campaign_seed"]),
                "variant": variant,
                "model": model_name,
                "step": step,
                "query_index": int(recommendation.query_index),
                "observed_yield": observed,
                "predicted_mean": float(recommendation.predicted_mean),
                "predicted_std": float(recommendation.predicted_std),
                "acquisition_name": recommendation.acquisition_name,
                "acquisition_score": float(recommendation.acquisition_value),
                "acquisition_margin": acquisition_margin,
                "alternate_acquisition_score": alternate_score,
                "alternate_query_index": (
                    -1 if alternate_local < 0 else int(candidate_ids[alternate_local])
                ),
                "posterior_mean_margin": mean_margin,
                "alternate_posterior_mean": alternate_mean,
                "mean_alternate_query_index": (
                    -1
                    if mean_alternate_local < 0
                    else int(candidate_ids[mean_alternate_local])
                ),
                "incumbent_before_query": best_before,
                **context,
                **calibration,
                "lengthscales_json": json.dumps(geometry["lengthscales"], sort_keys=True),
                "component_amplitudes_json": json.dumps(
                    geometry["component_amplitudes"], sort_keys=True
                ),
                "component_shares_json": json.dumps(
                    geometry["component_shares"], sort_keys=True
                ),
                "output_scale": geometry["output_scale"],
                "noise": geometry["noise"],
                "pair_scale": geometry["pair_scale"],
                "optimization_objective": geometry["optimization"].get("objective"),
                "optimization_success": geometry["optimization"].get("success"),
                "n_observed_before_query": len(session.observed),
                "n_candidates_before_query": len(candidate_ids),
            }
        )
        record["diagnostics"] = {
            **context,
            **calibration,
            "acquisition_margin": acquisition_margin,
            "posterior_mean_margin": mean_margin,
            "geometry": geometry,
        }
    elapsed = time.perf_counter() - started
    result = session.finalize(global_best=oracle.global_best)
    payload = result.to_payload()
    metrics = replication.campaign_metrics(payload["trajectory"], oracle.global_best, BUDGET)
    coverage = _coverage_context(source, episode)
    payload["atlas_metrics"] = metrics
    payload["atlas_contract"] = {
        "phase": PHASE,
        "variant": variant,
        "model": model_name,
        "train_only": True,
        "public_test_labels_used": False,
        "public_searchspace_labels_used": False,
        "retired_810_outcomes_used": False,
        "protected_920_or_930_episodes_opened": False,
        "target_transform": "standard",
        "acquisition": "posterior mean query 1, stable LogEI queries 2-5",
        "episode": episode,
        "coverage": coverage,
    }
    output_root = Path(output_root_string)
    trajectory_path = output_root / "trajectories" / variant / f"{episode['episode']}.json"
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    trajectory_path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8"
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
            "global_best": float(oracle.global_best),
            **coverage,
            **metrics,
            "elapsed_seconds": elapsed,
        },
        "rounds": round_diagnostics,
        "candidate_scores": candidate_scores,
    }


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
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
    records: list[dict] = []
    for metric in SUMMARY_METRICS:
        delta = candidate.loc[common, metric] - baseline.loc[common, metric]
        mean, sd, low, high = mean_ci(delta)
        higher = metric not in {"round_to_95_global_best", "elapsed_seconds"}
        wins = int((delta > 1e-10).sum()) if higher else int((delta < -1e-10).sum())
        losses = int((delta < -1e-10).sum()) if higher else int((delta > 1e-10).sum())
        non_ties = wins + losses
        sign_p = (
            float(binomtest(min(wins, losses), non_ties, 0.5).pvalue)
            if non_ties
            else 1.0
        )
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
                "two_sided_sign_test_p": sign_p,
            }
        )
    return pd.DataFrame(records)


def tail_summary(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    for variant, group in rows.groupby("variant", sort=True):
        values = group["AUC_best_so_far"].astype(float)
        threshold = float(values.quantile(0.10, interpolation="linear"))
        count = max(1, int(math.ceil(0.10 * len(values))))
        worst = values.nsmallest(count)
        records.append(
            {
                "variant": variant,
                "n_episodes": len(values),
                "auc_10th_percentile": threshold,
                "worst_decile_count": count,
                "worst_decile_auc_mean": float(worst.mean()),
                "worst_decile_auc_minimum": float(worst.min()),
            }
        )
    return pd.DataFrame(records)


def classify_auc_delta(value: float) -> str:
    if abs(value) <= 1e-10:
        return "tie"
    if value >= LARGE_WIN_AUC_DELTA:
        return "large_win"
    if value > 0.0:
        return "small_win"
    return "loss"


def episode_analysis(metrics: pd.DataFrame, rounds: pd.DataFrame) -> pd.DataFrame:
    pivot = metrics.pivot(index="episode", columns="variant", values="AUC_best_so_far")
    records: list[dict] = []
    for episode, values in pivot.iterrows():
        section = rounds[rounds["episode"].eq(episode)]
        fixed = section[section["variant"].eq(BASELINE)].set_index("step")
        ard = section[section["variant"].eq(CANDIDATE)].set_index("step")
        differing = [
            step
            for step in range(1, BUDGET + 1)
            if int(fixed.loc[step, "query_index"]) != int(ard.loc[step, "query_index"])
        ]
        first = differing[0] if differing else None
        delta = float(values[CANDIDATE] - values[BASELINE])
        record: dict[str, object] = {
            "episode": episode,
            "split_seed": int(fixed.iloc[0]["split_seed"]),
            "fixed_auc": float(values[BASELINE]),
            "ard_auc": float(values[CANDIDATE]),
            "auc_delta_ard_minus_fixed": delta,
            "outcome_class": classify_auc_delta(delta),
            "first_divergence_round": first,
            "policies_never_diverged": first is None,
            "n_differing_rounds": len(differing),
        }
        if first is not None:
            for prefix, table in (("fixed", fixed), ("ard", ard)):
                row = table.loc[first]
                record.update(
                    {
                        f"{prefix}_divergence_query_index": int(row["query_index"]),
                        f"{prefix}_divergence_yield": float(row["observed_yield"]),
                        f"{prefix}_divergence_pair": row["substrate_pair"],
                        f"{prefix}_divergence_pair_seen_initial": bool(
                            row["pair_seen_in_initial"]
                        ),
                        f"{prefix}_divergence_acquisition_margin": float(
                            row["acquisition_margin"]
                        ),
                        f"{prefix}_divergence_predicted_std": float(
                            row["predicted_std"]
                        ),
                        f"{prefix}_divergence_standardized_residual": float(
                            row["standardized_residual"]
                        ),
                    }
                )
            shares = json.loads(ard.loc[first, "component_shares_json"])
            record.update(
                {f"ard_share_{name}": float(value) for name, value in shares.items()}
            )
        records.append(record)
    return pd.DataFrame(records).sort_values("episode")


def outcome_group_summary(episodes: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "auc_delta_ard_minus_fixed",
        "first_divergence_round",
        "n_differing_rounds",
        "fixed_divergence_yield",
        "ard_divergence_yield",
        "fixed_divergence_acquisition_margin",
        "ard_divergence_acquisition_margin",
        "fixed_divergence_predicted_std",
        "ard_divergence_predicted_std",
        "fixed_divergence_standardized_residual",
        "ard_divergence_standardized_residual",
    ]
    numeric.extend(column for column in episodes if column.startswith("ard_share_"))
    records: list[dict] = []
    for outcome, group in episodes.groupby("outcome_class", sort=False):
        record: dict[str, object] = {
            "outcome_class": outcome,
            "n_episodes": len(group),
            "never_diverged_fraction": float(group["policies_never_diverged"].mean()),
        }
        for column in numeric:
            if column in group:
                values = pd.to_numeric(group[column], errors="coerce").dropna()
                record[f"mean_{column}"] = (
                    float(values.mean()) if len(values) else float("nan")
                )
        for prefix in ("fixed", "ard"):
            column = f"{prefix}_divergence_pair_seen_initial"
            values = group[column].dropna() if column in group else pd.Series(dtype=float)
            record[f"fraction_{column}"] = (
                float(values.astype(bool).mean()) if len(values) else float("nan")
            )
        records.append(record)
    return pd.DataFrame(records)


def calibration_summary(rounds: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    for variant, group in rounds.groupby("variant", sort=True):
        record: dict[str, object] = {
            "variant": variant,
            "n_predictions": len(group),
            "mean_residual": float(group["residual"].mean()),
            "mean_absolute_error": float(group["absolute_residual"].mean()),
            "root_mean_squared_error": float(
                np.sqrt(np.mean(np.square(group["residual"].astype(float))))
            ),
            "mean_standardized_residual": float(group["standardized_residual"].mean()),
            "standardized_residual_sd": float(group["standardized_residual"].std(ddof=1)),
            "mean_gaussian_nll": float(group["gaussian_nll"].mean()),
        }
        for level in CALIBRATION_LEVELS:
            column = f"covered_{int(level * 100)}"
            record[f"empirical_coverage_{int(level * 100)}"] = float(group[column].mean())
        records.append(record)
    return pd.DataFrame(records)


def pair_context_summary(rounds: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    for (variant, seen), group in rounds.groupby(
        ["variant", "pair_seen_in_initial"], sort=True
    ):
        records.append(
            {
                "variant": variant,
                "pair_seen_in_initial": bool(seen),
                "n_queries": len(group),
                "mean_observed_yield": float(group["observed_yield"].mean()),
                "mean_absolute_residual": float(group["absolute_residual"].mean()),
                "mean_gaussian_nll": float(group["gaussian_nll"].mean()),
                "coverage_95": float(group["covered_95"].mean()),
                "mean_acquisition_margin": float(group["acquisition_margin"].mean()),
            }
        )
    return pd.DataFrame(records)


def explicit_twenty_run(rows: pd.DataFrame) -> pd.DataFrame:
    ids = [f"random_{PHASE}_{index:02d}" for index in range(20)]
    result = rows[rows["episode"].isin(ids)].copy()
    if len(result) != 20 * len(VARIANTS) or set(result["episode"]) != set(ids):
        raise RuntimeError("The deterministic first-20 atlas view is incomplete.")
    return result.sort_values(["episode", "variant"])


def audit_results(metrics: pd.DataFrame, rounds: pd.DataFrame, output_root: Path) -> dict:
    expected_metric_rows = EPISODE_COUNT * len(VARIANTS)
    expected_round_rows = expected_metric_rows * BUDGET
    seeds = metrics["split_seed"].astype(int)
    trajectory_files = list((output_root / "trajectories").glob("*/*.json"))
    checks = {
        "metric_row_count": len(metrics) == expected_metric_rows,
        "episode_count": metrics["episode"].nunique() == EPISODE_COUNT,
        "paired_split_seeds": bool(metrics.groupby("episode")["split_seed"].nunique().eq(1).all()),
        "round_row_count": len(rounds) == expected_round_rows,
        "round_budget_complete": bool(
            rounds.groupby(["episode", "variant"])["step"].nunique().eq(BUDGET).all()
        ),
        "trajectory_count": len(trajectory_files) == expected_metric_rows,
        "only_910_series": int(seeds.min()) >= SPLIT_SEED_START
        and int(seeds.max()) < SPLIT_SEED_STOP,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Atlas result audit failed: {checks}")
    return {
        "passed": True,
        "checks": checks,
        "n_episodes": EPISODE_COUNT,
        "n_metric_rows": len(metrics),
        "n_round_rows": len(rounds),
        "n_trajectories": len(trajectory_files),
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
    parser.add_argument("--limit-episodes", type=int, default=None, help="Smoke testing only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.jobs <= 0:
        raise ValueError("--jobs must be positive.")
    episodes = make_episodes()
    if args.limit_episodes is not None:
        if args.limit_episodes <= 0 or args.limit_episodes > EPISODE_COUNT:
            raise ValueError("--limit-episodes must be in [1, 100].")
        episodes = episodes[: args.limit_episodes]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_seeds = [int(item["split_seed"]) for item in episodes]
    manifest = {
        "stage": PHASE,
        "diagnostic_only": True,
        "train_only": True,
        "public_test_labels_used": False,
        "public_searchspace_labels_used": False,
        "retired_810_outcomes_used": False,
        "protected_920_or_930_episodes_opened": False,
        "input": {
            "path": str(TRAIN_FILE.relative_to(DATASET_ROOT)),
            "sha256": file_sha256(TRAIN_FILE),
            "n_rows": len(pd.read_csv(TRAIN_FILE)),
        },
        "implementations": {
            "atlas_runner": {
                "path": str(Path(__file__).resolve().relative_to(DATASET_ROOT)),
                "sha256": file_sha256(Path(__file__).resolve()),
            },
            "frozen_problem_builder": {
                "path": str(Path(replication.__file__).resolve().relative_to(DATASET_ROOT)),
                "sha256": file_sha256(Path(replication.__file__).resolve()),
            },
            "source_config": {
                "path": str(CONFIG_FILE.relative_to(DATASET_ROOT)),
                "sha256": file_sha256(CONFIG_FILE),
            },
        },
        "variants": list(VARIANTS),
        "models": MODEL_NAMES,
        "episode_count": len(episodes),
        "full_episode_count": EPISODE_COUNT,
        "budget": BUDGET,
        "split_seed_start": SPLIT_SEED_START,
        "split_seed_stop_exclusive": SPLIT_SEED_STOP,
        "split_seed_min": min(split_seeds),
        "split_seed_max": max(split_seeds),
        "protected_seed_starts": list(PROTECTED_SEED_STARTS),
        "target_transform": "standard",
        "acquisition": "posterior mean query 1, stable LogEI queries 2-5",
        "large_win_auc_delta_threshold": LARGE_WIN_AUC_DELTA,
        "calibration_interval_levels": list(CALIBRATION_LEVELS),
        "t95_global_best": "held candidate partition through OfflineTableOracle.global_best",
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
    metric_rows: list[dict] = []
    round_rows: list[dict] = []
    candidate_rows: list[dict] = []
    wall_started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(run_one, job): job for job in jobs}
        for future in as_completed(futures):
            result = future.result()
            metric_rows.append(result["metrics"])
            round_rows.extend(result["rounds"])
            candidate_rows.extend(result["candidate_scores"])
            row = result["metrics"]
            print(
                f"{row['variant']} {row['episode']}: AUC={row['AUC_best_so_far']:.3f}",
                flush=True,
            )
    wall_seconds = time.perf_counter() - wall_started
    metrics = pd.DataFrame(metric_rows).sort_values(["episode", "variant"])
    rounds = pd.DataFrame(round_rows).sort_values(["episode", "variant", "step"])
    candidates = pd.DataFrame(candidate_rows).sort_values(
        ["episode", "variant", "step", "query_index"]
    )
    if args.limit_episodes is not None:
        # Smoke outputs intentionally omit full-only summaries and cannot be
        # mistaken for the scored B_dev artifact.
        metrics.to_csv(args.output_dir / "smoke_per_episode.csv", index=False)
        rounds.to_csv(args.output_dir / "smoke_round_diagnostics.csv", index=False)
        candidates.to_csv(args.output_dir / "smoke_candidate_scores.csv", index=False)
        return

    audit = audit_results(metrics, rounds, args.output_dir)
    summary = summarize(metrics)
    paired = paired_summary(metrics)
    tails = tail_summary(metrics)
    episodes_table = episode_analysis(metrics, rounds)
    groups = outcome_group_summary(episodes_table)
    calibration = calibration_summary(rounds)
    pair_context = pair_context_summary(rounds)
    metrics.to_csv(args.output_dir / "per_episode.csv", index=False)
    rounds.to_csv(args.output_dir / "round_diagnostics.csv", index=False)
    candidates.to_csv(args.output_dir / "candidate_scores.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    paired.to_csv(args.output_dir / "paired_ard_vs_fixed.csv", index=False)
    tails.to_csv(args.output_dir / "auc_tail_summary.csv", index=False)
    episodes_table.to_csv(args.output_dir / "episode_divergence_analysis.csv", index=False)
    groups.to_csv(args.output_dir / "outcome_group_summary.csv", index=False)
    calibration.to_csv(args.output_dir / "calibration_summary.csv", index=False)
    pair_context.to_csv(args.output_dir / "pair_context_summary.csv", index=False)
    twenty = explicit_twenty_run(metrics)
    twenty.to_csv(args.output_dir / "twenty_run_per_episode.csv", index=False)
    summarize(twenty).to_csv(args.output_dir / "twenty_run_summary.csv", index=False)
    paired_summary(twenty).to_csv(
        args.output_dir / "twenty_run_paired_ard_vs_fixed.csv", index=False
    )
    audit["wall_seconds"] = wall_seconds
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
