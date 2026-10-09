#!/usr/bin/env python3
"""Train-only 910-series evaluation of the Suzuki pair-context exact GP."""

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

import run_suzuki_component_ard_replication as replication  # noqa: E402
import run_suzuki_holdout_proxy as suzuki_proxy  # noqa: E402
from bgolearn_rxn.campaign import CampaignRunner  # noqa: E402
from bgolearn_rxn.config import load_run_config  # noqa: E402
from bgolearn_rxn.datasets import DatasetSpec, ReactionDataset  # noqa: E402
from bgolearn_rxn.factory import build_ensemble, prepare_problem  # noqa: E402
from bgolearn_rxn.oracle import OfflineTableOracle  # noqa: E402


PHASE = "phase5_pair_context_gp"
BASELINE = "component_ard"
CANDIDATE = "pair_context"
VARIANTS = (BASELINE, CANDIDATE)
MODEL_NAMES = {
    BASELINE: "component_ard_gp",
    CANDIDATE: "pair_context_coregionalized_gp",
}
TRAIN_FILE = DATASET_ROOT / "Suzuki" / "suzuki_train.csv"
CONFIG_FILE = EXTENSION_ROOT / "configs" / "suzuki.yaml"
OUTPUT_ROOT = (
    DATASET_ROOT
    / "submission"
    / "results"
    / "experimental_investigations"
    / "phase5-node3-pair-context-gp"
)
SPLIT_SEED_START = 910_000
SPLIT_SEED_STOP = 920_000
PROTECTED_SEED_STARTS = (810_000, 920_000, 930_000)
EPISODE_COUNT = 60
BUDGET = 5
PAIR_COLUMNS = ("Electrophile", "Nucleophile")
METRICS = replication.CORE_METRICS
SUMMARY_METRICS = METRICS + ("elapsed_seconds",)
CALIBRATION_LEVELS = (0.50, 0.80, 0.90, 0.95)
LPO_PAIR_COUNT = 13
LPO_TRAIN_ROWS = 32
LPO_TEST_ROWS = 8


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def mean_ci(values) -> tuple[float, float, float, float]:
    array = np.asarray(values, dtype=float).reshape(-1)
    mean = float(np.mean(array))
    if len(array) == 1:
        return mean, 0.0, mean, mean
    sd = float(np.std(array, ddof=1))
    half = float(t.ppf(0.975, len(array) - 1) * sd / np.sqrt(len(array)))
    return mean, sd, mean - half, mean + half


def make_episodes() -> list[dict]:
    source = pd.read_csv(TRAIN_FILE)
    episodes = suzuki_proxy._phase_random_episodes(
        source,
        phase=PHASE,
        seed_start=SPLIT_SEED_START,
        count=EPISODE_COUNT,
    )
    seeds = [int(item["split_seed"]) for item in episodes]
    if len(episodes) != EPISODE_COUNT or len(set(seeds)) != EPISODE_COUNT:
        raise RuntimeError("Pair-context episode contract is incomplete.")
    if min(seeds) < SPLIT_SEED_START or max(seeds) >= SPLIT_SEED_STOP:
        raise RuntimeError("Pair-context episodes escaped the 910-series family.")
    return episodes


def _configure_model(base, model_name: str, *, budget: int, seed: int):
    return replace(
        base,
        dataset=replace(base.dataset, budget=budget, seeds=(seed,)),
        gp=replace(base.gp, target_transform="standard"),
        ensemble=replace(
            base.ensemble,
            enabled_models=(model_name,),
            initial_weights={model_name: 1.0},
        ),
        policy=replace(
            base.policy,
            first_query=model_name,
            early_end=budget,
            final_start=budget + 1,
            challenge_every=0,
            novelty_disagreement_mode="off",
        ),
    )


def build_problem(episode: dict, variant: str):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    base, dataset, lookup = suzuki_proxy.build_problem(episode, "gp_only")
    model_name = MODEL_NAMES[variant]
    config = _configure_model(
        base,
        model_name,
        budget=int(episode["budget"]),
        seed=int(episode["campaign_seed"]),
    )
    return config, dataset, lookup, model_name


def _calibration(observed: float, mean: float, std: float) -> dict[str, object]:
    safe_std = max(float(std), 1e-12)
    residual = float(observed - mean)
    standardized = residual / safe_std
    result: dict[str, object] = {
        "residual": residual,
        "absolute_residual": abs(residual),
        "squared_residual": residual**2,
        "standardized_residual": standardized,
        "gaussian_nll": float(
            0.5 * standardized**2
            + math.log(safe_std)
            + 0.5 * math.log(2.0 * math.pi)
        ),
    }
    for level in CALIBRATION_LEVELS:
        z = float(norm.ppf(0.5 + level / 2.0))
        result[f"covered_{int(level * 100)}"] = bool(abs(residual) <= z * safe_std)
    return result


def run_one(job: tuple[str, dict, str]) -> dict:
    variant, episode, output_root_string = job
    source = pd.read_csv(TRAIN_FILE).reset_index(drop=True)
    config, dataset, lookup, model_name = build_problem(episode, variant)
    oracle = OfflineTableOracle(dataset=dataset, lookup=lookup)
    runner = CampaignRunner(
        dataset, config, seed=int(episode["campaign_seed"]), backend="native"
    )
    session = runner.start_session(budget=BUDGET)
    initial = source.loc[np.asarray(episode["initial_indices"], dtype=int)]
    initial_pairs = set(
        map(tuple, initial.loc[:, PAIR_COLUMNS].astype(str).to_numpy())
    )
    calibration_rows: list[dict] = []
    started = time.perf_counter()
    while not session.is_complete:
        recommendation = session.recommend()
        query = source.loc[int(recommendation.query_index)]
        pair = tuple(str(query[column]) for column in PAIR_COLUMNS)
        observed = float(oracle.query(recommendation.query_index))
        calibration_rows.append(
            {
                "episode": episode["episode"],
                "split_seed": int(episode["split_seed"]),
                "variant": variant,
                "model": model_name,
                "step": int(session.step),
                "query_index": int(recommendation.query_index),
                "observed_yield": observed,
                "predicted_mean": float(recommendation.predicted_mean),
                "predicted_std": float(recommendation.predicted_std),
                "pair_seen_in_initial": pair in initial_pairs,
                "substrate_pair": " || ".join(pair),
                **_calibration(
                    observed,
                    recommendation.predicted_mean,
                    recommendation.predicted_std,
                ),
            }
        )
        session.observe(recommendation.query_index, observed)
    elapsed = time.perf_counter() - started
    result = session.finalize(global_best=oracle.global_best)
    payload = result.to_payload()
    metrics = replication.campaign_metrics(
        payload["trajectory"], float(oracle.global_best), BUDGET
    )
    model = runner.ensemble.models[model_name]
    amplitudes = dict(getattr(model, "component_amplitudes_", {}))
    shares = dict(getattr(model, "component_shares_", {}))
    pair_diagonal = getattr(model, "pair_diagonal_", None)
    pair_contrasts = getattr(model, "pair_residual_contrasts_", None)
    payload["pair_context_contract"] = {
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
        "component_amplitudes": amplitudes or None,
        "pair_diagonal": _jsonable(pair_diagonal),
        "pair_residual_contrasts": _jsonable(pair_contrasts),
    }
    trajectory_path = (
        Path(output_root_string)
        / "trajectories"
        / variant
        / f"{episode['episode']}.json"
    )
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
            **metrics,
            "elapsed_seconds": elapsed,
            "component_amplitudes_json": json.dumps(amplitudes, sort_keys=True),
            "component_shares_json": json.dumps(shares, sort_keys=True),
            "pair_diagonal_json": (
                None if pair_diagonal is None else json.dumps(_jsonable(pair_diagonal))
            ),
            "pair_residual_contrasts_json": (
                None if pair_contrasts is None else json.dumps(_jsonable(pair_contrasts))
            ),
        },
        "calibration": calibration_rows,
    }


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
    for variant, group in rows.groupby("variant", sort=True):
        auc = group["AUC_best_so_far"].astype(float)
        count = max(1, int(math.ceil(0.10 * len(auc))))
        worst = auc.nsmallest(count)
        records.append(
            {
                "variant": variant,
                "n_episodes": len(auc),
                "worst_decile_count": count,
                "auc_10th_percentile": float(auc.quantile(0.10)),
                "cvar_worst_decile_auc": float(worst.mean()),
                "minimum_auc": float(auc.min()),
            }
        )
    return pd.DataFrame(records)


def calibration_summary(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for variant, group in rows.groupby("variant", sort=True):
        record = {
            "variant": variant,
            "n_predictions": len(group),
            "mean_absolute_error": float(group["absolute_residual"].mean()),
            "root_mean_squared_error": float(np.sqrt(group["squared_residual"].mean())),
            "mean_gaussian_nll": float(group["gaussian_nll"].mean()),
            "standardized_residual_sd": float(
                group["standardized_residual"].std(ddof=1)
            ),
        }
        for level in CALIBRATION_LEVELS:
            record[f"empirical_coverage_{int(level * 100)}"] = float(
                group[f"covered_{int(level * 100)}"].mean()
            )
        records.append(record)
    return pd.DataFrame(records)


def pair_geometry_summary(rows: pd.DataFrame) -> pd.DataFrame:
    candidate = rows[rows["variant"].eq(CANDIDATE)]
    records = []
    for _, row in candidate.iterrows():
        diagonal = np.asarray(json.loads(row["pair_diagonal_json"]), dtype=float)
        contrasts = np.asarray(
            json.loads(row["pair_residual_contrasts_json"]), dtype=float
        )
        amplitudes = json.loads(row["component_amplitudes_json"])
        records.append(
            {
                "episode": row["episode"],
                "n_pair_contexts": len(diagonal),
                "pair_diagonal_geometric_mean": float(
                    np.exp(np.mean(np.log(diagonal)))
                ),
                "pair_diagonal_minimum": float(diagonal.min()),
                "pair_diagonal_maximum": float(diagonal.max()),
                "pair_diagonal_ratio": float(diagonal.max() / diagonal.min()),
                "residual_contrast_l2": float(np.linalg.norm(contrasts)),
                "residual_contrast_max_abs": float(np.max(np.abs(contrasts))),
                "constituent_product_amplitude": float(
                    amplitudes["pair_context:constituent_product"]
                ),
                "exact_identity_amplitude": float(
                    amplitudes["pair_context:exact_identity"]
                ),
            }
        )
    return pd.DataFrame(records)


def _lpo_problem(
    source: pd.DataFrame,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    model_name: str,
    seed: int,
):
    base = load_run_config(CONFIG_FILE)
    config = _configure_model(base, model_name, budget=1, seed=seed)
    features = list(config.dataset.feature_columns)
    initial = source.loc[
        train_indices, features + [config.dataset.target_column]
    ].copy()
    candidates = source.loc[test_indices, features].copy()
    candidates.insert(0, "query_index", test_indices.astype(int))
    dataset = ReactionDataset(
        spec=DatasetSpec.from_config(config.dataset),
        initial=initial,
        candidates=candidates,
    )
    prepared = prepare_problem(dataset, config)
    model = build_ensemble(config, prepared, seed=seed).models[model_name]
    y = initial[config.dataset.target_column].to_numpy(dtype=float)
    model.fit(prepared.X_initial, y)
    return model.predict(prepared.X_candidates)


def leave_pair_out_diagnostic() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    source = pd.read_csv(TRAIN_FILE).reset_index(drop=True)
    pair_values = source.loc[:, PAIR_COLUMNS].astype(str)
    pairs = sorted(map(tuple, pair_values.drop_duplicates().to_numpy()))[:LPO_PAIR_COUNT]
    rows = []
    for pair_index, pair in enumerate(pairs):
        held_mask = pair_values.eq(pair).all(axis=1).to_numpy()
        held = np.flatnonzero(held_mask)
        available = np.flatnonzero(~held_mask)
        rng = np.random.default_rng(910_700 + pair_index)
        train_indices = np.sort(
            rng.choice(available, size=min(LPO_TRAIN_ROWS, len(available)), replace=False)
        )
        test_indices = np.sort(
            rng.choice(held, size=min(LPO_TEST_ROWS, len(held)), replace=False)
        )
        observed = source.loc[test_indices, "Yield"].to_numpy(dtype=float)
        for variant in VARIANTS:
            posterior = _lpo_problem(
                source,
                train_indices,
                test_indices,
                MODEL_NAMES[variant],
                seed=910_800 + pair_index,
            )
            for row_index, query_index in enumerate(test_indices):
                rows.append(
                    {
                        "held_pair_index": pair_index,
                        "held_pair": " || ".join(pair),
                        "variant": variant,
                        "query_index": int(query_index),
                        "observed_yield": float(observed[row_index]),
                        "predicted_mean": float(posterior.mean[row_index]),
                        "predicted_std": float(posterior.std[row_index]),
                        **_calibration(
                            observed[row_index],
                            posterior.mean[row_index],
                            posterior.std[row_index],
                        ),
                    }
                )
    detail = pd.DataFrame(rows)
    summaries = []
    for variant, group in detail.groupby("variant", sort=True):
        summaries.append(
            {
                "variant": variant,
                "n_held_pairs": group["held_pair"].nunique(),
                "n_predictions": len(group),
                "mean_absolute_error": float(group["absolute_residual"].mean()),
                "root_mean_squared_error": float(
                    np.sqrt(group["squared_residual"].mean())
                ),
                "mean_gaussian_nll": float(group["gaussian_nll"].mean()),
                "coverage_95": float(group["covered_95"].mean()),
            }
        )
    summary = pd.DataFrame(summaries)
    indexed = summary.set_index("variant")
    mae_improvement = float(
        indexed.loc[BASELINE, "mean_absolute_error"]
        - indexed.loc[CANDIDATE, "mean_absolute_error"]
    )
    diagnostic = {
        "positive": mae_improvement > 0.0,
        "primary": "baseline MAE minus candidate MAE on deterministic held pairs",
        "mae_improvement": mae_improvement,
        "n_held_pairs": LPO_PAIR_COUNT,
        "train_rows_per_pair": LPO_TRAIN_ROWS,
        "test_rows_per_pair": LPO_TEST_ROWS,
        "selection_used_public_labels": False,
    }
    return detail, summary, diagnostic


def gate_decision(
    paired: pd.DataFrame, tails: pd.DataFrame, lpo: dict
) -> dict[str, object]:
    metrics = paired.set_index("metric")
    tail = tails.set_index("variant")
    checks = {
        "random_auc_positive": float(metrics.loc["AUC_best_so_far", "mean_delta"]) > 0.0,
        "initial_noninferior": float(
            metrics.loc["initial_round_found_best", "mean_delta"]
        )
        >= 0.0,
        "best_noninferior": float(metrics.loc["best_found", "mean_delta"]) >= 0.0,
        "t95_noninferior": float(
            metrics.loc["round_to_95_global_best", "mean_delta"]
        )
        <= 0.0,
        "worst_decile_noninferior": float(
            tail.loc[CANDIDATE, "auc_10th_percentile"]
        )
        >= float(tail.loc[BASELINE, "auc_10th_percentile"]),
        "cvar_noninferior": float(tail.loc[CANDIDATE, "cvar_worst_decile_auc"])
        >= float(tail.loc[BASELINE, "cvar_worst_decile_auc"]),
        "structured_diagnostic_positive": bool(lpo["positive"]),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "decision": "eligible_for_920_validation" if all(checks.values()) else "stop_before_920",
        "public_test_labels_used": False,
        "protected_920_or_930_episodes_opened": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--limit-episodes", type=int, default=None)
    parser.add_argument(
        "--lpo-only",
        action="store_true",
        help="Refresh the deterministic leave-pair-out diagnostic from saved pilot summaries.",
    )
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
    manifest = {
        "stage": PHASE,
        "train_only": True,
        "public_test_labels_used": False,
        "public_searchspace_labels_used": False,
        "retired_810_outcomes_used": False,
        "protected_920_or_930_episodes_opened": False,
        "input": {
            "path": str(TRAIN_FILE.relative_to(DATASET_ROOT)),
            "sha256": file_sha256(TRAIN_FILE),
        },
        "implementations": {
            "driver": {
                "path": str(Path(__file__).resolve().relative_to(DATASET_ROOT)),
                "sha256": file_sha256(Path(__file__).resolve()),
            },
            "pair_context_surrogate": {
                "path": str(
                    (EXTENSION_ROOT / "src/bgolearn_rxn/surrogates/pair_context_gp.py")
                    .resolve()
                    .relative_to(DATASET_ROOT)
                ),
                "sha256": file_sha256(
                    EXTENSION_ROOT / "src/bgolearn_rxn/surrogates/pair_context_gp.py"
                ),
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
        "protected_seed_starts": list(PROTECTED_SEED_STARTS),
        "target_transform": "standard",
        "acquisition": "posterior mean query 1, stable LogEI queries 2-5",
        "pair_parameterization": (
            "constituent main kernels used once + constituent product + exact-pair "
            "diagonal with zero-sum Helmert log residuals"
        ),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    if args.lpo_only:
        paired = pd.read_csv(args.output_dir / "paired_summary.csv")
        tails = pd.read_csv(args.output_dir / "auc_tail_summary.csv")
        metrics = pd.read_csv(args.output_dir / "per_episode.csv")
        lpo_detail, lpo_summary, lpo_diagnostic = leave_pair_out_diagnostic()
        gate = gate_decision(paired, tails, lpo_diagnostic)
        lpo_detail.to_csv(
            args.output_dir / "leave_pair_out_predictions.csv", index=False
        )
        lpo_summary.to_csv(args.output_dir / "leave_pair_out_summary.csv", index=False)
        pair_geometry_summary(metrics).to_csv(
            args.output_dir / "pair_geometry_summary.csv", index=False
        )
        (args.output_dir / "leave_pair_out_diagnostic.json").write_text(
            json.dumps(lpo_diagnostic, indent=2, sort_keys=True), encoding="utf-8"
        )
        (args.output_dir / "gate_decision.json").write_text(
            json.dumps(gate, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"gate_passed={str(gate['passed']).lower()}", flush=True)
        return
    jobs = [
        (variant, episode, str(args.output_dir))
        for episode in episodes
        for variant in VARIANTS
    ]
    metric_rows = []
    calibration_rows = []
    wall_started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(run_one, job): job for job in jobs}
        for future in as_completed(futures):
            result = future.result()
            metric_rows.append(result["metrics"])
            calibration_rows.extend(result["calibration"])
            row = result["metrics"]
            print(
                f"{row['variant']} {row['episode']}: AUC={row['AUC_best_so_far']:.3f}",
                flush=True,
            )
    wall_seconds = time.perf_counter() - wall_started
    metrics = pd.DataFrame(metric_rows).sort_values(["episode", "variant"])
    calibration = pd.DataFrame(calibration_rows).sort_values(
        ["episode", "variant", "step"]
    )
    expected = len(episodes) * len(VARIANTS)
    if len(metrics) != expected or not metrics.groupby("episode")["split_seed"].nunique().eq(1).all():
        raise RuntimeError("Paired result audit failed.")
    if args.limit_episodes is not None:
        metrics.to_csv(args.output_dir / "smoke_per_episode.csv", index=False)
        calibration.to_csv(args.output_dir / "smoke_calibration_predictions.csv", index=False)
        print(
            f"absolute_mean_AUC_best_so_far={metrics[metrics['variant'].eq(CANDIDATE)]['AUC_best_so_far'].mean():.8f}",
            flush=True,
        )
        return

    summary = absolute_summary(metrics)
    paired = paired_summary(metrics)
    tails = tail_summary(metrics)
    calibration_table = calibration_summary(calibration)
    geometry = pair_geometry_summary(metrics)
    lpo_detail, lpo_summary, lpo_diagnostic = leave_pair_out_diagnostic()
    gate = gate_decision(paired, tails, lpo_diagnostic)
    first20_ids = [f"random_{PHASE}_{index:02d}" for index in range(20)]
    first20 = metrics[metrics["episode"].isin(first20_ids)].copy()
    if len(first20) != 20 * len(VARIANTS):
        raise RuntimeError("Deterministic first-20 evidence is incomplete.")
    metrics.to_csv(args.output_dir / "per_episode.csv", index=False)
    calibration.to_csv(args.output_dir / "calibration_predictions.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    paired.to_csv(args.output_dir / "paired_summary.csv", index=False)
    tails.to_csv(args.output_dir / "auc_tail_summary.csv", index=False)
    calibration_table.to_csv(args.output_dir / "calibration_summary.csv", index=False)
    geometry.to_csv(args.output_dir / "pair_geometry_summary.csv", index=False)
    first20.to_csv(args.output_dir / "first20_per_episode.csv", index=False)
    paired_summary(first20).to_csv(
        args.output_dir / "first20_paired_summary.csv", index=False
    )
    lpo_detail.to_csv(args.output_dir / "leave_pair_out_predictions.csv", index=False)
    lpo_summary.to_csv(args.output_dir / "leave_pair_out_summary.csv", index=False)
    audit = {
        "passed": True,
        "n_episodes": len(episodes),
        "n_metric_rows": len(metrics),
        "n_calibration_predictions": len(calibration),
        "n_trajectories": len(list((args.output_dir / "trajectories").glob("*/*.json"))),
        "split_seed_min": int(metrics["split_seed"].min()),
        "split_seed_max": int(metrics["split_seed"].max()),
        "wall_seconds": wall_seconds,
        "public_test_labels_used": False,
        "public_searchspace_labels_used": False,
        "retired_810_outcomes_used": False,
        "protected_920_or_930_episodes_opened": False,
    }
    (args.output_dir / "leave_pair_out_diagnostic.json").write_text(
        json.dumps(lpo_diagnostic, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.output_dir / "gate_decision.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    candidate_auc = float(
        summary[
            summary["variant"].eq(CANDIDATE)
            & summary["metric"].eq("AUC_best_so_far")
        ]["mean"].iloc[0]
    )
    print(f"absolute_mean_AUC_best_so_far={candidate_auc:.8f}", flush=True)
    print(f"gate_passed={str(gate['passed']).lower()}", flush=True)


if __name__ == "__main__":
    main()
