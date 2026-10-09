#!/usr/bin/env python3
"""Train and seal a pre-query Suzuki fixed-GP/component-ARD selector."""

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
from scipy.optimize import minimize
from scipy.stats import rankdata, t


EXPERIMENT_ROOT = Path(__file__).resolve().parent
CODE_ROOT = EXPERIMENT_ROOT.parent
DATASET_ROOT = CODE_ROOT.parents[1]
EXTENSION_ROOT = CODE_ROOT / "bgolearn_rxn_extension"
sys.path.insert(0, str(EXTENSION_ROOT / "src"))
sys.path.insert(0, str(EXPERIMENT_ROOT))

import run_suzuki_component_ard_replication as ard_replication  # noqa: E402
import run_suzuki_holdout_proxy as suzuki_proxy  # noqa: E402
from bgolearn_rxn.factory import prepare_problem  # noqa: E402
from bgolearn_rxn.surrogates.anova_gp import StructuredANOVAGP  # noqa: E402
from bgolearn_rxn.surrogates.component_ard_gp import (  # noqa: E402
    ComponentARDStructuredGP,
)


FIXED = "fixed"
PURE_ARD = "component_ard"
SELECTOR = "episodic_selector"
VARIANTS = (FIXED, PURE_ARD)
CORE_METRICS = (
    "initial_round_found_best",
    "best_found",
    "round_to_95_global_best",
    "AUC_best_so_far",
)
SUMMARY_METRICS = CORE_METRICS + ("elapsed_seconds",)
FEATURE_NAMES = (
    "posterior_rank_correlation",
    "top1_agreement",
    "top5_jaccard",
    "mean_abs_gap_scaled",
    "log_mean_uncertainty_ratio",
    "ard_pair_component_share",
)
TRAIN_FILE = DATASET_ROOT / "Suzuki" / "suzuki_train.csv"
CONFIG_FILE = EXTENSION_ROOT / "configs" / "suzuki.yaml"
SEED_FAMILY_START = 710_000
PROTECTED_PHASE3_START = 610_000
PROTECTED_CONFIRMATION_START = 810_000
TOTAL_EPISODES = 60
DEVELOPMENT_COUNT = 30
VALIDATION_COUNT = 30
L2_PENALTY = 1.0
ARD_CONFIDENCE_THRESHOLD = 0.65


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_episode_partitions() -> tuple[list[dict], list[dict]]:
    frame = pd.read_csv(TRAIN_FILE)
    episodes = suzuki_proxy._phase_random_episodes(
        frame,
        phase="episodic_selector",
        seed_start=SEED_FAMILY_START,
        count=TOTAL_EPISODES,
    )
    development = episodes[:DEVELOPMENT_COUNT]
    validation = episodes[DEVELOPMENT_COUNT:]
    if len(development) != DEVELOPMENT_COUNT or len(validation) != VALIDATION_COUNT:
        raise RuntimeError("Selector partition sizes changed.")
    development_ids = {item["episode"] for item in development}
    validation_ids = {item["episode"] for item in validation}
    if development_ids.intersection(validation_ids):
        raise RuntimeError("Development and validation episode IDs overlap.")
    seeds = [int(item["split_seed"]) for item in episodes]
    if len(set(seeds)) != TOTAL_EPISODES:
        raise RuntimeError("Selector split seeds are not unique.")
    if min(seeds) < SEED_FAMILY_START or max(seeds) >= PROTECTED_CONFIRMATION_START:
        raise RuntimeError("Selector escaped the 710k development family.")
    if any(PROTECTED_PHASE3_START <= seed < SEED_FAMILY_START for seed in seeds):
        raise RuntimeError("Selector entered the protected 610k family.")
    return development, validation


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = rankdata(left, method="average")
    right_rank = rankdata(right, method="average")
    if np.std(left_rank) <= 1e-12 or np.std(right_rank) <= 1e-12:
        return 1.0 if np.allclose(left_rank, right_rank) else 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _top_indices(values: np.ndarray, query_ids: np.ndarray, count: int) -> np.ndarray:
    order = np.lexsort((query_ids, -np.asarray(values, dtype=float)))
    return order[: min(count, len(order))]


def prequery_features(episode: dict) -> dict:
    config, dataset, _, _ = ard_replication.build_problem(episode, FIXED)
    prepared = prepare_problem(dataset, config)
    schema = prepared.encoder.schema
    if schema is None:
        raise RuntimeError("Feature schema is unavailable.")
    seed = int(episode["campaign_seed"])
    fixed = StructuredANOVAGP(schema=schema, config=config.gp, seed=seed + 11)
    ard = ComponentARDStructuredGP(schema=schema, config=config.gp, seed=seed + 17)
    y_initial = dataset.initial[dataset.spec.target_column].to_numpy(dtype=float)
    fixed.fit(prepared.X_initial, y_initial)
    ard.fit(prepared.X_initial, y_initial)
    fixed_posterior = fixed.predict(prepared.X_candidates)
    ard_posterior = ard.predict(prepared.X_candidates)
    query_ids = dataset.candidates["query_index"].to_numpy(dtype=int)
    fixed_top = _top_indices(fixed_posterior.mean, query_ids, 5)
    ard_top = _top_indices(ard_posterior.mean, query_ids, 5)
    fixed_set = set(fixed_top.tolist())
    ard_set = set(ard_top.tolist())
    target_scale = max(float(np.std(y_initial, ddof=1)), 1e-8)
    pair_share = float(
        sum(
            value
            for name, value in ard.component_shares_.items()
            if name.startswith("pair:")
        )
    )
    values = {
        "posterior_rank_correlation": _safe_correlation(
            fixed_posterior.mean, ard_posterior.mean
        ),
        "top1_agreement": float(fixed_top[0] == ard_top[0]),
        "top5_jaccard": float(len(fixed_set & ard_set) / len(fixed_set | ard_set)),
        "mean_abs_gap_scaled": float(
            np.mean(np.abs(fixed_posterior.mean - ard_posterior.mean)) / target_scale
        ),
        "log_mean_uncertainty_ratio": float(
            np.log(
                max(float(np.mean(ard_posterior.std)), 1e-12)
                / max(float(np.mean(fixed_posterior.std)), 1e-12)
            )
        ),
        "ard_pair_component_share": pair_share,
    }
    if set(values) != set(FEATURE_NAMES) or not np.isfinite(list(values.values())).all():
        raise RuntimeError("Pre-query selector features are incomplete or non-finite.")
    return {
        "episode": episode["episode"],
        "split_seed": int(episode["split_seed"]),
        "campaign_seed": int(episode["campaign_seed"]),
        **values,
        "feature_observation_cutoff": "before_query_1",
        "candidate_labels_accessed": False,
        "public_test_labels_used": False,
    }


def _fit_logistic(X: np.ndarray, y: np.ndarray) -> dict:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    mean = X.mean(axis=0)
    scale = X.std(axis=0, ddof=0)
    scale[scale < 1e-10] = 1.0
    normalized = (X - mean) / scale
    prevalence = float(np.clip(y.mean(), 1e-4, 1.0 - 1e-4))
    initial = np.zeros(X.shape[1] + 1, dtype=float)
    initial[0] = float(np.log(prevalence / (1.0 - prevalence)))

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        logits = parameters[0] + normalized @ parameters[1:]
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
        eps = 1e-12
        loss = -float(
            np.mean(y * np.log(probabilities + eps) + (1.0 - y) * np.log(1.0 - probabilities + eps))
        )
        loss += 0.5 * L2_PENALTY * float(np.sum(parameters[1:] ** 2))
        residual = probabilities - y
        gradient = np.concatenate(
            ([float(np.mean(residual))], normalized.T @ residual / len(y))
        )
        gradient[1:] += L2_PENALTY * parameters[1:]
        return loss, gradient

    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"Frozen selector fit failed: {result.message}")
    return {
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "intercept": float(result.x[0]),
        "coefficients": result.x[1:].tolist(),
        "optimizer_success": True,
        "optimizer_message": str(result.message),
    }


def selector_probability(model: dict, feature_row: dict) -> float:
    values = np.asarray([feature_row[name] for name in FEATURE_NAMES], dtype=float)
    mean = np.asarray(model["feature_mean"], dtype=float)
    scale = np.asarray(model["feature_scale"], dtype=float)
    coefficients = np.asarray(model["coefficients"], dtype=float)
    logit = float(model["intercept"] + ((values - mean) / scale) @ coefficients)
    return float(1.0 / (1.0 + np.exp(-np.clip(logit, -40.0, 40.0))))


def route_from_probability(probability: float) -> tuple[str, str]:
    if probability >= ARD_CONFIDENCE_THRESHOLD:
        return PURE_ARD, "ard_probability_at_or_above_frozen_threshold"
    return FIXED, "low_confidence_fixed_fallback"


def leave_one_out_diagnostics(features: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    rows: list[dict] = []
    X = features.loc[:, FEATURE_NAMES].to_numpy(dtype=float)
    y = labels.to_numpy(dtype=int)
    for held_index in range(len(features)):
        train_mask = np.arange(len(features)) != held_index
        model = _fit_logistic(X[train_mask], y[train_mask])
        probability = selector_probability(model, features.iloc[held_index].to_dict())
        route, reason = route_from_probability(probability)
        rows.append(
            {
                "episode": features.iloc[held_index]["episode"],
                "observed_ard_win": int(y[held_index]),
                "cross_fitted_ard_probability": probability,
                "cross_fitted_route": route,
                "route_reason": reason,
                "correct": int((route == PURE_ARD) == bool(y[held_index])),
            }
        )
    return pd.DataFrame(rows)


def _pair_worker(job: tuple[dict, str, bool]) -> tuple[list[dict], dict | None]:
    episode, output_dir, include_features = job
    feature_row = prequery_features(episode) if include_features else None
    rows = [
        ard_replication.run_one((variant, episode, output_dir)) for variant in VARIANTS
    ]
    return rows, feature_row


def run_pairs(
    episodes: list[dict], output_dir: Path, jobs_count: int, *, include_features: bool
) -> tuple[pd.DataFrame, pd.DataFrame]:
    jobs = [(episode, str(output_dir), include_features) for episode in episodes]
    metric_rows: list[dict] = []
    feature_rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=jobs_count) as executor:
        futures = {executor.submit(_pair_worker, job): job for job in jobs}
        for future in as_completed(futures):
            rows, feature_row = future.result()
            metric_rows.extend(rows)
            if feature_row is not None:
                feature_rows.append(feature_row)
            print(
                f"{rows[0]['episode']}: fixed AUC={rows[0]['AUC_best_so_far']:.3f}, "
                f"ARD AUC={rows[1]['AUC_best_so_far']:.3f}",
                flush=True,
            )
    metrics = pd.DataFrame(metric_rows).sort_values(["episode", "variant"])
    features = pd.DataFrame(feature_rows).sort_values("episode") if feature_rows else pd.DataFrame()
    return metrics, features


def run_feature_jobs(episodes: list[dict], jobs_count: int) -> pd.DataFrame:
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=jobs_count) as executor:
        futures = {executor.submit(prequery_features, episode): episode for episode in episodes}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(f"routed features: {row['episode']}", flush=True)
    return pd.DataFrame(rows).sort_values("episode")


def mean_ci(values: pd.Series) -> tuple[float, float, float, float]:
    array = values.to_numpy(dtype=float)
    mean = float(array.mean())
    if len(array) == 1:
        return mean, 0.0, mean, mean
    sd = float(array.std(ddof=1))
    half = float(t.ppf(0.975, len(array) - 1) * sd / np.sqrt(len(array)))
    return mean, sd, mean - half, mean + half


def summarize(rows: pd.DataFrame, *, phase: str = "episodic_selector_validation") -> pd.DataFrame:
    records: list[dict] = []
    for variant, group in rows.groupby("variant", sort=True):
        for metric in SUMMARY_METRICS:
            mean, sd, low, high = mean_ci(group[metric])
            records.append(
                {
                    "dataset": "Suzuki",
                    "phase": phase,
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


def paired_summary(rows: pd.DataFrame, candidate: str, baseline: str) -> pd.DataFrame:
    candidate_rows = rows[rows["variant"].eq(candidate)].set_index("episode")
    baseline_rows = rows[rows["variant"].eq(baseline)].set_index("episode")
    common = candidate_rows.index.intersection(baseline_rows.index)
    records: list[dict] = []
    for metric in SUMMARY_METRICS:
        delta = candidate_rows.loc[common, metric] - baseline_rows.loc[common, metric]
        mean, sd, low, high = mean_ci(delta)
        higher_is_better = metric not in {"round_to_95_global_best", "elapsed_seconds"}
        wins = int((delta > 1e-10).sum()) if higher_is_better else int((delta < -1e-10).sum())
        losses = int((delta < -1e-10).sum()) if higher_is_better else int((delta > 1e-10).sum())
        records.append(
            {
                "dataset": "Suzuki",
                "phase": "episodic_selector_validation",
                "candidate": candidate,
                "baseline": baseline,
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


def fit_frozen_selector(features: pd.DataFrame, metrics: pd.DataFrame, development: list[dict]) -> tuple[dict, pd.DataFrame]:
    expected_ids = [item["episode"] for item in development]
    if sorted(features["episode"].tolist()) != sorted(expected_ids):
        raise RuntimeError("Development features do not match the frozen partition.")
    pivot = metrics.pivot(index="episode", columns="variant", values="AUC_best_so_far")
    labels_by_id = (pivot[PURE_ARD] > pivot[FIXED] + 1e-10).astype(int)
    ordered = features.set_index("episode").loc[expected_ids].reset_index()
    labels = ordered["episode"].map(labels_by_id).astype(int)
    cross_fit = leave_one_out_diagnostics(ordered, labels)
    model = _fit_logistic(ordered.loc[:, FEATURE_NAMES].to_numpy(dtype=float), labels.to_numpy(dtype=int))
    artifact = {
        "selector_type": "l2_regularized_logistic_regression",
        "feature_names": list(FEATURE_NAMES),
        "model": model,
        "l2_penalty": L2_PENALTY,
        "ard_confidence_threshold": ARD_CONFIDENCE_THRESHOLD,
        "fallback_policy": FIXED,
        "training_target": "component_ard AUC_best_so_far strictly greater than fixed AUC",
        "training_episode_ids": expected_ids,
        "training_split_seeds": [int(item["split_seed"]) for item in development],
        "validation_episode_ids_used_for_training": [],
        "n_training_episodes": len(expected_ids),
        "n_ard_wins": int(labels.sum()),
        "n_fixed_or_tie": int(len(labels) - labels.sum()),
        "cross_fit_accuracy": float(cross_fit["correct"].mean()),
        "cross_fit_ard_route_count": int(cross_fit["cross_fitted_route"].eq(PURE_ARD).sum()),
        "classifier_sweep": False,
        "threshold_sweep": False,
        "feature_observation_cutoff": "before_query_1",
        "input": {"path": str(TRAIN_FILE.relative_to(DATASET_ROOT)), "sha256": file_sha256(TRAIN_FILE)},
        "config": {"path": str(CONFIG_FILE.relative_to(DATASET_ROOT)), "sha256": file_sha256(CONFIG_FILE)},
        "protected_seed_families_not_used": [PROTECTED_PHASE3_START, PROTECTED_CONFIRMATION_START],
        "train_only": True,
        "public_test_labels_used": False,
        "production_default_changed": False,
    }
    return artifact, cross_fit


def validate_selector_artifact(artifact: dict, development: list[dict]) -> None:
    if artifact["feature_names"] != list(FEATURE_NAMES):
        raise ValueError("Frozen selector feature schema changed.")
    if artifact["training_episode_ids"] != [item["episode"] for item in development]:
        raise ValueError("Frozen selector training episodes changed.")
    if artifact["training_split_seeds"] != [int(item["split_seed"]) for item in development]:
        raise ValueError("Frozen selector training seeds changed.")
    if artifact["validation_episode_ids_used_for_training"]:
        raise ValueError("Validation episodes were used to train the selector.")
    if artifact["ard_confidence_threshold"] != ARD_CONFIDENCE_THRESHOLD:
        raise ValueError("Frozen selector confidence threshold changed.")
    if artifact["input"]["sha256"] != file_sha256(TRAIN_FILE):
        raise ValueError("Frozen selector train-table hash changed.")
    if artifact["config"]["sha256"] != file_sha256(CONFIG_FILE):
        raise ValueError("Frozen selector config hash changed.")
    if artifact["public_test_labels_used"] is not False:
        raise ValueError("Frozen selector provenance is not train-only.")


def create_routes(features: pd.DataFrame, artifact: dict, validation: list[dict]) -> pd.DataFrame:
    expected = [item["episode"] for item in validation]
    ordered = features.set_index("episode").loc[expected].reset_index()
    rows: list[dict] = []
    for _, feature_row in ordered.iterrows():
        feature = feature_row.to_dict()
        probability = selector_probability(artifact["model"], feature)
        route, reason = route_from_probability(probability)
        rows.append(
            {
                **feature,
                "ard_probability": probability,
                "selected_policy": route,
                "route_reason": reason,
                "route_committed_before_query_1": True,
                "validation_outcomes_accessed_for_route": False,
                "selector_threshold": ARD_CONFIDENCE_THRESHOLD,
            }
        )
    return pd.DataFrame(rows)


def validate_routes(routes: pd.DataFrame, validation: list[dict]) -> None:
    expected_ids = [item["episode"] for item in validation]
    if routes["episode"].tolist() != expected_ids:
        raise ValueError("Route file does not exactly match validation episodes.")
    if not routes["route_committed_before_query_1"].astype(bool).all():
        raise ValueError("A route was not committed before query 1.")
    if routes["validation_outcomes_accessed_for_route"].astype(bool).any():
        raise ValueError("A validation route accessed an outcome.")
    if not set(routes["selected_policy"]).issubset(set(VARIANTS)):
        raise ValueError("Route file contains an unknown constituent policy.")


def compose_selector_rows(metrics: pd.DataFrame, routes: pd.DataFrame) -> pd.DataFrame:
    by_episode = routes.set_index("episode")
    selected: list[dict] = []
    for episode, route in by_episode.iterrows():
        source = metrics[
            metrics["episode"].eq(episode) & metrics["variant"].eq(route["selected_policy"])
        ]
        if len(source) != 1:
            raise RuntimeError(f"Missing selected constituent row for {episode}.")
        row = source.iloc[0].to_dict()
        row["selected_from_variant"] = row["variant"]
        row["variant"] = SELECTOR
        row["ard_probability"] = float(route["ard_probability"])
        row["route_reason"] = route["route_reason"]
        selected.append(row)
    return pd.DataFrame(selected)


def validation_gate(rows: pd.DataFrame) -> dict:
    summary = summarize(rows).set_index(["variant", "metric"])
    selector_fixed = paired_summary(rows, SELECTOR, FIXED).set_index("metric")
    ard_fixed = paired_summary(rows, PURE_ARD, FIXED).set_index("metric")
    selector_auc = float(summary.loc[(SELECTOR, "AUC_best_so_far"), "mean"])
    fixed_auc = float(summary.loc[(FIXED, "AUC_best_so_far"), "mean"])
    ard_auc = float(summary.loc[(PURE_ARD, "AUC_best_so_far"), "mean"])
    selector_initial = float(summary.loc[(SELECTOR, "initial_round_found_best"), "mean"])
    fixed_initial = float(summary.loc[(FIXED, "initial_round_found_best"), "mean"])
    selector_best = float(summary.loc[(SELECTOR, "best_found"), "mean"])
    fixed_best = float(summary.loc[(FIXED, "best_found"), "mean"])
    selector_t95 = float(summary.loc[(SELECTOR, "round_to_95_global_best"), "mean"])
    better_t95 = min(
        float(summary.loc[(FIXED, "round_to_95_global_best"), "mean"]),
        float(summary.loc[(PURE_ARD, "round_to_95_global_best"), "mean"]),
    )
    selector_losses = int(selector_fixed.loc["AUC_best_so_far", "losses"])
    ard_losses = int(ard_fixed.loc["AUC_best_so_far", "losses"])
    checks = {
        "selector_auc_greater_than_fixed": selector_auc > fixed_auc,
        "selector_auc_greater_than_pure_ard": selector_auc > ard_auc,
        "initial_no_worse_than_fixed": selector_initial >= fixed_initial,
        "best_no_worse_than_fixed": selector_best >= fixed_best,
        "t95_no_worse_than_better_constituent": selector_t95 <= better_t95,
        "auc_loss_frequency_lower_than_pure_ard": selector_losses < ard_losses,
    }
    return {
        "checks": checks,
        "passed": bool(all(checks.values())),
        "decision": "qualify_selector" if all(checks.values()) else "reject_selector",
        "selector_auc": selector_auc,
        "fixed_auc": fixed_auc,
        "pure_ard_auc": ard_auc,
        "selector_auc_losses_vs_fixed": selector_losses,
        "pure_ard_auc_losses_vs_fixed": ard_losses,
        "production_default_changed": False,
        "confirmation_executed": False,
        "public_test_labels_used": False,
    }


def _common_manifest(stage: str, development: list[dict], validation: list[dict]) -> dict:
    return {
        "stage": stage,
        "train_only": True,
        "public_test_labels_used": False,
        "input": {"path": str(TRAIN_FILE.relative_to(DATASET_ROOT)), "sha256": file_sha256(TRAIN_FILE)},
        "config": {"path": str(CONFIG_FILE.relative_to(DATASET_ROOT)), "sha256": file_sha256(CONFIG_FILE)},
        "development_episode_ids": [item["episode"] for item in development],
        "development_split_seeds": [int(item["split_seed"]) for item in development],
        "validation_episode_ids": [item["episode"] for item in validation],
        "validation_split_seeds": [int(item["split_seed"]) for item in validation],
        "protected_seed_families_not_used": [PROTECTED_PHASE3_START, PROTECTED_CONFIRMATION_START],
        "feature_observation_cutoff": "before_query_1",
        "classifier_sweep": False,
        "threshold_sweep": False,
        "production_default_changed": False,
        "confirmation_executed": False,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("development", "route-validation", "validation"), required=True)
    parser.add_argument("--jobs", type=int, default=min(6, os.cpu_count() or 1))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selector-file", type=Path)
    parser.add_argument("--routes-file", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.jobs <= 0:
        raise ValueError("--jobs must be positive.")
    development, validation = make_episode_partitions()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    common = _common_manifest(args.stage, development, validation)

    if args.stage == "development":
        if args.selector_file is not None or args.routes_file is not None:
            raise ValueError("Development accepts no frozen selector or routes.")
        metrics, features = run_pairs(
            development, args.output_dir, args.jobs, include_features=True
        )
        if len(metrics) != DEVELOPMENT_COUNT * len(VARIANTS) or len(features) != DEVELOPMENT_COUNT:
            raise RuntimeError("Development outputs are incomplete.")
        artifact, cross_fit = fit_frozen_selector(features, metrics, development)
        metrics.to_csv(args.output_dir / "per_episode.csv", index=False)
        features.to_csv(args.output_dir / "prequery_features.csv", index=False)
        summarize(metrics, phase="episodic_selector_development").to_csv(
            args.output_dir / "summary.csv", index=False
        )
        paired_summary(metrics, PURE_ARD, FIXED).to_csv(
            args.output_dir / "paired_ard_vs_fixed.csv", index=False
        )
        cross_fit.to_csv(args.output_dir / "leave_one_out_diagnostics.csv", index=False)
        selector_path = args.output_dir / "frozen_selector.json"
        selector_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
        manifest = {
            **common,
            "variants": list(VARIANTS),
            "episode_count": DEVELOPMENT_COUNT,
            "frozen_selector": str(selector_path),
            "frozen_selector_sha256": file_sha256(selector_path),
            "validation_outcomes_accessed": False,
        }
        (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        return

    if args.selector_file is None:
        raise ValueError(f"{args.stage} requires --selector-file.")
    artifact = json.loads(args.selector_file.read_text(encoding="utf-8"))
    validate_selector_artifact(artifact, development)

    if args.stage == "route-validation":
        if args.routes_file is not None:
            raise ValueError("Route stage creates, rather than accepts, routes.")
        features = run_feature_jobs(validation, args.jobs)
        routes = create_routes(features, artifact, validation)
        routes_path = args.output_dir / "sealed_routes.csv"
        routes.to_csv(routes_path, index=False)
        manifest = {
            **common,
            "episode_count": VALIDATION_COUNT,
            "selector_file": str(args.selector_file),
            "selector_sha256": file_sha256(args.selector_file),
            "routes_file": str(routes_path),
            "routes_sha256": file_sha256(routes_path),
            "ard_routes": int(routes["selected_policy"].eq(PURE_ARD).sum()),
            "fixed_routes": int(routes["selected_policy"].eq(FIXED).sum()),
            "validation_outcomes_accessed": False,
        }
        (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        return

    if args.routes_file is None:
        raise ValueError("Validation requires --routes-file.")
    routes = pd.read_csv(args.routes_file)
    validate_routes(routes, validation)
    metrics, unused_features = run_pairs(
        validation, args.output_dir, args.jobs, include_features=False
    )
    if len(metrics) != VALIDATION_COUNT * len(VARIANTS) or not unused_features.empty:
        raise RuntimeError("Validation constituent outputs are incomplete.")
    selected = compose_selector_rows(metrics, routes)
    all_rows = pd.concat([metrics, selected], ignore_index=True).sort_values(["episode", "variant"])
    all_rows.to_csv(args.output_dir / "per_episode.csv", index=False)
    summary = summarize(all_rows)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    for candidate, baseline, filename in (
        (PURE_ARD, FIXED, "paired_ard_vs_fixed.csv"),
        (SELECTOR, FIXED, "paired_selector_vs_fixed.csv"),
        (SELECTOR, PURE_ARD, "paired_selector_vs_ard.csv"),
    ):
        paired_summary(all_rows, candidate, baseline).to_csv(args.output_dir / filename, index=False)
    decision = validation_gate(all_rows)
    (args.output_dir / "validation_decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")
    first_twenty_ids = [item["episode"] for item in validation[:20]]
    twenty = all_rows[all_rows["episode"].isin(first_twenty_ids)].copy()
    twenty.to_csv(args.output_dir / "twenty_run_per_episode.csv", index=False)
    summarize(twenty).to_csv(args.output_dir / "twenty_run_summary.csv", index=False)
    for candidate, baseline, filename in (
        (PURE_ARD, FIXED, "twenty_run_paired_ard_vs_fixed.csv"),
        (SELECTOR, FIXED, "twenty_run_paired_selector_vs_fixed.csv"),
        (SELECTOR, PURE_ARD, "twenty_run_paired_selector_vs_ard.csv"),
    ):
        paired_summary(twenty, candidate, baseline).to_csv(args.output_dir / filename, index=False)
    manifest = {
        **common,
        "variants": [FIXED, PURE_ARD, SELECTOR],
        "episode_count": VALIDATION_COUNT,
        "selector_file": str(args.selector_file),
        "selector_sha256": file_sha256(args.selector_file),
        "routes_file": str(args.routes_file),
        "routes_sha256": file_sha256(args.routes_file),
        "routes_validated_before_campaigns": True,
        "validation_outcomes_accessed_after_route_commit": True,
        "ard_routes": int(routes["selected_policy"].eq(PURE_ARD).sum()),
        "fixed_routes": int(routes["selected_policy"].eq(FIXED).sum()),
        "validation_gate": decision,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
