#!/usr/bin/env python3
"""Leakage-safe 40-round Suzuki development landscapes.

The pseudo-oracles in this module are fitted or calibrated exclusively from
``suzuki_train.csv`` and the categorical identities in
``suzuki_test_features.csv``.  Organizer-side candidate yields are never read.
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
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge


EXPERIMENT_ROOT = Path(__file__).resolve().parent
CODE_ROOT = EXPERIMENT_ROOT.parent
DATASET_ROOT = CODE_ROOT.parents[1]
EXTENSION_ROOT = CODE_ROOT / "bgolearn_rxn_extension"
sys.path.insert(0, str(EXTENSION_ROOT / "src"))

from bgolearn_rxn.campaign import CampaignRunner  # noqa: E402
from bgolearn_rxn.config import load_run_config  # noqa: E402
from bgolearn_rxn.datasets import DatasetSpec, load_reaction_dataset  # noqa: E402
from bgolearn_rxn.metrics import compute_campaign_metrics  # noqa: E402
from bgolearn_rxn.oracle import OfflineTableOracle  # noqa: E402


TRAIN_FILE = DATASET_ROOT / "Suzuki" / "suzuki_train.csv"
CANDIDATE_FILE = DATASET_ROOT / "Suzuki" / "suzuki_test_features.csv"
OPTIONS_FILE = DATASET_ROOT / "Suzuki" / "options.json"
METADATA_FILE = DATASET_ROOT / "Suzuki" / "metadata.json"
CONFIG_FILE = EXTENSION_ROOT / "configs" / "suzuki_metric_frontier.yaml"
PROTECTED_FILES = (
    DATASET_ROOT / "Suzuki" / "suzuki_test.csv",
    DATASET_ROOT / "Suzuki" / "suzuki_searchspace.csv",
)
FEATURE_COLUMNS = ("Electrophile", "Nucleophile", "Ligand", "Base", "Solvent")
INTERACTION_PAIRS = (
    ("Electrophile", "Ligand"),
    ("Nucleophile", "Base"),
    ("Ligand", "Base"),
    ("Base", "Solvent"),
)
TEACHER_FAMILIES = ("additive", "interaction_low_rank", "tree", "rugged")
HELD_OUT_TEACHER_FAMILIES = ("sparse_pair", "local_neighborhood")
BUDGET = 40
VARIANTS = (
    "component_ard",
    "stage_split",
    "full_mean_jackknife",
    "prequential_ard_delete",
    "stage_split_late33",
    "early_delete_then_full",
    "query1_rank_consensus",
    "localized_jackknife",
    "surprise_gated_consensus",
    "credible_support_maximin",
)
MODEL_NAMES = {
    "component_ard": "component_ard_gp",
    "stage_split": "stage_split_delete_fold_ard_gp",
    "full_mean_jackknife": "full_mean_jackknife_variance_ard_gp",
    "prequential_ard_delete": (
        "component_ard_gp+delete_fold_component_ard_gp"
    ),
    "stage_split_late33": "stage_split_delete_fold_ard_gp",
    "early_delete_then_full": "early_delete_fold_then_full_ard_gp",
    "query1_rank_consensus": "query_one_rank_consensus_ard_gp",
    "localized_jackknife": "incumbent_localized_jackknife_variance_ard_gp",
    "surprise_gated_consensus": "surprise_gated_rank_consensus_ard_gp",
    "credible_support_maximin": "credible_support_maximin_ard_gp",
}
VARIANT_CONFIGS = {
    "component_ard": {
        "enabled_models": ("component_ard_gp",),
        "initial_weights": {"component_ard_gp": 1.0},
        "first_query": "component_ard_gp",
        "early_end": BUDGET,
        "final_start": BUDGET + 1,
        "final_beta": 0.15,
    },
    "stage_split": {
        "enabled_models": ("stage_split_delete_fold_ard_gp",),
        "initial_weights": {"stage_split_delete_fold_ard_gp": 1.0},
        "first_query": "stage_split_delete_fold_ard_gp",
        "early_end": BUDGET,
        "final_start": BUDGET + 1,
        "final_beta": 0.15,
    },
    "full_mean_jackknife": {
        "enabled_models": ("full_mean_jackknife_variance_ard_gp",),
        "initial_weights": {"full_mean_jackknife_variance_ard_gp": 1.0},
        "first_query": "posterior_mean",
        "early_end": BUDGET,
        "final_start": BUDGET + 1,
        "final_beta": 0.15,
    },
    "prequential_ard_delete": {
        "enabled_models": (
            "component_ard_gp",
            "delete_fold_component_ard_gp",
        ),
        "initial_weights": {
            "component_ard_gp": 0.5,
            "delete_fold_component_ard_gp": 0.5,
        },
        "first_query": "component_ard_gp",
        "early_end": BUDGET,
        "final_start": BUDGET + 1,
        "final_beta": 0.15,
    },
    "stage_split_late33": {
        "enabled_models": ("stage_split_delete_fold_ard_gp",),
        "initial_weights": {"stage_split_delete_fold_ard_gp": 1.0},
        "first_query": "stage_split_delete_fold_ard_gp",
        "early_end": 32,
        "final_start": 33,
        "final_beta": 0.15,
    },
    "early_delete_then_full": {
        "enabled_models": ("early_delete_fold_then_full_ard_gp",),
        "initial_weights": {"early_delete_fold_then_full_ard_gp": 1.0},
        "first_query": "early_delete_fold_then_full_ard_gp",
        "early_end": BUDGET,
        "final_start": BUDGET + 1,
        "final_beta": 0.15,
    },
    "query1_rank_consensus": {
        "enabled_models": ("query_one_rank_consensus_ard_gp",),
        "initial_weights": {"query_one_rank_consensus_ard_gp": 1.0},
        "first_query": "query_one_rank_consensus_ard_gp",
        "early_end": BUDGET,
        "final_start": BUDGET + 1,
        "final_beta": 0.15,
    },
    "localized_jackknife": {
        "enabled_models": ("incumbent_localized_jackknife_variance_ard_gp",),
        "initial_weights": {
            "incumbent_localized_jackknife_variance_ard_gp": 1.0
        },
        "first_query": "incumbent_localized_jackknife_variance_ard_gp",
        "early_end": BUDGET,
        "final_start": BUDGET + 1,
        "final_beta": 0.15,
    },
    "surprise_gated_consensus": {
        "enabled_models": ("surprise_gated_rank_consensus_ard_gp",),
        "initial_weights": {"surprise_gated_rank_consensus_ard_gp": 1.0},
        "first_query": "surprise_gated_rank_consensus_ard_gp",
        "early_end": BUDGET,
        "final_start": BUDGET + 1,
        "final_beta": 0.15,
    },
    "credible_support_maximin": {
        "enabled_models": ("credible_support_maximin_ard_gp",),
        "initial_weights": {"credible_support_maximin_ard_gp": 1.0},
        "first_query": "credible_support_maximin_ard_gp",
        "early_end": BUDGET,
        "final_start": BUDGET + 1,
        "final_beta": 0.15,
    },
}
SPLIT_SEED_STARTS = {
    "dev": 7_100_000,
    "test": 7_200_000,
    "phase8_dev": 8_100_000,
    "phase8_test": 8_200_000,
}
SPLIT_TEACHER_FAMILIES = {
    "dev": TEACHER_FAMILIES,
    "test": TEACHER_FAMILIES,
    "phase8_dev": TEACHER_FAMILIES,
    "phase8_test": HELD_OUT_TEACHER_FAMILIES,
}
LANDSCAPES_PER_SPLIT = 20
CORE_METRICS = (
    "initial_round_found_best",
    "best_found",
    "round_to_95_global_best",
    "AUC_best_so_far",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_sha256(value: object) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def landscape_specs(split: str, count: int = LANDSCAPES_PER_SPLIT) -> list[dict]:
    if split not in SPLIT_SEED_STARTS:
        raise ValueError(f"Unknown split: {split!r}")
    if count <= 0 or count > LANDSCAPES_PER_SPLIT:
        raise ValueError(f"count must be in [1, {LANDSCAPES_PER_SPLIT}]")
    start = SPLIT_SEED_STARTS[split]
    families = SPLIT_TEACHER_FAMILIES[split]
    return [
        {
            "landscape": (
                f"{split}_{families[index % len(families)]}_"
                f"{index // len(families):02d}"
            ),
            "split": split,
            "family": families[index % len(families)],
            "teacher_seed": start + index,
            "campaign_seed": start + 10_000 + index,
        }
        for index in range(count)
    ]


def _category_maps(all_rows: pd.DataFrame) -> dict[str, tuple[str, ...]]:
    return {
        column: tuple(sorted(all_rows[column].astype(str).unique()))
        for column in FEATURE_COLUMNS
    }


def _one_hot(rows: pd.DataFrame, levels: dict[str, tuple[str, ...]]) -> np.ndarray:
    blocks = []
    for column in FEATURE_COLUMNS:
        values = rows[column].astype(str).to_numpy()
        block = np.column_stack([values == level for level in levels[column]])
        blocks.append(block.astype(float))
    return np.hstack(blocks)


def _sample_weights(seed: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = np.exp(rng.normal(0.0, 0.22, size=n))
    return values / values.mean()


def _ridge_predictions(
    train_X: np.ndarray,
    candidate_X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(train_X, y, sample_weight=_sample_weights(seed, len(y)))
    return model.predict(train_X), model.predict(candidate_X)


def _low_rank_features(
    train: pd.DataFrame,
    candidates: pd.DataFrame,
    levels: dict[str, tuple[str, ...]],
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_blocks: dict[str, np.ndarray] = {}
    candidate_blocks: dict[str, np.ndarray] = {}
    for column in FEATURE_COLUMNS:
        embedding = rng.normal(0.0, 1.0, size=(len(levels[column]), 4))
        lookup = {value: embedding[index] for index, value in enumerate(levels[column])}
        train_blocks[column] = np.vstack([lookup[value] for value in train[column].astype(str)])
        candidate_blocks[column] = np.vstack(
            [lookup[value] for value in candidates[column].astype(str)]
        )
    train_features = [train_blocks[column] for column in FEATURE_COLUMNS]
    candidate_features = [candidate_blocks[column] for column in FEATURE_COLUMNS]
    for left, right in INTERACTION_PAIRS:
        train_features.append(train_blocks[left] * train_blocks[right])
        candidate_features.append(candidate_blocks[left] * candidate_blocks[right])
    return np.hstack(train_features), np.hstack(candidate_features)


def _rugged_field(
    train: pd.DataFrame,
    candidates: pd.DataFrame,
    levels: dict[str, tuple[str, ...]],
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_field = np.zeros(len(train), dtype=float)
    candidate_field = np.zeros(len(candidates), dtype=float)
    for column in FEATURE_COLUMNS:
        values = {level: value for level, value in zip(levels[column], rng.normal(size=len(levels[column])), strict=True)}
        train_field += 0.35 * train[column].astype(str).map(values).to_numpy(dtype=float)
        candidate_field += 0.35 * candidates[column].astype(str).map(values).to_numpy(dtype=float)
    for left, right in INTERACTION_PAIRS:
        keys = sorted(
            set(zip(train[left].astype(str), train[right].astype(str), strict=True))
            | set(zip(candidates[left].astype(str), candidates[right].astype(str), strict=True))
        )
        values = {key: value for key, value in zip(keys, rng.normal(size=len(keys)), strict=True)}
        train_field += np.asarray(
            [values[key] for key in zip(train[left].astype(str), train[right].astype(str), strict=True)]
        )
        candidate_field += np.asarray(
            [values[key] for key in zip(candidates[left].astype(str), candidates[right].astype(str), strict=True)]
        )
    return train_field, candidate_field


def _sparse_pair_features(
    train: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode a seeded sparse subset of exact category-pair interactions.

    Levels are learned only from pairs present in the labeled training rows.
    Candidate-only pairs consequently receive all-zero columns, making this a
    materially different inductive bias from the dense low-rank teacher.
    """

    rng = np.random.default_rng(seed)
    column_pairs = [
        (left, right)
        for index, left in enumerate(FEATURE_COLUMNS)
        for right in FEATURE_COLUMNS[index + 1 :]
    ]
    selected = rng.choice(len(column_pairs), size=6, replace=False)
    train_blocks = []
    candidate_blocks = []
    for pair_index in selected:
        left, right = column_pairs[int(pair_index)]
        train_keys = list(zip(train[left].astype(str), train[right].astype(str), strict=True))
        candidate_keys = list(
            zip(candidates[left].astype(str), candidates[right].astype(str), strict=True)
        )
        levels = tuple(sorted(set(train_keys)))
        train_blocks.append(
            np.column_stack(
                [np.asarray([key == level for key in train_keys]) for level in levels]
            )
        )
        candidate_blocks.append(
            np.column_stack(
                [
                    np.asarray([key == level for key in candidate_keys])
                    for level in levels
                ]
            )
        )
    return np.hstack(train_blocks).astype(float), np.hstack(candidate_blocks).astype(float)


def _local_neighborhood_predictions(
    train: pd.DataFrame,
    candidates: pd.DataFrame,
    y: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Categorical-kernel regression with seeded role-specific distances."""

    rng = np.random.default_rng(seed)
    role_weights = rng.uniform(0.7, 1.8, size=len(FEATURE_COLUMNS))
    bandwidth = float(rng.uniform(0.8, 1.6))
    train_values = train.loc[:, FEATURE_COLUMNS].astype(str).to_numpy()
    candidate_values = candidates.loc[:, FEATURE_COLUMNS].astype(str).to_numpy()

    def predict(query_values: np.ndarray, *, leave_self_out: bool) -> np.ndarray:
        output = np.empty(len(query_values), dtype=float)
        for start in range(0, len(query_values), 512):
            query = query_values[start : start + 512]
            distances = (
                (query[:, None, :] != train_values[None, :, :]) * role_weights
            ).sum(axis=2)
            weights = np.exp(-distances / bandwidth)
            if leave_self_out:
                indices = np.arange(start, start + len(query))
                weights[np.arange(len(query)), indices] = 0.0
            denominators = weights.sum(axis=1)
            output[start : start + len(query)] = np.divide(
                weights @ y,
                denominators,
                out=np.full(len(query), float(np.mean(y))),
                where=denominators > 1e-12,
            )
        return output

    return predict(train_values, leave_self_out=True), predict(
        candidate_values, leave_self_out=False
    )


def _calibrate(
    raw_train: np.ndarray, raw_candidates: np.ndarray, y: np.ndarray
) -> np.ndarray:
    center = float(np.mean(raw_train))
    scale = float(np.std(raw_train, ddof=1))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = max(float(np.std(raw_candidates, ddof=1)), 1.0)
    standardized = np.clip((raw_candidates - center) / scale, -2.4, 2.9)
    calibrated = float(np.mean(y)) + float(np.std(y, ddof=1)) * standardized
    return np.clip(calibrated, 0.0, 100.0)


def build_landscape(spec: dict) -> tuple[pd.DataFrame, dict]:
    # These are the only tabular inputs opened by the teacher builder.
    train = pd.read_csv(TRAIN_FILE)
    candidates = pd.read_csv(CANDIDATE_FILE)
    y = train["Yield"].to_numpy(dtype=float)
    all_rows = pd.concat(
        [train.loc[:, FEATURE_COLUMNS], candidates.loc[:, FEATURE_COLUMNS]],
        ignore_index=True,
    )
    levels = _category_maps(all_rows)
    main_train = _one_hot(train, levels)
    main_candidates = _one_hot(candidates, levels)
    family = str(spec["family"])
    seed = int(spec["teacher_seed"])

    if family == "additive":
        alpha = (0.75, 2.0, 5.0, 10.0, 18.0)[seed % 5]
        raw_train, raw_candidates = _ridge_predictions(
            main_train, main_candidates, y, seed=seed, alpha=alpha
        )
    elif family == "interaction_low_rank":
        train_X, candidate_X = _low_rank_features(
            train, candidates, levels, seed=seed
        )
        raw_train, raw_candidates = _ridge_predictions(
            train_X, candidate_X, y, seed=seed + 31, alpha=4.0
        )
    elif family == "tree":
        model = ExtraTreesRegressor(
            n_estimators=192,
            max_features=0.8,
            min_samples_leaf=2,
            bootstrap=True,
            max_samples=0.85,
            random_state=seed,
            n_jobs=1,
        )
        model.fit(main_train, y, sample_weight=_sample_weights(seed, len(y)))
        raw_train = model.predict(main_train)
        raw_candidates = model.predict(main_candidates)
    elif family == "rugged":
        base_train, base_candidates = _ridge_predictions(
            main_train, main_candidates, y, seed=seed + 53, alpha=3.0
        )
        field_train, field_candidates = _rugged_field(
            train, candidates, levels, seed=seed
        )
        field_scale = max(float(np.std(field_train, ddof=1)), 1e-6)
        raw_train = base_train + 0.75 * np.std(y, ddof=1) * field_train / field_scale
        raw_candidates = (
            base_candidates
            + 0.75 * np.std(y, ddof=1) * field_candidates / field_scale
        )
    elif family == "sparse_pair":
        train_X, candidate_X = _sparse_pair_features(
            train, candidates, seed=seed
        )
        raw_train, raw_candidates = _ridge_predictions(
            train_X, candidate_X, y, seed=seed + 71, alpha=2.5
        )
    elif family == "local_neighborhood":
        raw_train, raw_candidates = _local_neighborhood_predictions(
            train, candidates, y, seed=seed
        )
    else:
        raise ValueError(f"Unknown teacher family: {family!r}")

    yields = _calibrate(raw_train, raw_candidates, y)
    lookup = candidates.loc[:, FEATURE_COLUMNS].copy()
    lookup.insert(0, "query_index", np.arange(len(lookup), dtype=int))
    lookup["Yield"] = yields
    diagnostics = {
        **spec,
        "n_initial": len(train),
        "n_candidates": len(candidates),
        "candidate_mean": float(np.mean(yields)),
        "candidate_sample_sd": float(np.std(yields, ddof=1)),
        "candidate_min": float(np.min(yields)),
        "candidate_max": float(np.max(yields)),
        "candidate_yield_sha256": hashlib.sha256(yields.astype("<f8").tobytes()).hexdigest(),
    }
    return lookup, diagnostics


def environment_diagnostics(specs: list[dict]) -> dict:
    arrays = []
    rows = []
    for spec in specs:
        lookup, diagnostic = build_landscape(spec)
        arrays.append(lookup["Yield"].to_numpy(dtype=float))
        rows.append(diagnostic)
    matrix = np.vstack(arrays)
    family_means = np.vstack(
        [matrix[[row["family"] == family for row in rows]].mean(axis=0) for family in sorted({row["family"] for row in rows})]
    )
    if len(family_means) > 1:
        correlations = np.corrcoef(family_means)
        upper = correlations[np.triu_indices(len(correlations), 1)]
        rmses = []
        for left in range(len(family_means)):
            for right in range(left + 1, len(family_means)):
                rmses.append(float(np.sqrt(np.mean((family_means[left] - family_means[right]) ** 2))))
    else:
        upper = np.asarray([], dtype=float)
        rmses = []
    checks = {
        "all_landscapes_finite": bool(np.isfinite(matrix).all()),
        "all_yields_in_0_100": bool(np.all((matrix >= 0.0) & (matrix <= 100.0))),
        "meaningful_candidate_variation": bool(min(row["candidate_sample_sd"] for row in rows) >= 5.0),
        "nontrivial_cross_family_disagreement": bool(not rmses or min(rmses) >= 3.0),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Landscape validity contract failed: {checks}")
    return {
        "checks": checks,
        "landscapes": rows,
        "family_mean_pairwise_correlation_min": float(upper.min()) if len(upper) else None,
        "family_mean_pairwise_correlation_max": float(upper.max()) if len(upper) else None,
        "family_mean_pairwise_rmse_min": min(rmses) if rmses else None,
        "family_mean_pairwise_rmse_max": max(rmses) if rmses else None,
        "landscape_matrix_sha256": hashlib.sha256(matrix.astype("<f8").tobytes()).hexdigest(),
    }


def build_problem(variant: str):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant!r}")
    config = load_run_config(CONFIG_FILE)
    model_name = MODEL_NAMES[variant]
    variant_config = VARIANT_CONFIGS[variant]
    config = replace(
        config,
        dataset=replace(config.dataset, budget=BUDGET),
        ensemble=replace(
            config.ensemble,
            enabled_models=variant_config["enabled_models"],
            initial_weights=variant_config["initial_weights"],
        ),
        policy=replace(
            config.policy,
            first_query=variant_config["first_query"],
            early_end=variant_config["early_end"],
            final_start=variant_config["final_start"],
            challenge_every=0,
            final_beta=variant_config["final_beta"],
            novelty_disagreement_mode="off",
        ),
    )
    expected = {
        "enabled_models": tuple(variant_config["enabled_models"]),
        "initial_weights": dict(variant_config["initial_weights"]),
        "first_query": variant_config["first_query"],
        "early_end": variant_config["early_end"],
        "final_start": variant_config["final_start"],
        "challenge_every": 0,
        "final_beta": variant_config["final_beta"],
    }
    actual = {
        "enabled_models": config.ensemble.enabled_models,
        "initial_weights": config.ensemble.initial_weights,
        "first_query": config.policy.first_query,
        "early_end": config.policy.early_end,
        "final_start": config.policy.final_start,
        "challenge_every": config.policy.challenge_every,
        "final_beta": config.policy.final_beta,
    }
    if actual != expected:
        raise RuntimeError(
            f"Variant configuration audit failed for {variant!r}: "
            f"expected={expected}, actual={actual}"
        )
    spec = DatasetSpec.from_config(config.dataset)
    dataset = load_reaction_dataset(TRAIN_FILE, CANDIDATE_FILE, spec)
    return config, dataset, model_name


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
    variant, landscape_spec, output_root_string = job
    config, dataset, model_name = build_problem(variant)
    lookup, teacher = build_landscape(landscape_spec)
    oracle = OfflineTableOracle(dataset=dataset, lookup=lookup)
    runner = CampaignRunner(
        dataset, config, seed=int(landscape_spec["campaign_seed"]), backend="native"
    )
    started = time.perf_counter()
    result = runner.run(oracle, budget=BUDGET)
    elapsed = time.perf_counter() - started
    payload = result.to_payload()
    metrics = campaign_metrics(payload["trajectory"], oracle.global_best, BUDGET)
    first20 = campaign_metrics(payload["trajectory"][:20], oracle.global_best, 20)
    queried = [int(row["query_index"]) for row in payload["trajectory"]]
    schedule_audit = None
    if variant == "early_delete_then_full":
        schedule_model = runner.ensemble.models[model_name]
        schedule_audit = schedule_model.audit_state()
        expected_stages = (
            ["full_component_ard_query_1"]
            + ["delete_fold_component_ard_early"] * 9
            + ["full_component_ard_mature"] * 30
        )
        expected_counts = list(range(29, 69))
        if (
            schedule_audit["stage_history"] != expected_stages
            or schedule_audit["observed_count_history"] != expected_counts
            or schedule_audit["maturity_threshold_observed_count"] != 39
            or schedule_audit["mature_transition_observed_count"] != 39
        ):
            raise RuntimeError(
                f"Early-delete schedule audit failed: {schedule_audit}"
            )
    elif variant == "query1_rank_consensus":
        schedule_model = runner.ensemble.models[model_name]
        first_query_position = int(
            np.flatnonzero(dataset.candidate_ids == queried[0])[0]
        )
        schedule_model.record_query_one_selection(first_query_position)
        schedule_audit = schedule_model.audit_state()
        expected_stages = ["median_rank_consensus_query_1"] + [
            "component_ard_after_feedback"
        ] * 39
        if (
            schedule_audit["stage_history"] != expected_stages
            or schedule_audit["observed_count_history"] != list(range(29, 69))
            or schedule_audit["feedback_observed_count"] != 30
            or schedule_audit["selected_consensus_rank"] is None
        ):
            raise RuntimeError(
                f"Query-one rank-consensus audit failed: {schedule_audit}"
            )
    elif variant == "localized_jackknife":
        schedule_model = runner.ensemble.models[model_name]
        schedule_audit = dict(schedule_model.audit_state_)
        expected_stages = ["full_component_ard_query_1"] + [
            "full_mean_plus_delete_fold_variance"
        ] * 39
        if (
            schedule_audit["fit_calls"] != BUDGET
            or schedule_audit["stage_history"] != expected_stages
            or schedule_audit["initial_observed_count"] != 29
            or schedule_audit["transition_observed_count"] != 30
            or not schedule_audit["localization"].get("finite", False)
        ):
            raise RuntimeError(
                f"Localized-jackknife audit failed: {schedule_audit}"
            )
    elif variant == "surprise_gated_consensus":
        schedule_model = runner.ensemble.models[model_name]
        schedule_audit = schedule_model.audit_state()
        residuals = schedule_audit["residual_history"]
        expected_stages = [
            "surprise_rank_consensus" if active else "component_ard_nominal"
            for active in schedule_audit["activation_history"][1:]
        ]
        if (
            schedule_audit["fit_calls"] != BUDGET
            or schedule_audit["stage_history"][0] != "component_ard_query_1"
            or schedule_audit["stage_history"][1:] != expected_stages
            or schedule_audit["observed_count_history"] != list(range(29, 69))
            or len(residuals) != BUDGET - 1
            or schedule_audit["severe_negative_z"] != -1.0
            or schedule_audit["uncertainty_authority"] != "component_ard_gp"
            or not all(
                np.isfinite(float(record["standardized_residual"]))
                and bool(record["activated"])
                == (float(record["standardized_residual"]) <= -1.0)
                for record in residuals
            )
        ):
            raise RuntimeError(
                f"Surprise-gated consensus audit failed: {schedule_audit}"
            )
    elif variant == "credible_support_maximin":
        schedule_model = runner.ensemble.models[model_name]
        schedule_audit = schedule_model.audit_state()
        expected_stages = ["credible_support_maximin_query_1"] + [
            "component_ard_after_feedback"
        ] * 39
        if (
            schedule_audit["stage_history"] != expected_stages
            or schedule_audit["observed_count_history"] != list(range(29, 69))
            or schedule_audit["feedback_observed_count"] != 30
            or not 1 <= int(schedule_audit["support_size"]) <= 5731
            or schedule_audit["selected_candidate_position"] is None
            or schedule_audit["selected_maximin_rank"] is None
            or set(schedule_audit["selected_expert_ranks"])
            != {"component_ard_gp", "interaction", "trees"}
            or schedule_audit["uncertainty_authority"] != "component_ard_gp"
        ):
            raise RuntimeError(
                f"Credible-support maximin audit failed: {schedule_audit}"
            )
    contract = {
        "split": landscape_spec["split"],
        "landscape": landscape_spec["landscape"],
        "teacher_family": landscape_spec["family"],
        "variant": variant,
        "model": model_name,
        "n_initial": len(dataset.initial),
        "n_candidates": len(dataset.candidates),
        "budget": BUDGET,
        "all_queries_unique": len(set(queried)) == BUDGET,
        "all_queries_in_candidate_pool": set(queried).issubset(set(dataset.candidate_ids)),
        "public_test_labels_used": False,
        "public_searchspace_labels_used": False,
        "production_default_changed": False,
        "surrogate_schedule_audit": schedule_audit,
        "teacher": teacher,
    }
    payload["landscape_metrics"] = metrics
    payload["first20_metrics"] = first20
    payload["landscape_contract"] = contract
    path = Path(output_root_string) / "trajectories" / variant / f"{landscape_spec['landscape']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "dataset": "Suzuki",
        **landscape_spec,
        "variant": variant,
        "model": model_name,
        "n_initial": len(dataset.initial),
        "n_candidates": len(dataset.candidates),
        "budget": BUDGET,
        "global_best": float(oracle.global_best),
        **metrics,
        "elapsed_seconds": elapsed,
        "query_ids_sha256": payload_sha256(queried),
        "contract": contract,
        "first20": first20,
    }


def mean_ci(values: pd.Series) -> tuple[float, float, float, float]:
    array = values.to_numpy(dtype=float)
    mean = float(array.mean())
    if len(array) == 1:
        return mean, 0.0, mean, mean
    sd = float(array.std(ddof=1))
    half = float(t.ppf(0.975, len(array) - 1) * sd / math.sqrt(len(array)))
    return mean, sd, mean - half, mean + half


def absolute_summary(rows: pd.DataFrame, *, group_columns: tuple[str, ...] = ("variant",)) -> pd.DataFrame:
    records = []
    for keys, group in rows.groupby(list(group_columns), sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        prefix = dict(zip(group_columns, keys, strict=True))
        for metric in CORE_METRICS + ("elapsed_seconds",):
            mean, sd, low, high = mean_ci(group[metric])
            records.append({**prefix, "metric": metric, "n": len(group), "mean": mean, "sample_sd": sd, "ci95_low": low, "ci95_high": high})
    return pd.DataFrame(records)


def paired_summary(rows: pd.DataFrame, baseline: str) -> pd.DataFrame:
    baseline_rows = rows[rows["variant"].eq(baseline)].set_index("landscape")
    records = []
    for variant in sorted(set(rows["variant"]) - {baseline}):
        candidate = rows[rows["variant"].eq(variant)].set_index("landscape")
        common = baseline_rows.index.intersection(candidate.index)
        for metric in CORE_METRICS + ("elapsed_seconds",):
            delta = candidate.loc[common, metric] - baseline_rows.loc[common, metric]
            mean, sd, low, high = mean_ci(delta)
            lower_better = metric in {"round_to_95_global_best", "elapsed_seconds"}
            wins = int((delta < -1e-10).sum() if lower_better else (delta > 1e-10).sum())
            losses = int((delta > 1e-10).sum() if lower_better else (delta < -1e-10).sum())
            records.append({"candidate": variant, "baseline": baseline, "metric": metric, "n_pairs": len(common), "mean_delta": mean, "sample_sd_delta": sd, "ci95_low": low, "ci95_high": high, "wins": wins, "ties": len(common) - wins - losses, "losses": losses})
    return pd.DataFrame(records)


def tail_summary(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for variant, group in rows.groupby("variant", sort=True):
        values = np.sort(group["AUC_best_so_far"].to_numpy(dtype=float))
        count = max(1, int(math.ceil(0.10 * len(values))))
        records.append({"variant": variant, "n": len(values), "tail_count": count, "auc_cvar10": float(values[:count].mean())})
    return pd.DataFrame(records)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=tuple(SPLIT_SEED_STARTS), default="dev")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--episode-count", type=int, default=LANDSCAPES_PER_SPLIT)
    parser.add_argument("--smoke", action="store_true", help="Run two full 40-round landscapes.")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.jobs <= 0:
        raise ValueError("--jobs must be positive")
    count = 2 if args.smoke else args.episode_count
    specs = landscape_specs(args.split, count)
    output = args.output_dir or (
        DATASET_ROOT / "submission" / "results" / "experimental_investigations"
        / f"phase7-node1-landscape-zoo-{args.split}{'-smoke' if args.smoke else ''}"
    )
    output.mkdir(parents=True, exist_ok=True)
    environment = environment_diagnostics(specs)
    manifest = {
        "phase": (
            "phase8_40round_landscape_zoo"
            if args.split.startswith("phase8_")
            else "phase7_40round_landscape_zoo"
        ),
        "split": args.split,
        "smoke": bool(args.smoke),
        "variants": args.variants,
        "landscape_count": len(specs),
        "teacher_families": list(SPLIT_TEACHER_FAMILIES[args.split]),
        "balanced_full_panel": count == 20,
        "landscape_specs": specs,
        "landscape_specs_sha256": payload_sha256(specs),
        "official_geometry": {"n_initial": 29, "n_candidates": 5731, "budget": BUDGET},
        "allowed_inputs": {
            str(path.relative_to(DATASET_ROOT)): file_sha256(path)
            for path in (TRAIN_FILE, CANDIDATE_FILE, OPTIONS_FILE, METADATA_FILE, CONFIG_FILE)
        },
        "protected_inputs": {
            str(path.relative_to(DATASET_ROOT)): {"opened": False, "hashed": False}
            for path in PROTECTED_FILES
        },
        "public_test_labels_used": False,
        "public_searchspace_labels_used": False,
        "historical_public_oracle_outcomes_used": False,
        "production_default_changed": False,
        "environment_diagnostics": environment,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    jobs = [(variant, spec, str(output)) for spec in specs for variant in args.variants]
    results = []
    wall_started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=min(args.jobs, len(jobs))) as executor:
        futures = {executor.submit(run_one, job): job for job in jobs}
        for future in as_completed(futures):
            row = future.result()
            results.append(row)
            print(f"{row['variant']} {row['landscape']}: AUC={row['AUC_best_so_far']:.3f}", flush=True)
    wall_seconds = time.perf_counter() - wall_started
    contracts = [row.pop("contract") for row in results]
    first20 = [dict(row, **row.pop("first20")) for row in results]
    rows = pd.DataFrame(results).sort_values(["landscape", "variant"])
    first20_rows = pd.DataFrame(first20).sort_values(["landscape", "variant"])
    rows.to_csv(output / "per_landscape.csv", index=False)
    first20_rows.to_csv(output / "first20_per_landscape.csv", index=False)
    absolute_summary(rows).to_csv(output / "summary.csv", index=False)
    absolute_summary(rows, group_columns=("family", "variant")).to_csv(output / "per_teacher_summary.csv", index=False)
    absolute_summary(first20_rows).to_csv(output / "first20_summary.csv", index=False)
    baseline = "component_ard" if "component_ard" in args.variants else args.variants[0]
    paired = paired_summary(rows, baseline) if len(args.variants) > 1 else pd.DataFrame()
    paired.to_csv(output / "paired_vs_component_ard.csv", index=False)
    tail_summary(rows).to_csv(output / "tail_summary.csv", index=False)
    query_checks = [contract["all_queries_unique"] and contract["all_queries_in_candidate_pool"] for contract in contracts]
    audit = {
        "passed": bool(
            len(rows) == len(specs) * len(args.variants)
            and rows["n_initial"].eq(29).all()
            and rows["n_candidates"].eq(5731).all()
            and rows["budget"].eq(40).all()
            and all(query_checks)
            and all(environment["checks"].values())
        ),
        "result_rows": len(rows),
        "trajectory_files": len(list((output / "trajectories").glob("*/*.json"))),
        "unique_landscapes": int(rows["landscape"].nunique()),
        "query_contracts_passed": int(sum(query_checks)),
        "wall_seconds": wall_seconds,
        "protected_inputs_opened": False,
        "production_default_changed": False,
    }
    if not audit["passed"]:
        raise RuntimeError(f"Landscape campaign audit failed: {audit}")
    (output / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(output), "audit": audit, "paired": paired.to_dict(orient="records")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
