#!/usr/bin/env python3
"""Source-only pairwise ranking prior for Buchwald query-one warm starts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.linear_model import LogisticRegression


EXPERIMENT_ROOT = Path(__file__).resolve().parent
CODE_ROOT = EXPERIMENT_ROOT.parent
DATASET_ROOT = CODE_ROOT.parents[1]
EXTENSION_ROOT = CODE_ROOT / "bgolearn_rxn_extension"
SESSION_ROOT = DATASET_ROOT / ".arbor" / "sessions" / "bgolearn_frontier_20260728"
sys.path.insert(0, str(EXTENSION_ROOT / "src"))
sys.path.insert(0, str(EXPERIMENT_ROOT))

import run_buchwald_source_meta_selection as meta  # noqa: E402
from bgolearn_rxn.acquisition import zscore  # noqa: E402
from bgolearn_rxn.campaign import CampaignRunner  # noqa: E402
from bgolearn_rxn.config import load_run_config  # noqa: E402
from bgolearn_rxn.datasets import DatasetSpec  # noqa: E402
from bgolearn_rxn.features import FrozenReactionEncoder  # noqa: E402
from bgolearn_rxn.oracle import OfflineTableOracle  # noqa: E402
from bgolearn_rxn.schemas import Posterior, Recommendation  # noqa: E402
from bgolearn_rxn.utils import deterministic_argmax  # noqa: E402


TRAIN_FILE = meta.TRAIN_FILE
CONFIG_FILE = meta.CONFIG_FILE
MODEL_FILE = meta.MODEL_FILE
FACTORY_FILE = meta.FACTORY_FILE
OFFICIAL_TARGET = meta.OFFICIAL_TARGET
MODEL_NAME = meta.MODEL_NAME
CORE_METRICS = meta.CORE_METRICS


@dataclass(frozen=True)
class RankSetting:
    name: str
    roles: tuple[str, ...]
    block: str
    l2_c: float


# Ordered for deterministic tie breaking. All choices are intentionally strongly
# regularized because each held-product fit has only 63 unique directed pairs.
RANK_GRID = (
    RankSetting("shared_identity_c001", ("Ligand", "Base"), "identity", 0.001),
    RankSetting("shared_identity_c0001", ("Ligand", "Base"), "identity", 0.0001),
    RankSetting("shared_identity_c01", ("Ligand", "Base"), "identity", 0.01),
    RankSetting(
        "all_condition_c001",
        ("Reactant2", "Ligand", "Additive", "Base"),
        "full",
        0.001,
    ),
    RankSetting(
        "all_condition_c0001",
        ("Reactant2", "Ligand", "Additive", "Base"),
        "full",
        0.0001,
    ),
    RankSetting(
        "all_condition_c01",
        ("Reactant2", "Ligand", "Additive", "Base"),
        "full",
        0.01,
    ),
)
SETTING_BY_NAME = {setting.name: setting for setting in RANK_GRID}
BLEND_GRID = (0.0, 0.25, 0.50, 1.0, 2.0)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_products() -> list[str]:
    _, products = meta.product_rows()
    return [product for product in products if product != OFFICIAL_TARGET]


def _feature_columns(encoder: FrozenReactionEncoder, setting: RankSetting) -> np.ndarray:
    if encoder.schema is None:
        raise RuntimeError("Rank encoder is not fitted.")
    specs = (
        encoder.schema.identity_slices
        if setting.block == "identity"
        else encoder.schema.role_slices
    )
    return np.concatenate(
        [
            np.arange(specs[role].start, specs[role].stop, dtype=int)
            for role in setting.roles
        ]
    )


def _pairwise_training_data(
    X: np.ndarray, yields: np.ndarray, products: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    differences: list[np.ndarray] = []
    labels: list[int] = []
    for product in sorted(set(products.astype(str))):
        indices = np.flatnonzero(products.astype(str) == product)
        for left, right in combinations(indices.tolist(), 2):
            if np.isclose(yields[left], yields[right]):
                continue
            label = int(yields[left] > yields[right])
            delta = X[left] - X[right]
            differences.extend([delta, -delta])
            labels.extend([label, 1 - label])
    return np.asarray(differences, dtype=float), np.asarray(labels, dtype=int)


@dataclass
class FittedRanker:
    encoder: FrozenReactionEncoder
    columns: np.ndarray
    model: LogisticRegression
    setting: RankSetting
    training_products: tuple[str, ...]

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        features = self.encoder.transform(frame)[:, self.columns]
        return np.asarray(self.model.decision_function(features), dtype=float)


@lru_cache(maxsize=32)
def fit_ranker(setting_name: str, held_product: str | None) -> FittedRanker:
    setting = SETTING_BY_NAME[setting_name]
    frame = pd.read_csv(TRAIN_FILE)
    products = source_products()
    training_products = tuple(
        product for product in products if held_product is None or product != held_product
    )
    training = frame[frame["Product"].astype(str).isin(training_products)].copy()
    if OFFICIAL_TARGET in set(training["Product"].astype(str)):
        raise AssertionError("Official-target labels entered the rank model.")

    config = load_run_config(CONFIG_FILE)
    feature_columns = list(config.dataset.feature_columns)
    encoder = FrozenReactionEncoder(
        DatasetSpec.from_config(config.dataset), config.features
    ).fit(training[feature_columns])
    columns = _feature_columns(encoder, setting)
    X = encoder.transform(training[feature_columns])[:, columns]
    pair_X, pair_y = _pairwise_training_data(
        X,
        training[config.dataset.target_column].to_numpy(dtype=float),
        training["Product"].astype(str).to_numpy(),
    )
    model = LogisticRegression(
        C=setting.l2_c,
        fit_intercept=False,
        solver="liblinear",
        random_state=0,
        max_iter=2_000,
    ).fit(pair_X, pair_y)
    return FittedRanker(
        encoder=encoder,
        columns=columns,
        model=model,
        setting=setting,
        training_products=training_products,
    )


def held_product_rank_validation() -> pd.DataFrame:
    frame = pd.read_csv(TRAIN_FILE)
    config = load_run_config(CONFIG_FILE)
    feature_columns = list(config.dataset.feature_columns)
    records = []
    for grid_index, setting in enumerate(RANK_GRID):
        for held_product in source_products():
            held = frame[frame["Product"].astype(str).eq(held_product)].copy()
            scores = fit_ranker(setting.name, held_product).score(held[feature_columns])
            yields = held[config.dataset.target_column].to_numpy(dtype=float)
            correct = 0.0
            ties = 0
            n_pairs = 0
            for left, right in combinations(range(len(held)), 2):
                score_delta = float(scores[left] - scores[right])
                yield_delta = float(yields[left] - yields[right])
                if np.isclose(score_delta, 0.0, atol=1e-12):
                    correct += 0.5
                    ties += 1
                else:
                    correct += float(score_delta * yield_delta > 0.0)
                n_pairs += 1
            records.append(
                {
                    "grid_index": grid_index,
                    "setting": setting.name,
                    "held_product": held_product,
                    "correct_credit": correct,
                    "n_pairs": n_pairs,
                    "score_ties": ties,
                    "pairwise_accuracy": correct / n_pairs,
                }
            )
    return pd.DataFrame(records)


def select_rank_setting(validation: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    summary = (
        validation.groupby(["grid_index", "setting"], as_index=False)
        .agg(
            correct_credit=("correct_credit", "sum"),
            n_pairs=("n_pairs", "sum"),
            score_ties=("score_ties", "sum"),
            product_accuracy_mean=("pairwise_accuracy", "mean"),
            product_accuracy_sd=("pairwise_accuracy", "std"),
        )
        .sort_values("grid_index")
    )
    summary["pooled_pairwise_accuracy"] = (
        summary["correct_credit"] / summary["n_pairs"]
    )
    ranked = summary.sort_values(
        ["pooled_pairwise_accuracy", "grid_index"],
        ascending=[False, True],
        kind="stable",
    )
    return str(ranked.iloc[0]["setting"]), summary


class QueryOneRankBlendPolicy:
    """Blend source rank scores only at query 1, delegate exactly thereafter."""

    def __init__(self, base_policy, *, blend_weight: float) -> None:
        self.base_policy = base_policy
        self.blend_weight = float(blend_weight)

    def select(
        self,
        posterior: Posterior,
        candidate_ids: np.ndarray,
        *,
        best_f: float,
        step: int,
        budget: int,
        prior_scores: np.ndarray | None = None,
        novelty_scores: np.ndarray | None = None,
    ) -> Recommendation:
        if step != 1 or self.blend_weight <= 0.0:
            return self.base_policy.select(
                posterior,
                candidate_ids,
                best_f=best_f,
                step=step,
                budget=budget,
                prior_scores=None,
                novelty_scores=novelty_scores,
            )
        if prior_scores is None:
            raise ValueError("Query-one rank blending requires rank scores.")
        model_mean = np.asarray(posterior.component_means[MODEL_NAME], dtype=float)
        rank_scores = np.asarray(prior_scores, dtype=float).reshape(-1)
        combined = zscore(model_mean) + self.blend_weight * zscore(rank_scores)
        candidate_ids = np.asarray(candidate_ids, dtype=int)
        local_index = deterministic_argmax(
            combined, candidate_ids, seed=self.base_policy.seed + step
        )
        return Recommendation(
            local_index=local_index,
            query_index=int(candidate_ids[local_index]),
            acquisition_name="query_one_pairwise_rank_blend",
            acquisition_value=float(combined[local_index]),
            predicted_mean=float(posterior.mean[local_index]),
            predicted_std=float(posterior.std[local_index]),
            component_means={
                name: float(values[local_index])
                for name, values in posterior.component_means.items()
            },
            component_stds={
                name: float(values[local_index])
                for name, values in posterior.component_stds.items()
            },
            model_weights=dict(posterior.weights),
            prior_weight=self.blend_weight,
        )


def blend_name(weight: float) -> str:
    return f"blend_{weight:.2f}".replace(".", "p")


def run_one(job: tuple[dict, str, float, bool, str]) -> dict:
    episode, rank_setting_name, blend_weight, strict_source_isolation, output_root = job
    config, dataset, lookup, observed_products = meta.build_problem(
        episode, strict_source_isolation=strict_source_isolation
    )
    held_product = episode["target_task"] if strict_source_isolation else None
    ranker = fit_ranker(rank_setting_name, held_product)
    feature_columns = list(config.dataset.feature_columns)
    rank_scores = ranker.score(dataset.candidates[feature_columns])
    oracle = OfflineTableOracle(dataset=dataset, lookup=lookup)
    started = time.perf_counter()
    runner = CampaignRunner(
        dataset,
        config,
        seed=int(episode["seed"]),
        backend="native",
        prior_scores=rank_scores,
    )
    runner.policy = QueryOneRankBlendPolicy(
        runner.policy, blend_weight=blend_weight
    )
    result = runner.run(oracle, budget=int(episode["budget"]))
    elapsed = time.perf_counter() - started
    payload = result.to_payload()
    metrics = meta._campaign_metrics(
        payload["trajectory"], float(oracle.global_best), int(episode["budget"])
    )
    if blend_weight > 0.0:
        if payload["trajectory"][0]["acquisition_name"] != "query_one_pairwise_rank_blend":
            raise AssertionError("Rank blend was not active at query 1.")
        if any(
            record["acquisition_name"] != "stable_log_ei"
            for record in payload["trajectory"][1:]
        ):
            raise AssertionError("Frontier LogEI was not restored after query 1.")
    payload["pairwise_rank_contract"] = {
        "train_only": True,
        "public_test_labels_used": False,
        "selection_target_labels_used": False,
        "strict_source_isolation": strict_source_isolation,
        "rank_training_products": list(ranker.training_products),
        "official_target_in_rank_training": OFFICIAL_TARGET in ranker.training_products,
        "rank_setting": asdict(ranker.setting),
        "blend_weight": blend_weight,
        "rank_scores": rank_scores.tolist(),
        "observed_products": observed_products,
        "episode": episode,
        "metrics": metrics,
    }
    name = blend_name(blend_weight)
    output_path = (
        Path(output_root)
        / "trajectories"
        / episode["phase"]
        / name
        / f"{episode['episode']}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "phase": episode["phase"],
        "episode": episode["episode"],
        "target_task": episode["target_task"],
        "rank_setting": rank_setting_name,
        "blend_weight": blend_weight,
        "variant": name,
        "budget": int(episode["budget"]),
        "global_best": float(oracle.global_best),
        **metrics,
        "query_1_acquisition": payload["trajectory"][0]["acquisition_name"],
        "query_2_acquisition": payload["trajectory"][1]["acquisition_name"],
        "elapsed_seconds": elapsed,
    }


def run_jobs(jobs: list[tuple[dict, str, float, bool, str]], workers: int) -> pd.DataFrame:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    rows = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_one, job): job for job in jobs}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"{row['phase']} {row['variant']} {row['episode']}: "
                f"AUC={row['AUC_best_so_far']:.3f}",
                flush=True,
            )
    return pd.DataFrame(rows).sort_values(["phase", "episode", "blend_weight"])


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (phase, variant), group in rows.groupby(["phase", "variant"], sort=True):
        for metric in CORE_METRICS:
            mean, sd, low, high = meta.mean_ci(group[metric])
            records.append(
                {
                    "phase": phase,
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


def paired_summary(rows: pd.DataFrame, selected_variant: str) -> pd.DataFrame:
    baseline = rows[rows["variant"].eq(blend_name(0.0))].set_index("episode")
    candidate = rows[rows["variant"].eq(selected_variant)].set_index("episode")
    common = baseline.index.intersection(candidate.index)
    records = []
    for metric in CORE_METRICS:
        delta = candidate.loc[common, metric] - baseline.loc[common, metric]
        mean, sd, low, high = meta.mean_ci(delta)
        higher = metric != "round_to_95_global_best"
        wins = int((delta > 1e-10).sum()) if higher else int((delta < -1e-10).sum())
        losses = int((delta < -1e-10).sum()) if higher else int((delta > 1e-10).sum())
        records.append(
            {
                "candidate": selected_variant,
                "baseline": blend_name(0.0),
                "metric": metric,
                "n_pairs": len(common),
                "mean_delta_candidate_minus_baseline": mean,
                "sample_sd_delta": sd,
                "ci95_low": low,
                "ci95_high": high,
                "wins": wins,
                "ties": len(common) - wins - losses,
                "losses": losses,
            }
        )
    return pd.DataFrame(records)


def all_paired_summaries(rows: pd.DataFrame) -> pd.DataFrame:
    variants = [
        variant
        for variant in rows["variant"].drop_duplicates().tolist()
        if variant != blend_name(0.0)
    ]
    return pd.concat(
        [paired_summary(rows, variant) for variant in variants],
        ignore_index=True,
    )


def select_blend(rows: pd.DataFrame) -> float:
    means = rows.groupby("blend_weight", sort=False)["AUC_best_so_far"].mean()
    return float(means.sort_values(ascending=False, kind="stable").index[0])


def deterministic_twenty(rows: pd.DataFrame) -> pd.DataFrame:
    episodes = sorted(rows["episode"].unique())[:20]
    sliced = rows[rows["episode"].isin(episodes)].copy()
    if len(sliced) != 20 * rows["variant"].nunique():
        raise ValueError("Incomplete deterministic 20-run slice.")
    return sliced.sort_values(["episode", "blend_weight"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=min(6, os.cpu_count() or 1))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            SESSION_ROOT
            / "experiments"
            / "6"
            / "results"
            / "6-pairwise-rank-warmstart"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.jobs <= 0:
        raise ValueError("--jobs must be positive.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    validation = held_product_rank_validation()
    selected_rank, rank_summary = select_rank_setting(validation)
    validation.to_csv(args.output_dir / "held_product_rank_accuracy.csv", index=False)
    rank_summary.to_csv(args.output_dir / "rank_setting_summary.csv", index=False)

    source_episodes = meta.make_source_meta_episodes()
    blend_grid = BLEND_GRID
    if args.smoke:
        source_episodes = source_episodes[:1]
        blend_grid = BLEND_GRID[:2]
    manifest = {
        "study": "Buchwald source-product pairwise rank query-one warm start",
        "train_only": True,
        "public_test_labels_used": False,
        "selection_target_labels_used": False,
        "selection_excludes_official_target_rows": True,
        "official_target": OFFICIAL_TARGET,
        "rank_grid": [asdict(setting) for setting in RANK_GRID],
        "blend_grid": list(BLEND_GRID),
        "selected_rank_setting": selected_rank,
        "rank_selection_rule": "highest pooled leave-one-source-product pairwise accuracy",
        "blend_selection_rule": "highest mean source-only sequential AUC over 84 episodes",
        "query_policy": "rank blend only at query 1; exact frontier stable LogEI thereafter",
        "inputs": {
            "train": {"path": str(TRAIN_FILE), "sha256": file_sha256(TRAIN_FILE)},
            "config": {"path": str(CONFIG_FILE), "sha256": file_sha256(CONFIG_FILE)},
            "model": {"path": str(MODEL_FILE), "sha256": file_sha256(MODEL_FILE)},
            "factory": {"path": str(FACTORY_FILE), "sha256": file_sha256(FACTORY_FILE)},
        },
        "source_episode_count": len(source_episodes),
        "python": platform.python_version(),
        "smoke": bool(args.smoke),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    rank_accuracy = float(
        rank_summary.loc[
            rank_summary["setting"].eq(selected_rank), "pooled_pairwise_accuracy"
        ].iloc[0]
    )
    gate = {
        "selected_rank_setting": selected_rank,
        "held_product_pairwise_accuracy": rank_accuracy,
        "chance_threshold": 0.5,
        "rank_gate_passed": rank_accuracy > 0.5,
    }
    (args.output_dir / "rank_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True), encoding="utf-8"
    )
    if not gate["rank_gate_passed"]:
        return

    source_jobs = [
        (episode, selected_rank, blend, True, str(args.output_dir))
        for blend in blend_grid
        for episode in source_episodes
    ]
    source_rows = run_jobs(source_jobs, args.jobs)
    source_rows.to_csv(args.output_dir / "source_blend_per_episode.csv", index=False)
    summarize(source_rows).to_csv(args.output_dir / "source_blend_summary.csv", index=False)
    selected_blend = select_blend(source_rows)
    selected_variant = blend_name(selected_blend)
    paired_summary(source_rows, selected_variant).to_csv(
        args.output_dir / "source_paired_selected_vs_frontier.csv", index=False
    )
    all_paired_summaries(source_rows).to_csv(
        args.output_dir / "source_all_paired_vs_frontier.csv", index=False
    )
    selection = {
        "selected_rank_setting": selected_rank,
        "held_product_pairwise_accuracy": rank_accuracy,
        "selected_blend_weight": selected_blend,
        "selected_variant": selected_variant,
        "source_mean_auc_by_blend": {
            str(key): float(value)
            for key, value in source_rows.groupby("blend_weight")[
                "AUC_best_so_far"
            ].mean().items()
        },
    }
    (args.output_dir / "selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True), encoding="utf-8"
    )
    if args.smoke:
        return

    confirmation_episodes = meta.make_official_confirmation_episodes()
    confirmation_blends = [0.0]
    if selected_blend != 0.0:
        confirmation_blends.append(selected_blend)
    confirmation_jobs = [
        (episode, selected_rank, blend, False, str(args.output_dir))
        for blend in confirmation_blends
        for episode in confirmation_episodes
    ]
    confirmation = run_jobs(confirmation_jobs, args.jobs)
    confirmation.to_csv(args.output_dir / "official_confirmation_per_episode.csv", index=False)
    summarize(confirmation).to_csv(
        args.output_dir / "official_confirmation_summary.csv", index=False
    )
    paired_summary(confirmation, selected_variant).to_csv(
        args.output_dir / "paired_selected_vs_frontier.csv", index=False
    )
    twenty = deterministic_twenty(confirmation)
    twenty.to_csv(args.output_dir / "official_confirmation_20run_per_run.csv", index=False)
    summarize(twenty).to_csv(
        args.output_dir / "official_confirmation_20run_summary.csv", index=False
    )
    paired_summary(twenty, selected_variant).to_csv(
        args.output_dir / "official_confirmation_20run_paired.csv", index=False
    )

    prior = pd.read_csv(
        DATASET_ROOT
        / "submission"
        / "results"
        / "experimental_investigations"
        / "5.1-component-ard-study"
        / "per_episode.csv"
    )
    prior = prior[
        prior["dataset"].eq("Buchwald_sub4")
        & prior["variant"].eq("component_ard")
    ].set_index("episode")
    fresh = confirmation[confirmation["blend_weight"].eq(0.0)].set_index("episode")
    common = prior.index.intersection(fresh.index)
    differences = {
        metric: float(
            np.max(
                np.abs(
                    fresh.loc[common, metric].to_numpy(dtype=float)
                    - prior.loc[common, metric].to_numpy(dtype=float)
                )
            )
        )
        for metric in CORE_METRICS
    }
    check = {
        "n_common_episodes": len(common),
        "max_abs_metric_differences": differences,
        "exact_metric_match": all(value <= 1e-12 for value in differences.values()),
    }
    (args.output_dir / "baseline_reproducibility.json").write_text(
        json.dumps(check, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
