#!/usr/bin/env python3
"""Train-only sequential proxy evaluation for multitask reaction transfer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


CODE_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path(__file__).resolve().parents[3]
EXTENSION_ROOT = CODE_ROOT / "bgolearn_rxn_extension"
sys.path.insert(0, str(EXTENSION_ROOT / "src"))

from bgolearn_rxn.campaign import CampaignRunner  # noqa: E402
from bgolearn_rxn.config import load_run_config  # noqa: E402
from bgolearn_rxn.datasets import DatasetSpec, ReactionDataset  # noqa: E402
from bgolearn_rxn.oracle import OfflineTableOracle  # noqa: E402


TRAIN_FILES = {
    "Buchwald_sub4": DATASET_ROOT / "Buchwald_sub4" / "buchwald_sub4_train.csv",
    "Suzuki": DATASET_ROOT / "Suzuki" / "suzuki_train.csv",
}
CONFIG_FILES = {
    "Buchwald_sub4": EXTENSION_ROOT / "configs" / "buchwald_sub4.yaml",
    "Suzuki": EXTENSION_ROOT / "configs" / "suzuki.yaml",
}
VARIANTS = {
    "Buchwald_sub4": (
        "target_only_gp",
        "pooled_gp_scalar_task",
        "pooled_gp_task_descriptor_025",
        "pooled_gp_task_descriptor_100",
        "pooled_gp_task_descriptor_400",
        "pooled_gp_task_mean_100",
        "pooled_gp_task_mean_300",
        "pooled_gp_task_mean_1000",
        "residual_transfer_gp",
        "safe_transfer_80_20",
        "safe_transfer_50_50",
        "source_expert_gp",
        "low_rank_task_gp",
        "pooled_gp_interaction",
        "pooled_gp_interaction_task_pairs",
        "pooled_interaction_task_pairs",
    ),
    "Suzuki": (
        "target_only_gp",
        "pooled_pair_task_gp",
        "pooled_role_task_gp",
        "pooled_constituent_task_gp",
        "pooled_constituent_gp_interaction",
        "pooled_interaction_tensor",
    ),
}
SUZUKI_PAIR_VARIANTS = tuple(
    variant for variant in VARIANTS["Suzuki"] if variant != "pooled_role_task_gp"
)
SUZUKI_ROLE_VARIANTS = ("target_only_gp", "pooled_role_task_gp")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _logei_config(config, *, budget: int, seed: int):
    return replace(
        config,
        dataset=replace(config.dataset, budget=budget, seeds=(seed,)),
        policy=replace(
            config.policy,
            first_query="gp",
            early_end=budget,
            final_start=budget + 1,
            challenge_every=0,
            novelty_disagreement_mode="off",
        ),
    )


def _with_suzuki_condition_roles(config, roles: tuple[str, ...]):
    allowed = set(roles) - {"SubstratePair"}
    interaction_pairs = tuple(
        pair
        for pair in config.gp.interaction_pairs
        if pair[0] in allowed and pair[1] in allowed
    )
    interaction_weights = {
        "|".join(pair): config.gp.interaction_weights["|".join(pair)]
        for pair in interaction_pairs
        if "|".join(pair) in config.gp.interaction_weights
    }
    return replace(
        config,
        dataset=replace(config.dataset, condition_roles=roles),
        gp=replace(
            config.gp,
            interaction_pairs=interaction_pairs,
            interaction_weights=interaction_weights,
        ),
    )


def configure_variant(
    base,
    dataset: str,
    variant: str,
    target_task: str,
    budget: int,
    seed: int,
    *,
    task_column: str | None = None,
):
    config = _logei_config(base, budget=budget, seed=seed)
    if dataset == "Buchwald_sub4":
        config = replace(
            config,
            dataset=replace(config.dataset, target_task_value=target_task),
        )
        if variant in {
            "target_only_gp",
            "pooled_gp_scalar_task",
            "pooled_gp_task_descriptor_025",
            "pooled_gp_task_descriptor_100",
            "pooled_gp_task_descriptor_400",
            "pooled_gp_task_mean_100",
            "pooled_gp_task_mean_300",
            "pooled_gp_task_mean_1000",
        }:
            descriptor_scale = {
                "pooled_gp_task_descriptor_025": 0.25,
                "pooled_gp_task_descriptor_100": 1.0,
                "pooled_gp_task_descriptor_400": 4.0,
            }.get(variant, 0.0)
            task_mean_prior_count = {
                "pooled_gp_task_mean_100": 1.0,
                "pooled_gp_task_mean_300": 3.0,
                "pooled_gp_task_mean_1000": 10.0,
            }.get(variant)
            return replace(
                config,
                gp=replace(
                    config.gp,
                    task_descriptor_scale=descriptor_scale,
                    task_mean_prior_count=task_mean_prior_count,
                ),
                ensemble=replace(config.ensemble, enabled_models=("gp",)),
            )
        if variant == "residual_transfer_gp":
            return replace(
                config,
                ensemble=replace(
                    config.ensemble,
                    enabled_models=("residual_transfer",),
                    initial_weights={"residual_transfer": 1.0},
                ),
                policy=replace(config.policy, first_query="residual_transfer"),
            )
        if variant in {"safe_transfer_80_20", "safe_transfer_50_50"}:
            gp_weight = 0.8 if variant == "safe_transfer_80_20" else 0.5
            return replace(
                config,
                ensemble=replace(
                    config.ensemble,
                    enabled_models=("gp", "residual_transfer"),
                    initial_weights={
                        "gp": gp_weight,
                        "residual_transfer": 1.0 - gp_weight,
                    },
                    minimum_weight=0.05,
                ),
                policy=replace(config.policy, first_query="gp"),
            )
        if variant == "source_expert_gp":
            return replace(
                config,
                ensemble=replace(
                    config.ensemble,
                    enabled_models=("source_expert",),
                    initial_weights={"source_expert": 1.0},
                ),
                policy=replace(config.policy, first_query="source_expert"),
            )
        if variant == "low_rank_task_gp":
            return replace(
                config,
                ensemble=replace(
                    config.ensemble,
                    enabled_models=("low_rank_task_gp",),
                    initial_weights={"low_rank_task_gp": 1.0},
                ),
                policy=replace(config.policy, first_query="low_rank_task_gp"),
            )
        if variant == "pooled_gp_interaction":
            return replace(
                config,
                ensemble=replace(config.ensemble, enabled_models=("gp", "interaction")),
            )
        task_pairs = (
            ("Product", "Reactant2"),
            ("Product", "Ligand"),
            ("Product", "Additive"),
            ("Product", "Base"),
        )
        interaction = replace(
            config.interaction_model,
            interaction_pairs=config.interaction_model.interaction_pairs + task_pairs,
        )
        enabled = (
            ("interaction",)
            if variant == "pooled_interaction_task_pairs"
            else ("gp", "interaction")
        )
        first_query = "interaction" if enabled == ("interaction",) else "gp"
        return replace(
            config,
            interaction_model=interaction,
            ensemble=replace(config.ensemble, enabled_models=enabled),
            policy=replace(config.policy, first_query=first_query),
        )

    task_column = task_column or "SubstratePair"
    config = replace(
        config,
        dataset=replace(
            config.dataset,
            task_column=task_column,
            target_task_value=target_task,
        ),
    )
    if task_column in {"Electrophile", "Nucleophile"}:
        if variant not in SUZUKI_ROLE_VARIANTS:
            raise ValueError(
                f"Variant {variant} is incompatible with Suzuki role episodes."
            )
        complementary_role = (
            "Nucleophile" if task_column == "Electrophile" else "Electrophile"
        )
        config = _with_suzuki_condition_roles(
            config,
            (complementary_role, "Ligand", "Base", "Solvent"),
        )
        return replace(
            config,
            ensemble=replace(config.ensemble, enabled_models=("gp",)),
        )
    if task_column != "SubstratePair":
        raise ValueError(f"Unsupported Suzuki task column: {task_column}")
    if variant in {"target_only_gp", "pooled_pair_task_gp"}:
        config = _with_suzuki_condition_roles(
            config, ("Ligand", "Base", "Solvent")
        )
        return replace(
            config,
            ensemble=replace(config.ensemble, enabled_models=("gp",)),
        )
    if variant == "pooled_constituent_task_gp":
        config = _with_suzuki_condition_roles(
            config,
            (
                "Electrophile",
                "Nucleophile",
                "Ligand",
                "Base",
                "Solvent",
            ),
        )
        return replace(
            config,
            ensemble=replace(config.ensemble, enabled_models=("gp",)),
        )
    if variant == "pooled_constituent_gp_interaction":
        config = _with_suzuki_condition_roles(
            config,
            (
                "Electrophile",
                "Nucleophile",
                "Ligand",
                "Base",
                "Solvent",
            ),
        )
        return replace(
            config,
            ensemble=replace(config.ensemble, enabled_models=("gp", "interaction")),
        )
    if variant == "pooled_interaction_tensor":
        config = _with_suzuki_condition_roles(
            config,
            (
                "Electrophile",
                "Nucleophile",
                "Ligand",
                "Base",
                "Solvent",
            ),
        )
        return replace(
            config,
            ensemble=replace(config.ensemble, enabled_models=("interaction",)),
            policy=replace(config.policy, first_query="interaction"),
        )
    raise ValueError(f"Unknown {dataset} variant: {variant}")


def make_buchwald_episodes(repeats: int) -> list[dict]:
    frame = pd.read_csv(TRAIN_FILES["Buchwald_sub4"]).reset_index(names="source_index")
    products = sorted(frame["Product"].astype(str).unique())
    episodes = []
    for product_index, product in enumerate(products):
        target_indices = frame.index[frame["Product"].astype(str).eq(product)].to_numpy()
        for repeat in range(repeats):
            split_seed = 910_000 + product_index * 1_000 + repeat
            rng = np.random.default_rng(split_seed)
            initial_indices = np.sort(rng.choice(target_indices, size=2, replace=False))
            candidate_indices = np.setdiff1d(target_indices, initial_indices)
            episodes.append(
                {
                    "episode": f"product_{product_index}_split_{repeat}",
                    "episode_family": "Product",
                    "task_column": "Product",
                    "target_task": product,
                    "initial_indices": initial_indices.tolist(),
                    "candidate_indices": candidate_indices.tolist(),
                    "budget": 3,
                    "seed": 100 + 100 * repeat + product_index,
                }
            )
    return episodes


def make_buchwald_exhaustive_pair_episodes(product_index: int) -> list[dict]:
    frame = pd.read_csv(TRAIN_FILES["Buchwald_sub4"]).reset_index(names="source_index")
    products = sorted(frame["Product"].astype(str).unique())
    if not 0 <= product_index < len(products):
        raise ValueError(
            f"Buchwald product index must lie in [0, {len(products) - 1}]."
        )
    product = products[product_index]
    target_indices = frame.index[frame["Product"].astype(str).eq(product)].to_numpy()
    if len(target_indices) < 2:
        raise ValueError("Exhaustive Buchwald pairs require at least two target rows.")

    episodes = []
    for pair_index, initial_pair in enumerate(combinations(target_indices.tolist(), 2)):
        initial_indices = np.asarray(initial_pair, dtype=int)
        candidate_indices = np.setdiff1d(target_indices, initial_indices)
        episodes.append(
            {
                "episode": (
                    f"product_{product_index}_pair_{initial_pair[0]}_{initial_pair[1]}"
                ),
                "episode_family": "Product",
                "task_column": "Product",
                "target_task": product,
                "initial_indices": initial_indices.tolist(),
                "candidate_indices": candidate_indices.tolist(),
                "budget": min(3, len(candidate_indices)),
                "seed": 50_000 + product_index * 1_000 + pair_index,
            }
        )
    return episodes


def make_suzuki_pair_episodes() -> list[dict]:
    frame = pd.read_csv(TRAIN_FILES["Suzuki"]).reset_index(names="source_index")
    pairs = frame["Electrophile"].astype(str) + " || " + frame["Nucleophile"].astype(str)
    episodes = []
    eligible = sorted(pair for pair, count in pairs.value_counts().items() if count >= 3)
    for pair_index, pair in enumerate(eligible):
        target_indices = frame.index[pairs.eq(pair)].to_numpy()
        for local_index, initial_index in enumerate(target_indices):
            candidate_indices = target_indices[target_indices != initial_index]
            episodes.append(
                {
                    "episode": f"pair_{pair_index}_initial_{local_index}",
                    "episode_family": "SubstratePair",
                    "task_column": "SubstratePair",
                    "target_task": pair,
                    "initial_indices": [int(initial_index)],
                    "candidate_indices": candidate_indices.tolist(),
                    "budget": min(3, len(candidate_indices)),
                    "seed": 2_000 + pair_index * 100 + local_index,
                }
            )
    return episodes


def make_suzuki_role_episodes() -> list[dict]:
    frame = pd.read_csv(TRAIN_FILES["Suzuki"]).reset_index(names="source_index")
    episodes = []
    for family_index, task_column in enumerate(("Electrophile", "Nucleophile")):
        values = frame[task_column].astype(str)
        counts = values.value_counts()
        eligible = sorted(level for level, count in counts.items() if count >= 4)
        for level_index, level in enumerate(eligible):
            target_indices = frame.index[values.eq(level)].to_numpy()
            for local_index, initial_index in enumerate(target_indices):
                candidate_indices = target_indices[target_indices != initial_index]
                episodes.append(
                    {
                        "episode": (
                            f"{task_column.lower()}_{level_index}_initial_{local_index}"
                        ),
                        "episode_family": task_column,
                        "task_column": task_column,
                        "target_task": level,
                        "initial_indices": [int(initial_index)],
                        "candidate_indices": candidate_indices.tolist(),
                        "budget": min(3, len(candidate_indices)),
                        "seed": (
                            3_000
                            + family_index * 10_000
                            + level_index * 100
                            + local_index
                        ),
                    }
                )
    return episodes


def make_suzuki_episodes(mode: str = "pair") -> list[dict]:
    if mode == "pair":
        return make_suzuki_pair_episodes()
    if mode == "role":
        return make_suzuki_role_episodes()
    raise ValueError(f"Unknown Suzuki episode mode: {mode}")


def build_problem(dataset_name: str, variant: str, episode: dict):
    source = pd.read_csv(TRAIN_FILES[dataset_name]).reset_index(names="source_index")
    base = load_run_config(CONFIG_FILES[dataset_name])
    config = configure_variant(
        base,
        dataset_name,
        variant,
        episode["target_task"],
        episode["budget"],
        episode["seed"],
        task_column=episode.get("task_column"),
    )
    feature_columns = list(config.dataset.feature_columns)
    initial_indices = np.asarray(episode["initial_indices"], dtype=int)
    candidate_indices = np.asarray(episode["candidate_indices"], dtype=int)
    target_only = variant.startswith("target_only")
    if target_only:
        observed_indices = initial_indices
    else:
        observed_indices = np.setdiff1d(source.index.to_numpy(), candidate_indices)

    initial = source.loc[observed_indices, feature_columns + [config.dataset.target_column]].copy()
    candidates = source.loc[candidate_indices, feature_columns].copy()
    candidates.insert(0, "query_index", source.loc[candidate_indices, "source_index"].astype(int))
    lookup = source.loc[candidate_indices, feature_columns + [config.dataset.target_column]].copy()
    lookup.insert(0, "query_index", source.loc[candidate_indices, "source_index"].astype(int))
    spec = DatasetSpec.from_config(config.dataset)
    reaction_dataset = ReactionDataset(spec=spec, initial=initial, candidates=candidates)
    return config, reaction_dataset, lookup


def run_one(job: tuple[str, str, dict, str]) -> dict:
    dataset_name, variant, episode, output_root_string = job
    config, dataset, lookup = build_problem(dataset_name, variant, episode)
    oracle = OfflineTableOracle(dataset=dataset, lookup=lookup)
    started = time.perf_counter()
    runner = CampaignRunner(dataset, config, seed=episode["seed"], backend="native")
    result = runner.run(
        oracle,
        budget=episode["budget"],
    )
    elapsed = time.perf_counter() - started
    payload = result.to_payload()
    payload["proxy_contract"] = {
        "train_only": True,
        "source_file": str(TRAIN_FILES[dataset_name].relative_to(DATASET_ROOT)),
        "source_sha256": file_sha256(TRAIN_FILES[dataset_name]),
        "public_test_labels_used": False,
        "episode": episode,
        "variant": variant,
    }
    task_model = runner.ensemble.models.get("low_rank_task_gp")
    if task_model is not None and task_model.task_correlation_ is not None:
        task = task_model.schema.task_role
        payload["proxy_contract"]["task_correlation"] = (
            task_model.task_correlation_.tolist()
        )
        payload["proxy_contract"]["task_identity_names"] = list(
            task_model.schema.feature_names[
                task_model.schema.identity_slices[task].start
                : task_model.schema.identity_slices[task].stop
            ]
        )
    output_path = (
        Path(output_root_string)
        / dataset_name
        / variant
        / f"{episode['episode']}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    metrics = payload["metrics"]
    nll_values = []
    coverage80 = []
    coverage95 = []
    for record in payload["trajectory"]:
        std = max(float(record["predicted_std"]), 2.0)
        residual = float(record["observed_yield"] - record["predicted_yield"])
        z = residual / std
        nll_values.append(0.5 * z**2 + np.log(std) + 0.5 * np.log(2.0 * np.pi))
        coverage80.append(abs(z) <= 1.2815515655446004)
        coverage95.append(abs(z) <= 1.959963984540054)

    t95 = metrics["round_to_95_global_best"]
    return {
        "dataset": dataset_name,
        "variant": variant,
        "episode": episode["episode"],
        "episode_family": episode["episode_family"],
        "task_column": episode["task_column"],
        "target_task": episode["target_task"],
        "seed": episode["seed"],
        "budget": episode["budget"],
        "n_initial_target": len(episode["initial_indices"]),
        "n_candidates": len(episode["candidate_indices"]),
        "initial_round_found_best": metrics["initial_round_found_best"],
        "best_found": metrics["best_found"],
        "AUC_best_so_far": metrics["auc_best_so_far"],
        "final_simple_regret": metrics["final_simple_regret"],
        "t95_penalized": episode["budget"] + 1 if t95 is None else t95,
        "t95_reached": t95 is not None,
        "mean_selected_nll": float(np.mean(nll_values)),
        "coverage80": float(np.mean(coverage80)),
        "coverage95": float(np.mean(coverage95)),
        "elapsed_seconds": elapsed,
        "task_correlation_json": (
            json.dumps(payload["proxy_contract"].get("task_correlation"))
            if "task_correlation" in payload["proxy_contract"]
            else None
        ),
    }


def mean_ci(values: pd.Series) -> tuple[float, float, float, float]:
    array = values.to_numpy(dtype=float)
    mean = float(np.mean(array))
    if len(array) == 1:
        return mean, 0.0, mean, mean
    sd = float(np.std(array, ddof=1))
    half = float(t.ppf(0.975, len(array) - 1) * sd / np.sqrt(len(array)))
    return mean, sd, mean - half, mean + half


def summarize(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = (
        "initial_round_found_best",
        "best_found",
        "AUC_best_so_far",
        "final_simple_regret",
        "t95_penalized",
        "mean_selected_nll",
        "coverage80",
        "coverage95",
    )
    summary_rows = []
    for (dataset, variant), group in rows.groupby(["dataset", "variant"], sort=True):
        for metric in metrics:
            mean, sd, low, high = mean_ci(group[metric])
            summary_rows.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "metric": metric,
                    "n_episodes": len(group),
                    "mean": mean,
                    "sample_sd": sd,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )

    paired_rows = []
    for dataset, dataset_rows in rows.groupby("dataset", sort=True):
        baseline = dataset_rows[dataset_rows["variant"].eq("target_only_gp")].set_index("episode")
        for variant, group in dataset_rows.groupby("variant", sort=True):
            if variant == "target_only_gp":
                continue
            candidate = group.set_index("episode")
            common = baseline.index.intersection(candidate.index)
            for metric in metrics:
                delta = candidate.loc[common, metric] - baseline.loc[common, metric]
                mean, sd, low, high = mean_ci(delta)
                higher_is_better = metric not in {
                    "final_simple_regret",
                    "t95_penalized",
                    "mean_selected_nll",
                }
                tolerance = 1e-10
                wins = int((delta > tolerance).sum()) if higher_is_better else int((delta < -tolerance).sum())
                losses = int((delta < -tolerance).sum()) if higher_is_better else int((delta > tolerance).sum())
                paired_rows.append(
                    {
                        "dataset": dataset,
                        "variant": variant,
                        "baseline": "target_only_gp",
                        "metric": metric,
                        "n_pairs": len(common),
                        "mean_delta_variant_minus_baseline": mean,
                        "sample_sd_delta": sd,
                        "ci95_low": low,
                        "ci95_high": high,
                        "wins": wins,
                        "ties": len(common) - wins - losses,
                        "losses": losses,
                    }
                )
    return pd.DataFrame(summary_rows), pd.DataFrame(paired_rows)


def summarize_by_family(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    aggregate_frames = []
    paired_frames = []
    for (dataset, family), group in rows.groupby(
        ["dataset", "episode_family"], sort=True
    ):
        aggregate, paired = summarize(group)
        aggregate.insert(1, "episode_family", family)
        paired.insert(1, "episode_family", family)
        aggregate_frames.append(aggregate)
        paired_frames.append(paired)
    return (
        pd.concat(aggregate_frames, ignore_index=True),
        pd.concat(paired_frames, ignore_index=True),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=tuple(TRAIN_FILES), default=list(TRAIN_FILES))
    parser.add_argument("--buchwald-repeats", type=int, default=8)
    parser.add_argument(
        "--buchwald-exhaustive-product-index",
        type=int,
        default=None,
        help="Opt-in exhaustive 2-row initializations for one sorted Buchwald product.",
    )
    parser.add_argument(
        "--suzuki-episode-mode",
        choices=("pair", "role"),
        default="pair",
        help="Suzuki train-only proxy family; production configuration is unchanged.",
    )
    parser.add_argument("--jobs", type=int, default=min(6, os.cpu_count() or 1))
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(sorted({variant for values in VARIANTS.values() for variant in values})),
        default=None,
        help="Optional ordered variant subset; every name must support every selected dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATASET_ROOT / "submission" / "results" / "experimental_investigations" / "transfer_proxy",
    )
    return parser.parse_args()


def resolve_variants(
    datasets: list[str],
    requested: list[str] | None,
    *,
    suzuki_episode_mode: str = "pair",
) -> dict[str, tuple[str, ...]]:
    if requested is not None and len(requested) != len(set(requested)):
        raise ValueError("--variants must not contain duplicates.")
    selected = None if requested is None else tuple(requested)
    output = {}
    for dataset in datasets:
        if dataset == "Suzuki":
            allowed = (
                SUZUKI_ROLE_VARIANTS
                if suzuki_episode_mode == "role"
                else SUZUKI_PAIR_VARIANTS
            )
        else:
            allowed = VARIANTS[dataset]
        dataset_selection = allowed if selected is None else selected
        unsupported = [variant for variant in dataset_selection if variant not in allowed]
        if unsupported:
            raise ValueError(
                f"Variants {unsupported} are unavailable for dataset {dataset} "
                f"in {suzuki_episode_mode!r} Suzuki episode mode."
            )
        output[dataset] = tuple(dataset_selection)
    return output


def main() -> None:
    args = parse_args()
    if args.buchwald_repeats <= 0 or args.jobs <= 0:
        raise ValueError("Repeat and worker counts must be positive.")
    variant_map = resolve_variants(
        args.datasets,
        args.variants,
        suzuki_episode_mode=args.suzuki_episode_mode,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    episode_map = {}
    for dataset in args.datasets:
        if dataset == "Buchwald_sub4":
            if args.buchwald_exhaustive_product_index is None:
                episode_map[dataset] = make_buchwald_episodes(args.buchwald_repeats)
            else:
                episode_map[dataset] = make_buchwald_exhaustive_pair_episodes(
                    args.buchwald_exhaustive_product_index
                )
        else:
            episode_map[dataset] = make_suzuki_episodes(args.suzuki_episode_mode)
    jobs = [
        (dataset, variant, episode, str(args.output_dir))
        for dataset in args.datasets
        for variant in variant_map[dataset]
        for episode in episode_map[dataset]
    ]
    manifest = {
        "train_only": True,
        "public_test_labels_used": False,
        "input_files": {
            dataset: {
                "path": str(TRAIN_FILES[dataset].relative_to(DATASET_ROOT)),
                "sha256": file_sha256(TRAIN_FILES[dataset]),
            }
            for dataset in args.datasets
        },
        "datasets": args.datasets,
        "variants": variant_map,
        "suzuki_episode_mode": args.suzuki_episode_mode,
        "buchwald_exhaustive_product_index": args.buchwald_exhaustive_product_index,
        "episode_counts": {dataset: len(episode_map[dataset]) for dataset in args.datasets},
        "episode_family_counts": {
            dataset: pd.Series(
                [episode["episode_family"] for episode in episode_map[dataset]]
            ).value_counts().sort_index().to_dict()
            for dataset in args.datasets
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    rows = []
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(run_one, job): job for job in jobs}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"{row['dataset']} {row['variant']} {row['episode']}: "
                f"AUC={row['AUC_best_so_far']:.3f}, "
                f"regret={row['final_simple_regret']:.3f}, "
                f"NLL={row['mean_selected_nll']:.3f}",
                flush=True,
            )

    per_episode = pd.DataFrame(rows).sort_values(["dataset", "variant", "episode"])
    aggregate, paired = summarize(per_episode)
    family_aggregate, family_paired = summarize_by_family(per_episode)
    per_episode.to_csv(args.output_dir / "per_episode.csv", index=False)
    aggregate.to_csv(args.output_dir / "aggregate.csv", index=False)
    paired.to_csv(args.output_dir / "paired_vs_target_only.csv", index=False)
    family_aggregate.to_csv(args.output_dir / "aggregate_by_family.csv", index=False)
    family_paired.to_csv(args.output_dir / "paired_by_family_vs_target_only.csv", index=False)
    print("\nAUC summary:")
    print(aggregate[aggregate["metric"].eq("AUC_best_so_far")].to_string(index=False))
    print("\nFinal-regret summary:")
    print(aggregate[aggregate["metric"].eq("final_simple_regret")].to_string(index=False))


if __name__ == "__main__":
    main()
