#!/usr/bin/env python3
"""Train-only ablation of frozen label-free IUPAC name subword features."""

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

import numpy as np
import pandas as pd
from scipy.stats import t


EXPERIMENT_ROOT = Path(__file__).resolve().parent
CODE_ROOT = EXPERIMENT_ROOT.parent
DATASET_ROOT = CODE_ROOT.parents[1]
EXTENSION_ROOT = CODE_ROOT / "bgolearn_rxn_extension"
sys.path.insert(0, str(EXTENSION_ROOT / "src"))
sys.path.insert(0, str(EXPERIMENT_ROOT))

import run_suzuki_holdout_proxy as suzuki_proxy  # noqa: E402
import run_transfer_proxy as transfer_proxy  # noqa: E402
from bgolearn_rxn.campaign import CampaignRunner  # noqa: E402
from bgolearn_rxn.config import load_run_config  # noqa: E402
from bgolearn_rxn.datasets import DatasetSpec, ReactionDataset  # noqa: E402
from bgolearn_rxn.metrics import compute_campaign_metrics  # noqa: E402
from bgolearn_rxn.oracle import OfflineTableOracle  # noqa: E402


VARIANTS = ("base", "name_subword")
BASELINE = "base"
CORE_METRICS = (
    "initial_round_found_best",
    "best_found",
    "round_to_95_global_best",
    "AUC_best_so_far",
)
INPUT_FILES = {
    "Buchwald_sub4": DATASET_ROOT / "Buchwald_sub4" / "buchwald_sub4_train.csv",
    "Suzuki": DATASET_ROOT / "Suzuki" / "suzuki_train.csv",
}
BUCHWALD_CONFIG = EXTENSION_ROOT / "configs" / "buchwald_sub4.yaml"
OFFICIAL_BUCHWALD_PRODUCT = "N-(4-ethylphenyl)-4-methylaniline"
SUBWORD_DESIGN = {
    "analyzer": "char_wb",
    "ngram_range": [3, 5],
    "tfidf_norm": "l2",
    "sublinear_tf": True,
    "fit_names": "sorted unique levels from initial plus unlabeled candidates",
    "svd_algorithm": "randomized",
    "svd_random_state": 0,
    "svd_n_iter": 7,
    "max_rank": 8,
    "rank_cap": "min(8, unique_levels - 1, vocabulary_width - 1)",
    "standardized": True,
    "scale": 0.35,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_official_buchwald_episodes() -> list[dict]:
    episodes = []
    for episode in transfer_proxy.make_buchwald_exhaustive_pair_episodes(1):
        item = dict(episode)
        item["phase"] = "official_target_exhaustive"
        if item["target_task"] != OFFICIAL_BUCHWALD_PRODUCT:
            raise ValueError("Sorted Buchwald product index 1 is not the official target.")
        episodes.append(item)
    return episodes


def make_buchwald_component_episodes() -> list[dict]:
    """Build cold-component holdouts with enough rows for sequential ranking."""
    frame = pd.read_csv(INPUT_FILES["Buchwald_sub4"])
    episodes = []
    for family_index, role in enumerate(("Additive", "Ligand")):
        values = frame[role].astype(str)
        eligible = sorted(
            level for level, count in values.value_counts().items() if count >= 3
        )
        for level_index, level in enumerate(eligible):
            candidates = np.flatnonzero(values.eq(level).to_numpy())
            initial = np.flatnonzero(~values.eq(level).to_numpy())
            episodes.append(
                {
                    "episode": f"component_{role.lower()}_{level_index:02d}",
                    "episode_family": f"leave_{role.lower()}_out",
                    "phase": "component_holdout",
                    "held_role": role,
                    "held_level": level,
                    "initial_indices": initial.tolist(),
                    "candidate_indices": candidates.tolist(),
                    "budget": min(5, len(candidates)),
                    "seed": 610_000 + family_index * 1_000 + level_index,
                }
            )
    return episodes


def make_episodes() -> dict[str, list[dict]]:
    suzuki = suzuki_proxy.make_random_episodes()
    return {
        "Buchwald_sub4": (
            make_official_buchwald_episodes() + make_buchwald_component_episodes()
        ),
        "Suzuki": suzuki,
    }


def _with_feature_variant(config, variant: str):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown encoder variant: {variant}")
    return replace(
        config,
        features=replace(
            config.features,
            use_name_subwords=variant == "name_subword",
            name_subword_scale=0.35,
            name_subword_max_rank=8,
        ),
    )


def _build_component_problem(episode: dict):
    source = pd.read_csv(INPUT_FILES["Buchwald_sub4"]).reset_index(
        names="source_index"
    )
    base = load_run_config(BUCHWALD_CONFIG)
    budget = int(episode["budget"])
    seed = int(episode["seed"])
    feature_columns = tuple(base.dataset.feature_columns)
    config = replace(
        base,
        dataset=replace(
            base.dataset,
            task_column=None,
            target_task_value=None,
            condition_roles=feature_columns,
            budget=budget,
            seeds=(seed,),
        ),
        ensemble=replace(
            base.ensemble,
            enabled_models=("gp",),
            initial_weights={"gp": 1.0},
        ),
        policy=replace(
            base.policy,
            first_query="gp",
            early_end=budget,
            final_start=budget + 1,
            challenge_every=0,
            novelty_disagreement_mode="off",
        ),
    )
    initial_indices = np.asarray(episode["initial_indices"], dtype=int)
    candidate_indices = np.asarray(episode["candidate_indices"], dtype=int)
    features = list(feature_columns)
    initial = source.loc[
        initial_indices, features + [config.dataset.target_column]
    ].copy()
    candidates = source.loc[candidate_indices, features].copy()
    candidates.insert(
        0, "query_index", source.loc[candidate_indices, "source_index"].astype(int)
    )
    lookup = source.loc[
        candidate_indices, features + [config.dataset.target_column]
    ].copy()
    lookup.insert(
        0, "query_index", source.loc[candidate_indices, "source_index"].astype(int)
    )
    dataset = ReactionDataset(
        spec=DatasetSpec.from_config(config.dataset),
        initial=initial,
        candidates=candidates,
    )
    return config, dataset, lookup


def build_problem(dataset_name: str, episode: dict, variant: str):
    if dataset_name == "Buchwald_sub4":
        if episode["phase"] == "official_target_exhaustive":
            config, dataset, lookup = transfer_proxy.build_problem(
                dataset_name, "low_rank_task_gp", episode
            )
            expected_model = "low_rank_task_gp"
        elif episode["phase"] == "component_holdout":
            config, dataset, lookup = _build_component_problem(episode)
            expected_model = "gp"
        else:
            raise ValueError(f"Unknown Buchwald phase: {episode['phase']}")
    elif dataset_name == "Suzuki":
        config, dataset, lookup = suzuki_proxy.build_problem(episode, "gp_only")
        expected_model = "gp"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    config = _with_feature_variant(config, variant)
    if config.gp.target_transform != "standard":
        raise ValueError("Name-subword study requires the standard target transform.")
    if tuple(config.ensemble.enabled_models) != (expected_model,):
        raise ValueError("Name-subword study changed the incumbent surrogate.")
    return config, dataset, lookup


def _metrics_from_trajectory(
    trajectory: list[dict], *, global_best: float, budget: int
) -> dict[str, float | int | bool]:
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


def run_one(job: tuple[str, str, dict, str]) -> dict:
    dataset_name, variant, episode, output_root_string = job
    config, dataset, lookup = build_problem(dataset_name, episode, variant)
    oracle = OfflineTableOracle(dataset=dataset, lookup=lookup)
    seed = int(episode.get("seed", episode.get("campaign_seed")))
    started = time.perf_counter()
    runner = CampaignRunner(dataset, config, seed=seed, backend="native")
    result = runner.run(oracle, budget=int(episode["budget"]))
    elapsed = time.perf_counter() - started
    payload = result.to_payload()
    global_best = (
        float(oracle.global_best)
        if episode["phase"] == "official_target_exhaustive"
        else float(lookup[dataset.spec.target_column].max())
    )
    metrics = _metrics_from_trajectory(
        payload["trajectory"],
        global_best=global_best,
        budget=int(episode["budget"]),
    )
    schema = runner.prepared.encoder.schema
    if schema is None:
        raise RuntimeError("Encoder schema is unavailable after campaign setup.")
    subword_widths = {
        role: sum(
            "::subword::" in name
            for name in schema.feature_names[
                schema.role_slices[role].start : schema.role_slices[role].stop
            ]
        )
        for role in schema.roles
    }
    payload["name_subword_metrics"] = metrics
    payload["name_subword_contract"] = {
        "encoder_variant": variant,
        "train_only": True,
        "public_test_labels_used": False,
        "input_file": str(INPUT_FILES[dataset_name].relative_to(DATASET_ROOT)),
        "input_sha256": file_sha256(INPUT_FILES[dataset_name]),
        "fit_reference": "initial features plus unlabeled candidate features",
        "preprocessing_frozen": True,
        "subword_widths": subword_widths,
        "episode": episode,
    }
    output_path = (
        Path(output_root_string)
        / "trajectories"
        / dataset_name
        / episode["phase"]
        / episode["episode_family"]
        / variant
        / f"{episode['episode']}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {
        "dataset": dataset_name,
        "phase": episode["phase"],
        "episode_family": episode["episode_family"],
        "episode": episode["episode"],
        "target_task": episode.get("target_task"),
        "held_role": episode.get("held_role"),
        "held_level": episode.get("held_level"),
        "variant": variant,
        "budget": episode["budget"],
        "n_initial": len(dataset.initial),
        "n_candidates": len(dataset.candidates),
        "global_best": global_best,
        "feature_width": schema.n_features,
        "subword_width": sum(subword_widths.values()),
        **metrics,
        "elapsed_seconds": elapsed,
    }


def mean_ci(values: pd.Series) -> tuple[float, float, float, float]:
    array = values.to_numpy(dtype=float)
    mean = float(array.mean())
    if len(array) == 1:
        return mean, 0.0, mean, mean
    sd = float(array.std(ddof=1))
    half = float(t.ppf(0.975, len(array) - 1) * sd / np.sqrt(len(array)))
    return mean, sd, mean - half, mean + half


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (dataset, phase, family, variant), group in rows.groupby(
        ["dataset", "phase", "episode_family", "variant"], sort=True
    ):
        for metric in CORE_METRICS:
            mean, sd, low, high = mean_ci(group[metric])
            records.append(
                {
                    "dataset": dataset,
                    "phase": phase,
                    "episode_family": family,
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
    records = []
    for (dataset, phase, family), section in rows.groupby(
        ["dataset", "phase", "episode_family"], sort=True
    ):
        baseline = section[section["variant"].eq(BASELINE)].set_index("episode")
        candidate = section[section["variant"].eq("name_subword")].set_index(
            "episode"
        )
        common = baseline.index.intersection(candidate.index)
        for metric in CORE_METRICS:
            delta = candidate.loc[common, metric] - baseline.loc[common, metric]
            mean, sd, low, high = mean_ci(delta)
            higher = metric != "round_to_95_global_best"
            wins = int((delta > 1e-10).sum()) if higher else int((delta < -1e-10).sum())
            losses = int((delta < -1e-10).sum()) if higher else int((delta > 1e-10).sum())
            records.append(
                {
                    "dataset": dataset,
                    "phase": phase,
                    "episode_family": family,
                    "candidate": "name_subword",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=min(6, os.cpu_count() or 1))
    parser.add_argument("--limit-episodes", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            DATASET_ROOT
            / "submission"
            / "results"
            / "experimental_investigations"
            / "6.1-name-subword-study"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.jobs <= 0:
        raise ValueError("--jobs must be positive.")
    episodes = make_episodes()
    if args.limit_episodes is not None:
        if args.limit_episodes <= 0:
            raise ValueError("--limit-episodes must be positive.")
        episodes = {
            dataset: values[: args.limit_episodes]
            for dataset, values in episodes.items()
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "train_only": True,
        "public_test_labels_used": False,
        "unlabeled_candidate_features_used": True,
        "encoder_fit_once": True,
        "round_refitting": False,
        "variants": list(VARIANTS),
        "production_default_enabled": False,
        "fixed_subword_design": SUBWORD_DESIGN,
        "parameter_sweep_performed": False,
        "target_transform": "standard",
        "acquisition": "posterior mean query 1, stable LogEI thereafter",
        "models": {
            "Buchwald_sub4/official_target_exhaustive": "low_rank_task_gp",
            "Buchwald_sub4/component_holdout": "structured_anova_gp",
            "Suzuki/random_holdout": "structured_anova_gp",
        },
        "official_buchwald_target": OFFICIAL_BUCHWALD_PRODUCT,
        "component_holdout_contract": {
            "roles": ["Additive", "Ligand"],
            "minimum_level_rows": 3,
            "budget": "min(5, held level rows)",
        },
        "inputs": {
            dataset: {
                "path": str(path.relative_to(DATASET_ROOT)),
                "sha256": file_sha256(path),
            }
            for dataset, path in INPUT_FILES.items()
        },
        "episode_counts": {
            dataset: len(values) for dataset, values in episodes.items()
        },
        "phase_counts": {
            dataset: pd.Series([e["phase"] for e in values]).value_counts().to_dict()
            for dataset, values in episodes.items()
        },
        "python": platform.python_version(),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    jobs = [
        (dataset, variant, episode, str(args.output_dir))
        for dataset, values in episodes.items()
        for variant in VARIANTS
        for episode in values
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(run_one, job): job for job in jobs}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"{row['dataset']} {row['phase']} {row['variant']} "
                f"{row['episode']}: AUC={row['AUC_best_so_far']:.3f}",
                flush=True,
            )
    detail = pd.DataFrame(rows).sort_values(
        ["dataset", "phase", "episode", "variant"]
    )
    detail.to_csv(args.output_dir / "per_episode.csv", index=False)
    summarize(detail).to_csv(args.output_dir / "summary.csv", index=False)
    paired_summary(detail).to_csv(
        args.output_dir / "paired_subword_vs_base.csv", index=False
    )


if __name__ == "__main__":
    main()
