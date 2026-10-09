#!/usr/bin/env python3
"""Train-only Suzuki holdout proxy for sequential optimizer comparison."""

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


CODE_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path(__file__).resolve().parents[3]
EXTENSION_ROOT = CODE_ROOT / "bgolearn_rxn_extension"
TRAIN_FILE = DATASET_ROOT / "Suzuki" / "suzuki_train.csv"
CONFIG_FILE = EXTENSION_ROOT / "configs" / "suzuki.yaml"
ROLE_COLUMNS = ("Electrophile", "Nucleophile", "Ligand", "Base", "Solvent")
VARIANTS = (
    "gp_only",
    "interaction_only",
    "gp_interaction_gp_first",
    "full_ensemble",
)
BASELINE = "gp_only"
CORE_METRICS = (
    "initial_round_found_best",
    "best_found",
    "round_to_95_global_best",
    "AUC_best_so_far",
)

sys.path.insert(0, str(EXTENSION_ROOT / "src"))

from bgolearn_rxn.campaign import CampaignRunner  # noqa: E402
from bgolearn_rxn.config import load_run_config  # noqa: E402
from bgolearn_rxn.datasets import DatasetSpec, ReactionDataset  # noqa: E402
from bgolearn_rxn.metrics import compute_campaign_metrics  # noqa: E402
from bgolearn_rxn.oracle import OfflineTableOracle  # noqa: E402


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _covered(frame: pd.DataFrame, initial: np.ndarray, candidates: np.ndarray) -> bool:
    """Return whether every candidate level occurs in the initial observations."""
    for role in ROLE_COLUMNS:
        initial_levels = set(frame.loc[initial, role].astype(str))
        candidate_levels = set(frame.loc[candidates, role].astype(str))
        if not candidate_levels.issubset(initial_levels):
            return False
    return True


def _phase_random_episodes(
    frame: pd.DataFrame,
    *,
    phase: str,
    seed_start: int,
    count: int = 20,
) -> list[dict]:
    episodes: list[dict] = []
    source_seed = seed_start
    while len(episodes) < count:
        rng = np.random.default_rng(source_seed)
        initial = np.sort(rng.choice(len(frame), size=20, replace=False))
        candidates = np.setdiff1d(np.arange(len(frame)), initial)
        if _covered(frame, initial, candidates):
            episode_number = len(episodes)
            episodes.append(
                {
                    "episode": f"random_{phase}_{episode_number:02d}",
                    "episode_family": "random_holdout",
                    "phase": phase,
                    "split_seed": source_seed,
                    "campaign_seed": 41_000 + seed_start + episode_number,
                    "initial_indices": initial.tolist(),
                    "candidate_indices": candidates.tolist(),
                    "budget": 5,
                }
            )
        source_seed += 1
        if source_seed - seed_start > 1_000_000:
            raise RuntimeError(f"Could not construct {count} covered {phase} episodes.")
    return episodes


def make_random_episodes() -> list[dict]:
    """Build disjoint, deterministic screening and confirmation episodes."""
    frame = pd.read_csv(TRAIN_FILE)
    screening = _phase_random_episodes(
        frame, phase="screening", seed_start=110_000
    )
    confirmation = _phase_random_episodes(
        frame, phase="confirmation", seed_start=210_000
    )
    return screening + confirmation


def make_block_episodes() -> list[dict]:
    """Build leave-one-ligand/base-level-out cold-level diagnostics."""
    frame = pd.read_csv(TRAIN_FILE)
    episodes: list[dict] = []
    for family_index, role in enumerate(("Ligand", "Base")):
        values = frame[role].astype(str)
        eligible = sorted(
            level for level, n_rows in values.value_counts().items() if n_rows >= 2
        )
        for level_index, level in enumerate(eligible):
            candidates = np.flatnonzero(values.eq(level).to_numpy())
            initial = np.flatnonzero(~values.eq(level).to_numpy())
            episodes.append(
                {
                    "episode": f"block_{role.lower()}_{level_index:02d}",
                    "episode_family": f"leave_{role.lower()}_out",
                    "phase": "block_diagnostic",
                    "held_role": role,
                    "held_level": level,
                    "split_seed": None,
                    "campaign_seed": 510_000 + family_index * 1_000 + level_index,
                    "initial_indices": initial.tolist(),
                    "candidate_indices": candidates.tolist(),
                    "budget": min(5, len(candidates)),
                }
            )
    return episodes


def make_episodes(include_blocks: bool = True) -> list[dict]:
    episodes = make_random_episodes()
    if include_blocks:
        episodes.extend(make_block_episodes())
    return episodes


def configure_variant(base, variant: str, *, budget: int, seed: int):
    policy = replace(
        base.policy,
        early_end=budget,
        final_start=budget + 1,
        challenge_every=0,
        novelty_disagreement_mode="off",
    )
    if variant == "gp_only":
        enabled, first_query = ("gp",), "gp"
    elif variant == "interaction_only":
        enabled, first_query = ("interaction",), "interaction"
    elif variant == "gp_interaction_gp_first":
        enabled, first_query = ("gp", "interaction"), "gp"
    elif variant == "full_ensemble":
        enabled, first_query = base.ensemble.enabled_models, base.policy.first_query
    else:
        raise ValueError(f"Unknown variant: {variant}")
    return replace(
        base,
        dataset=replace(base.dataset, budget=budget, seeds=(seed,)),
        ensemble=replace(base.ensemble, enabled_models=enabled),
        policy=replace(policy, first_query=first_query),
    )


def build_problem(episode: dict, variant: str):
    source = pd.read_csv(TRAIN_FILE).reset_index(names="source_index")
    initial_indices = np.asarray(episode["initial_indices"], dtype=int)
    candidate_indices = np.asarray(episode["candidate_indices"], dtype=int)
    if set(initial_indices).intersection(candidate_indices):
        raise ValueError("Initial and candidate indices overlap.")
    if len(initial_indices) + len(candidate_indices) != len(source):
        raise ValueError("Episode indices must partition the Suzuki train table.")

    base = load_run_config(CONFIG_FILE)
    config = configure_variant(
        base,
        variant,
        budget=int(episode["budget"]),
        seed=int(episode["campaign_seed"]),
    )
    features = list(config.dataset.feature_columns)
    initial = source.loc[initial_indices, features + [config.dataset.target_column]].copy()
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


def _proxy_metrics(trajectory: list[dict], candidate_best: float, budget: int) -> dict:
    queried = [float(record["observed_yield"]) for record in trajectory]
    metrics = compute_campaign_metrics(queried, global_best=candidate_best).to_dict()
    t95 = metrics["round_to_95_global_best"]
    return {
        "initial_round_found_best": metrics["initial_round_found_best"],
        "best_found": metrics["best_found"],
        "round_to_95_global_best": budget + 1 if t95 is None else int(t95),
        "t95_reached": t95 is not None,
        "AUC_best_so_far": metrics["auc_best_so_far"],
    }


def run_one(job: tuple[str, dict, str]) -> dict:
    variant, episode, output_root_string = job
    config, dataset, lookup = build_problem(episode, variant)
    oracle = OfflineTableOracle(dataset=dataset, lookup=lookup)
    started = time.perf_counter()
    result = CampaignRunner(
        dataset,
        config,
        seed=int(episode["campaign_seed"]),
        backend="native",
    ).run(oracle, budget=int(episode["budget"]))
    elapsed = time.perf_counter() - started
    payload = result.to_payload()
    candidate_best = float(lookup[config.dataset.target_column].max())
    metrics = _proxy_metrics(payload["trajectory"], candidate_best, episode["budget"])
    payload["proxy_metrics"] = metrics
    payload["proxy_contract"] = {
        "train_only": True,
        "source_file": str(TRAIN_FILE.relative_to(DATASET_ROOT)),
        "source_sha256": file_sha256(TRAIN_FILE),
        "public_test_labels_used": False,
        "held_candidate_global_best": candidate_best,
        "episode": episode,
        "variant": variant,
    }
    output_path = (
        Path(output_root_string)
        / "trajectories"
        / variant
        / f"{episode['episode']}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "variant": variant,
        "episode": episode["episode"],
        "episode_family": episode["episode_family"],
        "phase": episode["phase"],
        "split_seed": episode["split_seed"],
        "campaign_seed": episode["campaign_seed"],
        "n_initial": len(episode["initial_indices"]),
        "n_candidates": len(episode["candidate_indices"]),
        "budget": episode["budget"],
        "candidate_global_best": candidate_best,
        **metrics,
        "elapsed_seconds": elapsed,
    }


def mean_ci(values: pd.Series) -> tuple[float, float, float, float]:
    array = values.to_numpy(dtype=float)
    mean = float(np.mean(array))
    if len(array) == 1:
        return mean, 0.0, mean, mean
    sd = float(np.std(array, ddof=1))
    half_width = float(t.ppf(0.975, len(array) - 1) * sd / np.sqrt(len(array)))
    return mean, sd, mean - half_width, mean + half_width


def summarize(rows: pd.DataFrame, group_columns: tuple[str, ...]) -> pd.DataFrame:
    records: list[dict] = []
    for group_key, group in rows.groupby(list(group_columns) + ["variant"], sort=True):
        keys = group_key if isinstance(group_key, tuple) else (group_key,)
        identifiers = dict(zip(group_columns + ("variant",), keys, strict=True))
        for metric in CORE_METRICS:
            mean, sd, low, high = mean_ci(group[metric])
            records.append(
                {
                    **identifiers,
                    "metric": metric,
                    "n_episodes": len(group),
                    "mean": mean,
                    "sample_sd": sd,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    return pd.DataFrame(records)


def paired_summary(rows: pd.DataFrame, group_columns: tuple[str, ...]) -> pd.DataFrame:
    records: list[dict] = []
    group_arg = list(group_columns) if group_columns else [lambda _: "all"]
    for group_key, section in rows.groupby(group_arg, sort=True):
        keys = group_key if isinstance(group_key, tuple) else (group_key,)
        identifiers = (
            dict(zip(group_columns, keys, strict=True)) if group_columns else {}
        )
        baseline = section[section["variant"].eq(BASELINE)].set_index("episode")
        for variant, candidate in section.groupby("variant", sort=True):
            if variant == BASELINE:
                continue
            candidate = candidate.set_index("episode")
            common = baseline.index.intersection(candidate.index)
            for metric in CORE_METRICS:
                delta = candidate.loc[common, metric] - baseline.loc[common, metric]
                mean, sd, low, high = mean_ci(delta)
                higher_is_better = metric != "round_to_95_global_best"
                wins = int((delta > 1e-10).sum()) if higher_is_better else int((delta < -1e-10).sum())
                losses = int((delta < -1e-10).sum()) if higher_is_better else int((delta > 1e-10).sum())
                records.append(
                    {
                        **identifiers,
                        "variant": variant,
                        "baseline": BASELINE,
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
    return pd.DataFrame(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=min(6, os.cpu_count() or 1))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--random-only", action="store_true")
    parser.add_argument("--limit-episodes", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            DATASET_ROOT
            / "submission"
            / "results"
            / "experimental_investigations"
            / "2.1-suzuki-holdout-proxy"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.jobs <= 0:
        raise ValueError("--jobs must be positive.")
    if len(args.variants) != len(set(args.variants)):
        raise ValueError("--variants must not contain duplicates.")
    episodes = make_episodes(include_blocks=not args.random_only)
    if args.limit_episodes is not None:
        if args.limit_episodes <= 0:
            raise ValueError("--limit-episodes must be positive.")
        episodes = episodes[: args.limit_episodes]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(TRAIN_FILE)
    manifest = {
        "train_only": True,
        "public_test_labels_used": False,
        "input_file": {
            "path": str(TRAIN_FILE.relative_to(DATASET_ROOT)),
            "sha256": file_sha256(TRAIN_FILE),
            "n_rows": len(frame),
        },
        "variants": args.variants,
        "episode_count": len(episodes),
        "episode_family_counts": pd.Series(
            [episode["episode_family"] for episode in episodes]
        ).value_counts().sort_index().to_dict(),
        "phase_counts": pd.Series(
            [episode["phase"] for episode in episodes]
        ).value_counts().sort_index().to_dict(),
        "random_split_contract": {
            "n_initial": 20,
            "n_candidates": 9,
            "budget": 5,
            "candidate_levels_must_occur_in_initial": list(ROLE_COLUMNS),
            "screening_seed_start": 110_000,
            "confirmation_seed_start": 210_000,
        },
        "t95_miss_penalty": "episode budget + 1",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    jobs = [
        (variant, episode, str(args.output_dir))
        for variant in args.variants
        for episode in episodes
    ]
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(run_one, job): job for job in jobs}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"{row['variant']} {row['episode']}: "
                f"first={row['initial_round_found_best']:.3f}, "
                f"best={row['best_found']:.3f}, "
                f"t95={row['round_to_95_global_best']}, "
                f"AUC={row['AUC_best_so_far']:.3f}",
                flush=True,
            )

    detail = pd.DataFrame(rows).sort_values(["phase", "episode", "variant"])
    detail.to_csv(args.output_dir / "per_episode.csv", index=False)
    summarize(detail, ()).to_csv(args.output_dir / "aggregate_overall.csv", index=False)
    summarize(detail, ("episode_family",)).to_csv(
        args.output_dir / "aggregate_by_family.csv", index=False
    )
    summarize(detail, ("phase",)).to_csv(
        args.output_dir / "aggregate_by_phase.csv", index=False
    )
    paired_summary(detail, ()).to_csv(
        args.output_dir / "paired_vs_gp_overall.csv", index=False
    )
    paired_summary(detail, ("episode_family",)).to_csv(
        args.output_dir / "paired_vs_gp_by_family.csv", index=False
    )
    paired_summary(detail, ("phase",)).to_csv(
        args.output_dir / "paired_vs_gp_by_phase.csv", index=False
    )
    auc = summarize(detail, ("phase",))
    print("\nAUC by phase:")
    print(auc[auc["metric"].eq("AUC_best_so_far")].to_string(index=False))


if __name__ == "__main__":
    main()
