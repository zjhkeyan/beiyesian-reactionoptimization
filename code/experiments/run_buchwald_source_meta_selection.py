#!/usr/bin/env python3
"""Source-only episodic selection of Buchwald component/task GP priors."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass, replace
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t


EXPERIMENT_ROOT = Path(__file__).resolve().parent
CODE_ROOT = EXPERIMENT_ROOT.parent
DATASET_ROOT = CODE_ROOT.parents[1]
EXTENSION_ROOT = CODE_ROOT / "bgolearn_rxn_extension"
SESSION_ROOT = DATASET_ROOT / ".arbor" / "sessions" / "bgolearn_frontier_20260728"
sys.path.insert(0, str(EXTENSION_ROOT / "src"))

from bgolearn_rxn.campaign import CampaignRunner  # noqa: E402
from bgolearn_rxn.config import load_run_config  # noqa: E402
from bgolearn_rxn.datasets import DatasetSpec, ReactionDataset  # noqa: E402
from bgolearn_rxn.metrics import compute_campaign_metrics  # noqa: E402
from bgolearn_rxn.oracle import OfflineTableOracle  # noqa: E402
from bgolearn_rxn.surrogates.component_ard_gp import (  # noqa: E402
    ComponentARDLowRankTaskGP,
)


TRAIN_FILE = DATASET_ROOT / "Buchwald_sub4" / "buchwald_sub4_train.csv"
CONFIG_FILE = EXTENSION_ROOT / "configs" / "buchwald_sub4_metric_frontier.yaml"
MODEL_FILE = (
    EXTENSION_ROOT
    / "src"
    / "bgolearn_rxn"
    / "surrogates"
    / "component_ard_gp.py"
)
FACTORY_FILE = EXTENSION_ROOT / "src" / "bgolearn_rxn" / "factory.py"
OFFICIAL_TARGET = "N-(4-ethylphenyl)-4-methylaniline"
MODEL_NAME = "component_ard_low_rank_task_gp"
CORE_METRICS = (
    "initial_round_found_best",
    "best_found",
    "round_to_95_global_best",
    "AUC_best_so_far",
)


@dataclass(frozen=True)
class PriorSetting:
    name: str
    amplitude_prior_scale: float
    task_prior_scale: float


# Predeclared before any source episode is evaluated. Values bracket the frozen
# phase-2 defaults with progressively stronger MAP shrinkage for the two most
# weakly identified parameter groups in a 28-row source-only fit.
PRIOR_GRID = (
    PriorSetting("frontier_default", 0.60, 0.35),
    PriorSetting("strong_component", 0.35, 0.35),
    PriorSetting("strong_task", 0.60, 0.20),
    PriorSetting("strong_joint", 0.35, 0.20),
    PriorSetting("weak_joint", 0.90, 0.55),
)
SETTING_BY_NAME = {setting.name: setting for setting in PRIOR_GRID}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def product_rows() -> tuple[pd.DataFrame, list[str]]:
    frame = pd.read_csv(TRAIN_FILE).reset_index(names="source_index")
    products = sorted(frame["Product"].astype(str).unique())
    if OFFICIAL_TARGET not in products or len(products) != 5:
        raise ValueError("Unexpected Buchwald product topology.")
    return frame, products


def make_pair_episodes(target_product: str, *, phase: str) -> list[dict]:
    frame, products = product_rows()
    if target_product not in products:
        raise ValueError(f"Unknown Buchwald product: {target_product}")
    indices = frame.index[frame["Product"].astype(str).eq(target_product)].to_numpy()
    episodes = []
    product_index = products.index(target_product)
    for pair_index, pair in enumerate(combinations(indices.tolist(), 2)):
        initial = np.asarray(pair, dtype=int)
        candidates = np.setdiff1d(indices, initial)
        episodes.append(
            {
                "episode": f"product_{product_index}_pair_{pair[0]}_{pair[1]}",
                "phase": phase,
                "target_task": target_product,
                "initial_indices": initial.tolist(),
                "candidate_indices": candidates.tolist(),
                "budget": min(3, len(candidates)),
                # Match the established exhaustive Buchwald proxy exactly so
                # the frontier-default confirmation is bitwise comparable.
                "seed": 50_000 + product_index * 1_000 + pair_index,
            }
        )
    if len(episodes) != 21:
        raise ValueError("Each Buchwald product must yield exactly 21 initial pairs.")
    return episodes


def make_source_meta_episodes() -> list[dict]:
    _, products = product_rows()
    episodes = []
    for product in products:
        if product != OFFICIAL_TARGET:
            episodes.extend(make_pair_episodes(product, phase="source_meta_selection"))
    return episodes


def make_official_confirmation_episodes() -> list[dict]:
    return make_pair_episodes(OFFICIAL_TARGET, phase="official_target_confirmation")


def configure(episode: dict):
    base = load_run_config(CONFIG_FILE)
    return replace(
        base,
        dataset=replace(
            base.dataset,
            target_task_value=episode["target_task"],
            budget=int(episode["budget"]),
            seeds=(int(episode["seed"]),),
        ),
        gp=replace(base.gp, target_transform="standard"),
        ensemble=replace(
            base.ensemble,
            enabled_models=(MODEL_NAME,),
            initial_weights={MODEL_NAME: 1.0},
        ),
        policy=replace(
            base.policy,
            first_query=MODEL_NAME,
            early_end=int(episode["budget"]),
            final_start=int(episode["budget"]) + 1,
            challenge_every=0,
            novelty_disagreement_mode="off",
        ),
    )


def build_problem(episode: dict, *, strict_source_isolation: bool):
    frame, products = product_rows()
    config = configure(episode)
    initial_indices = np.asarray(episode["initial_indices"], dtype=int)
    candidate_indices = np.asarray(episode["candidate_indices"], dtype=int)
    feature_columns = list(config.dataset.feature_columns)

    if strict_source_isolation:
        if episode["target_task"] == OFFICIAL_TARGET:
            raise ValueError("The official target cannot be used for meta-selection.")
        allowed = frame.index[~frame["Product"].astype(str).eq(OFFICIAL_TARGET)].to_numpy()
        observed_indices = np.setdiff1d(allowed, candidate_indices)
    else:
        if episode["target_task"] != OFFICIAL_TARGET:
            raise ValueError("Confirmation is reserved for the official target.")
        observed_indices = np.setdiff1d(frame.index.to_numpy(), candidate_indices)

    initial = frame.loc[
        observed_indices, feature_columns + [config.dataset.target_column]
    ].copy()
    candidates = frame.loc[candidate_indices, feature_columns].copy()
    candidates.insert(
        0,
        "query_index",
        frame.loc[candidate_indices, "source_index"].astype(int),
    )
    lookup = frame.loc[
        candidate_indices, feature_columns + [config.dataset.target_column]
    ].copy()
    lookup.insert(
        0,
        "query_index",
        frame.loc[candidate_indices, "source_index"].astype(int),
    )

    if strict_source_isolation:
        used_products = set(initial["Product"].astype(str)) | set(
            candidates["Product"].astype(str)
        )
        if OFFICIAL_TARGET in used_products:
            raise AssertionError("Official-target rows entered source meta-selection.")
        if len(initial) != 23:
            raise AssertionError("Source meta-selection must use 3x7 auxiliary + 2 target rows.")
    elif len(initial) != 30:
        raise AssertionError("Official confirmation must use 4x7 source + 2 target rows.")

    dataset = ReactionDataset(
        spec=DatasetSpec.from_config(config.dataset),
        initial=initial,
        candidates=candidates,
    )
    return config, dataset, lookup, sorted(set(initial["Product"].astype(str)))


def _campaign_metrics(trajectory: list[dict], global_best: float, budget: int) -> dict:
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


def run_one(job: tuple[dict, str, bool, str]) -> dict:
    episode, setting_name, strict_source_isolation, output_root_string = job
    setting = SETTING_BY_NAME[setting_name]
    config, dataset, lookup, observed_products = build_problem(
        episode, strict_source_isolation=strict_source_isolation
    )
    oracle = OfflineTableOracle(dataset=dataset, lookup=lookup)
    started = time.perf_counter()
    runner = CampaignRunner(
        dataset, config, seed=int(episode["seed"]), backend="native"
    )

    # Experiment-local injection: production classes/configuration remain unchanged.
    tuned_model = ComponentARDLowRankTaskGP(
        schema=runner.prepared.encoder.schema,
        config=config.gp,
        seed=int(episode["seed"]) + 79,
    )
    tuned_model.amplitude_prior_scale = setting.amplitude_prior_scale
    tuned_model.task_prior_scale = setting.task_prior_scale
    runner.ensemble.models[MODEL_NAME] = tuned_model

    result = runner.run(oracle, budget=int(episode["budget"]))
    elapsed = time.perf_counter() - started
    payload = result.to_payload()
    metrics = _campaign_metrics(
        payload["trajectory"], float(oracle.global_best), int(episode["budget"])
    )
    payload["source_meta_contract"] = {
        "train_only": True,
        "public_test_labels_used": False,
        "strict_source_isolation": strict_source_isolation,
        "official_target_rows_used": not strict_source_isolation,
        "official_target_role": (
            "confirmation_target" if not strict_source_isolation else "excluded"
        ),
        "observed_products": observed_products,
        "setting": asdict(setting),
        "episode": episode,
        "metrics": metrics,
    }
    output_path = (
        Path(output_root_string)
        / "trajectories"
        / episode["phase"]
        / setting_name
        / f"{episode['episode']}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "phase": episode["phase"],
        "episode": episode["episode"],
        "target_task": episode["target_task"],
        "setting": setting_name,
        "amplitude_prior_scale": setting.amplitude_prior_scale,
        "task_prior_scale": setting.task_prior_scale,
        "budget": int(episode["budget"]),
        "global_best": float(oracle.global_best),
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
    for (phase, setting), group in rows.groupby(["phase", "setting"], sort=True):
        for metric in CORE_METRICS:
            mean, sd, low, high = mean_ci(group[metric])
            records.append(
                {
                    "phase": phase,
                    "setting": setting,
                    "metric": metric,
                    "n_episodes": len(group),
                    "mean": mean,
                    "sample_sd": sd,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    return pd.DataFrame(records)


def select_setting(meta_rows: pd.DataFrame) -> tuple[str, dict]:
    auc = meta_rows.pivot(index="episode", columns="setting", values="AUC_best_so_far")
    overall = auc.mean(axis=0).sort_values(ascending=False, kind="stable")
    product_means = meta_rows.pivot_table(
        index="target_task", columns="setting", values="AUC_best_so_far", aggfunc="mean"
    )
    products = sorted(product_means.index)
    loo_winners = {}
    for held_out in products:
        train_products = [product for product in products if product != held_out]
        means = product_means.loc[train_products].mean(axis=0)
        loo_winners[held_out] = str(means.sort_values(ascending=False, kind="stable").index[0])

    correlations = []
    for left, right in combinations(products, 2):
        correlation = spearmanr(product_means.loc[left], product_means.loc[right]).statistic
        correlations.append(float(correlation))
    finite_correlations = [value for value in correlations if np.isfinite(value)]
    selected = str(overall.index[0])
    stability = {
        "selection_metric": "macro/micro mean source-only AUC_best_so_far (21 episodes per product)",
        "selected_setting": selected,
        "overall_mean_auc": {name: float(value) for name, value in overall.items()},
        "per_product_mean_auc": {
            product: {name: float(value) for name, value in row.items()}
            for product, row in product_means.to_dict(orient="index").items()
        },
        "leave_one_product_out_winners": loo_winners,
        "selected_in_leave_one_product_out_folds": int(
            sum(winner == selected for winner in loo_winners.values())
        ),
        "mean_pairwise_product_rank_spearman": (
            float(np.mean(finite_correlations)) if finite_correlations else None
        ),
    }
    return selected, stability


def paired_summary(confirmation: pd.DataFrame, selected: str) -> pd.DataFrame:
    baseline = confirmation[
        confirmation["setting"].eq("frontier_default")
    ].set_index("episode")
    candidate = confirmation[confirmation["setting"].eq(selected)].set_index("episode")
    common = baseline.index.intersection(candidate.index)
    records = []
    for metric in CORE_METRICS:
        delta = candidate.loc[common, metric] - baseline.loc[common, metric]
        mean, sd, low, high = mean_ci(delta)
        higher = metric != "round_to_95_global_best"
        wins = int((delta > 1e-10).sum()) if higher else int((delta < -1e-10).sum())
        losses = int((delta < -1e-10).sum()) if higher else int((delta > 1e-10).sum())
        records.append(
            {
                "candidate": selected,
                "baseline": "frontier_default",
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


def deterministic_twenty_run_slice(confirmation: pd.DataFrame) -> pd.DataFrame:
    """Return the first 20 lexicographic episodes for every compared setting."""

    episode_ids = sorted(confirmation["episode"].unique())[:20]
    sliced = confirmation[confirmation["episode"].isin(episode_ids)].copy()
    expected = 20 * confirmation["setting"].nunique()
    if len(sliced) != expected:
        raise ValueError("The deterministic 20-run slice is incomplete.")
    return sliced.sort_values(["episode", "setting"])


def run_jobs(jobs: list[tuple[dict, str, bool, str]], workers: int) -> pd.DataFrame:
    # ProcessPoolExecutor is imported lazily so module import remains cheap in tests.
    from concurrent.futures import ProcessPoolExecutor, as_completed

    rows = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_one, job): job for job in jobs}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"{row['phase']} {row['setting']} {row['episode']}: "
                f"AUC={row['AUC_best_so_far']:.3f}",
                flush=True,
            )
    return pd.DataFrame(rows).sort_values(["phase", "episode", "setting"])


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
            / "3"
            / "results"
            / "3-source-only-meta-selection"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.jobs <= 0:
        raise ValueError("--jobs must be positive.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta_episodes = make_source_meta_episodes()
    if args.smoke:
        meta_episodes = meta_episodes[:1]
        settings = PRIOR_GRID[:2]
    else:
        settings = PRIOR_GRID

    manifest = {
        "study": "Buchwald source-only episodic GP-prior meta-selection",
        "train_only": True,
        "public_test_labels_used": False,
        "selection_target_labels_used": False,
        "selection_excludes_official_target_rows": True,
        "official_target": OFFICIAL_TARGET,
        "predeclared_prior_grid": [asdict(setting) for setting in PRIOR_GRID],
        "selection_rule": "highest mean AUC_best_so_far over all 84 source-only episodes",
        "tie_break": "predeclared grid order via stable sort",
        "acquisition": "posterior mean query 1, stable LogEI thereafter",
        "target_transform": "standard",
        "default_behavior_expected": {
            "amplitude_prior_scale": 0.60,
            "task_prior_scale": 0.35,
            "task_noise_enabled": False,
            "model": MODEL_NAME,
        },
        "inputs": {
            "train": {"path": str(TRAIN_FILE), "sha256": file_sha256(TRAIN_FILE)},
            "config": {"path": str(CONFIG_FILE), "sha256": file_sha256(CONFIG_FILE)},
            "model": {"path": str(MODEL_FILE), "sha256": file_sha256(MODEL_FILE)},
            "factory": {"path": str(FACTORY_FILE), "sha256": file_sha256(FACTORY_FILE)},
        },
        "meta_episode_count": len(meta_episodes),
        "official_confirmation_episode_count": 0 if args.smoke else 21,
        "python": platform.python_version(),
        "smoke": bool(args.smoke),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    meta_jobs = [
        (episode, setting.name, True, str(args.output_dir))
        for setting in settings
        for episode in meta_episodes
    ]
    meta_rows = run_jobs(meta_jobs, args.jobs)
    meta_rows.to_csv(args.output_dir / "source_meta_per_episode.csv", index=False)
    summarize(meta_rows).to_csv(args.output_dir / "source_meta_summary.csv", index=False)
    selected, stability = select_setting(meta_rows)
    (args.output_dir / "selection.json").write_text(
        json.dumps(stability, indent=2, sort_keys=True), encoding="utf-8"
    )
    paired_summary(meta_rows, selected).to_csv(
        args.output_dir / "source_meta_paired_selected_vs_frontier.csv", index=False
    )

    if args.smoke:
        return

    # Selection is now frozen. No official-target episode is used above this line.
    confirmation_episodes = make_official_confirmation_episodes()
    confirmation_settings = ["frontier_default"]
    if selected != "frontier_default":
        confirmation_settings.append(selected)
    confirmation_jobs = [
        (episode, setting_name, False, str(args.output_dir))
        for setting_name in confirmation_settings
        for episode in confirmation_episodes
    ]
    confirmation = run_jobs(confirmation_jobs, args.jobs)
    confirmation.to_csv(args.output_dir / "official_confirmation_per_episode.csv", index=False)
    summarize(confirmation).to_csv(
        args.output_dir / "official_confirmation_summary.csv", index=False
    )
    paired_summary(confirmation, selected).to_csv(
        args.output_dir / "paired_selected_vs_frontier.csv", index=False
    )
    twenty_runs = deterministic_twenty_run_slice(confirmation)
    twenty_runs.to_csv(
        args.output_dir / "official_confirmation_20run_per_run.csv", index=False
    )
    summarize(twenty_runs).to_csv(
        args.output_dir / "official_confirmation_20run_summary.csv", index=False
    )
    paired_summary(twenty_runs, selected).to_csv(
        args.output_dir / "official_confirmation_20run_paired.csv", index=False
    )

    prior_phase2 = pd.read_csv(
        DATASET_ROOT
        / "submission"
        / "results"
        / "experimental_investigations"
        / "5.1-component-ard-study"
        / "per_episode.csv"
    )
    prior_phase2 = prior_phase2[
        prior_phase2["dataset"].eq("Buchwald_sub4")
        & prior_phase2["variant"].eq("component_ard")
    ].set_index("episode")
    fresh_default = confirmation[
        confirmation["setting"].eq("frontier_default")
    ].set_index("episode")
    common = prior_phase2.index.intersection(fresh_default.index)
    baseline_deltas = {
        metric: float(
            np.max(
                np.abs(
                    fresh_default.loc[common, metric].to_numpy(dtype=float)
                    - prior_phase2.loc[common, metric].to_numpy(dtype=float)
                )
            )
        )
        for metric in CORE_METRICS
    }
    baseline_check = {
        "prior_phase2_path": str(
            DATASET_ROOT
            / "submission"
            / "results"
            / "experimental_investigations"
            / "5.1-component-ard-study"
            / "per_episode.csv"
        ),
        "n_common_episodes": len(common),
        "max_abs_metric_differences": baseline_deltas,
        "exact_metric_match": all(value <= 1e-12 for value in baseline_deltas.values()),
    }
    (args.output_dir / "baseline_reproducibility.json").write_text(
        json.dumps(baseline_check, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
