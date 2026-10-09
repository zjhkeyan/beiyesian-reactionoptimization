#!/usr/bin/env python3
"""Train-only fixed-component versus MAP component-ARD GP study."""

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
from bgolearn_rxn.metrics import compute_campaign_metrics  # noqa: E402
from bgolearn_rxn.oracle import OfflineTableOracle  # noqa: E402


VARIANTS = ("fixed", "component_ard")
BASELINE = "fixed"
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
OFFICIAL_BUCHWALD_PRODUCT = "N-(4-ethylphenyl)-4-methylaniline"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_episodes() -> dict[str, list[dict]]:
    buchwald = []
    for episode in transfer_proxy.make_buchwald_exhaustive_pair_episodes(1):
        item = dict(episode)
        item["phase"] = "official_target_exhaustive"
        if item["target_task"] != OFFICIAL_BUCHWALD_PRODUCT:
            raise ValueError("Sorted product index 1 is not the official target.")
        buchwald.append(item)
    return {
        "Buchwald_sub4": buchwald,
        "Suzuki": suzuki_proxy.make_random_episodes() + suzuki_proxy.make_block_episodes(),
    }


def build_problem(dataset_name: str, episode: dict, variant: str):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    if dataset_name == "Buchwald_sub4":
        config, dataset, lookup = transfer_proxy.build_problem(
            dataset_name, "low_rank_task_gp", episode
        )
        fixed_model = "low_rank_task_gp"
        ard_model = "component_ard_low_rank_task_gp"
    elif dataset_name == "Suzuki":
        config, dataset, lookup = suzuki_proxy.build_problem(episode, "gp_only")
        fixed_model = "gp"
        ard_model = "component_ard_gp"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    model = fixed_model if variant == "fixed" else ard_model
    config = replace(
        config,
        gp=replace(config.gp, target_transform="standard"),
        ensemble=replace(
            config.ensemble,
            enabled_models=(model,),
            initial_weights={model: 1.0},
        ),
        policy=replace(config.policy, first_query=model),
    )
    return config, dataset, lookup, model


def _campaign_metrics(trajectory, global_best: float, budget: int) -> dict:
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
    config, dataset, lookup, model_name = build_problem(dataset_name, episode, variant)
    oracle = OfflineTableOracle(dataset=dataset, lookup=lookup)
    seed = int(episode.get("seed", episode.get("campaign_seed")))
    started = time.perf_counter()
    runner = CampaignRunner(dataset, config, seed=seed, backend="native")
    result = runner.run(oracle, budget=int(episode["budget"]))
    elapsed = time.perf_counter() - started
    payload = result.to_payload()
    global_best = (
        float(oracle.global_best)
        if dataset_name == "Buchwald_sub4"
        else float(lookup[dataset.spec.target_column].max())
    )
    metrics = _campaign_metrics(payload["trajectory"], global_best, episode["budget"])
    model = runner.ensemble.models[model_name]
    amplitudes = dict(getattr(model, "component_amplitudes_", {}))
    shares = dict(getattr(model, "component_shares_", {}))
    payload["ard_metrics"] = metrics
    payload["ard_contract"] = {
        "variant": variant,
        "model": model_name,
        "target_transform": config.gp.target_transform,
        "train_only": True,
        "public_test_labels_used": False,
        "input_file": str(INPUT_FILES[dataset_name].relative_to(DATASET_ROOT)),
        "input_sha256": file_sha256(INPUT_FILES[dataset_name]),
        "component_amplitudes": amplitudes or None,
        "component_shares": shares or None,
        "amplitude_prior_scale": getattr(model, "amplitude_prior_scale", None),
        "episode": episode,
    }
    output_path = (
        Path(output_root_string)
        / "trajectories"
        / dataset_name
        / episode["phase"]
        / variant
        / f"{episode['episode']}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "dataset": dataset_name,
        "phase": episode["phase"],
        "episode_family": episode["episode_family"],
        "episode": episode["episode"],
        "variant": variant,
        "model": model_name,
        "budget": episode["budget"],
        "global_best": global_best,
        **metrics,
        "elapsed_seconds": elapsed,
        "component_amplitudes_json": json.dumps(amplitudes, sort_keys=True) if amplitudes else None,
        "component_shares_json": json.dumps(shares, sort_keys=True) if shares else None,
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
        candidate = section[section["variant"].eq("component_ard")].set_index("episode")
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
                    "candidate": "component_ard",
                    "baseline": BASELINE,
                    "metric": metric,
                    "n_pairs": len(common),
                    "mean_delta_ard_minus_fixed": mean,
                    "sample_sd_delta": sd,
                    "ci95_low": low,
                    "ci95_high": high,
                    "wins": wins,
                    "ties": len(common) - wins - losses,
                    "losses": losses,
                }
            )
    return pd.DataFrame(records)


def amplitude_summary(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    ard = rows[rows["variant"].eq("component_ard")]
    for (dataset, phase, family), group in ard.groupby(
        ["dataset", "phase", "episode_family"], sort=True
    ):
        parsed = [json.loads(value) for value in group["component_shares_json"]]
        names = sorted(set().union(*(values.keys() for values in parsed)))
        for name in names:
            values = np.asarray([item[name] for item in parsed], dtype=float)
            records.append(
                {
                    "dataset": dataset,
                    "phase": phase,
                    "episode_family": family,
                    "component": name,
                    "n_episodes": len(values),
                    "mean_share": float(values.mean()),
                    "sample_sd_share": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "min_share": float(values.min()),
                    "max_share": float(values.max()),
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
            / "5.1-component-ard-study"
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
        episodes = {dataset: values[: args.limit_episodes] for dataset, values in episodes.items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "train_only": True,
        "public_test_labels_used": False,
        "variants": list(VARIANTS),
        "target_transform": "standard",
        "acquisition": "posterior mean query 1, stable LogEI thereafter",
        "amplitude_prior_scale": 0.60,
        "amplitude_parameter_sweep": False,
        "shared_bias_amplitude": 1.0,
        "models": {
            "Buchwald_sub4": ["low_rank_task_gp", "component_ard_low_rank_task_gp"],
            "Suzuki": ["structured_anova_gp", "component_ard_gp"],
        },
        "official_buchwald_target": OFFICIAL_BUCHWALD_PRODUCT,
        "inputs": {
            dataset: {"path": str(path.relative_to(DATASET_ROOT)), "sha256": file_sha256(path)}
            for dataset, path in INPUT_FILES.items()
        },
        "episode_counts": {dataset: len(values) for dataset, values in episodes.items()},
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
    detail = pd.DataFrame(rows).sort_values(["dataset", "phase", "episode", "variant"])
    detail.to_csv(args.output_dir / "per_episode.csv", index=False)
    summarize(detail).to_csv(args.output_dir / "summary.csv", index=False)
    paired_summary(detail).to_csv(args.output_dir / "paired_ard_vs_fixed.csv", index=False)
    amplitude_summary(detail).to_csv(args.output_dir / "amplitude_stability.csv", index=False)


if __name__ == "__main__":
    main()
