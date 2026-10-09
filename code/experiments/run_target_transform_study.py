#!/usr/bin/env python3
"""Train-only standard-versus-logit target-transform study."""

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


TRANSFORMS = ("standard", "logit")
BASELINE = "standard"
INPUT_FILES = {
    "Buchwald_sub4": DATASET_ROOT / "Buchwald_sub4" / "buchwald_sub4_train.csv",
    "Suzuki": DATASET_ROOT / "Suzuki" / "suzuki_train.csv",
}
CORE_METRICS = (
    "initial_round_found_best",
    "best_found",
    "round_to_95_global_best",
    "AUC_best_so_far",
)
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
            raise ValueError("Sorted Buchwald product index 1 is not the official target.")
        buchwald.append(item)
    suzuki = suzuki_proxy.make_random_episodes() + suzuki_proxy.make_block_episodes()
    return {"Buchwald_sub4": buchwald, "Suzuki": suzuki}


def build_problem(dataset_name: str, episode: dict, transform: str):
    if transform not in TRANSFORMS:
        raise ValueError(f"Unknown transform: {transform}")
    if dataset_name == "Buchwald_sub4":
        config, dataset, lookup = transfer_proxy.build_problem(
            dataset_name, "low_rank_task_gp", episode
        )
        expected_model = "low_rank_task_gp"
    elif dataset_name == "Suzuki":
        config, dataset, lookup = suzuki_proxy.build_problem(episode, "gp_only")
        expected_model = "gp"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    config = replace(
        config,
        gp=replace(config.gp, target_transform=transform),
    )
    if tuple(config.ensemble.enabled_models) != (expected_model,):
        raise ValueError("Transform study did not preserve the incumbent GP model.")
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
    dataset_name, transform, episode, output_root_string = job
    config, dataset, lookup = build_problem(dataset_name, episode, transform)
    oracle = OfflineTableOracle(dataset=dataset, lookup=lookup)
    seed = int(episode.get("seed", episode.get("campaign_seed")))
    started = time.perf_counter()
    runner = CampaignRunner(dataset, config, seed=seed, backend="native")
    result = runner.run(oracle, budget=int(episode["budget"]))
    elapsed = time.perf_counter() - started
    payload = result.to_payload()

    if dataset_name == "Buchwald_sub4":
        # The seven official-target train rows define the exhaustive proxy.
        global_best = float(oracle.global_best)
    else:
        # Match the established Suzuki holdout evaluator's held-pool contract.
        global_best = float(lookup[dataset.spec.target_column].max())
    metrics = _metrics_from_trajectory(
        payload["trajectory"], global_best=global_best, budget=int(episode["budget"])
    )
    payload["transform_metrics"] = metrics
    payload["transform_contract"] = {
        "target_transform": transform,
        "train_only": True,
        "public_test_labels_used": False,
        "input_file": str(INPUT_FILES[dataset_name].relative_to(DATASET_ROOT)),
        "input_sha256": file_sha256(INPUT_FILES[dataset_name]),
        "global_best_contract": (
            "all_seven_official_target_rows"
            if dataset_name == "Buchwald_sub4"
            else "held_candidate_pool"
        ),
        "global_best": global_best,
        "episode": episode,
    }
    output_path = (
        Path(output_root_string)
        / "trajectories"
        / dataset_name
        / episode["phase"]
        / transform
        / f"{episode['episode']}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "dataset": dataset_name,
        "phase": episode["phase"],
        "episode_family": episode["episode_family"],
        "episode": episode["episode"],
        "target_task": episode.get("target_task"),
        "transform": transform,
        "budget": episode["budget"],
        "n_initial": len(dataset.initial),
        "n_candidates": len(dataset.candidates),
        "global_best": global_best,
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
    for (dataset, phase, family, transform), group in rows.groupby(
        ["dataset", "phase", "episode_family", "transform"], sort=True
    ):
        for metric in CORE_METRICS:
            mean, sd, low, high = mean_ci(group[metric])
            records.append(
                {
                    "dataset": dataset,
                    "phase": phase,
                    "episode_family": family,
                    "transform": transform,
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
        baseline = section[section["transform"].eq(BASELINE)].set_index("episode")
        candidate = section[section["transform"].eq("logit")].set_index("episode")
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
                    "candidate": "logit",
                    "baseline": BASELINE,
                    "metric": metric,
                    "n_pairs": len(common),
                    "mean_delta_logit_minus_standard": mean,
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
            / "4.1-target-transform-study"
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
            dataset: values[: args.limit_episodes] for dataset, values in episodes.items()
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "train_only": True,
        "public_test_labels_used": False,
        "transforms": list(TRANSFORMS),
        "transform_parameters_selected": False,
        "acquisition": "posterior mean query 1, stable LogEI thereafter",
        "models": {
            "Buchwald_sub4": "low_rank_task_gp",
            "Suzuki": "structured_anova_gp",
        },
        "official_buchwald_target": OFFICIAL_BUCHWALD_PRODUCT,
        "inputs": {
            dataset: {
                "path": str(path.relative_to(DATASET_ROOT)),
                "sha256": file_sha256(path),
            }
            for dataset, path in INPUT_FILES.items()
        },
        "episode_counts": {dataset: len(values) for dataset, values in episodes.items()},
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
        (dataset, transform, episode, str(args.output_dir))
        for dataset, values in episodes.items()
        for transform in TRANSFORMS
        for episode in values
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(run_one, job): job for job in jobs}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"{row['dataset']} {row['phase']} {row['transform']} "
                f"{row['episode']}: AUC={row['AUC_best_so_far']:.3f}",
                flush=True,
            )
    detail = pd.DataFrame(rows).sort_values(
        ["dataset", "phase", "episode", "transform"]
    )
    detail.to_csv(args.output_dir / "per_episode.csv", index=False)
    summarize(detail).to_csv(args.output_dir / "summary.csv", index=False)
    paired_summary(detail).to_csv(args.output_dir / "paired_logit_vs_standard.csv", index=False)


if __name__ == "__main__":
    main()
