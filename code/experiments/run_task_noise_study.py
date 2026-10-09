#!/usr/bin/env python3
"""Staged train-only evaluation of task-conditioned Buchwald GP noise."""

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

import run_transfer_proxy as transfer_proxy  # noqa: E402
from bgolearn_rxn.campaign import CampaignRunner  # noqa: E402
from bgolearn_rxn.metrics import compute_campaign_metrics  # noqa: E402
from bgolearn_rxn.oracle import OfflineTableOracle  # noqa: E402


BASELINE = "component_ard"
CANDIDATE = "component_ard_task_noise"
VARIANTS = (BASELINE, CANDIDATE)
MODEL_NAMES = {
    BASELINE: "component_ard_low_rank_task_gp",
    CANDIDATE: "component_ard_low_rank_task_noise_gp",
}
CORE_METRICS = (
    "initial_round_found_best",
    "best_found",
    "round_to_95_global_best",
    "AUC_best_so_far",
)
SUMMARY_METRICS = CORE_METRICS + ("elapsed_seconds",)
TRAIN_FILE = DATASET_ROOT / "Buchwald_sub4" / "buchwald_sub4_train.csv"
OFFICIAL_TARGET = "N-(4-ethylphenyl)-4-methylaniline"
TASK_NOISE_DESIGN = {
    "parameterization": "orthonormal Helmert log-noise contrasts",
    "identifiability": "sum of task log-noise deviations equals zero",
    "global_noise_meaning": "geometric mean task noise",
    "contrast_prior": "independent zero-mean Gaussian",
    "contrast_prior_scale": 0.35,
    "contrast_bounds": [-1.25, 1.25],
    "global_noise_bounds": [1e-4, 0.75],
    "parameter_sweep": False,
}
PILOT_GATE = {
    "minimum_auc_delta": 0.0,
    "minimum_initial_delta": -0.5,
    "minimum_best_delta": -0.5,
    "maximum_t95_delta": 0.25,
    "contrast_bound_margin": 1e-3,
    "global_noise_bound_fraction_margin": 0.01,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_pilot_episodes() -> list[dict]:
    """Use auxiliary-product targets so official confirmation stays untouched."""
    episodes = []
    for episode in transfer_proxy.make_buchwald_episodes(2):
        if episode["target_task"] == OFFICIAL_TARGET:
            continue
        item = dict(episode)
        item["phase"] = "auxiliary_product_pilot"
        episodes.append(item)
    if len(episodes) != 8:
        raise RuntimeError("Expected eight auxiliary-product pilot episodes.")
    return episodes


def make_confirmation_episodes() -> list[dict]:
    episodes = []
    for episode in transfer_proxy.make_buchwald_exhaustive_pair_episodes(1):
        item = dict(episode)
        item["phase"] = "official_target_confirmation"
        if item["target_task"] != OFFICIAL_TARGET:
            raise ValueError("Sorted Buchwald product index 1 is not official target.")
        episodes.append(item)
    return episodes


def episodes_for_stage(stage: str) -> list[dict]:
    if stage == "smoke":
        return make_pilot_episodes()[:2]
    if stage == "pilot":
        return make_pilot_episodes()
    if stage == "confirmation":
        return make_confirmation_episodes()
    raise ValueError(f"Unknown stage: {stage}")


def build_problem(episode: dict, variant: str):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown task-noise variant: {variant}")
    config, dataset, lookup = transfer_proxy.build_problem(
        "Buchwald_sub4", "low_rank_task_gp", episode
    )
    model_name = MODEL_NAMES[variant]
    config = replace(
        config,
        gp=replace(config.gp, target_transform="standard"),
        ensemble=replace(
            config.ensemble,
            enabled_models=(model_name,),
            initial_weights={model_name: 1.0},
        ),
        policy=replace(config.policy, first_query=model_name),
    )
    return config, dataset, lookup, model_name


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


def run_one(job: tuple[str, dict, str]) -> dict:
    variant, episode, output_root_string = job
    config, dataset, lookup, model_name = build_problem(episode, variant)
    oracle = OfflineTableOracle(dataset=dataset, lookup=lookup)
    seed = int(episode["seed"])
    started = time.perf_counter()
    runner = CampaignRunner(dataset, config, seed=seed, backend="native")
    result = runner.run(oracle, budget=int(episode["budget"]))
    elapsed = time.perf_counter() - started
    payload = result.to_payload()
    global_best = float(oracle.global_best)
    metrics = _campaign_metrics(payload["trajectory"], global_best, episode["budget"])
    model = runner.ensemble.models[model_name]
    amplitudes = dict(getattr(model, "component_amplitudes_", {}))
    task_noises = dict(getattr(model, "task_noises_", {}))
    task_noise_deviations = dict(
        getattr(model, "task_noise_log_deviations_", {})
    )
    contrasts = getattr(model, "task_noise_contrasts_", None)
    global_noise = float(model._unpack(model.theta_)[2])
    payload["task_noise_metrics"] = metrics
    payload["task_noise_contract"] = {
        "variant": variant,
        "model": model_name,
        "target_transform": config.gp.target_transform,
        "train_only": True,
        "public_test_labels_used": False,
        "input_file": str(TRAIN_FILE.relative_to(DATASET_ROOT)),
        "input_sha256": file_sha256(TRAIN_FILE),
        "task_noise_design": TASK_NOISE_DESIGN if variant == CANDIDATE else None,
        "global_noise": global_noise,
        "task_noises": task_noises or None,
        "task_noise_log_deviations": task_noise_deviations or None,
        "task_noise_contrasts": None if contrasts is None else contrasts.tolist(),
        "component_amplitudes": amplitudes,
        "episode": episode,
    }
    output_path = (
        Path(output_root_string)
        / "trajectories"
        / episode["phase"]
        / variant
        / f"{episode['episode']}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {
        "dataset": "Buchwald_sub4",
        "phase": episode["phase"],
        "episode_family": episode["episode_family"],
        "episode": episode["episode"],
        "target_task": episode["target_task"],
        "variant": variant,
        "model": model_name,
        "budget": episode["budget"],
        "global_best": global_best,
        **metrics,
        "elapsed_seconds": elapsed,
        "global_noise": global_noise,
        "task_noises_json": json.dumps(task_noises, sort_keys=True) if task_noises else None,
        "task_noise_deviations_json": (
            json.dumps(task_noise_deviations, sort_keys=True)
            if task_noise_deviations
            else None
        ),
        "task_noise_contrasts_json": (
            json.dumps(contrasts.tolist()) if contrasts is not None else None
        ),
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
    for (phase, variant), group in rows.groupby(["phase", "variant"], sort=True):
        for metric in SUMMARY_METRICS:
            mean, sd, low, high = mean_ci(group[metric])
            records.append(
                {
                    "dataset": "Buchwald_sub4",
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


def paired_summary(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for phase, section in rows.groupby("phase", sort=True):
        baseline = section[section["variant"].eq(BASELINE)].set_index("episode")
        candidate = section[section["variant"].eq(CANDIDATE)].set_index("episode")
        common = baseline.index.intersection(candidate.index)
        for metric in SUMMARY_METRICS:
            delta = candidate.loc[common, metric] - baseline.loc[common, metric]
            mean, sd, low, high = mean_ci(delta)
            higher = metric not in {"round_to_95_global_best", "elapsed_seconds"}
            wins = int((delta > 1e-10).sum()) if higher else int((delta < -1e-10).sum())
            losses = int((delta < -1e-10).sum()) if higher else int((delta > 1e-10).sum())
            records.append(
                {
                    "dataset": "Buchwald_sub4",
                    "phase": phase,
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


def task_noise_summary(rows: pd.DataFrame) -> pd.DataFrame:
    candidate = rows[rows["variant"].eq(CANDIDATE)]
    records = []
    for phase, group in candidate.groupby("phase", sort=True):
        parsed = [json.loads(value) for value in group["task_noises_json"]]
        for task in sorted(set().union(*(values.keys() for values in parsed))):
            values = pd.Series([item[task] for item in parsed], dtype=float)
            mean, sd, low, high = mean_ci(values)
            records.append(
                {
                    "dataset": "Buchwald_sub4",
                    "phase": phase,
                    "task": task,
                    "n_episodes": len(values),
                    "mean_noise": mean,
                    "sample_sd_noise": sd,
                    "ci95_low": low,
                    "ci95_high": high,
                    "min_noise": float(values.min()),
                    "max_noise": float(values.max()),
                }
            )
    return pd.DataFrame(records)


def pilot_gate(rows: pd.DataFrame, paired: pd.DataFrame) -> dict[str, object]:
    metric_delta = {
        row.metric: float(row.mean_delta) for row in paired.itertuples()
    }
    candidate = rows[rows["variant"].eq(CANDIDATE)]
    contrast_values = np.concatenate(
        [
            np.asarray(json.loads(value), dtype=float)
            for value in candidate["task_noise_contrasts_json"]
        ]
    )
    lower_noise, upper_noise = TASK_NOISE_DESIGN["global_noise_bounds"]
    margin = PILOT_GATE["global_noise_bound_fraction_margin"]
    contrast_interior = bool(
        np.max(np.abs(contrast_values))
        < 1.25 - PILOT_GATE["contrast_bound_margin"]
    )
    global_noise_interior = bool(
        (candidate["global_noise"] > lower_noise * (1.0 + margin)).all()
        and (candidate["global_noise"] < upper_noise * (1.0 - margin)).all()
    )
    checks = {
        "auc": metric_delta["AUC_best_so_far"] >= PILOT_GATE["minimum_auc_delta"],
        "initial": metric_delta["initial_round_found_best"]
        >= PILOT_GATE["minimum_initial_delta"],
        "best": metric_delta["best_found"] >= PILOT_GATE["minimum_best_delta"],
        "t95": metric_delta["round_to_95_global_best"]
        <= PILOT_GATE["maximum_t95_delta"],
        "contrast_interior": contrast_interior,
        "global_noise_interior": global_noise_interior,
    }
    return {
        "gate": PILOT_GATE,
        "mean_deltas": metric_delta,
        "max_abs_contrast": float(np.max(np.abs(contrast_values))),
        "global_noise_min": float(candidate["global_noise"].min()),
        "global_noise_max": float(candidate["global_noise"].max()),
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def deterministic_twenty_run_view(rows: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Return the frozen 20-run reporting view of 21 exhaustive episodes."""
    episode_ids = sorted(rows["episode"].unique())
    if len(episode_ids) != 21:
        raise ValueError("The 20-run view requires all 21 exhaustive episodes.")
    selected = set(episode_ids[:20])
    omitted = episode_ids[20]
    view = rows[rows["episode"].isin(selected)].copy()
    if len(view) != 20 * len(VARIANTS):
        raise RuntimeError("The deterministic 20-run view is not fully paired.")
    return view, omitted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("smoke", "pilot", "confirmation"), required=True
    )
    parser.add_argument("--jobs", type=int, default=min(6, os.cpu_count() or 1))
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.jobs <= 0:
        raise ValueError("--jobs must be positive.")
    episodes = episodes_for_stage(args.stage)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "stage": args.stage,
        "train_only": True,
        "public_test_labels_used": False,
        "input": {
            "path": str(TRAIN_FILE.relative_to(DATASET_ROOT)),
            "sha256": file_sha256(TRAIN_FILE),
        },
        "variants": list(VARIANTS),
        "models": MODEL_NAMES,
        "target_transform": "standard",
        "acquisition": "posterior mean query 1, stable LogEI thereafter",
        "task_noise_design": TASK_NOISE_DESIGN,
        "pilot_gate": PILOT_GATE,
        "official_target_excluded_from_pilot": args.stage != "confirmation",
        "official_target": OFFICIAL_TARGET,
        "episode_count": len(episodes),
        "parameter_sweep_performed": False,
        "production_default_changed": False,
        "python": platform.python_version(),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    jobs = [
        (variant, episode, str(args.output_dir))
        for variant in VARIANTS
        for episode in episodes
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(run_one, job): job for job in jobs}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"{row['phase']} {row['variant']} {row['episode']}: "
                f"AUC={row['AUC_best_so_far']:.3f}",
                flush=True,
            )
    detail = pd.DataFrame(rows).sort_values(["episode", "variant"])
    summary = summarize(detail)
    paired = paired_summary(detail)
    detail.to_csv(args.output_dir / "per_episode.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    paired.to_csv(args.output_dir / "paired_task_noise_vs_baseline.csv", index=False)
    task_noise_summary(detail).to_csv(
        args.output_dir / "task_noise_stability.csv", index=False
    )
    if args.stage == "confirmation":
        view, omitted_episode = deterministic_twenty_run_view(detail)
        view.to_csv(args.output_dir / "per_run_20.csv", index=False)
        summarize(view).to_csv(args.output_dir / "summary_20run.csv", index=False)
        paired_summary(view).to_csv(
            args.output_dir / "paired_20run_task_noise_vs_baseline.csv", index=False
        )
        reporting_contract = {
            "selection": "lexicographically first 20 exhaustive episode IDs",
            "selection_uses_metrics": False,
            "n_runs": 20,
            "omitted_episode": omitted_episode,
            "omitted_episode_retained_in_primary_exhaustive_artifacts": True,
        }
        (args.output_dir / "reporting_20run_contract.json").write_text(
            json.dumps(reporting_contract, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if args.stage in {"smoke", "pilot"}:
        decision = pilot_gate(detail, paired)
        (args.output_dir / "gate_decision.json").write_text(
            json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
