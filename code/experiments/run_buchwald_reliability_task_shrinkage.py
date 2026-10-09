#!/usr/bin/env python3
"""Evaluate source-reliability shrinkage on 84 Buchwald proxy episodes."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
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

import run_buchwald_source_meta_selection as meta  # noqa: E402
from bgolearn_rxn.campaign import CampaignRunner  # noqa: E402
from bgolearn_rxn.oracle import OfflineTableOracle  # noqa: E402
from bgolearn_rxn.surrogates.component_ard_gp import (  # noqa: E402
    ComponentARDLowRankTaskGP,
)
from phase9_buchwald_reliability_gp import (  # noqa: E402
    ReliabilityShrunkComponentARDLowRankTaskGP,
)


BASELINE = "component_ard"
CORE_METRICS = meta.CORE_METRICS


@dataclass(frozen=True)
class ReliabilitySetting:
    name: str
    reliability_floor: float
    reliability_power: float = 1.0
    ridge: float = 0.10


SETTINGS = (
    ReliabilitySetting("reliability_floor_015", 0.15),
    ReliabilitySetting("reliability_floor_035", 0.35),
    ReliabilitySetting("reliability_floor_060", 0.60),
)
SETTING_BY_NAME = {setting.name: setting for setting in SETTINGS}


def mean_ci(values: pd.Series) -> tuple[float, float, float, float]:
    data = values.to_numpy(dtype=float)
    mean = float(np.mean(data))
    if len(data) < 2:
        return mean, 0.0, mean, mean
    sd = float(np.std(data, ddof=1))
    half = float(t.ppf(0.975, len(data) - 1) * sd / np.sqrt(len(data)))
    return mean, sd, mean - half, mean + half


def install_model(runner: CampaignRunner, variant: str) -> None:
    schema = runner.prepared.encoder.schema
    if schema is None:
        raise RuntimeError("The fitted encoder has no feature schema.")
    if variant == BASELINE:
        model = ComponentARDLowRankTaskGP(
            schema=schema,
            config=runner.config.gp,
            seed=runner.seed + 79,
        )
    else:
        setting = SETTING_BY_NAME[variant]
        model = ReliabilityShrunkComponentARDLowRankTaskGP(
            schema=schema,
            config=runner.config.gp,
            seed=runner.seed + 79,
            reliability_floor=setting.reliability_floor,
            reliability_power=setting.reliability_power,
            ridge=setting.ridge,
        )
    runner.ensemble.models[meta.MODEL_NAME] = model


def run_one(job: tuple[dict, str, str]) -> dict:
    episode, variant, output_root = job
    config, dataset, lookup, observed_products = meta.build_problem(
        episode, strict_source_isolation=True
    )
    oracle = OfflineTableOracle(dataset=dataset, lookup=lookup)
    runner = CampaignRunner(dataset, config, seed=int(episode["seed"]), backend="native")
    install_model(runner, variant)
    started = time.perf_counter()
    result = runner.run(oracle, budget=int(episode["budget"]))
    elapsed = time.perf_counter() - started
    payload = result.to_payload()
    metrics = meta._campaign_metrics(
        payload["trajectory"], float(oracle.global_best), int(episode["budget"])
    )
    first = payload["trajectory"][0]
    model = runner.ensemble.models[meta.MODEL_NAME]
    reliability_contract = None
    if isinstance(model, ReliabilityShrunkComponentARDLowRankTaskGP):
        eigenvalues = np.linalg.eigvalsh(model.task_correlation_)
        reliability_contract = {
            "target_task_index": model.target_task_index_,
            "source_reliability": {
                str(key): value for key, value in model.source_reliability_.items()
            },
            "task_multipliers": model.task_multipliers_.tolist(),
            "minimum_task_correlation_eigenvalue": float(np.min(eigenvalues)),
        }
        if reliability_contract["minimum_task_correlation_eigenvalue"] < -1e-8:
            raise AssertionError("Reliability-shrunk task correlation is not PSD.")
    payload["phase9_reliability_contract"] = {
        "train_only": True,
        "public_test_labels_used": False,
        "official_target_rows_used": False,
        "strict_source_isolation": True,
        "variant": variant,
        "setting": (
            None if variant == BASELINE else asdict(SETTING_BY_NAME[variant])
        ),
        "observed_product_count": len(observed_products),
        "episode": episode,
        "metrics": metrics,
        "reliability": reliability_contract,
    }
    output_path = (
        Path(output_root)
        / "trajectories"
        / variant
        / f"{episode['episode']}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "episode": episode["episode"],
        "target_task": episode["target_task"],
        "variant": variant,
        "budget": int(episode["budget"]),
        "global_best": float(oracle.global_best),
        **metrics,
        "query_1_index": int(first["query_index"]),
        "query_1_acquisition": first["acquisition_name"],
        "elapsed_seconds": elapsed,
        "source_reliability_json": json.dumps(
            {} if reliability_contract is None else reliability_contract["source_reliability"],
            sort_keys=True,
        ),
        "task_multipliers_json": json.dumps(
            [] if reliability_contract is None else reliability_contract["task_multipliers"]
        ),
        "minimum_task_correlation_eigenvalue": (
            np.nan
            if reliability_contract is None
            else reliability_contract["minimum_task_correlation_eigenvalue"]
        ),
    }


def run_jobs(jobs: list[tuple[dict, str, str]], workers: int) -> pd.DataFrame:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    rows = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_one, job): job for job in jobs}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"{row['variant']} {row['episode']}: "
                f"AUC={row['AUC_best_so_far']:.3f}",
                flush=True,
            )
    return pd.DataFrame(rows).sort_values(["episode", "variant"])


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for variant, group in rows.groupby("variant", sort=True):
        for metric in (*CORE_METRICS, "elapsed_seconds"):
            mean, sd, low, high = mean_ci(group[metric])
            records.append(
                {
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
    baseline = rows[rows["variant"].eq(BASELINE)].set_index("episode")
    records = []
    for setting in SETTINGS:
        if setting.name not in set(rows["variant"]):
            continue
        candidate = rows[rows["variant"].eq(setting.name)].set_index("episode")
        common = baseline.index.intersection(candidate.index)
        for metric in (*CORE_METRICS, "elapsed_seconds"):
            delta = candidate.loc[common, metric] - baseline.loc[common, metric]
            mean, sd, low, high = mean_ci(delta)
            lower_is_better = metric in {
                "round_to_95_global_best",
                "elapsed_seconds",
            }
            wins = int((delta < -1e-12).sum()) if lower_is_better else int((delta > 1e-12).sum())
            losses = int((delta > 1e-12).sum()) if lower_is_better else int((delta < -1e-12).sum())
            records.append(
                {
                    "candidate": setting.name,
                    "baseline": BASELINE,
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


def lower_tail_summary(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for variant, group in rows.groupby("variant", sort=True):
        auc = np.sort(group["AUC_best_so_far"].to_numpy(dtype=float))
        tail_count = max(1, int(math.ceil(0.10 * len(auc))))
        records.append(
            {
                "variant": variant,
                "n_episodes": len(auc),
                "minimum_AUC": float(auc[0]),
                "q10_AUC": float(np.quantile(auc, 0.10)),
                "CVaR10_AUC": float(np.mean(auc[:tail_count])),
                "tail_count": tail_count,
            }
        )
    return pd.DataFrame(records)


def select_and_gate(rows: pd.DataFrame, tails: pd.DataFrame) -> dict:
    means = rows.groupby("variant")[list(CORE_METRICS) + ["elapsed_seconds"]].mean()
    candidates = [setting.name for setting in SETTINGS if setting.name in means.index]
    selected = max(candidates, key=lambda name: means.loc[name, "AUC_best_so_far"])
    baseline = rows[rows["variant"].eq(BASELINE)].set_index("episode")
    candidate = rows[rows["variant"].eq(selected)].set_index("episode")
    common = baseline.index.intersection(candidate.index)
    deltas = {
        metric: candidate.loc[common, metric] - baseline.loc[common, metric]
        for metric in CORE_METRICS
    }
    tail_table = tails.set_index("variant")
    checks = {
        "all_episode_AUC_strictly_positive": bool(
            (deltas["AUC_best_so_far"] > 1e-12).all()
        ),
        "all_episode_initial_noninferior": bool(
            (deltas["initial_round_found_best"] >= -1e-12).all()
        ),
        "all_episode_best_found_noninferior": bool(
            (deltas["best_found"] >= -1e-12).all()
        ),
        "all_episode_t95_noninferior": bool(
            (deltas["round_to_95_global_best"] <= 1e-12).all()
        ),
        "CVaR10_AUC_noninferior": bool(
            tail_table.loc[selected, "CVaR10_AUC"]
            >= tail_table.loc[BASELINE, "CVaR10_AUC"] - 1e-12
        ),
    }
    return {
        "selected_variant": selected,
        "strict_gate_passed": all(checks.values()),
        "checks": checks,
        "absolute_mean_metrics": {
            variant: {metric: float(value) for metric, value in row.items()}
            for variant, row in means.to_dict(orient="index").items()
        },
        "paired_counts": {
            metric: {
                "positive": int((delta > 1e-12).sum()),
                "zero": int(np.isclose(delta, 0.0, atol=1e-12).sum()),
                "negative": int((delta < -1e-12).sum()),
            }
            for metric, delta in deltas.items()
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=min(6, os.cpu_count() or 1))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=(BASELINE,) + tuple(SETTING_BY_NAME),
        default=[BASELINE] + list(SETTING_BY_NAME),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            DATASET_ROOT
            / "submission"
            / "results"
            / "experimental_investigations"
            / "phase9-2.1-buchwald-reliability-dev"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.jobs <= 0:
        raise ValueError("--jobs must be positive.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    episodes = meta.make_source_meta_episodes()
    if args.smoke:
        episodes = episodes[:1]
    manifest = {
        "study": "leave-one-source-out reliability shrinkage inside Buchwald task covariance",
        "train_only": True,
        "public_test_labels_used": False,
        "official_target_rows_used": False,
        "selection_excludes_official_target_rows": True,
        "source_episode_count": len(episodes),
        "variants": args.variants,
        "settings": [asdict(setting) for setting in SETTINGS],
        "reliability_estimator": "LOSO fixed condition-kernel Spearman, target outcomes excluded",
        "covariance_map": "D R D + diag(1-D^2)",
        "python": platform.python_version(),
        "smoke": bool(args.smoke),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    jobs = [
        (episode, variant, str(args.output_dir))
        for variant in args.variants
        for episode in episodes
    ]
    rows = run_jobs(jobs, args.jobs)
    rows.to_csv(args.output_dir / "per_episode.csv", index=False)
    summarize(rows).to_csv(args.output_dir / "summary.csv", index=False)
    paired_summary(rows).to_csv(args.output_dir / "paired_vs_component_ard.csv", index=False)
    tails = lower_tail_summary(rows)
    tails.to_csv(args.output_dir / "lower_tail_summary.csv", index=False)
    candidate_variants = set(rows["variant"]) - {BASELINE}
    if candidate_variants:
        gate = select_and_gate(rows, tails)
        (args.output_dir / "gate.json").write_text(
            json.dumps(gate, indent=2, sort_keys=True), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
