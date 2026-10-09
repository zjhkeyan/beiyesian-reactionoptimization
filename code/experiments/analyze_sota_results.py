#!/usr/bin/env python3
"""Build 20-seed core-metric tables with sample SD and Student-t 95% CIs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


DATASET_ROOT = Path(__file__).resolve().parents[3]
RESULT_ROOT = DATASET_ROOT / "submission" / "results" / "experimental_investigations"
SOTA_ROOT = RESULT_ROOT / "sota_search"
OUTPUT_ROOT = SOTA_ROOT / "final"
SEEDS = tuple(range(100, 2001, 100))
BUDGET = 40

METHOD_SOURCES = {
    "optimized_gp_interaction_gp_warmstart_logei": (
        "refinement_screening",
        "refinement_confirmation",
        "gp_interaction_gp_warmstart_logei",
    ),
    "gp_logei": ("screening", "confirmation_holdout", "gp_logei"),
    "gp_interaction_logei": (
        "screening",
        "confirmation_holdout",
        "gp_interaction_logei",
    ),
    "full_logei": ("screening", "confirmation_holdout", "full_logei"),
}


def load_payload(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_method(dataset: str, label: str, sources: tuple[str, str, str]) -> list[dict]:
    screening_phase, confirmation_phase, variant = sources
    rows = []
    for seed in SEEDS:
        phase = screening_phase if seed <= 300 else confirmation_phase
        payload = load_payload(
            SOTA_ROOT / phase / dataset / variant / f"seed_{seed}.json"
        )
        rows.append(row_from_payload(dataset, label, seed, payload))
    return rows


def row_from_payload(dataset: str, method: str, seed: int, payload: dict) -> dict:
    metrics = payload["metrics"]
    t95 = metrics["round_to_95_global_best"]
    return {
        "dataset": dataset,
        "method": method,
        "seed": seed,
        "initial_round_found_best": metrics["initial_round_found_best"],
        "best_found": metrics["best_found"],
        "round_to_95_global_best": np.nan if t95 is None else float(t95),
        "round_to_95_global_best_penalized": BUDGET + 1 if t95 is None else float(t95),
        "AUC_best_so_far": metrics["auc_best_so_far"],
    }


def load_random(dataset: str) -> list[dict]:
    rows = []
    for seed in SEEDS:
        payload = load_payload(
            RESULT_ROOT / dataset / "random_search" / f"seed_{seed}.json"
        )
        rows.append(row_from_payload(dataset, "random_search", seed, payload))
    return rows


def summarize(values: pd.Series) -> tuple[float, float, float, float]:
    array = values.to_numpy(dtype=float)
    n = len(array)
    mean = float(np.mean(array))
    sd = float(np.std(array, ddof=1))
    half_width = float(t.ppf(0.975, df=n - 1) * sd / np.sqrt(n))
    return mean, sd, mean - half_width, mean + half_width


def main() -> None:
    rows = []
    for dataset in ("Buchwald_sub4", "Suzuki"):
        for label, sources in METHOD_SOURCES.items():
            rows.extend(load_method(dataset, label, sources))
        if dataset == "Buchwald_sub4":
            rows.extend(
                load_method(
                    dataset,
                    "full_logei_strain_025",
                    ("screening", "confirmation_holdout", "full_logei_strain_025"),
                )
            )
        rows.extend(load_random(dataset))

    per_seed = pd.DataFrame(rows).sort_values(["dataset", "method", "seed"])
    if not (per_seed.groupby(["dataset", "method"]).size() == 20).all():
        raise RuntimeError("Every reported dataset/method pair must contain 20 runs.")

    metric_specs = (
        ("initial_round_found_best", "higher"),
        ("best_found", "higher"),
        ("round_to_95_global_best_penalized", "lower"),
        ("AUC_best_so_far", "higher"),
    )
    summaries = []
    diagnostics = []
    for (dataset, method), group in per_seed.groupby(["dataset", "method"], sort=True):
        for metric, direction in metric_specs:
            mean, sd, low, high = summarize(group[metric])
            summaries.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "metric": metric,
                    "direction": direction,
                    "n_runs": len(group),
                    "mean": mean,
                    "sample_sd": sd,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
        reached = group["round_to_95_global_best"].dropna()
        diagnostics.append(
            {
                "dataset": dataset,
                "method": method,
                "n_runs": len(group),
                "n_reached_t95": len(reached),
                "t95_success_rate": len(reached) / len(group),
                "mean_t95_when_reached": reached.mean(),
                "non_attainment_penalty": BUDGET + 1,
            }
        )

    summary = pd.DataFrame(summaries)
    wide = summary.pivot(
        index=["dataset", "method"],
        columns="metric",
        values=["mean", "sample_sd", "ci95_low", "ci95_high"],
    )
    wide.columns = [f"{metric}_{stat}" for stat, metric in wide.columns]
    wide = wide.reset_index()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(OUTPUT_ROOT / "core_metrics_per_seed.csv", index=False)
    summary.to_csv(OUTPUT_ROOT / "core_metrics_summary.csv", index=False)
    wide.to_csv(OUTPUT_ROOT / "method_comparison_wide.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(OUTPUT_ROOT / "t95_diagnostics.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
