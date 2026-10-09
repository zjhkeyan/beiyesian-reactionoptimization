#!/usr/bin/env python3
"""Build and validate the official 20-seed, 40-round result artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

import pandas as pd
import torch
from scipy.stats import t as student_t


OFFICIAL_SEEDS = tuple(range(100, 2001, 100))
METHOD = "optimized_gp_interaction_gp_warmstart_logei"
SOURCE_VARIANT = "gp_interaction_gp_warmstart_logei"
DATASETS = {
    "Buchwald_sub4": {
        "features": "Buchwald_sub4/buchwald_sub4_test_features.csv",
        "roles": ("Product", "Reactant2", "Ligand", "Additive", "Base"),
    },
    "Suzuki": {
        "features": "Suzuki/suzuki_test_features.csv",
        "roles": ("Electrophile", "Nucleophile", "Ligand", "Base", "Solvent"),
    },
}
METRICS = (
    ("initial_round_found_best", "higher"),
    ("best_found", "higher"),
    ("round_to_95_global_best", "lower"),
    ("AUC_best_so_far", "higher"),
)


def source_path(result_root: Path, dataset: str, seed: int) -> Path:
    phase = "refinement_screening" if seed <= 300 else "refinement_confirmation"
    return result_root / phase / dataset / SOURCE_VARIANT / f"seed_{seed}.json"


def normalise_record(raw: dict, best_before: float) -> tuple[dict, float]:
    observed_value = raw["observed_yield"] if "observed_yield" in raw else raw["actual_yield"]
    observed = float(observed_value)
    best_so_far = max(best_before, observed)
    record = dict(raw)
    record.update(
        {
            "round": int(raw["step"]),
            "step": int(raw["step"]),
            "candidate_id": int(raw["query_index"]),
            "query_index": int(raw["query_index"]),
            "observed_yield": observed,
            "actual_yield": observed,
            "best_so_far": best_so_far,
            "uncertainty": (
                None if raw.get("predicted_std") is None else float(raw["predicted_std"])
            ),
            "rationale": (
                f"{METHOD}; acquisition={raw.get('acquisition_name', 'not_recorded')}"
            ),
        }
    )
    return record, best_so_far


def recompute_metrics(trajectory: list[dict], global_best: float) -> dict[str, float]:
    best_curve = [float(row["best_so_far"]) for row in trajectory]
    threshold = 0.95 * global_best
    t95 = next(
        (int(row["step"]) for row in trajectory if row["best_so_far"] >= threshold),
        41,
    )
    return {
        "initial_round_found_best": best_curve[0],
        "best_found": best_curve[-1],
        "round_to_95_global_best": float(t95),
        "AUC_best_so_far": mean(best_curve),
    }


def assert_close(left: float, right: float, label: str) -> None:
    if not math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-8):
        raise ValueError(f"{label}: {left} != {right}")


def validate_conditions(
    dataset: str,
    trajectory: list[dict],
    candidate_frame: pd.DataFrame,
) -> None:
    roles = DATASETS[dataset]["roles"]
    if len(trajectory) != 40:
        raise ValueError(f"{dataset}: expected 40 records, got {len(trajectory)}")
    indices = [int(row["query_index"]) for row in trajectory]
    if len(set(indices)) != 40:
        raise ValueError(f"{dataset}: duplicate query_index in trajectory")
    for expected_step, row in enumerate(trajectory, start=1):
        if row["step"] != expected_step or row["round"] != expected_step:
            raise ValueError(f"{dataset}: non-consecutive step at {expected_step}")
        index = int(row["query_index"])
        if index < 0 or index >= len(candidate_frame):
            raise ValueError(f"{dataset}: query_index {index} is outside candidate pool")
        expected = {role: str(candidate_frame.iloc[index][role]) for role in roles}
        actual = {role: str(row["condition"][role]) for role in roles}
        if actual != expected:
            raise ValueError(f"{dataset}: condition mismatch at query_index {index}")


def build(root: Path) -> tuple[list[dict], list[dict]]:
    investigation_root = (
        root / "submission/results/experimental_investigations/sota_search"
    )
    output_root = root / "submission/results/optimization_trajectories"
    per_seed_rows: list[dict] = []
    audit_rows: list[dict] = []

    for dataset, spec in DATASETS.items():
        candidate_frame = pd.read_csv(root / spec["features"], dtype=str, keep_default_na=False)
        dataset_output = output_root / dataset
        dataset_output.mkdir(parents=True, exist_ok=True)
        expected_names = {f"seed_{seed}.pt" for seed in OFFICIAL_SEEDS}
        unexpected = {path.name for path in dataset_output.glob("*.pt")} - expected_names
        if unexpected:
            raise ValueError(f"Unexpected trajectory files in {dataset_output}: {sorted(unexpected)}")

        for seed in OFFICIAL_SEEDS:
            source = source_path(investigation_root, dataset, seed)
            if not source.is_file():
                raise FileNotFoundError(source)
            payload = json.loads(source.read_text(encoding="utf-8"))
            raw_trajectory = payload["trajectory"]
            trajectory: list[dict] = []
            running_best = -math.inf
            for raw in raw_trajectory:
                record, running_best = normalise_record(raw, running_best)
                trajectory.append(record)

            validate_conditions(dataset, trajectory, candidate_frame)
            metrics = recompute_metrics(trajectory, float(payload["metrics"]["global_best"]))
            source_metrics = payload["metrics"]
            assert_close(
                metrics["initial_round_found_best"],
                float(source_metrics["initial_round_found_best"]),
                f"{dataset} seed {seed} first",
            )
            assert_close(
                metrics["best_found"],
                float(source_metrics["best_found"]),
                f"{dataset} seed {seed} best",
            )
            assert_close(
                metrics["AUC_best_so_far"],
                float(source_metrics["auc_best_so_far"]),
                f"{dataset} seed {seed} AUC",
            )
            source_t95 = source_metrics.get("round_to_95_global_best")
            source_t95_penalized = 41 if source_t95 is None else int(source_t95)
            if int(metrics["round_to_95_global_best"]) != source_t95_penalized:
                raise ValueError(f"{dataset} seed {seed}: t95 mismatch")

            artifact = {
                "format_version": "1.0",
                "dataset": dataset,
                "seed": seed,
                "method": METHOD,
                "budget": 40,
                "metrics": metrics,
                "trajectory": trajectory,
            }
            destination = dataset_output / f"seed_{seed}.pt"
            torch.save(artifact, destination)
            loaded = torch.load(destination, map_location="cpu", weights_only=True)
            validate_conditions(dataset, loaded["trajectory"], candidate_frame)
            if loaded["seed"] != seed or loaded["method"] != METHOD:
                raise ValueError(f"{destination}: serialization round-trip mismatch")

            per_seed_rows.append(
                {"dataset": dataset, "method": METHOD, "seed": seed, **metrics}
            )
            audit_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "source": str(source.relative_to(root)),
                    "output": str(destination.relative_to(root)),
                    "records": len(trajectory),
                    "unique_query_indices": len({x["query_index"] for x in trajectory}),
                    "condition_match": True,
                    "metric_match": True,
                    "roundtrip_load": True,
                }
            )

    return per_seed_rows, audit_rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarise(per_seed_rows: list[dict]) -> list[dict]:
    summary: list[dict] = []
    for dataset in DATASETS:
        dataset_rows = [row for row in per_seed_rows if row["dataset"] == dataset]
        for metric, direction in METRICS:
            values = [float(row[metric]) for row in dataset_rows]
            sample_sd = stdev(values)
            half_width = student_t.ppf(0.975, len(values) - 1) * sample_sd / math.sqrt(len(values))
            summary.append(
                {
                    "dataset": dataset,
                    "method": METHOD,
                    "metric": metric,
                    "direction": direction,
                    "n_runs": len(values),
                    "mean": mean(values),
                    "sample_sd": sample_sd,
                    "ci95_low": mean(values) - half_width,
                    "ci95_high": mean(values) + half_width,
                }
            )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
    )
    return parser.parse_args()


def main() -> None:
    root = parse_args().project_root.resolve()
    per_seed_rows, audit_rows = build(root)
    result_root = root / "submission/results"
    write_csv(
        result_root / "per_seed_metrics.csv",
        per_seed_rows,
        ["dataset", "method", "seed", *(metric for metric, _ in METRICS)],
    )
    write_csv(
        result_root / "summary_metrics.csv",
        summarise(per_seed_rows),
        [
            "dataset",
            "method",
            "metric",
            "direction",
            "n_runs",
            "mean",
            "sample_sd",
            "ci95_low",
            "ci95_high",
        ],
    )
    write_csv(
        result_root / "trajectory_validation.csv",
        audit_rows,
        [
            "dataset",
            "seed",
            "source",
            "output",
            "records",
            "unique_query_indices",
            "condition_match",
            "metric_match",
            "roundtrip_load",
        ],
    )
    print("Built 40 validated trajectory files (2 datasets x 20 seeds).")


if __name__ == "__main__":
    main()
