#!/usr/bin/env python3
"""Validate the official trajectory layout without accessing target labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch


SEEDS = tuple(range(100, 2001, 100))
DATASETS = {
    "Buchwald_sub4": (
        "Buchwald_sub4/buchwald_sub4_test_features.csv",
        ("Product", "Reactant2", "Ligand", "Additive", "Base"),
    ),
    "Suzuki": (
        "Suzuki/suzuki_test_features.csv",
        ("Electrophile", "Nucleophile", "Ligand", "Base", "Solvent"),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for dataset, (relative_features, roles) in DATASETS.items():
        directory = args.submission_root / "results/optimization_trajectories" / dataset
        expected = {f"seed_{seed}.pt" for seed in SEEDS}
        actual = {path.name for path in directory.glob("*.pt")}
        if actual != expected:
            raise ValueError(f"{dataset}: file set mismatch: {sorted(actual ^ expected)}")
        candidates = pd.read_csv(
            args.dataset_root / relative_features,
            dtype=str,
            keep_default_na=False,
        )
        for seed in SEEDS:
            path = directory / f"seed_{seed}.pt"
            payload = torch.load(path, map_location="cpu", weights_only=True)
            trajectory = payload["trajectory"]
            if payload["seed"] != seed or payload["dataset"] != dataset:
                raise ValueError(f"{path}: seed or dataset mismatch")
            if len(trajectory) != 40:
                raise ValueError(f"{path}: expected 40 records")
            indices = [int(row["query_index"]) for row in trajectory]
            if len(indices) != len(set(indices)):
                raise ValueError(f"{path}: duplicate query_index")
            best = float("-inf")
            for step, row in enumerate(trajectory, start=1):
                if int(row["step"]) != step or int(row["round"]) != step:
                    raise ValueError(f"{path}: invalid step sequence")
                index = int(row["query_index"])
                if index < 0 or index >= len(candidates):
                    raise ValueError(f"{path}: invalid query_index {index}")
                expected_condition = {
                    role: str(candidates.iloc[index][role]) for role in roles
                }
                actual_condition = {role: str(row["condition"][role]) for role in roles}
                if actual_condition != expected_condition:
                    raise ValueError(f"{path}: condition mismatch at step {step}")
                best = max(best, float(row["observed_yield"]))
                if abs(float(row["best_so_far"]) - best) > 1e-8:
                    raise ValueError(f"{path}: invalid best_so_far at step {step}")
        print(f"{dataset}: 20 files x 40 unique, condition-matched records")


if __name__ == "__main__":
    main()
