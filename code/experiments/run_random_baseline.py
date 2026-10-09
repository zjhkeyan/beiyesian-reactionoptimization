#!/usr/bin/env python3
"""Run the official-seed random-search baseline through the isolated oracle."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


CODE_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path(__file__).resolve().parents[3]
EXTENSION_ROOT = CODE_ROOT / "bgolearn_rxn_extension"
sys.path.insert(0, str(EXTENSION_ROOT / "src"))

from bgolearn_rxn.config import load_run_config  # noqa: E402
from bgolearn_rxn.datasets import DatasetSpec, load_reaction_dataset  # noqa: E402
from bgolearn_rxn.metrics import compute_campaign_metrics, summarize_metric  # noqa: E402
from bgolearn_rxn.oracle import OfflineTableOracle  # noqa: E402
from bgolearn_rxn.serialization import save_json  # noqa: E402


DATASETS = {
    "Suzuki": ("suzuki.yaml", "suzuki_train.csv", "suzuki_test_features.csv", "suzuki_test.csv"),
    "Buchwald_sub4": (
        "buchwald_sub4.yaml",
        "buchwald_sub4_train.csv",
        "buchwald_sub4_test_features.csv",
        "buchwald_sub4_test.csv",
    ),
}


def main() -> None:
    output_root = DATASET_ROOT / "submission" / "results" / "experimental_investigations"
    rows: list[dict] = []
    for dataset_name, (config_name, train_name, candidates_name, oracle_name) in DATASETS.items():
        config = load_run_config(EXTENSION_ROOT / "configs" / config_name)
        directory = DATASET_ROOT / dataset_name
        dataset = load_reaction_dataset(
            directory / train_name,
            directory / candidates_name,
            DatasetSpec.from_config(config.dataset),
        )
        run_dir = output_root / dataset_name / "random_search"
        for seed in config.dataset.seeds:
            oracle = OfflineTableOracle.from_csv(dataset, directory / oracle_name)
            rng = np.random.default_rng(seed)
            selected_ids = rng.permutation(dataset.candidate_ids)[: config.dataset.budget]
            trajectory = []
            observed = []
            for step, query_index in enumerate(selected_ids, start=1):
                value = float(oracle.query(int(query_index)))
                observed.append(value)
                trajectory.append(
                    {
                        "step": step,
                        "query_index": int(query_index),
                        "observed_yield": value,
                        "best_so_far": float(max(observed)),
                    }
                )
            metrics = compute_campaign_metrics(observed, global_best=oracle.global_best)
            payload = {
                "dataset": dataset_name,
                "seed": seed,
                "method": "uniform_random_without_replacement",
                "trajectory": trajectory,
                "metrics": metrics.to_dict(),
                "investigation": {
                    "public_lookup_diagnostic": True,
                    "eligible_for_hidden_model_selection": False,
                },
            }
            save_json(payload, run_dir / f"seed_{seed}.json")
            rows.append({"dataset": dataset_name, "seed": seed, **metrics.to_dict()})

    per_seed = pd.DataFrame(rows).sort_values(["dataset", "seed"])
    per_seed.to_csv(output_root / "random_baseline_per_seed.csv", index=False)
    summary = {}
    for dataset_name, group in per_seed.groupby("dataset"):
        summary[dataset_name] = {
            key: summarize_metric(group[key])
            for key in ("initial_round_found_best", "best_found", "auc_best_so_far")
        }
        t95 = group["round_to_95_global_best"].dropna()
        summary[dataset_name]["t95_success_rate"] = float(len(t95) / len(group))
        summary[dataset_name]["t95_conditional"] = (
            summarize_metric(t95) if len(t95) else None
        )
    save_json(summary, output_root / "random_baseline_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
