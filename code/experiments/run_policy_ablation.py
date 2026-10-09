#!/usr/bin/env python3
"""Run leakage-isolated policy diagnostics for the two public reaction tasks."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pandas as pd


CODE_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path(__file__).resolve().parents[3]
EXTENSION_ROOT = CODE_ROOT / "bgolearn_rxn_extension"
sys.path.insert(0, str(EXTENSION_ROOT / "src"))

from bgolearn_rxn.campaign import CampaignRunner  # noqa: E402
from bgolearn_rxn.config import load_run_config  # noqa: E402
from bgolearn_rxn.datasets import DatasetSpec, load_reaction_dataset  # noqa: E402
from bgolearn_rxn.oracle import OfflineTableOracle  # noqa: E402
from bgolearn_rxn.serialization import save_json  # noqa: E402


CORRECT_SUZUKI_PRODUCT = (
    "6-(5-methyl-1-(tetrahydro-2H-pyran-2-yl)-1H-indazol-4-yl)quinoline"
)
BORONATE_NUCLEOPHILE = (
    "5-methyl-1-(oxan-2-yl)-4-(4,4,5,5-tetramethyl-1,3,2-dioxaborolan-2-yl)indazole"
)

DATASETS = {
    "Suzuki": {
        "config": "suzuki.yaml",
        "train": "Suzuki/suzuki_train.csv",
        "candidates": "Suzuki/suzuki_test_features.csv",
        "oracle": "Suzuki/suzuki_test.csv",
    },
    "Buchwald_sub4": {
        "config": "buchwald_sub4.yaml",
        "train": "Buchwald_sub4/buchwald_sub4_train.csv",
        "candidates": "Buchwald_sub4/buchwald_sub4_test_features.csv",
        "oracle": "Buchwald_sub4/buchwald_sub4_test.csv",
    },
}


def validate_suzuki_product_contract(config, dataset) -> None:
    metadata = json.loads((DATASET_ROOT / "Suzuki" / "metadata.json").read_text())
    product = metadata["fixed_components"]["Product"]
    direct = metadata["model_feature_policy"]["direct_features"]
    fixed_context = metadata["model_feature_policy"]["fixed_context_features"]
    if product["standardized_name"] != CORRECT_SUZUKI_PRODUCT:
        raise ValueError("Suzuki fixed-product metadata does not contain the corrected product.")
    if not product["direct_model_feature"] or fixed_context != {"ProductContext": "Product"}:
        raise ValueError("Suzuki corrected Product must be enabled as fixed reaction context.")
    if "Product" in direct:
        raise ValueError("Suzuki Product must not become a selectable CSV variable.")
    if "Product" in config.dataset.feature_columns:
        raise ValueError("Suzuki experiment config incorrectly includes Product as a feature.")
    if config.dataset.fixed_roles.get("ProductContext") != CORRECT_SUZUKI_PRODUCT:
        raise ValueError("Suzuki experiment config lacks the corrected ProductContext.")
    nucleophiles = set(dataset.initial["Nucleophile"]) | set(dataset.candidates["Nucleophile"])
    if BORONATE_NUCLEOPHILE not in nucleophiles:
        raise ValueError("The valid boronate nucleophile was incorrectly removed from the task.")


def policy_variants(base_config):
    policy = base_config.policy
    return {
        "metric_aware": base_config,
        "logei_only": replace(
            base_config,
            policy=replace(
                policy,
                early_end=base_config.dataset.budget,
                final_start=base_config.dataset.budget + 1,
                challenge_every=0,
            ),
        ),
        "greedy_mean": replace(
            base_config,
            policy=replace(
                policy,
                final_start=2,
                final_beta=0.0,
                challenge_every=0,
            ),
        ),
    }


def run_dataset(
    dataset_name: str,
    seeds: list[int],
    budget: int,
    output_root: Path,
    selected_variants: set[str],
) -> list[dict]:
    paths = DATASETS[dataset_name]
    base_config = load_run_config(EXTENSION_ROOT / "configs" / paths["config"])
    spec = DatasetSpec.from_config(base_config.dataset)
    dataset = load_reaction_dataset(
        DATASET_ROOT / paths["train"],
        DATASET_ROOT / paths["candidates"],
        spec,
    )
    if dataset_name == "Suzuki":
        validate_suzuki_product_contract(base_config, dataset)

    summaries: list[dict] = []
    for variant_name, config in policy_variants(base_config).items():
        if variant_name not in selected_variants:
            continue
        for seed in seeds:
            oracle = OfflineTableOracle.from_csv(dataset, DATASET_ROOT / paths["oracle"])
            started = time.perf_counter()
            result = CampaignRunner(dataset, config, seed=seed, backend="native").run(
                oracle,
                budget=budget,
            )
            elapsed = time.perf_counter() - started
            payload = result.to_payload()
            payload["investigation"] = {
                "variant": variant_name,
                "public_lookup_diagnostic": True,
                "eligible_for_hidden_model_selection": False,
                "suzuki_product_contract": (
                    "corrected_fixed_reaction_context" if dataset_name == "Suzuki" else None
                ),
            }
            run_dir = output_root / dataset_name / variant_name
            save_json(payload, run_dir / f"seed_{seed}.json")
            metrics = payload["metrics"]
            acquisitions = Counter(row["acquisition_name"] for row in payload["trajectory"])
            row = {
                "dataset": dataset_name,
                "variant": variant_name,
                "seed": seed,
                "budget": budget,
                "first_query_index": payload["trajectory"][0]["query_index"],
                "first_query_yield": metrics["initial_round_found_best"],
                "best_found": metrics["best_found"],
                "auc_best_so_far": metrics["auc_best_so_far"],
                "round_to_95_global_best": metrics["round_to_95_global_best"],
                "final_simple_regret": metrics["final_simple_regret"],
                "elapsed_seconds": elapsed,
                "acquisition_counts": json.dumps(acquisitions, sort_keys=True),
            }
            summaries.append(row)
            print(
                f"{dataset_name} {variant_name} seed={seed}: "
                f"first={row['first_query_yield']:.4f}, best={row['best_found']:.4f}, "
                f"AUC={row['auc_best_so_far']:.4f}, t95={row['round_to_95_global_best']}, "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(DATASETS),
        default=list(DATASETS),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[100])
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("metric_aware", "logei_only", "greedy_mean"),
        default=["metric_aware", "logei_only", "greedy_mean"],
    )
    parser.add_argument("--budget", type=int, default=40)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATASET_ROOT / "submission" / "results" / "experimental_investigations",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.budget <= 0 or args.budget > 40:
        raise ValueError("Diagnostic budget must be between 1 and 40.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for dataset_name in args.datasets:
        rows.extend(
            run_dataset(
                dataset_name,
                args.seeds,
                args.budget,
                args.output_dir,
                set(args.variants),
            )
        )
    summary = pd.DataFrame(rows).sort_values(["dataset", "variant", "seed"])
    summary_path = args.output_dir / "policy_ablation_summary.csv"
    if summary_path.exists():
        previous = pd.read_csv(summary_path)
        summary = pd.concat([previous, summary], ignore_index=True)
        summary = summary.drop_duplicates(
            ["dataset", "variant", "seed", "budget"], keep="last"
        ).sort_values(["dataset", "variant", "seed"])
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
