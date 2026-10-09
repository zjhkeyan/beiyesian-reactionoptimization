#!/usr/bin/env python3
"""Screen and confirm public-benchmark reaction-optimization configurations."""

from __future__ import annotations

import argparse
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

VARIANTS = (
    "full_metric_aware",
    "full_logei",
    "gp_logei",
    "interaction_logei",
    "trees_logei",
    "gp_interaction_logei",
    "gp_interaction_gp_warmstart_logei",
    "full_gp_warmstart_logei",
    "full_gp_warmstart_strain_025",
    "full_strain_pure",
    "full_logei_strain_025",
    "full_logei_strain_050",
    "full_logei_strain_100",
    "full_logei_no_product_context",
)


def _logei_only(config):
    return replace(
        config,
        policy=replace(
            config.policy,
            early_end=config.dataset.budget,
            final_start=config.dataset.budget + 1,
            challenge_every=0,
        ),
    )


def make_variant(base_config, variant: str, dataset_name: str):
    if variant == "full_metric_aware":
        return base_config
    if variant == "full_logei":
        return _logei_only(base_config)
    if variant in {"gp_logei", "interaction_logei", "trees_logei"}:
        model = variant.removesuffix("_logei")
        config = replace(
            base_config,
            ensemble=replace(base_config.ensemble, enabled_models=(model,)),
        )
        return _logei_only(config)
    if variant == "gp_interaction_logei":
        config = replace(
            base_config,
            ensemble=replace(
                base_config.ensemble,
                enabled_models=("gp", "interaction"),
            ),
        )
        return _logei_only(config)
    if variant == "gp_interaction_gp_warmstart_logei":
        config = replace(
            base_config,
            ensemble=replace(
                base_config.ensemble,
                enabled_models=("gp", "interaction"),
            ),
            policy=replace(base_config.policy, first_query="gp"),
        )
        return _logei_only(config)
    if variant == "full_gp_warmstart_logei":
        config = replace(
            base_config,
            policy=replace(base_config.policy, first_query="gp"),
        )
        return _logei_only(config)
    if variant == "full_gp_warmstart_strain_025":
        config = _logei_only(
            replace(
                base_config,
                policy=replace(base_config.policy, first_query="gp"),
            )
        )
        return replace(
            config,
            policy=replace(
                config.policy,
                novelty_disagreement_mode="hybrid",
                novelty_disagreement_weight=0.25,
            ),
        )
    if variant == "full_strain_pure":
        return replace(
            base_config,
            policy=replace(
                base_config.policy,
                novelty_disagreement_mode="pure",
            ),
        )
    if variant.startswith("full_logei_strain_"):
        weight = int(variant.rsplit("_", 1)[1]) / 100.0
        config = _logei_only(base_config)
        return replace(
            config,
            policy=replace(
                config.policy,
                novelty_disagreement_mode="hybrid",
                novelty_disagreement_weight=weight,
            ),
        )
    if variant == "full_logei_no_product_context":
        if dataset_name != "Suzuki":
            raise ValueError("The product-context ablation applies only to Suzuki.")
        config = replace(
            base_config,
            dataset=replace(base_config.dataset, fixed_roles={}),
        )
        return _logei_only(config)
    raise ValueError(f"Unknown variant: {variant}")


def run_one(job: tuple[str, str, int, int, str, str]) -> dict:
    dataset_name, variant, seed, budget, output_root_string, dataset_root_string = job
    dataset_root = Path(dataset_root_string)
    paths = DATASETS[dataset_name]
    base_config = load_run_config(EXTENSION_ROOT / "configs" / paths["config"])
    if dataset_name == "Suzuki":
        configured_product = base_config.dataset.fixed_roles.get("ProductContext")
        if configured_product != CORRECT_SUZUKI_PRODUCT:
            raise ValueError("Suzuki config does not contain the official corrected product.")
    config = make_variant(base_config, variant, dataset_name)
    spec = DatasetSpec.from_config(config.dataset)
    dataset = load_reaction_dataset(
        dataset_root / paths["train"],
        dataset_root / paths["candidates"],
        spec,
    )
    oracle = OfflineTableOracle.from_csv(dataset, dataset_root / paths["oracle"])
    started = time.perf_counter()
    result = CampaignRunner(dataset, config, seed=seed, backend="native").run(
        oracle,
        budget=budget,
    )
    elapsed = time.perf_counter() - started
    payload = result.to_payload()
    payload["investigation"] = {
        "variant": variant,
        "public_lookup_diagnostic": True,
        "eligible_for_hidden_model_selection": False,
        "selection_warning": (
            "Public labels are used only by the offline oracle. Comparing variants on "
            "these outcomes can overfit this public table and is not hidden-track evidence."
        ),
        "suzuki_product_context": (
            "official_corrected_fixed_context"
            if dataset_name == "Suzuki" and config.dataset.fixed_roles
            else "ablated"
            if dataset_name == "Suzuki"
            else None
        ),
        "strain_sensor_adaptation": (
            "nearest-observation encoded distance times between-model disagreement; "
            "real candidate rows only"
            if "strain" in variant
            else None
        ),
    }
    output_root = Path(output_root_string)
    save_json(
        payload,
        output_root / dataset_name / variant / f"seed_{seed}.json",
    )
    metrics = payload["metrics"]
    return {
        "dataset": dataset_name,
        "variant": variant,
        "seed": seed,
        "budget": budget,
        "first_query_index": payload["trajectory"][0]["query_index"],
        "first_query_yield": metrics["initial_round_found_best"],
        "best_found": metrics["best_found"],
        "auc_best_so_far": metrics["auc_best_so_far"],
        "round_to_95_global_best": metrics["round_to_95_global_best"],
        "final_simple_regret": metrics["final_simple_regret"],
        "elapsed_seconds": elapsed,
    }


def aggregate(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (dataset, variant), group in rows.groupby(["dataset", "variant"], sort=True):
        t95 = pd.to_numeric(group["round_to_95_global_best"], errors="coerce")
        records.append(
            {
                "dataset": dataset,
                "variant": variant,
                "n_seeds": len(group),
                "mean_first_query_yield": group["first_query_yield"].mean(),
                "sd_first_query_yield": group["first_query_yield"].std(ddof=0),
                "mean_best_found": group["best_found"].mean(),
                "sd_best_found": group["best_found"].std(ddof=0),
                "mean_auc_best_so_far": group["auc_best_so_far"].mean(),
                "sd_auc_best_so_far": group["auc_best_so_far"].std(ddof=0),
                "t95_success_rate": t95.notna().mean(),
                "mean_t95_when_reached": t95.mean(),
                "mean_final_simple_regret": group["final_simple_regret"].mean(),
                "total_elapsed_seconds": group["elapsed_seconds"].sum(),
            }
        )
    return pd.DataFrame(records).sort_values(
        ["dataset", "mean_auc_best_so_far", "mean_best_found"],
        ascending=[True, False, False],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=tuple(DATASETS), default=list(DATASETS))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[100, 200, 300])
    parser.add_argument("--budget", type=int, default=40)
    parser.add_argument("--jobs", type=int, default=min(6, os.cpu_count() or 1))
    parser.add_argument("--phase", default="screening")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DATASET_ROOT,
        help="Directory containing the Suzuki/ and Buchwald_sub4/ folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATASET_ROOT / "submission" / "results" / "experimental_investigations" / "sota_search",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.budget <= 40:
        raise ValueError("Budget must be between 1 and 40.")
    if args.jobs <= 0:
        raise ValueError("--jobs must be positive.")
    invalid_seeds = sorted(set(args.seeds) - set(range(100, 2001, 100)))
    if invalid_seeds:
        raise ValueError(f"Only official benchmark seeds are accepted: {invalid_seeds}")

    phase_root = args.output_dir / args.phase
    phase_root.mkdir(parents=True, exist_ok=True)
    jobs = []
    for dataset_name in args.datasets:
        for variant in args.variants:
            if variant == "full_logei_no_product_context" and dataset_name != "Suzuki":
                continue
            for seed in args.seeds:
                jobs.append(
                    (
                        dataset_name,
                        variant,
                        seed,
                        args.budget,
                        str(phase_root),
                        str(args.dataset_root.resolve()),
                    )
                )

    manifest = {
        "phase": args.phase,
        "datasets": args.datasets,
        "variants": args.variants,
        "seeds": args.seeds,
        "budget": args.budget,
        "jobs": args.jobs,
        "dataset_root": str(args.dataset_root.resolve()),
        "public_lookup_diagnostic": True,
        "eligible_for_hidden_model_selection": False,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    save_json(manifest, phase_root / "manifest.json")

    rows = []
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(run_one, job): job for job in jobs}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"{row['dataset']} {row['variant']} seed={row['seed']}: "
                f"first={row['first_query_yield']:.4f}, best={row['best_found']:.4f}, "
                f"AUC={row['auc_best_so_far']:.4f}, "
                f"t95={row['round_to_95_global_best']}, "
                f"elapsed={row['elapsed_seconds']:.1f}s",
                flush=True,
            )

    detail = pd.DataFrame(rows).sort_values(["dataset", "variant", "seed"])
    summary = aggregate(detail)
    detail.to_csv(phase_root / "run_summary.csv", index=False)
    summary.to_csv(phase_root / "aggregate_summary.csv", index=False)
    print("\nRanked aggregate summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
