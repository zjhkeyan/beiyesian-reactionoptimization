#!/usr/bin/env python3
"""Calibrate Suzuki component-ARD shrinkage on 30 episodes, then validate once."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENT_ROOT = Path(__file__).resolve().parent
CODE_ROOT = EXPERIMENT_ROOT.parent
DATASET_ROOT = CODE_ROOT.parents[1]
EXTENSION_ROOT = CODE_ROOT / "bgolearn_rxn_extension"
sys.path.insert(0, str(EXTENSION_ROOT / "src"))
sys.path.insert(0, str(EXPERIMENT_ROOT))

import run_suzuki_amplitude_distillation as amplitude_study  # noqa: E402
from bgolearn_rxn.surrogates.anova_gp import StructuredANOVAGP  # noqa: E402
from bgolearn_rxn.surrogates.component_ard_gp import (  # noqa: E402
    ComponentARDStructuredGP,
)


SCALES = (0.25, 0.40, 0.60, 0.90)
DEFAULT_SCALE = 0.60
FIXED = "fixed"
DEFAULT_ARD = "default_ard"
SELECTED_ARD = "selected_ard"
CORE_METRICS = amplitude_study.CORE_METRICS
SUMMARY_METRICS = amplitude_study.SUMMARY_METRICS
TRAIN_FILE = amplitude_study.TRAIN_FILE
CONFIG_FILE = amplitude_study.CONFIG_FILE
PROTECTED_SEED_FAMILIES = (610_000, 810_000)
SELECTION_RULE = (
    "highest mean AUC_best_so_far; then lowest mean penalized t95; then lower "
    "episode-level AUC loss frequency versus fixed GP; then stronger shrinkage"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scale_variant(scale: float) -> str:
    if scale not in SCALES:
        raise ValueError(f"Scale {scale} is outside the frozen calibration grid.")
    return f"ard_scale_{scale:.2f}".replace(".", "p")


DEVELOPMENT_VARIANTS = (FIXED,) + tuple(scale_variant(scale) for scale in SCALES)
VALIDATION_VARIANTS = (FIXED, DEFAULT_ARD, SELECTED_ARD)


class CalibratedShrinkageComponentARDGP(ComponentARDStructuredGP):
    """Experiment-local component ARD with a frozen log-amplitude prior scale."""

    def __init__(self, schema, config, scale: float, *, seed: int = 0) -> None:
        if scale not in SCALES:
            raise ValueError(f"Scale {scale} is outside the frozen calibration grid.")
        super().__init__(schema, config, seed=seed)
        self.amplitude_prior_scale = float(scale)
        self.name = f"component_ard_gp_scale_{scale:.2f}"


def make_episode_partitions() -> tuple[list[dict], list[dict]]:
    development, validation = amplitude_study.make_episode_partitions()
    if [item["episode"] for item in development] != [
        f"random_amplitude_distillation_{index:02d}" for index in range(30)
    ]:
        raise RuntimeError("Development episode IDs differ from the frozen contract.")
    if [item["episode"] for item in validation] != [
        f"random_amplitude_distillation_{index:02d}" for index in range(30, 60)
    ]:
        raise RuntimeError("Validation episode IDs differ from the frozen contract.")
    all_seeds = [int(item["split_seed"]) for item in development + validation]
    if any(610_000 <= seed < 710_000 or seed >= 810_000 for seed in all_seeds):
        raise RuntimeError("Episode split entered a protected seed family.")
    return development, validation


def _make_model(variant: str, schema, config, seed: int, scale: float | None):
    if variant == FIXED:
        if scale is not None:
            raise ValueError("Fixed GP cannot receive an amplitude prior scale.")
        return StructuredANOVAGP(schema, config.gp, seed=seed + 11)
    if scale is None:
        raise ValueError("Component ARD requires an amplitude prior scale.")
    return CalibratedShrinkageComponentARDGP(
        schema, config.gp, scale, seed=seed + 17
    )


def _run_job(job: tuple[str, dict, str, float | None, str]) -> dict:
    variant, episode, output_root, scale, phase = job
    original_factory = amplitude_study.make_model
    amplitude_study.make_model = (
        lambda _variant, schema, config, seed, _frozen: _make_model(
            variant, schema, config, seed, scale
        )
    )
    try:
        row, _ = amplitude_study.run_one((variant, episode, output_root, None))
    finally:
        amplitude_study.make_model = original_factory
    row["phase"] = phase
    row["amplitude_prior_scale"] = scale

    trajectory_path = (
        Path(output_root) / "trajectories" / variant / f"{episode['episode']}.json"
    )
    payload = json.loads(trajectory_path.read_text(encoding="utf-8"))
    payload["contract"].update(
        {
            "amplitude_prior_scale": scale,
            "amplitude_prior_scale_adaptive": scale is not None,
            "component_amplitudes_frozen": False,
            "production_default_changed": False,
            "protected_seed_families_not_used": list(PROTECTED_SEED_FAMILIES),
        }
    )
    trajectory_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return row


def execute_jobs(
    specifications: tuple[tuple[str, float | None], ...],
    episodes: list[dict],
    output_dir: Path,
    jobs_count: int,
    phase: str,
) -> pd.DataFrame:
    jobs = [
        (variant, episode, str(output_dir), scale, phase)
        for episode in episodes
        for variant, scale in specifications
    ]
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=jobs_count) as executor:
        futures = {executor.submit(_run_job, job): job for job in jobs}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"{phase} {row['variant']} {row['episode']}: "
                f"AUC={row['AUC_best_so_far']:.3f}",
                flush=True,
            )
    detail = pd.DataFrame(rows).sort_values(["episode", "variant"])
    expected = len(episodes) * len(specifications)
    if len(detail) != expected:
        raise RuntimeError(f"Expected {expected} result rows, found {len(detail)}.")
    if not detail.groupby("episode")["split_seed"].nunique().eq(1).all():
        raise RuntimeError("Variants do not share split seeds within episodes.")
    return detail


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    result = amplitude_study.summarize(rows)
    result["phase"] = rows["phase"].iloc[0]
    return result


def paired_summary(
    rows: pd.DataFrame, candidate: str, baseline: str
) -> pd.DataFrame:
    result = amplitude_study.paired_summary(rows, candidate, baseline)
    result["phase"] = rows["phase"].iloc[0]
    return result


def amplitude_diagnostics(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    for row in rows.itertuples(index=False):
        if pd.isna(row.amplitude_prior_scale):
            continue
        amplitudes = json.loads(row.final_component_amplitudes_json)
        values = np.asarray(list(amplitudes.values()), dtype=float)
        shares = values / float(values.sum())
        records.append(
            {
                "phase": row.phase,
                "episode": row.episode,
                "variant": row.variant,
                "amplitude_prior_scale": row.amplitude_prior_scale,
                "n_components": len(values),
                "amplitude_min": float(values.min()),
                "amplitude_max": float(values.max()),
                "amplitude_geometric_mean": float(np.exp(np.log(values).mean())),
                "log_amplitude_sample_sd": float(np.log(values).std(ddof=1)),
                "share_entropy": float(-np.sum(shares * np.log(shares))),
                "effective_components": float(np.exp(-np.sum(shares * np.log(shares)))),
            }
        )
    return pd.DataFrame(records).sort_values(["episode", "variant"])


def development_selection(rows: pd.DataFrame) -> dict:
    fixed = rows[rows["variant"].eq(FIXED)].set_index("episode")
    candidates: list[dict] = []
    for scale in SCALES:
        variant = scale_variant(scale)
        current = rows[rows["variant"].eq(variant)].set_index("episode")
        if len(current) != 30 or not current.index.equals(fixed.index):
            raise RuntimeError(f"Development pairing is incomplete for {variant}.")
        auc_losses = current["AUC_best_so_far"] < fixed["AUC_best_so_far"] - 1e-10
        candidates.append(
            {
                "variant": variant,
                "amplitude_prior_scale": scale,
                "mean_initial_round_found_best": float(
                    current["initial_round_found_best"].mean()
                ),
                "mean_best_found": float(current["best_found"].mean()),
                "mean_round_to_95_global_best": float(
                    current["round_to_95_global_best"].mean()
                ),
                "mean_AUC_best_so_far": float(current["AUC_best_so_far"].mean()),
                "auc_loss_frequency_vs_fixed": float(auc_losses.mean()),
                "auc_loss_count_vs_fixed": int(auc_losses.sum()),
                "mean_elapsed_seconds": float(current["elapsed_seconds"].mean()),
            }
        )
    ranked = sorted(
        candidates,
        key=lambda item: (
            -item["mean_AUC_best_so_far"],
            item["mean_round_to_95_global_best"],
            item["auc_loss_frequency_vs_fixed"],
            item["amplitude_prior_scale"],
        ),
    )
    selected = ranked[0]
    return {
        "decision": "freeze_selected_scale_for_sealed_validation",
        "selection_rule": SELECTION_RULE,
        "selected_variant": selected["variant"],
        "selected_amplitude_prior_scale": selected["amplitude_prior_scale"],
        "ranked_candidates": ranked,
        "development_episode_ids": sorted(fixed.index.tolist()),
        "validation_episode_ids_used_for_selection": [],
        "input_sha256": file_sha256(TRAIN_FILE),
        "config_sha256": file_sha256(CONFIG_FILE),
        "runner_sha256": file_sha256(Path(__file__)),
        "train_only": True,
        "public_test_labels_used": False,
        "production_default_changed": False,
        "protected_seed_families_not_used": list(PROTECTED_SEED_FAMILIES),
    }


def validate_selection(selection: dict, development: list[dict]) -> float:
    expected_ids = [item["episode"] for item in development]
    if selection["development_episode_ids"] != expected_ids:
        raise ValueError("Selection file has unexpected development episodes.")
    if selection["validation_episode_ids_used_for_selection"]:
        raise ValueError("Selection used sealed validation episodes.")
    if selection["input_sha256"] != file_sha256(TRAIN_FILE):
        raise ValueError("Selection file uses a different train table.")
    if selection["config_sha256"] != file_sha256(CONFIG_FILE):
        raise ValueError("Selection file uses a different config.")
    if selection["public_test_labels_used"] is not False:
        raise ValueError("Selection provenance is not train-only.")
    scale = float(selection["selected_amplitude_prior_scale"])
    if scale not in SCALES:
        raise ValueError("Selected scale is outside the frozen grid.")
    return scale


def validation_gate(rows: pd.DataFrame, selected_scale: float) -> dict:
    selected_default = paired_summary(rows, SELECTED_ARD, DEFAULT_ARD).set_index(
        "metric"
    )
    selected_fixed = paired_summary(rows, SELECTED_ARD, FIXED).set_index("metric")
    checks = {
        "auc_better_than_default_ard": float(
            selected_default.loc["AUC_best_so_far", "mean_delta"]
        )
        > 0.0,
        "t95_better_than_default_ard": float(
            selected_default.loc["round_to_95_global_best", "mean_delta"]
        )
        < 0.0,
        "initial_noninferior_to_fixed": float(
            selected_fixed.loc["initial_round_found_best", "mean_delta"]
        )
        >= 0.0,
        "best_noninferior_to_fixed": float(
            selected_fixed.loc["best_found", "mean_delta"]
        )
        >= 0.0,
        "auc_noninferior_to_fixed": float(
            selected_fixed.loc["AUC_best_so_far", "mean_delta"]
        )
        >= 0.0,
        "t95_noninferior_to_fixed": float(
            selected_fixed.loc["round_to_95_global_best", "mean_delta"]
        )
        <= 0.0,
    }
    return {
        "gate": {
            "selected_vs_default": "AUC delta > 0 and penalized t95 delta < 0",
            "selected_vs_fixed": (
                "initial/best/AUC deltas >= 0 and penalized t95 delta <= 0"
            ),
        },
        "checks": checks,
        "passed": bool(all(checks.values())),
        "decision": "pilot_pass" if all(checks.values()) else "reject",
        "selected_amplitude_prior_scale": selected_scale,
        "confirmation_executed": False,
        "public_test_labels_used": False,
        "production_default_changed": False,
    }


def explicit_twenty(rows: pd.DataFrame) -> pd.DataFrame:
    episode_ids = {
        f"random_amplitude_distillation_{index:02d}" for index in range(30, 50)
    }
    result = rows[rows["episode"].isin(episode_ids)].copy()
    if len(result) != 20 * len(VALIDATION_VARIANTS):
        raise RuntimeError("Explicit first-20 validation slice is incomplete.")
    return result.sort_values(["episode", "variant"])


def common_manifest(stage: str) -> dict:
    return {
        "stage": stage,
        "train_only": True,
        "public_test_labels_used": False,
        "confirmation_executed": False,
        "production_default_changed": False,
        "input": {
            "path": str(TRAIN_FILE.relative_to(DATASET_ROOT)),
            "sha256": file_sha256(TRAIN_FILE),
        },
        "config": {
            "path": str(CONFIG_FILE.relative_to(DATASET_ROOT)),
            "sha256": file_sha256(CONFIG_FILE),
        },
        "runner_sha256": file_sha256(Path(__file__)),
        "protected_seed_families_not_used": list(PROTECTED_SEED_FAMILIES),
        "target_transform": "standard",
        "acquisition": "posterior mean query 1, stable LogEI queries 2-5",
        "frozen_scale_grid": list(SCALES),
        "production_component_ard_scale": ComponentARDStructuredGP.amplitude_prior_scale,
        "selection_rule": SELECTION_RULE,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("development", "validation"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection-file", type=Path)
    parser.add_argument("--jobs", type=int, default=min(6, os.cpu_count() or 1))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.jobs <= 0:
        raise ValueError("--jobs must be positive.")
    development, validation = make_episode_partitions()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.stage == "development":
        if args.selection_file is not None:
            raise ValueError("Development does not accept --selection-file.")
        specifications = ((FIXED, None),) + tuple(
            (scale_variant(scale), scale) for scale in SCALES
        )
        manifest = {
            **common_manifest("development"),
            "episode_count": len(development),
            "episode_ids": [item["episode"] for item in development],
            "split_seed_min": min(int(item["split_seed"]) for item in development),
            "split_seed_max": max(int(item["split_seed"]) for item in development),
            "variants": [variant for variant, _ in specifications],
            "sealed_validation_episode_count": len(validation),
            "validation_outcomes_loaded": False,
        }
        (args.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        rows = execute_jobs(
            specifications, development, args.output_dir, args.jobs, "shrinkage_development"
        )
        rows.to_csv(args.output_dir / "per_episode.csv", index=False)
        summarize(rows).to_csv(args.output_dir / "summary.csv", index=False)
        amplitude_diagnostics(rows).to_csv(
            args.output_dir / "amplitude_diagnostics.csv", index=False
        )
        for scale in SCALES:
            variant = scale_variant(scale)
            paired_summary(rows, variant, FIXED).to_csv(
                args.output_dir / f"paired_{variant}_vs_fixed.csv", index=False
            )
        selection = development_selection(rows)
        (args.output_dir / "selection_decision.json").write_text(
            json.dumps(selection, indent=2, sort_keys=True), encoding="utf-8"
        )
        return

    if args.selection_file is None:
        raise ValueError("Validation requires --selection-file.")
    selection = json.loads(args.selection_file.read_text(encoding="utf-8"))
    selected_scale = validate_selection(selection, development)
    specifications = (
        (FIXED, None),
        (DEFAULT_ARD, DEFAULT_SCALE),
        (SELECTED_ARD, selected_scale),
    )
    manifest = {
        **common_manifest("validation"),
        "episode_count": len(validation),
        "episode_ids": [item["episode"] for item in validation],
        "split_seed_min": min(int(item["split_seed"]) for item in validation),
        "split_seed_max": max(int(item["split_seed"]) for item in validation),
        "variants": [variant for variant, _ in specifications],
        "selected_amplitude_prior_scale": selected_scale,
        "selection_file": str(args.selection_file),
        "selection_sha256": file_sha256(args.selection_file),
        "selection_provenance_validated": True,
        "interim_selection": False,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    rows = execute_jobs(
        specifications, validation, args.output_dir, args.jobs, "shrinkage_validation"
    )
    rows.to_csv(args.output_dir / "per_episode.csv", index=False)
    summarize(rows).to_csv(args.output_dir / "summary.csv", index=False)
    amplitude_diagnostics(rows).to_csv(
        args.output_dir / "amplitude_diagnostics.csv", index=False
    )
    for candidate, baseline, filename in (
        (SELECTED_ARD, DEFAULT_ARD, "paired_selected_vs_default_ard.csv"),
        (SELECTED_ARD, FIXED, "paired_selected_vs_fixed.csv"),
        (DEFAULT_ARD, FIXED, "paired_default_ard_vs_fixed.csv"),
    ):
        paired_summary(rows, candidate, baseline).to_csv(
            args.output_dir / filename, index=False
        )
    gate = validation_gate(rows, selected_scale)
    (args.output_dir / "pilot_decision.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True), encoding="utf-8"
    )
    twenty = explicit_twenty(rows)
    twenty.to_csv(args.output_dir / "twenty_run_per_episode.csv", index=False)
    summarize(twenty).to_csv(args.output_dir / "twenty_run_summary.csv", index=False)
    for candidate, baseline, filename in (
        (SELECTED_ARD, DEFAULT_ARD, "twenty_run_paired_selected_vs_default_ard.csv"),
        (SELECTED_ARD, FIXED, "twenty_run_paired_selected_vs_fixed.csv"),
        (DEFAULT_ARD, FIXED, "twenty_run_paired_default_ard_vs_fixed.csv"),
    ):
        paired_summary(twenty, candidate, baseline).to_csv(
            args.output_dir / filename, index=False
        )


if __name__ == "__main__":
    main()
