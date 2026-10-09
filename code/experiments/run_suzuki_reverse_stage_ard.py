#!/usr/bin/env python3
"""Frozen Suzuki pilot with component ARD only for the first query."""

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
from bgolearn_rxn.acquisition import stable_log_expected_improvement  # noqa: E402
from bgolearn_rxn.datasets import DatasetSpec, ReactionDataset  # noqa: E402
from bgolearn_rxn.factory import prepare_problem  # noqa: E402
from bgolearn_rxn.metrics import compute_campaign_metrics  # noqa: E402
from bgolearn_rxn.oracle import OfflineTableOracle  # noqa: E402
from bgolearn_rxn.surrogates.anova_gp import StructuredANOVAGP  # noqa: E402
from bgolearn_rxn.surrogates.component_ard_gp import (  # noqa: E402
    ComponentARDStructuredGP,
)
from bgolearn_rxn.utils import deterministic_argmax  # noqa: E402


VARIANTS = ("fixed", "pure_ard", "ard_q1_fixed_later")
BASELINE = "fixed"
CANDIDATE = "ard_q1_fixed_later"
PURE_ARD = "pure_ard"
CORE_METRICS = (
    "initial_round_found_best",
    "best_found",
    "round_to_95_global_best",
    "AUC_best_so_far",
)
SUMMARY_METRICS = CORE_METRICS + ("elapsed_seconds",)
TRAIN_FILE = DATASET_ROOT / "Suzuki" / "suzuki_train.csv"
CONFIG_FILE = EXTENSION_ROOT / "configs" / "suzuki.yaml"
PILOT_SPLIT_SEED_START = 710_000
PROTECTED_CONFIRMATION_SEED_START = 810_000
PILOT_EPISODE_COUNT = 60
IMPLEMENTATION_CHECK_COUNT = 4
PILOT_GATE = {
    "minimum_initial_delta_vs_fixed": 0.0,
    "minimum_best_delta_vs_fixed": 0.0,
    "minimum_auc_delta_vs_fixed": 0.0,
    "maximum_t95_delta_vs_fixed": 0.0,
    "minimum_runtime_reduction_seconds_vs_pure_ard": 0.25,
    "minimum_runtime_reduction_fraction_vs_pure_ard": 0.10,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_pilot_episodes() -> list[dict]:
    frame = pd.read_csv(TRAIN_FILE)
    episodes = suzuki_proxy._phase_random_episodes(
        frame,
        phase="reverse_stage_pilot",
        seed_start=PILOT_SPLIT_SEED_START,
        count=PILOT_EPISODE_COUNT,
    )
    if len(episodes) != PILOT_EPISODE_COUNT:
        raise RuntimeError("Reverse-stage pilot episode count changed.")
    split_seeds = {int(item["split_seed"]) for item in episodes}
    if len(split_seeds) != PILOT_EPISODE_COUNT:
        raise RuntimeError("Reverse-stage split seeds are not unique.")
    if min(split_seeds) < PILOT_SPLIT_SEED_START:
        raise RuntimeError("Reverse-stage pilot escaped the 710k seed family.")
    if max(split_seeds) >= PROTECTED_CONFIRMATION_SEED_START:
        raise RuntimeError("Reverse-stage pilot overlaps the protected 810k range.")
    return episodes


def episodes_for_stage(stage: str) -> list[dict]:
    episodes = make_pilot_episodes()
    if stage == "implementation_check":
        return episodes[:IMPLEMENTATION_CHECK_COUNT]
    if stage == "pilot":
        return episodes
    raise ValueError(f"Unknown stage: {stage}")


def _copy_dataset(dataset: ReactionDataset) -> ReactionDataset:
    return ReactionDataset(
        spec=DatasetSpec(
            name=dataset.spec.name,
            feature_columns=dataset.spec.feature_columns,
            target_column=dataset.spec.target_column,
            task_column=dataset.spec.task_column,
            target_task_value=dataset.spec.target_task_value,
            condition_roles=dataset.spec.condition_roles,
            derived_roles=dict(dataset.spec.derived_roles),
            fixed_roles=dict(dataset.spec.fixed_roles),
            budget=dataset.spec.budget,
            seeds=dataset.spec.seeds,
        ),
        initial=dataset.initial.copy(),
        candidates=dataset.candidates.copy(),
    )


def build_problem(episode: dict):
    config, dataset, lookup = suzuki_proxy.build_problem(episode, "gp_only")
    budget = int(episode["budget"])
    config = replace(
        config,
        gp=replace(config.gp, target_transform="standard"),
        policy=replace(
            config.policy,
            early_end=budget,
            final_start=budget + 1,
            challenge_every=0,
            novelty_disagreement_mode="off",
        ),
    )
    return config, _copy_dataset(dataset), lookup.copy()


def make_fixed_model(schema, config, seed: int):
    return StructuredANOVAGP(schema, config.gp, seed=seed + 11)


def make_ard_model(schema, config, seed: int):
    return ComponentARDStructuredGP(schema, config.gp, seed=seed + 17)


def model_family_for_step(variant: str, step: int) -> str:
    if variant == BASELINE:
        return "fixed"
    if variant == PURE_ARD:
        return "component_ard"
    if variant == CANDIDATE:
        return "component_ard" if step == 1 else "fixed"
    raise ValueError(f"Unknown surrogate schedule: {variant}")


def campaign_metrics(trajectory: list[dict], global_best: float, budget: int) -> dict:
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
    config, dataset, lookup = build_problem(episode)
    prepared = prepare_problem(dataset, config)
    schema = prepared.encoder.schema
    seed = int(episode["campaign_seed"])
    fixed_model = make_fixed_model(schema, config, seed) if variant == BASELINE else None
    ard_model = make_ard_model(schema, config, seed) if variant in {PURE_ARD, CANDIDATE} else None
    creation_steps: dict[str, int] = {}
    if fixed_model is not None:
        creation_steps["fixed"] = 0
    if ard_model is not None:
        creation_steps["component_ard"] = 0

    oracle = OfflineTableOracle(dataset=dataset, lookup=lookup)
    observed = dataset.initial.copy()
    X_observed = prepared.X_initial.copy()
    y_observed = observed[dataset.spec.target_column].to_numpy(dtype=float)
    remaining = np.arange(len(dataset.candidates), dtype=int)
    trajectory: list[dict] = []
    budget = int(episode["budget"])
    started = time.perf_counter()

    for step in range(1, budget + 1):
        family = model_family_for_step(variant, step)
        if family == "fixed":
            if fixed_model is None:
                # The reverse schedule constructs fixed GP after query-1 feedback.
                fixed_model = make_fixed_model(schema, config, seed)
                creation_steps["fixed"] = step
            model = fixed_model
        else:
            if ard_model is None:
                raise RuntimeError("Component-ARD model was not initialized.")
            model = ard_model

        X_remaining = prepared.X_candidates[remaining]
        candidate_rows = dataset.candidates.iloc[remaining]
        model.fit(X_observed, y_observed)
        posterior = model.predict(X_remaining)
        if step == 1:
            scores = posterior.mean.copy()
            acquisition_name = "posterior_mean"
        else:
            progress = (step - 1) / (budget - 1)
            scores = stable_log_expected_improvement(
                posterior.mean,
                posterior.std,
                dataset.target_incumbent(observed),
                xi=0.05 * (1.0 - progress),
            )
            acquisition_name = "stable_log_ei"

        candidate_ids = candidate_rows["query_index"].to_numpy(dtype=int)
        local_index = deterministic_argmax(scores, candidate_ids, seed=seed + step)
        query_index = int(candidate_ids[local_index])
        observed_yield = float(oracle.query(query_index))
        selected_global = int(remaining[local_index])
        selected_row = dataset.candidates.iloc[selected_global]
        trajectory.append(
            {
                "step": step,
                "query_index": query_index,
                "observed_yield": observed_yield,
                "predicted_yield": float(posterior.mean[local_index]),
                "predicted_std": float(posterior.std[local_index]),
                "model_family": family,
                "acquisition_name": acquisition_name,
                "acquisition_value": float(scores[local_index]),
                "component_amplitudes": dict(
                    getattr(model, "component_amplitudes_", {})
                )
                or None,
                "component_shares": dict(getattr(model, "component_shares_", {}))
                or None,
            }
        )
        new_observation = selected_row.loc[
            list(dataset.spec.feature_columns)
        ].to_dict()
        new_observation[dataset.spec.target_column] = observed_yield
        observed = pd.concat(
            [observed, pd.DataFrame([new_observation])], ignore_index=True
        )
        X_observed = np.vstack([X_observed, prepared.X_candidates[selected_global]])
        y_observed = np.append(y_observed, observed_yield)
        remaining = np.delete(remaining, local_index)

    elapsed = time.perf_counter() - started
    global_best = float(oracle.global_best)
    metrics = campaign_metrics(trajectory, global_best, budget)
    amplitudes = dict(getattr(ard_model, "component_amplitudes_", {}))
    shares = dict(getattr(ard_model, "component_shares_", {}))
    schedule = [model_family_for_step(variant, step) for step in range(1, budget + 1)]
    acquisition_schedule = ["posterior_mean"] + ["stable_log_ei"] * (budget - 1)
    payload = {
        "dataset": "Suzuki",
        "variant": variant,
        "episode": episode,
        "trajectory": trajectory,
        "metrics": metrics,
        "contract": {
            "train_only": True,
            "public_test_labels_used": False,
            "candidate_information_identical": True,
            "input_file": str(TRAIN_FILE.relative_to(DATASET_ROOT)),
            "input_sha256": file_sha256(TRAIN_FILE),
            "config_file": str(CONFIG_FILE.relative_to(DATASET_ROOT)),
            "config_sha256": file_sha256(CONFIG_FILE),
            "target_transform": "standard",
            "schedule": schedule,
            "acquisition_schedule": acquisition_schedule,
            "model_creation_steps": creation_steps,
            "episode": episode,
        },
    }
    output_path = (
        Path(output_root_string)
        / "trajectories"
        / variant
        / f"{episode['episode']}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    first = trajectory[0]
    return {
        "dataset": "Suzuki",
        "phase": episode["phase"],
        "episode_family": episode["episode_family"],
        "episode": episode["episode"],
        "split_seed": episode["split_seed"],
        "campaign_seed": episode["campaign_seed"],
        "variant": variant,
        "budget": budget,
        "global_best": global_best,
        **metrics,
        "elapsed_seconds": elapsed,
        "first_query_index": first["query_index"],
        "first_query_predicted_yield": first["predicted_yield"],
        "first_query_predicted_std": first["predicted_std"],
        "first_query_acquisition_value": first["acquisition_value"],
        "schedule_json": json.dumps(schedule),
        "acquisition_schedule_json": json.dumps(acquisition_schedule),
        "model_creation_steps_json": json.dumps(creation_steps, sort_keys=True),
        "component_amplitudes_json": (
            json.dumps(amplitudes, sort_keys=True) if amplitudes else None
        ),
        "component_shares_json": json.dumps(shares, sort_keys=True) if shares else None,
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
    records: list[dict] = []
    for variant, group in rows.groupby("variant", sort=True):
        for metric in SUMMARY_METRICS:
            mean, sd, low, high = mean_ci(group[metric])
            records.append(
                {
                    "dataset": "Suzuki",
                    "phase": "reverse_stage_pilot",
                    "episode_family": "random_holdout",
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


def paired_summary(rows: pd.DataFrame, baseline_name: str) -> pd.DataFrame:
    baseline = rows[rows["variant"].eq(baseline_name)].set_index("episode")
    candidate = rows[rows["variant"].eq(CANDIDATE)].set_index("episode")
    common = baseline.index.intersection(candidate.index)
    records: list[dict] = []
    for metric in SUMMARY_METRICS:
        delta = candidate.loc[common, metric] - baseline.loc[common, metric]
        mean, sd, low, high = mean_ci(delta)
        higher = metric not in {"round_to_95_global_best", "elapsed_seconds"}
        wins = int((delta > 1e-10).sum()) if higher else int((delta < -1e-10).sum())
        losses = int((delta < -1e-10).sum()) if higher else int((delta > 1e-10).sum())
        records.append(
            {
                "dataset": "Suzuki",
                "phase": "reverse_stage_pilot",
                "episode_family": "random_holdout",
                "candidate": CANDIDATE,
                "baseline": baseline_name,
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


def pilot_decision(rows: pd.DataFrame) -> dict:
    versus_fixed = paired_summary(rows, BASELINE).set_index("metric")
    versus_ard = paired_summary(rows, PURE_ARD).set_index("metric")
    initial = float(versus_fixed.loc["initial_round_found_best", "mean_delta"])
    best = float(versus_fixed.loc["best_found", "mean_delta"])
    auc = float(versus_fixed.loc["AUC_best_so_far", "mean_delta"])
    t95 = float(versus_fixed.loc["round_to_95_global_best", "mean_delta"])
    runtime_delta = float(versus_ard.loc["elapsed_seconds", "mean_delta"])
    ard_runtime = float(
        rows.loc[rows["variant"].eq(PURE_ARD), "elapsed_seconds"].mean()
    )
    runtime_reduction = -runtime_delta
    runtime_fraction = runtime_reduction / ard_runtime
    checks = {
        "initial_noninferior_vs_fixed": initial
        >= PILOT_GATE["minimum_initial_delta_vs_fixed"],
        "best_noninferior_vs_fixed": best
        >= PILOT_GATE["minimum_best_delta_vs_fixed"],
        "auc_noninferior_vs_fixed": auc
        >= PILOT_GATE["minimum_auc_delta_vs_fixed"],
        "t95_noninferior_vs_fixed": t95
        <= PILOT_GATE["maximum_t95_delta_vs_fixed"],
        "runtime_reduction_seconds_vs_pure_ard": runtime_reduction
        >= PILOT_GATE["minimum_runtime_reduction_seconds_vs_pure_ard"],
        "runtime_reduction_fraction_vs_pure_ard": runtime_fraction
        >= PILOT_GATE["minimum_runtime_reduction_fraction_vs_pure_ard"],
    }
    return {
        "gate": PILOT_GATE,
        "checks": checks,
        "passed": bool(all(checks.values())),
        "decision": "eligible_for_untouched_confirmation" if all(checks.values()) else "reject",
        "confirmation_executed": False,
        "production_default_changed": False,
        "public_test_labels_used": False,
        "mean_deltas_vs_fixed": {
            metric: float(versus_fixed.loc[metric, "mean_delta"])
            for metric in SUMMARY_METRICS
        },
        "mean_deltas_vs_pure_ard": {
            metric: float(versus_ard.loc[metric, "mean_delta"])
            for metric in SUMMARY_METRICS
        },
        "runtime_reduction_seconds_vs_pure_ard": runtime_reduction,
        "runtime_reduction_fraction_vs_pure_ard": runtime_fraction,
    }


def amplitude_summary(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    available = rows.dropna(subset=["component_shares_json"])
    for variant, group in available.groupby("variant", sort=True):
        parsed = [json.loads(value) for value in group["component_shares_json"]]
        for component in sorted(set().union(*(item.keys() for item in parsed))):
            values = pd.Series([item[component] for item in parsed], dtype=float)
            mean, sd, low, high = mean_ci(values)
            records.append(
                {
                    "dataset": "Suzuki",
                    "phase": "reverse_stage_pilot",
                    "variant": variant,
                    "component": component,
                    "n_episodes": len(values),
                    "mean_share": mean,
                    "sample_sd_share": sd,
                    "ci95_low": low,
                    "ci95_high": high,
                    "min_share": float(values.min()),
                    "max_share": float(values.max()),
                }
            )
    return pd.DataFrame(records)


def schedule_audit(rows: pd.DataFrame) -> dict:
    expected = {
        variant: [model_family_for_step(variant, step) for step in range(1, 6)]
        for variant in VARIANTS
    }
    expected_acquisition = ["posterior_mean"] + ["stable_log_ei"] * 4
    schedule_matches = bool(
        all(
            json.loads(row.schedule_json) == expected[row.variant]
            for row in rows.itertuples()
        )
    )
    acquisition_matches = bool(
        all(
            json.loads(value) == expected_acquisition
            for value in rows["acquisition_schedule_json"]
        )
    )
    first = rows.pivot(index="episode", columns="variant", values="first_query_index")
    q1_ids_match = bool((first[CANDIDATE] == first[PURE_ARD]).all())
    creation = {
        variant: sorted(
            {
                tuple(sorted(json.loads(value).items()))
                for value in rows.loc[
                    rows["variant"].eq(variant), "model_creation_steps_json"
                ]
            }
        )
        for variant in VARIANTS
    }
    candidate_creation_ok = creation[CANDIDATE] == [
        (("component_ard", 0), ("fixed", 2))
    ]
    return {
        "expected_schedules": expected,
        "expected_acquisition_schedule": expected_acquisition,
        "all_observed_schedules_match": schedule_matches,
        "all_acquisition_schedules_match": acquisition_matches,
        "candidate_query1_matches_pure_ard": q1_ids_match,
        "candidate_fixed_created_after_query1_at_step2": candidate_creation_ok,
        "observed_model_creation_steps": {
            variant: [dict(items) for items in values]
            for variant, values in creation.items()
        },
        "passed": bool(
            schedule_matches
            and acquisition_matches
            and q1_ids_match
            and candidate_creation_ok
        ),
    }


def explicit_twenty(rows: pd.DataFrame) -> pd.DataFrame:
    episode_ids = {
        f"random_reverse_stage_pilot_{index:02d}" for index in range(20)
    }
    view = rows[rows["episode"].isin(episode_ids)].copy()
    if len(view) != 20 * len(VARIANTS) or view["episode"].nunique() != 20:
        raise RuntimeError("The deterministic first-20 view is incomplete.")
    return view.sort_values(["episode", "variant"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("implementation_check", "pilot"), required=True
    )
    parser.add_argument("--jobs", type=int, default=min(6, os.cpu_count() or 1))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.jobs <= 0:
        raise ValueError("--jobs must be positive.")
    episodes = episodes_for_stage(args.stage)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_seeds = [int(item["split_seed"]) for item in episodes]
    manifest = {
        "stage": args.stage,
        "train_only": True,
        "public_test_labels_used": False,
        "input": {
            "path": str(TRAIN_FILE.relative_to(DATASET_ROOT)),
            "sha256": file_sha256(TRAIN_FILE),
        },
        "config": {
            "path": str(CONFIG_FILE.relative_to(DATASET_ROOT)),
            "sha256": file_sha256(CONFIG_FILE),
        },
        "runner_sha256": file_sha256(Path(__file__)),
        "variants": list(VARIANTS),
        "schedules": {
            variant: [model_family_for_step(variant, step) for step in range(1, 6)]
            for variant in VARIANTS
        },
        "target_transform": "standard",
        "acquisition": "posterior mean query 1, stable LogEI queries 2-5",
        "episode_count": len(episodes),
        "pilot_split_seed_start": PILOT_SPLIT_SEED_START,
        "split_seed_min": min(split_seeds),
        "split_seed_max": max(split_seeds),
        "protected_confirmation_seed_start": PROTECTED_CONFIRMATION_SEED_START,
        "confirmation_executed": False,
        "parameter_sweep": False,
        "interim_selection": False,
        "pilot_gate": PILOT_GATE,
        "t95_global_best": "full train partition via OfflineTableOracle.global_best",
        "t95_miss_penalty": "episode budget + 1",
        "production_default_changed": False,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    jobs = [
        (variant, episode, str(args.output_dir))
        for episode in episodes
        for variant in VARIANTS
    ]
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(run_one, job): job for job in jobs}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"{args.stage} {row['variant']} {row['episode']}: "
                f"AUC={row['AUC_best_so_far']:.3f}",
                flush=True,
            )

    detail = pd.DataFrame(rows).sort_values(["episode", "variant"])
    expected_rows = len(episodes) * len(VARIANTS)
    if len(detail) != expected_rows:
        raise RuntimeError("Reverse-stage result count is incomplete.")
    if not detail.groupby("episode")["split_seed"].nunique().eq(1).all():
        raise RuntimeError("Paired variants do not share split seeds.")
    audit = schedule_audit(detail)
    if not audit["passed"]:
        raise RuntimeError(f"Schedule audit failed: {audit}")

    detail.to_csv(args.output_dir / "per_episode.csv", index=False)
    summarize(detail).to_csv(args.output_dir / "summary.csv", index=False)
    paired_summary(detail, BASELINE).to_csv(
        args.output_dir / "paired_candidate_vs_fixed.csv", index=False
    )
    paired_summary(detail, PURE_ARD).to_csv(
        args.output_dir / "paired_candidate_vs_pure_ard.csv", index=False
    )
    amplitude_summary(detail).to_csv(
        args.output_dir / "amplitude_stability.csv", index=False
    )
    (args.output_dir / "schedule_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )

    if args.stage == "pilot":
        decision = pilot_decision(detail)
        (args.output_dir / "pilot_decision.json").write_text(
            json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8"
        )
        twenty = explicit_twenty(detail)
        twenty.to_csv(args.output_dir / "twenty_run_per_episode.csv", index=False)
        summarize(twenty).to_csv(
            args.output_dir / "twenty_run_summary.csv", index=False
        )
        paired_summary(twenty, BASELINE).to_csv(
            args.output_dir / "twenty_run_paired_candidate_vs_fixed.csv",
            index=False,
        )
        paired_summary(twenty, PURE_ARD).to_csv(
            args.output_dir / "twenty_run_paired_candidate_vs_pure_ard.csv",
            index=False,
        )


if __name__ == "__main__":
    main()
