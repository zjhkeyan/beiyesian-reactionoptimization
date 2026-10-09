#!/usr/bin/env python3
"""Apply the frozen phase-2 LogEI/Thompson schedule to fixed and ARD GPs."""

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
import run_suzuki_hybrid_acquisition as frozen_hybrid  # noqa: E402
from bgolearn_rxn.factory import prepare_problem  # noqa: E402
from bgolearn_rxn.metrics import compute_campaign_metrics  # noqa: E402
from bgolearn_rxn.oracle import OfflineTableOracle  # noqa: E402
from bgolearn_rxn.surrogates.anova_gp import StructuredANOVAGP  # noqa: E402
from bgolearn_rxn.surrogates.component_ard_gp import (  # noqa: E402
    ComponentARDStructuredGP,
)
from bgolearn_rxn.utils import deterministic_argmax  # noqa: E402


FIXED_LOGEI = "fixed_logei"
FIXED_HYBRID = "fixed_hybrid"
ARD_LOGEI = "ard_logei"
ARD_HYBRID = "ard_hybrid"
VARIANTS = (FIXED_LOGEI, FIXED_HYBRID, ARD_LOGEI, ARD_HYBRID)
CANDIDATE = ARD_HYBRID
SURROGATE = {
    FIXED_LOGEI: "fixed",
    FIXED_HYBRID: "fixed",
    ARD_LOGEI: "component_ard",
    ARD_HYBRID: "component_ard",
}
POLICY = {
    FIXED_LOGEI: frozen_hybrid.BASELINE_POLICY,
    FIXED_HYBRID: frozen_hybrid.HYBRID_POLICY,
    ARD_LOGEI: frozen_hybrid.BASELINE_POLICY,
    ARD_HYBRID: frozen_hybrid.HYBRID_POLICY,
}
CORE_METRICS = (
    "initial_round_found_best",
    "best_found",
    "round_to_95_global_best",
    "AUC_best_so_far",
)
SUMMARY_METRICS = CORE_METRICS + ("elapsed_seconds",)
TRAIN_FILE = DATASET_ROOT / "Suzuki" / "suzuki_train.csv"
FROZEN_RUNNER_FILE = EXPERIMENT_ROOT / "run_suzuki_hybrid_acquisition.py"
FROZEN_HELPER_FILE = EXPERIMENT_ROOT / "acquisition_study_helpers.py"
PILOT_SEED_START = 710_000
PILOT_EPISODES = 60
PROTECTED_CONFIRMATION_START = 810_000
BUDGET = 5
MATERIAL_LOSS_FREQUENCY_INCREASE = 0.01
PILOT_GATE = {
    "initial_delta_vs_ard_logei_tolerance": 0.0,
    "minimum_best_delta_vs_fixed_logei": 0.0,
    "minimum_auc_delta_vs_fixed_logei": 0.0,
    "maximum_t95_delta_vs_fixed_logei": 0.0,
    "maximum_loss_frequency_increase_vs_ard_logei": (
        MATERIAL_LOSS_FREQUENCY_INCREASE
    ),
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
        phase="phase4_ard_hybrid_pilot",
        seed_start=PILOT_SEED_START,
        count=PILOT_EPISODES,
    )
    if len(episodes) != PILOT_EPISODES:
        raise RuntimeError("Phase-4 ARD-hybrid pilot episode count changed.")
    seeds = [int(item["split_seed"]) for item in episodes]
    if len(set(seeds)) != PILOT_EPISODES:
        raise RuntimeError("Phase-4 ARD-hybrid split seeds are not unique.")
    if min(seeds) < PILOT_SEED_START or max(seeds) >= PROTECTED_CONFIRMATION_START:
        raise RuntimeError("Pilot escaped the frozen 710k development range.")
    if any(int(item["budget"]) != BUDGET for item in episodes):
        raise RuntimeError("The frozen hybrid requires budget 5.")
    return episodes


def schedule_for_variant(variant: str) -> list[str]:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    return [
        frozen_hybrid.policy_component(POLICY[variant], step)
        for step in range(1, BUDGET + 1)
    ]


def build_problem(episode: dict):
    config, dataset, lookup = suzuki_proxy.build_problem(episode, "gp_only")
    config = replace(config, gp=replace(config.gp, target_transform="standard"))
    return config, dataset, lookup


def make_model(variant: str, schema, config, campaign_seed: int):
    if SURROGATE[variant] == "fixed":
        return StructuredANOVAGP(
            schema=schema,
            config=config.gp,
            seed=campaign_seed + 11,
        )
    return ComponentARDStructuredGP(
        schema=schema,
        config=config.gp,
        seed=campaign_seed + 17,
    )


def campaign_metrics(trajectory: list[dict], global_best: float) -> dict:
    raw = compute_campaign_metrics(
        [float(record["observed_yield"]) for record in trajectory],
        global_best=global_best,
    ).to_dict()
    t95 = raw["round_to_95_global_best"]
    return {
        "initial_round_found_best": float(raw["initial_round_found_best"]),
        "best_found": float(raw["best_found"]),
        "round_to_95_global_best": BUDGET + 1 if t95 is None else int(t95),
        "t95_reached": t95 is not None,
        "AUC_best_so_far": float(raw["auc_best_so_far"]),
    }


def run_one(job: tuple[str, dict, str]) -> dict:
    variant, episode, output_root_string = job
    config, dataset, lookup = build_problem(episode)
    prepared = prepare_problem(dataset, config)
    campaign_seed = int(episode["campaign_seed"])
    model = make_model(variant, prepared.encoder.schema, config, campaign_seed)
    oracle = OfflineTableOracle(dataset=dataset, lookup=lookup)
    observed = dataset.initial.copy()
    X_observed = prepared.X_initial.copy()
    y_observed = observed[dataset.spec.target_column].to_numpy(dtype=float)
    remaining = np.arange(len(dataset.candidates), dtype=int)
    trajectory: list[dict] = []
    started = time.perf_counter()

    for step in range(1, BUDGET + 1):
        X_remaining = prepared.X_candidates[remaining]
        candidate_rows = dataset.candidates.iloc[remaining]
        model.fit(X_observed, y_observed)
        posterior = model.predict(X_remaining)
        component = frozen_hybrid.policy_component(POLICY[variant], step)
        covariance = (
            model.posterior_covariance(X_remaining)
            if component == frozen_hybrid.THOMPSON_POLICY
            else None
        )
        scores, component = frozen_hybrid.scheduled_scores(
            POLICY[variant],
            posterior.mean,
            posterior.std,
            best_f=dataset.target_incumbent(observed),
            step=step,
            covariance=covariance,
            campaign_seed=campaign_seed,
        )
        candidate_ids = candidate_rows["query_index"].to_numpy(dtype=int)
        local_index = deterministic_argmax(
            scores,
            candidate_ids,
            seed=campaign_seed + step,
        )
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
                "surrogate": SURROGATE[variant],
                "policy": POLICY[variant],
                "acquisition_component": component,
                "acquisition_value": float(scores[local_index]),
                "thompson_seed": (
                    campaign_seed
                    + 100_000 * step
                    + frozen_hybrid.THOMPSON_STREAM_ID
                    if component == frozen_hybrid.THOMPSON_POLICY
                    else None
                ),
            }
        )
        new_observation = selected_row.loc[
            list(dataset.spec.feature_columns)
        ].to_dict()
        new_observation[dataset.spec.target_column] = observed_yield
        observed = pd.concat(
            [observed, pd.DataFrame([new_observation])], ignore_index=True
        )
        X_observed = np.vstack(
            [X_observed, prepared.X_candidates[selected_global]]
        )
        y_observed = np.append(y_observed, observed_yield)
        remaining = np.delete(remaining, local_index)

    elapsed = time.perf_counter() - started
    global_best = float(lookup[dataset.spec.target_column].max())
    metrics = campaign_metrics(trajectory, global_best)
    amplitudes = dict(getattr(model, "component_amplitudes_", {}))
    shares = dict(getattr(model, "component_shares_", {}))
    payload = {
        "dataset": "Suzuki",
        "variant": variant,
        "episode": episode,
        "trajectory": trajectory,
        "metrics": metrics,
        "contract": {
            "train_only": True,
            "public_test_labels_used": False,
            "prior_610k_results_used": False,
            "protected_810k_confirmation_run": False,
            "input_file": str(TRAIN_FILE.relative_to(DATASET_ROOT)),
            "input_sha256": file_sha256(TRAIN_FILE),
            "surrogate": SURROGATE[variant],
            "frozen_phase2_policy": POLICY[variant],
            "schedule": schedule_for_variant(variant),
            "thompson_stream_id": frozen_hybrid.THOMPSON_STREAM_ID,
            "component_amplitudes": amplitudes or None,
            "component_shares": shares or None,
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
        "campaign_seed": campaign_seed,
        "variant": variant,
        "surrogate": SURROGATE[variant],
        "policy": POLICY[variant],
        "schedule_json": json.dumps(schedule_for_variant(variant)),
        "budget": BUDGET,
        "global_best": global_best,
        **metrics,
        "elapsed_seconds": elapsed,
        "first_query_index": first["query_index"],
        "first_query_observed_yield": first["observed_yield"],
        "first_query_predicted_yield": first["predicted_yield"],
        "first_query_predicted_std": first["predicted_std"],
        "first_query_acquisition_value": first["acquisition_value"],
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
                    "phase": "phase4_ard_hybrid_pilot",
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
    records: list[dict] = []
    for variant, candidate in rows.groupby("variant", sort=True):
        if variant == baseline_name:
            continue
        candidate = candidate.set_index("episode")
        common = baseline.index.intersection(candidate.index)
        for metric in SUMMARY_METRICS:
            delta = candidate.loc[common, metric] - baseline.loc[common, metric]
            mean, sd, low, high = mean_ci(delta)
            higher = metric not in {"round_to_95_global_best", "elapsed_seconds"}
            wins = int((delta > 1e-10).sum()) if higher else int((delta < -1e-10).sum())
            losses = int((delta < -1e-10).sum()) if higher else int((delta > 1e-10).sum())
            records.append(
                {
                    "dataset": "Suzuki",
                    "phase": "phase4_ard_hybrid_pilot",
                    "candidate": variant,
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


def risk_summary(rows: pd.DataFrame) -> pd.DataFrame:
    baseline = rows[rows["variant"].eq(FIXED_LOGEI)].set_index("episode")
    records: list[dict] = []
    for variant in (ARD_LOGEI, ARD_HYBRID):
        candidate = rows[rows["variant"].eq(variant)].set_index("episode")
        common = baseline.index.intersection(candidate.index)
        losses = pd.DataFrame(index=common)
        for metric in CORE_METRICS:
            delta = candidate.loc[common, metric] - baseline.loc[common, metric]
            losses[metric] = delta > 1e-10 if metric == "round_to_95_global_best" else delta < -1e-10
        loss_count = losses.sum(axis=1).astype(float)
        records.append(
            {
                "dataset": "Suzuki",
                "variant": variant,
                "baseline": FIXED_LOGEI,
                "n_pairs": len(common),
                "n_metric_comparisons": len(common) * len(CORE_METRICS),
                "loss_frequency": float(losses.to_numpy().mean()),
                "any_loss_frequency": float((loss_count > 0).mean()),
                "mean_loss_count": float(loss_count.mean()),
                "sample_variance_loss_count": float(loss_count.var(ddof=1)),
            }
        )
    return pd.DataFrame(records)


def validate_first_query_contract(rows: pd.DataFrame) -> dict[str, object]:
    checked_pairs = 0
    for _, section in rows.groupby("episode", sort=True):
        for logei_name, hybrid_name in (
            (FIXED_LOGEI, FIXED_HYBRID),
            (ARD_LOGEI, ARD_HYBRID),
        ):
            left = section[section["variant"].eq(logei_name)].iloc[0]
            right = section[section["variant"].eq(hybrid_name)].iloc[0]
            for column in (
                "first_query_index",
                "first_query_observed_yield",
                "first_query_predicted_yield",
                "first_query_predicted_std",
                "first_query_acquisition_value",
            ):
                if left[column] != right[column]:
                    raise RuntimeError(
                        f"First-query pair is not bit-identical: {column}."
                    )
            checked_pairs += 1
    return {
        "passed": True,
        "n_episodes": rows["episode"].nunique(),
        "n_surrogate_policy_pairs": checked_pairs,
        "fields": [
            "query_index",
            "observed_yield",
            "predicted_yield",
            "predicted_std",
            "acquisition_value",
        ],
    }


def validate_schedule_contract(
    rows: pd.DataFrame, output_root: Path
) -> dict[str, object]:
    expected_jobs = len(VARIANTS) * PILOT_EPISODES
    if len(rows) != expected_jobs:
        raise RuntimeError(
            f"Expected {expected_jobs} pilot jobs, received {len(rows)}."
        )
    if rows["episode"].nunique() != PILOT_EPISODES:
        raise RuntimeError("Schedule audit did not receive all pilot episodes.")
    seeds = rows["split_seed"].astype(int)
    if seeds.min() < PILOT_SEED_START or seeds.max() >= PROTECTED_CONFIRMATION_START:
        raise RuntimeError("Schedule audit found a protected split seed.")

    checked_trajectories = 0
    checked_thompson_steps = 0
    thompson_seeds: dict[tuple[str, int], set[int]] = {}
    for record in rows.to_dict(orient="records"):
        path = (
            output_root
            / "trajectories"
            / record["variant"]
            / f"{record['episode']}.json"
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        trajectory = payload["trajectory"]
        expected_schedule = schedule_for_variant(record["variant"])
        observed_schedule = [step["acquisition_component"] for step in trajectory]
        if observed_schedule != expected_schedule:
            raise RuntimeError(f"Frozen schedule mismatch in {path}.")
        if payload["contract"]["schedule"] != expected_schedule:
            raise RuntimeError(f"Trajectory contract mismatch in {path}.")
        for step in trajectory:
            expected_seed = None
            if step["acquisition_component"] == frozen_hybrid.THOMPSON_POLICY:
                expected_seed = (
                    int(record["campaign_seed"])
                    + 100_000 * int(step["step"])
                    + frozen_hybrid.THOMPSON_STREAM_ID
                )
                checked_thompson_steps += 1
                key = (record["episode"], int(step["step"]))
                thompson_seeds.setdefault(key, set()).add(expected_seed)
            if step["thompson_seed"] != expected_seed:
                raise RuntimeError(f"Thompson stream mismatch in {path}.")
        checked_trajectories += 1

    if any(len(values) != 1 for values in thompson_seeds.values()):
        raise RuntimeError("Surrogates did not share the frozen Thompson stream.")
    return {
        "passed": True,
        "checked_trajectories": checked_trajectories,
        "checked_thompson_steps": checked_thompson_steps,
        "shared_thompson_stream": True,
        "thompson_stream_id": frozen_hybrid.THOMPSON_STREAM_ID,
        "split_seed_min": int(seeds.min()),
        "split_seed_max": int(seeds.max()),
        "prior_610k_results_used": False,
        "protected_810k_confirmation_run": False,
        "schedules": {name: schedule_for_variant(name) for name in VARIANTS},
    }


def pilot_gate(rows: pd.DataFrame) -> dict[str, object]:
    versus_fixed = paired_summary(rows, FIXED_LOGEI)
    fixed = versus_fixed[versus_fixed["candidate"].eq(CANDIDATE)].set_index("metric")
    versus_ard = paired_summary(rows, ARD_LOGEI)
    ard = versus_ard[versus_ard["candidate"].eq(CANDIDATE)].set_index("metric")
    risks = risk_summary(rows).set_index("variant")
    loss_increase = float(
        risks.loc[CANDIDATE, "loss_frequency"]
        - risks.loc[ARD_LOGEI, "loss_frequency"]
    )
    deltas_fixed = {
        metric: float(fixed.loc[metric, "mean_delta"]) for metric in CORE_METRICS
    }
    deltas_ard = {
        metric: float(ard.loc[metric, "mean_delta"]) for metric in CORE_METRICS
    }
    checks = {
        "initial_preserved_vs_ard_logei": abs(
            deltas_ard["initial_round_found_best"]
        )
        <= PILOT_GATE["initial_delta_vs_ard_logei_tolerance"],
        "best_nonnegative_vs_fixed_logei": deltas_fixed["best_found"]
        >= PILOT_GATE["minimum_best_delta_vs_fixed_logei"],
        "auc_nonnegative_vs_fixed_logei": deltas_fixed["AUC_best_so_far"]
        >= PILOT_GATE["minimum_auc_delta_vs_fixed_logei"],
        "t95_nonpositive_vs_fixed_logei": deltas_fixed[
            "round_to_95_global_best"
        ]
        <= PILOT_GATE["maximum_t95_delta_vs_fixed_logei"],
        "loss_frequency_not_materially_higher": loss_increase
        <= PILOT_GATE["maximum_loss_frequency_increase_vs_ard_logei"],
    }
    return {
        "candidate": CANDIDATE,
        "gate": PILOT_GATE,
        "mean_deltas_vs_fixed_logei": deltas_fixed,
        "mean_deltas_vs_ard_logei": deltas_ard,
        "risk_vs_fixed_logei": {
            name: risks.loc[name].to_dict() for name in (ARD_LOGEI, CANDIDATE)
        },
        "loss_frequency_increase_vs_ard_logei": loss_increase,
        "checks": checks,
        "passed": bool(all(checks.values())),
        "confirmation_run": False,
        "production_default_changed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=min(6, os.cpu_count() or 1))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.jobs <= 0:
        raise ValueError("--jobs must be positive.")
    episodes = make_pilot_episodes()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "train_only": True,
        "public_test_labels_used": False,
        "prior_610k_results_used": False,
        "protected_810k_confirmation_run": False,
        "production_default_changed": False,
        "input": {
            "path": str(TRAIN_FILE.relative_to(DATASET_ROOT)),
            "sha256": file_sha256(TRAIN_FILE),
        },
        "frozen_phase2_implementation": {
            "runner_path": str(FROZEN_RUNNER_FILE.relative_to(DATASET_ROOT)),
            "runner_sha256": file_sha256(FROZEN_RUNNER_FILE),
            "helper_path": str(FROZEN_HELPER_FILE.relative_to(DATASET_ROOT)),
            "helper_sha256": file_sha256(FROZEN_HELPER_FILE),
            "thompson_stream_id": frozen_hybrid.THOMPSON_STREAM_ID,
        },
        "executor_implementation": {
            "runner_path": str(Path(__file__).resolve().relative_to(DATASET_ROOT)),
            "runner_sha256": file_sha256(Path(__file__).resolve()),
        },
        "episode_contract": {
            "split_seed_start": PILOT_SEED_START,
            "accepted_split_seed_min": min(item["split_seed"] for item in episodes),
            "accepted_split_seed_max": max(item["split_seed"] for item in episodes),
            "n_episodes": len(episodes),
            "budget": BUDGET,
            "protected_confirmation_seed_start": PROTECTED_CONFIRMATION_START,
        },
        "variants": list(VARIANTS),
        "surrogates": SURROGATE,
        "policies": POLICY,
        "schedules": {name: schedule_for_variant(name) for name in VARIANTS},
        "pilot_gate": PILOT_GATE,
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
                f"{row['variant']} {row['episode']}: "
                f"AUC={row['AUC_best_so_far']:.3f}",
                flush=True,
            )
    detail = pd.DataFrame(rows).sort_values(["episode", "variant"])
    first_query_audit = validate_first_query_contract(detail)
    schedule_audit = validate_schedule_contract(detail, args.output_dir)
    detail.to_csv(args.output_dir / "per_episode.csv", index=False)
    summarize(detail).to_csv(args.output_dir / "summary.csv", index=False)
    paired_summary(detail, FIXED_LOGEI).to_csv(
        args.output_dir / "paired_vs_fixed_logei.csv", index=False
    )
    paired_summary(detail, ARD_LOGEI).query(
        "candidate == @CANDIDATE"
    ).to_csv(args.output_dir / "paired_candidate_vs_ard_logei.csv", index=False)
    risk_summary(detail).to_csv(args.output_dir / "risk_summary.csv", index=False)
    decision = pilot_gate(detail)
    (args.output_dir / "pilot_gate.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.output_dir / "first_query_audit.json").write_text(
        json.dumps(first_query_audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.output_dir / "schedule_audit.json").write_text(
        json.dumps(schedule_audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    first_ids = {
        f"random_phase4_ard_hybrid_pilot_{index:02d}" for index in range(20)
    }
    twenty = detail[detail["episode"].isin(first_ids)].copy()
    if len(twenty) != 20 * len(VARIANTS) or twenty["episode"].nunique() != 20:
        raise RuntimeError("The explicit first-20 view is incomplete.")
    twenty.to_csv(args.output_dir / "twenty_run_per_episode.csv", index=False)
    summarize(twenty).to_csv(
        args.output_dir / "twenty_run_summary.csv", index=False
    )
    paired_summary(twenty, FIXED_LOGEI).to_csv(
        args.output_dir / "twenty_run_paired_vs_fixed_logei.csv", index=False
    )
    paired_summary(twenty, ARD_LOGEI).query(
        "candidate == @CANDIDATE"
    ).to_csv(
        args.output_dir / "twenty_run_paired_candidate_vs_ard_logei.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
