#!/usr/bin/env python3
"""Staged Suzuki fixed-query1/component-ARD-later surrogate study."""

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


VARIANTS = ("fixed", "pure_ard", "fixed_q1_ard_later")
BASELINE = "fixed"
CANDIDATE = "fixed_q1_ard_later"
CORE_METRICS = (
    "initial_round_found_best",
    "best_found",
    "round_to_95_global_best",
    "AUC_best_so_far",
)
SUMMARY_METRICS = CORE_METRICS + ("elapsed_seconds",)
TRAIN_FILE = DATASET_ROOT / "Suzuki" / "suzuki_train.csv"
PILOT_GATE = {
    "initial_delta_tolerance": 1e-12,
    "minimum_best_delta_vs_fixed": -0.5,
    "maximum_t95_delta_vs_fixed": 0.25,
    "strict_gain_metrics": ["AUC_best_so_far", "best_found"],
    "minimum_auc_delta_vs_pure_ard": 0.0,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_episodes() -> dict[str, list[dict]]:
    frame = pd.read_csv(TRAIN_FILE)
    return {
        "pilot": suzuki_proxy._phase_random_episodes(
            frame, phase="pilot", seed_start=310_000, count=20
        ),
        "confirmation": suzuki_proxy._phase_random_episodes(
            frame, phase="confirmation", seed_start=410_000, count=80
        ),
    }


def episodes_for_stage(stage: str) -> list[dict]:
    episodes = make_episodes()
    if stage == "smoke":
        return episodes["pilot"][:2]
    return episodes[stage]


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
    if variant == "fixed":
        return "fixed"
    if variant == "pure_ard":
        return "component_ard"
    if variant == CANDIDATE:
        return "fixed" if step == 1 else "component_ard"
    raise ValueError(f"Unknown surrogate schedule: {variant}")


def _metrics(trajectory: list[dict], global_best: float, budget: int) -> dict:
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
    fixed_model = None
    ard_model = None
    if variant in {"fixed", CANDIDATE}:
        fixed_model = make_fixed_model(schema, config, seed)
    if variant == "pure_ard":
        ard_model = make_ard_model(schema, config, seed)
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
                raise RuntimeError("Fixed model was not initialized.")
            model = fixed_model
        else:
            if ard_model is None:
                # The switched schedule creates ARD only after query-1 feedback.
                ard_model = make_ard_model(schema, config, seed)
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
        local_index = deterministic_argmax(
            scores, candidate_ids, seed=seed + step
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
                "model_family": family,
                "acquisition_name": acquisition_name,
                "acquisition_value": float(scores[local_index]),
                "component_amplitudes": dict(
                    getattr(model, "component_amplitudes_", {})
                ) or None,
                "component_shares": dict(
                    getattr(model, "component_shares_", {})
                ) or None,
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
    metrics = _metrics(trajectory, global_best, budget)
    final_ard = ard_model if ard_model is not None else None
    amplitudes = dict(getattr(final_ard, "component_amplitudes_", {}))
    shares = dict(getattr(final_ard, "component_shares_", {}))
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
            "target_transform": "standard",
            "schedule": [model_family_for_step(variant, step) for step in range(1, budget + 1)],
            "episode": episode,
        },
    }
    output_path = (
        Path(output_root_string)
        / "trajectories"
        / episode["phase"]
        / variant
        / f"{episode['episode']}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {
        "dataset": "Suzuki",
        "phase": episode["phase"],
        "episode_family": episode["episode_family"],
        "episode": episode["episode"],
        "variant": variant,
        "budget": budget,
        "global_best": global_best,
        **metrics,
        "elapsed_seconds": elapsed,
        "first_query_index": trajectory[0]["query_index"],
        "component_amplitudes_json": json.dumps(amplitudes, sort_keys=True) if amplitudes else None,
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
    for (phase, family, variant), group in rows.groupby(
        ["phase", "episode_family", "variant"], sort=True
    ):
        for metric in SUMMARY_METRICS:
            mean, sd, low, high = mean_ci(group[metric])
            records.append(
                {
                    "dataset": "Suzuki",
                    "phase": phase,
                    "episode_family": family,
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


def paired_summary(
    rows: pd.DataFrame, baseline_name: str = BASELINE
) -> pd.DataFrame:
    records: list[dict] = []
    for (phase, family), section in rows.groupby(
        ["phase", "episode_family"], sort=True
    ):
        baseline = section[section["variant"].eq(baseline_name)].set_index("episode")
        for variant, candidate in section.groupby("variant", sort=True):
            if variant == baseline_name:
                continue
            candidate = candidate.set_index("episode")
            common = baseline.index.intersection(candidate.index)
            for metric in SUMMARY_METRICS:
                delta = candidate.loc[common, metric] - baseline.loc[common, metric]
                mean, sd, low, high = mean_ci(delta)
                higher = metric not in {
                    "round_to_95_global_best",
                    "elapsed_seconds",
                }
                wins = int((delta > 1e-10).sum()) if higher else int((delta < -1e-10).sum())
                losses = int((delta < -1e-10).sum()) if higher else int((delta > 1e-10).sum())
                records.append(
                    {
                        "dataset": "Suzuki",
                        "phase": phase,
                        "episode_family": family,
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


def paired_between(rows: pd.DataFrame, candidate_name: str, baseline_name: str) -> dict[str, float]:
    candidate = rows[rows["variant"].eq(candidate_name)].set_index("episode")
    baseline = rows[rows["variant"].eq(baseline_name)].set_index("episode")
    common = candidate.index.intersection(baseline.index)
    return {
        metric: float((candidate.loc[common, metric] - baseline.loc[common, metric]).mean())
        for metric in CORE_METRICS
    }


def pilot_gate(rows: pd.DataFrame) -> dict[str, object]:
    versus_fixed = paired_between(rows, CANDIDATE, "fixed")
    versus_ard = paired_between(rows, CANDIDATE, "pure_ard")
    checks = {
        "initial_exact": abs(versus_fixed["initial_round_found_best"])
        <= PILOT_GATE["initial_delta_tolerance"],
        "best_safe": versus_fixed["best_found"]
        >= PILOT_GATE["minimum_best_delta_vs_fixed"],
        "t95_safe": versus_fixed["round_to_95_global_best"]
        <= PILOT_GATE["maximum_t95_delta_vs_fixed"],
        "strict_gain_vs_fixed": any(
            versus_fixed[metric] > 0.0
            for metric in PILOT_GATE["strict_gain_metrics"]
        ),
        "auc_noninferior_to_pure_ard": versus_ard["AUC_best_so_far"]
        >= PILOT_GATE["minimum_auc_delta_vs_pure_ard"],
    }
    return {
        "gate": PILOT_GATE,
        "mean_deltas_vs_fixed": versus_fixed,
        "mean_deltas_vs_pure_ard": versus_ard,
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def amplitude_summary(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    for (phase, variant), group in rows.dropna(
        subset=["component_shares_json"]
    ).groupby(["phase", "variant"], sort=True):
        parsed = [json.loads(value) for value in group["component_shares_json"]]
        for component in sorted(set().union(*(item.keys() for item in parsed))):
            values = pd.Series([item[component] for item in parsed], dtype=float)
            mean, sd, low, high = mean_ci(values)
            records.append(
                {
                    "dataset": "Suzuki",
                    "phase": phase,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("smoke", "pilot", "confirmation"), required=True
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
    manifest = {
        "stage": args.stage,
        "train_only": True,
        "public_test_labels_used": False,
        "input": {
            "path": str(TRAIN_FILE.relative_to(DATASET_ROOT)),
            "sha256": file_sha256(TRAIN_FILE),
        },
        "variants": list(VARIANTS),
        "schedules": {
            variant: [model_family_for_step(variant, step) for step in range(1, 6)]
            for variant in VARIANTS
        },
        "pilot_gate": PILOT_GATE,
        "target_transform": "standard",
        "acquisition": "posterior mean query 1, stable LogEI thereafter",
        "episode_count": len(episodes),
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
                f"{row['phase']} {row['variant']} {row['episode']}: "
                f"AUC={row['AUC_best_so_far']:.3f}",
                flush=True,
            )
    detail = pd.DataFrame(rows).sort_values(["episode", "variant"])
    detail.to_csv(args.output_dir / "per_episode.csv", index=False)
    summarize(detail).to_csv(args.output_dir / "summary.csv", index=False)
    paired_summary(detail).to_csv(
        args.output_dir / "paired_vs_fixed.csv", index=False
    )
    paired_summary(detail, baseline_name="pure_ard").query(
        "candidate == @CANDIDATE"
    ).to_csv(args.output_dir / "paired_switch_vs_pure_ard.csv", index=False)
    amplitude_summary(detail).to_csv(
        args.output_dir / "amplitude_stability.csv", index=False
    )
    first = detail.pivot(index="episode", columns="variant", values="first_query_index")
    if not (first["fixed"] == first[CANDIDATE]).all():
        raise RuntimeError("Switched schedule did not exactly share fixed query 1.")
    if args.stage == "pilot":
        decision = pilot_gate(detail)
        (args.output_dir / "pilot_gate.json").write_text(
            json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8"
        )
    if args.stage in {"pilot", "confirmation"}:
        if args.stage == "pilot":
            twenty = detail.copy()
        else:
            twenty_ids = {f"random_confirmation_{index:02d}" for index in range(20)}
            twenty = detail[detail["episode"].isin(twenty_ids)].copy()
        if len(twenty) != 20 * len(VARIANTS) or twenty["episode"].nunique() != 20:
            raise RuntimeError("The explicit 20-run view is incomplete.")
        twenty.to_csv(args.output_dir / "twenty_run_per_episode.csv", index=False)
        summarize(twenty).to_csv(
            args.output_dir / "twenty_run_summary.csv", index=False
        )
        paired_summary(twenty).to_csv(
            args.output_dir / "twenty_run_paired_vs_fixed.csv", index=False
        )
        paired_summary(twenty, baseline_name="pure_ard").query(
            "candidate == @CANDIDATE"
        ).to_csv(
            args.output_dir / "twenty_run_paired_switch_vs_pure_ard.csv",
            index=False,
        )


if __name__ == "__main__":
    main()
