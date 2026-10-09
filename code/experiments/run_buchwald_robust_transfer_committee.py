#!/usr/bin/env python3
"""Source-gated robust transfer/target-only Buchwald committee."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENT_ROOT = Path(__file__).resolve().parent
CODE_ROOT = EXPERIMENT_ROOT.parent
DATASET_ROOT = CODE_ROOT.parents[1]
EXTENSION_ROOT = CODE_ROOT / "bgolearn_rxn_extension"
SESSION_ROOT = DATASET_ROOT / ".arbor" / "sessions" / "bgolearn_frontier_phase4_20260728"
sys.path.insert(0, str(EXTENSION_ROOT / "src"))
sys.path.insert(0, str(EXPERIMENT_ROOT))

import run_buchwald_source_meta_selection as meta  # noqa: E402
from bgolearn_rxn.backend import NativeBackend  # noqa: E402
from bgolearn_rxn.campaign import CampaignRunner  # noqa: E402
from bgolearn_rxn.oracle import OfflineTableOracle  # noqa: E402
from bgolearn_rxn.schemas import FeatureSchema, Posterior  # noqa: E402
from bgolearn_rxn.surrogates.component_ard_gp import (  # noqa: E402
    ComponentARDStructuredGP,
)
from bgolearn_rxn.surrogates.ensemble import PrequentialEnsemble  # noqa: E402


TRAIN_FILE = meta.TRAIN_FILE
CONFIG_FILE = meta.CONFIG_FILE
MODEL_FILE = meta.MODEL_FILE
FACTORY_FILE = meta.FACTORY_FILE
OFFICIAL_TARGET = meta.OFFICIAL_TARGET
TRANSFER_MODEL = meta.MODEL_NAME
TARGET_MODEL = "target_only_component_ard_gp"
CORE_METRICS = meta.CORE_METRICS


@dataclass(frozen=True)
class CommitteeSetting:
    name: str
    transfer_weight: float
    target_weight: float
    learning_rate: float
    minimum_weight: float


COMMITTEE_GRID = (
    CommitteeSetting("conservative_lr005", 0.90, 0.10, 0.05, 0.05),
    CommitteeSetting("balanced_lr015", 0.80, 0.20, 0.15, 0.05),
    CommitteeSetting("fast_lr030", 0.80, 0.20, 0.30, 0.05),
)
SETTING_BY_NAME = {setting.name: setting for setting in COMMITTEE_GRID}
BASELINE = "frontier"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TargetOnlyComponentARDGP:
    """Experiment-local component GP fitted only on target-task observations."""

    name = TARGET_MODEL

    def __init__(
        self,
        schema: FeatureSchema,
        config,
        *,
        target_identity: np.ndarray,
        seed: int,
    ) -> None:
        if schema.task_role is None:
            raise ValueError("Target-only committee model requires a task role.")
        self.schema = schema
        self.target_identity = np.asarray(target_identity, dtype=float).reshape(-1)
        self.model = ComponentARDStructuredGP(schema, config, seed=seed)
        self.n_target_: int | None = None

    def _target_mask(self, X: np.ndarray) -> np.ndarray:
        task = self.schema.task_role
        if task is None:
            raise RuntimeError("Target-only model lost its task role.")
        block = np.asarray(X[:, self.schema.identity_slice(task)], dtype=float)
        return np.all(
            np.isclose(block, self.target_identity[None, :], atol=1e-12), axis=1
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TargetOnlyComponentARDGP":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        mask = self._target_mask(X)
        self.n_target_ = int(mask.sum())
        if self.n_target_ < 2:
            raise ValueError("Target-only committee model requires two target observations.")
        self.model.fit(X[mask], y[mask])
        return self

    def predict(self, X: np.ndarray) -> Posterior:
        if not self._target_mask(np.asarray(X, dtype=float)).all():
            raise ValueError("Target-only committee model predicts only the target task.")
        return self.model.predict(X)

    def posterior_covariance(self, X: np.ndarray) -> np.ndarray:
        if not self._target_mask(np.asarray(X, dtype=float)).all():
            raise ValueError("Target-only committee model predicts only the target task.")
        return self.model.posterior_covariance(X)


def _target_identity(runner: CampaignRunner) -> np.ndarray:
    schema = runner.prepared.encoder.schema
    if schema is None or schema.task_role is None:
        raise RuntimeError("Committee requires a fitted task-aware encoder.")
    block = runner.prepared.X_candidates[:, schema.identity_slice(schema.task_role)]
    identity = np.asarray(block[0], dtype=float).copy()
    if not np.all(np.isclose(block, identity[None, :], atol=1e-12)):
        raise ValueError("Candidate pool contains multiple target identities.")
    return identity


def install_committee(runner: CampaignRunner, setting: CommitteeSetting) -> None:
    transfer_model = runner.ensemble.models[TRANSFER_MODEL]
    schema = runner.prepared.encoder.schema
    if schema is None:
        raise RuntimeError("Committee encoder schema is missing.")
    target_model = TargetOnlyComponentARDGP(
        schema,
        runner.config.gp,
        target_identity=_target_identity(runner),
        seed=runner.seed + 211,
    )
    ensemble_config = replace(
        runner.config.ensemble,
        enabled_models=(TRANSFER_MODEL, TARGET_MODEL),
        initial_weights={
            TRANSFER_MODEL: setting.transfer_weight,
            TARGET_MODEL: setting.target_weight,
        },
        prequential_learning_rate=setting.learning_rate,
        minimum_weight=setting.minimum_weight,
    )
    ensemble = PrequentialEnsemble(
        OrderedDict(
            [(TRANSFER_MODEL, transfer_model), (TARGET_MODEL, target_model)]
        ),
        ensemble_config,
    )
    runner.ensemble = ensemble
    runner.backend = NativeBackend(ensemble)


def run_one(job: tuple[dict, str, bool, str]) -> dict:
    episode, variant, strict_source_isolation, output_root = job
    config, dataset, lookup, observed_products = meta.build_problem(
        episode, strict_source_isolation=strict_source_isolation
    )
    oracle = OfflineTableOracle(dataset=dataset, lookup=lookup)
    runner = CampaignRunner(
        dataset, config, seed=int(episode["seed"]), backend="native"
    )
    setting = None if variant == BASELINE else SETTING_BY_NAME[variant]
    if setting is not None:
        install_committee(runner, setting)
    started = time.perf_counter()
    result = runner.run(oracle, budget=int(episode["budget"]))
    elapsed = time.perf_counter() - started
    payload = result.to_payload()
    metrics = meta._campaign_metrics(
        payload["trajectory"], float(oracle.global_best), int(episode["budget"])
    )
    first = payload["trajectory"][0]
    if first["acquisition_name"] != f"{TRANSFER_MODEL}_posterior_mean":
        raise AssertionError("Committee changed the validated query-one policy.")
    weight_path = [record["model_weights_after_query"] for record in payload["trajectory"]]
    if setting is not None:
        if any(set(weights) != {TRANSFER_MODEL, TARGET_MODEL} for weights in weight_path):
            raise AssertionError("Committee weight path is incomplete.")
        if any(
            record["acquisition_name"] != "stable_log_ei"
            for record in payload["trajectory"][1:]
        ):
            raise AssertionError("Committee did not preserve LogEI after query 1.")
    payload["robust_committee_contract"] = {
        "train_only": True,
        "public_test_labels_used": False,
        "selection_target_labels_used": False,
        "strict_source_isolation": strict_source_isolation,
        "variant": variant,
        "setting": None if setting is None else asdict(setting),
        "observed_products": observed_products,
        "weight_path": weight_path,
        "episode": episode,
        "metrics": metrics,
    }
    output_path = (
        Path(output_root)
        / "trajectories"
        / episode["phase"]
        / variant
        / f"{episode['episode']}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "phase": episode["phase"],
        "episode": episode["episode"],
        "target_task": episode["target_task"],
        "variant": variant,
        "budget": int(episode["budget"]),
        "global_best": float(oracle.global_best),
        **metrics,
        "query_1_index": int(first["query_index"]),
        "query_1_acquisition": first["acquisition_name"],
        "query_2_acquisition": payload["trajectory"][1]["acquisition_name"],
        "final_transfer_weight": float(weight_path[-1][TRANSFER_MODEL]),
        "final_target_weight": float(weight_path[-1].get(TARGET_MODEL, 0.0)),
        "weight_path_json": json.dumps(weight_path, sort_keys=True),
        "elapsed_seconds": elapsed,
    }


def run_jobs(jobs: list[tuple[dict, str, bool, str]], workers: int) -> pd.DataFrame:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    rows = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_one, job): job for job in jobs}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"{row['phase']} {row['variant']} {row['episode']}: "
                f"AUC={row['AUC_best_so_far']:.3f}",
                flush=True,
            )
    return pd.DataFrame(rows).sort_values(["phase", "episode", "variant"])


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (phase, variant), group in rows.groupby(["phase", "variant"], sort=True):
        for metric in CORE_METRICS:
            mean, sd, low, high = meta.mean_ci(group[metric])
            records.append(
                {
                    "phase": phase,
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


def paired_summary(rows: pd.DataFrame, candidate_name: str) -> pd.DataFrame:
    baseline = rows[rows["variant"].eq(BASELINE)].set_index("episode")
    candidate = rows[rows["variant"].eq(candidate_name)].set_index("episode")
    common = baseline.index.intersection(candidate.index)
    records = []
    for metric in CORE_METRICS:
        delta = candidate.loc[common, metric] - baseline.loc[common, metric]
        mean, sd, low, high = meta.mean_ci(delta)
        higher = metric != "round_to_95_global_best"
        wins = int((delta > 1e-10).sum()) if higher else int((delta < -1e-10).sum())
        losses = int((delta < -1e-10).sum()) if higher else int((delta > 1e-10).sum())
        records.append(
            {
                "candidate": candidate_name,
                "baseline": BASELINE,
                "metric": metric,
                "n_pairs": len(common),
                "mean_delta_candidate_minus_baseline": mean,
                "sample_sd_delta": sd,
                "ci95_low": low,
                "ci95_high": high,
                "wins": wins,
                "ties": len(common) - wins - losses,
                "losses": losses,
            }
        )
    return pd.DataFrame(records)


def all_paired(rows: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(
        [
            paired_summary(rows, setting.name)
            for setting in COMMITTEE_GRID
            if setting.name in set(rows["variant"])
        ],
        ignore_index=True,
    )


def select_candidate(rows: pd.DataFrame) -> tuple[str, dict]:
    aggregate = rows.groupby("variant")[list(CORE_METRICS)].mean()
    baseline = aggregate.loc[BASELINE]
    eligible = []
    for setting in COMMITTEE_GRID:
        if setting.name not in aggregate.index:
            continue
        values = aggregate.loc[setting.name]
        first_preserved = abs(
            values["initial_round_found_best"]
            - baseline["initial_round_found_best"]
        ) <= 1e-10
        best_preserved = values["best_found"] >= baseline["best_found"] - 1e-10
        improves = (
            values["AUC_best_so_far"] > baseline["AUC_best_so_far"] + 1e-10
            or values["round_to_95_global_best"]
            < baseline["round_to_95_global_best"] - 1e-10
        )
        if first_preserved and best_preserved and improves:
            eligible.append(setting.name)
    selected = (
        max(eligible, key=lambda name: aggregate.loc[name, "AUC_best_so_far"])
        if eligible
        else BASELINE
    )
    gate = {
        "selected_variant": selected,
        "source_gate_passed": selected != BASELINE,
        "eligible_variants": eligible,
        "source_means": {
            variant: {metric: float(value) for metric, value in row.items()}
            for variant, row in aggregate.to_dict(orient="index").items()
        },
        "criteria": {
            "query_1_mean_exactly_preserved": True,
            "best_found_mean_not_lower": True,
            "AUC_higher_or_t95_lower": True,
        },
    }
    return selected, gate


def deterministic_twenty(rows: pd.DataFrame) -> pd.DataFrame:
    episodes = sorted(rows["episode"].unique())[:20]
    output = rows[rows["episode"].isin(episodes)].copy()
    if len(output) != 20 * rows["variant"].nunique():
        raise ValueError("Incomplete deterministic 20-run slice.")
    return output.sort_values(["episode", "variant"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=min(6, os.cpu_count() or 1))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--official-only",
        action="store_true",
        help="Resume the permitted official stage from an existing source gate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(SESSION_ROOT / "experiments" / "4" / "results" / "4-robust-transfer-committee"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.jobs <= 0:
        raise ValueError("--jobs must be positive.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    episodes = meta.make_source_meta_episodes()
    variants = (BASELINE,) + tuple(setting.name for setting in COMMITTEE_GRID)
    if args.smoke:
        episodes = episodes[:1]
        variants = variants[:2]
    manifest = {
        "study": "robust low-rank-transfer/target-only Buchwald committee",
        "train_only": True,
        "public_test_labels_used": False,
        "selection_target_labels_used": False,
        "selection_excludes_official_target_rows": True,
        "official_target": OFFICIAL_TARGET,
        "committee_grid": [asdict(setting) for setting in COMMITTEE_GRID],
        "source_gate": "preserve query1 and best-found; improve AUC or t95",
        "query_1_model": TRANSFER_MODEL,
        "post_query_1_acquisition": "stable_log_ei",
        "inputs": {
            "train": {"path": str(TRAIN_FILE), "sha256": file_sha256(TRAIN_FILE)},
            "config": {"path": str(CONFIG_FILE), "sha256": file_sha256(CONFIG_FILE)},
            "model": {"path": str(MODEL_FILE), "sha256": file_sha256(MODEL_FILE)},
            "factory": {"path": str(FACTORY_FILE), "sha256": file_sha256(FACTORY_FILE)},
        },
        "source_episode_count": len(episodes),
        "python": platform.python_version(),
        "smoke": bool(args.smoke),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    if args.official_only:
        source = pd.read_csv(args.output_dir / "source_per_episode.csv")
        gate = json.loads((args.output_dir / "source_gate.json").read_text(encoding="utf-8"))
        selected = str(gate["selected_variant"])
    else:
        jobs = [
            (episode, variant, True, str(args.output_dir))
            for variant in variants
            for episode in episodes
        ]
        source = run_jobs(jobs, args.jobs)
        source.to_csv(args.output_dir / "source_per_episode.csv", index=False)
        summarize(source).to_csv(args.output_dir / "source_summary.csv", index=False)
        all_paired(source).to_csv(args.output_dir / "source_all_paired_vs_frontier.csv", index=False)
        selected, gate = select_candidate(source)
        (args.output_dir / "source_gate.json").write_text(
            json.dumps(gate, indent=2, sort_keys=True), encoding="utf-8"
        )
    if args.smoke:
        return

    official_variants = [BASELINE]
    if gate["source_gate_passed"]:
        official_variants.append(selected)
    official_episodes = meta.make_official_confirmation_episodes()
    official_jobs = [
        (episode, variant, False, str(args.output_dir))
        for variant in official_variants
        for episode in official_episodes
    ]
    official = run_jobs(official_jobs, args.jobs)
    official.to_csv(args.output_dir / "official_per_episode.csv", index=False)
    summarize(official).to_csv(args.output_dir / "official_summary.csv", index=False)
    paired_summary(official, selected).to_csv(
        args.output_dir / "official_paired_selected_vs_frontier.csv", index=False
    )
    twenty = deterministic_twenty(official)
    twenty.to_csv(args.output_dir / "official_20run_per_run.csv", index=False)
    summarize(twenty).to_csv(args.output_dir / "official_20run_summary.csv", index=False)
    paired_summary(twenty, selected).to_csv(
        args.output_dir / "official_20run_paired.csv", index=False
    )

    prior = pd.read_csv(
        DATASET_ROOT / "submission" / "results" / "experimental_investigations"
        / "5.1-component-ard-study" / "per_episode.csv"
    )
    prior = prior[
        prior["dataset"].eq("Buchwald_sub4")
        & prior["variant"].eq("component_ard")
    ].set_index("episode")
    fresh = official[official["variant"].eq(BASELINE)].set_index("episode")
    common = prior.index.intersection(fresh.index)
    differences = {
        metric: float(np.max(np.abs(fresh.loc[common, metric] - prior.loc[common, metric])))
        for metric in CORE_METRICS
    }
    check = {
        "n_common_episodes": len(common),
        "max_abs_metric_differences": differences,
        "exact_metric_match": all(value <= 1e-12 for value in differences.values()),
    }
    (args.output_dir / "baseline_reproducibility.json").write_text(
        json.dumps(check, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
