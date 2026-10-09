#!/usr/bin/env python3
"""Freeze the Phase 7 node 2.2 B_dev comparison and promotion gate."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "submission" / "results" / "experimental_investigations"
BASELINE = RESULTS / "phase7-node1-landscape-zoo-dev"
CANDIDATE = RESULTS / "phase7-node2-full-mean-jackknife-dev"
BASELINES = ("component_ard", "stage_split")
CANDIDATE_NAME = "full_mean_jackknife"
CORE_METRICS = (
    "initial_round_found_best",
    "best_found",
    "round_to_95_global_best",
    "AUC_best_so_far",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean_ci(values: pd.Series) -> tuple[float, float, float, float]:
    array = values.to_numpy(dtype=float)
    mean = float(array.mean())
    if len(array) == 1:
        return mean, 0.0, mean, mean
    sd = float(array.std(ddof=1))
    half = float(t.ppf(0.975, len(array) - 1) * sd / math.sqrt(len(array)))
    return mean, sd, mean - half, mean + half


def absolute_summary(
    rows: pd.DataFrame,
    *,
    group_columns: tuple[str, ...] = ("variant",),
) -> pd.DataFrame:
    records = []
    for keys, group in rows.groupby(list(group_columns), sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        prefix = dict(zip(group_columns, keys, strict=True))
        for metric in CORE_METRICS + ("elapsed_seconds",):
            mean, sd, low, high = mean_ci(group[metric])
            records.append(
                {
                    **prefix,
                    "metric": metric,
                    "n": len(group),
                    "mean": mean,
                    "sample_sd": sd,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    return pd.DataFrame(records)


def paired_summary(rows: pd.DataFrame) -> pd.DataFrame:
    candidate = rows[rows["variant"].eq(CANDIDATE_NAME)].set_index("landscape")
    records = []
    for baseline_name in BASELINES:
        baseline = rows[rows["variant"].eq(baseline_name)].set_index("landscape")
        common = baseline.index.intersection(candidate.index)
        for metric in CORE_METRICS + ("elapsed_seconds",):
            delta = candidate.loc[common, metric] - baseline.loc[common, metric]
            mean, sd, low, high = mean_ci(delta)
            lower_better = metric in {
                "round_to_95_global_best",
                "elapsed_seconds",
            }
            wins = int(
                (delta < -1e-10).sum()
                if lower_better
                else (delta > 1e-10).sum()
            )
            losses = int(
                (delta > 1e-10).sum()
                if lower_better
                else (delta < -1e-10).sum()
            )
            records.append(
                {
                    "candidate": CANDIDATE_NAME,
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


def tail_summary(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for variant, group in rows.groupby("variant", sort=True):
        values = np.sort(group["AUC_best_so_far"].to_numpy(dtype=float))
        count = max(1, int(math.ceil(0.10 * len(values))))
        records.append(
            {
                "variant": variant,
                "n": len(values),
                "tail_count": count,
                "auc_cvar10": float(values[:count].mean()),
            }
        )
    return pd.DataFrame(records)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def trajectory_audit(root: Path, expected_variant: str) -> dict:
    paths = sorted((root / "trajectories" / expected_variant).glob("*.json"))
    checks = []
    for path in paths:
        payload = read_json(path)
        trajectory = payload["trajectory"]
        query_ids = [int(row["query_index"]) for row in trajectory]
        contract = payload["landscape_contract"]
        checks.append(
            len(trajectory) == 40
            and len(query_ids) == len(set(query_ids))
            and contract["n_initial"] == 29
            and contract["n_candidates"] == 5731
            and contract["budget"] == 40
            and contract["public_test_labels_used"] is False
            and contract["public_searchspace_labels_used"] is False
            and contract["production_default_changed"] is False
        )
    return {
        "variant": expected_variant,
        "trajectory_count": len(paths),
        "valid_trajectory_count": int(sum(checks)),
        "passed": len(paths) == 20 and all(checks),
    }


def main() -> None:
    baseline_rows = pd.read_csv(BASELINE / "per_landscape.csv")
    candidate_rows = pd.read_csv(CANDIDATE / "per_landscape.csv")
    baseline_first20 = pd.read_csv(BASELINE / "first20_per_landscape.csv")
    candidate_first20 = pd.read_csv(CANDIDATE / "first20_per_landscape.csv")
    rows = pd.concat([baseline_rows, candidate_rows], ignore_index=True).sort_values(
        ["landscape", "variant"]
    )
    first20 = pd.concat(
        [baseline_first20, candidate_first20], ignore_index=True
    ).sort_values(["landscape", "variant"])

    baseline_manifest = read_json(BASELINE / "manifest.json")
    candidate_manifest = read_json(CANDIDATE / "manifest.json")
    baseline_audit = read_json(BASELINE / "audit.json")
    candidate_audit = read_json(CANDIDATE / "audit.json")
    expected_landscapes = set(baseline_rows["landscape"])
    variants = set(rows["variant"])
    geometry_columns = [
        "landscape",
        "family",
        "teacher_seed",
        "campaign_seed",
        "n_initial",
        "n_candidates",
        "budget",
        "global_best",
    ]
    reference_geometry = (
        baseline_rows[baseline_rows["variant"].eq("component_ard")]
        .sort_values("landscape")[geometry_columns]
        .reset_index(drop=True)
    )
    candidate_geometry = (
        candidate_rows.sort_values("landscape")[geometry_columns].reset_index(drop=True)
    )
    geometry_exact = reference_geometry.equals(candidate_geometry)
    manifest_hashes_match = all(
        (
            baseline_manifest[key] == candidate_manifest[key]
            for key in (
                "split",
                "landscape_count",
                "landscape_specs_sha256",
                "official_geometry",
                "allowed_inputs",
                "protected_inputs",
                "environment_diagnostics",
            )
        )
    )
    panel_complete = (
        variants == {CANDIDATE_NAME, *BASELINES}
        and len(rows) == 60
        and all(
            set(rows[rows["variant"].eq(variant)]["landscape"])
            == expected_landscapes
            for variant in variants
        )
    )
    finite_metrics = bool(
        np.isfinite(rows.loc[:, CORE_METRICS + ("elapsed_seconds",)].to_numpy()).all()
    )
    candidate_trajectory_audit = trajectory_audit(CANDIDATE, CANDIDATE_NAME)

    rows.to_csv(CANDIDATE / "combined_per_landscape.csv", index=False)
    first20.to_csv(CANDIDATE / "combined_first20_per_landscape.csv", index=False)
    absolute_summary(rows).to_csv(CANDIDATE / "combined_summary.csv", index=False)
    absolute_summary(first20).to_csv(
        CANDIDATE / "combined_first20_summary.csv", index=False
    )
    absolute_summary(rows, group_columns=("family", "variant")).to_csv(
        CANDIDATE / "combined_per_teacher_summary.csv", index=False
    )
    paired = paired_summary(rows)
    paired.to_csv(CANDIDATE / "paired_vs_baselines.csv", index=False)
    paired_summary(first20).to_csv(
        CANDIDATE / "paired_first20_vs_baselines.csv", index=False
    )
    tails = tail_summary(rows)
    tails.to_csv(CANDIDATE / "combined_tail_summary.csv", index=False)

    elapsed = (
        rows.groupby("variant", sort=True)["elapsed_seconds"].mean().to_dict()
    )
    runtime_ratios = pd.DataFrame(
        [
            {
                "candidate": CANDIDATE_NAME,
                "baseline": baseline,
                "candidate_mean_seconds": elapsed[CANDIDATE_NAME],
                "baseline_mean_seconds": elapsed[baseline],
                "runtime_ratio": elapsed[CANDIDATE_NAME] / elapsed[baseline],
            }
            for baseline in BASELINES
        ]
    )
    runtime_ratios.to_csv(CANDIDATE / "runtime_ratios.csv", index=False)

    stage_deltas = paired[paired["baseline"].eq("stage_split")].set_index("metric")
    cvar = tails.set_index("variant")["auc_cvar10"]
    runtime_ratio = float(
        runtime_ratios[runtime_ratios["baseline"].eq("component_ard")][
            "runtime_ratio"
        ].iloc[0]
    )
    gate_checks = {
        "initial_delta_ge_0": bool(
            stage_deltas.loc["initial_round_found_best", "mean_delta"] >= 0.0
        ),
        "best_delta_ge_0": bool(
            stage_deltas.loc["best_found", "mean_delta"] >= 0.0
        ),
        "t95_delta_le_0": bool(
            stage_deltas.loc["round_to_95_global_best", "mean_delta"] <= 0.0
        ),
        "auc_delta_gt_0": bool(
            stage_deltas.loc["AUC_best_so_far", "mean_delta"] > 0.0
        ),
        "cvar10_delta_ge_0": bool(
            cvar[CANDIDATE_NAME] - cvar["stage_split"] >= 0.0
        ),
        "runtime_ratio_component_le_6": runtime_ratio <= 6.0,
        "geometry_exact": geometry_exact,
        "manifest_hashes_match": manifest_hashes_match,
        "panel_complete": panel_complete,
        "finite_metrics": finite_metrics,
        "candidate_trajectory_audit": candidate_trajectory_audit["passed"],
        "runner_audits_passed": bool(
            baseline_audit["passed"] and candidate_audit["passed"]
        ),
        "protected_inputs_closed": bool(
            not baseline_audit["protected_inputs_opened"]
            and not candidate_audit["protected_inputs_opened"]
        ),
    }
    gate = {
        "candidate": CANDIDATE_NAME,
        "baseline": "stage_split",
        "split": "dev",
        "passed": all(gate_checks.values()),
        "decision": "promote" if all(gate_checks.values()) else "reject",
        "checks": gate_checks,
        "mean_deltas_vs_stage_split": {
            metric: float(stage_deltas.loc[metric, "mean_delta"])
            for metric in CORE_METRICS
        },
        "auc_cvar10": {
            "candidate": float(cvar[CANDIDATE_NAME]),
            "stage_split": float(cvar["stage_split"]),
            "delta": float(cvar[CANDIDATE_NAME] - cvar["stage_split"]),
        },
        "runtime_ratio_vs_component": runtime_ratio,
        "test_split_opened": False,
        "production_default_changed": False,
    }
    (CANDIDATE / "gate_decision.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True), encoding="utf-8"
    )

    audit = {
        "passed": all(
            gate_checks[key]
            for key in (
                "geometry_exact",
                "manifest_hashes_match",
                "panel_complete",
                "finite_metrics",
                "candidate_trajectory_audit",
                "runner_audits_passed",
                "protected_inputs_closed",
            )
        ),
        "gate_passed": gate["passed"],
        "geometry_exact": geometry_exact,
        "manifest_hashes_match": manifest_hashes_match,
        "panel_complete": panel_complete,
        "finite_metrics": finite_metrics,
        "candidate_trajectory_audit": candidate_trajectory_audit,
        "protected_inputs_opened": False,
        "test_split_opened": False,
        "production_default_changed": False,
    }
    (CANDIDATE / "comparison_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )

    source_files = (
        BASELINE / "manifest.json",
        BASELINE / "audit.json",
        BASELINE / "per_landscape.csv",
        BASELINE / "first20_per_landscape.csv",
        CANDIDATE / "manifest.json",
        CANDIDATE / "audit.json",
        CANDIDATE / "per_landscape.csv",
        CANDIDATE / "first20_per_landscape.csv",
    )
    output_files = tuple(
        CANDIDATE / name
        for name in (
            "combined_per_landscape.csv",
            "combined_first20_per_landscape.csv",
            "combined_summary.csv",
            "combined_first20_summary.csv",
            "combined_per_teacher_summary.csv",
            "paired_vs_baselines.csv",
            "paired_first20_vs_baselines.csv",
            "combined_tail_summary.csv",
            "runtime_ratios.csv",
            "gate_decision.json",
            "comparison_audit.json",
        )
    )
    manifest = {
        "phase": "phase7_node2_full_mean_jackknife_dev_comparison",
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in source_files
        },
        "output_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in output_files
        },
        "baseline_landscape_specs_sha256": baseline_manifest[
            "landscape_specs_sha256"
        ],
        "candidate_landscape_specs_sha256": candidate_manifest[
            "landscape_specs_sha256"
        ],
        "landscape_matrix_sha256": candidate_manifest[
            "environment_diagnostics"
        ]["landscape_matrix_sha256"],
        "test_split_opened": False,
        "protected_inputs_opened": False,
    }
    (CANDIDATE / "comparison_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"audit": audit, "gate": gate}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
