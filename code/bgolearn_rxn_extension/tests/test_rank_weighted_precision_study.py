from __future__ import annotations

import importlib.util
from pathlib import Path


RUNNER_PATH = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "run_rank_weighted_precision_study.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_rank_weighted_precision_study", RUNNER_PATH
)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_rank_weighted_study_episode_contract() -> None:
    pilot = RUNNER.episodes_for_stage("pilot")
    confirmation = RUNNER.episodes_for_stage("confirmation")
    assert sum(name == "Buchwald_sub4" for name, _ in pilot) == 8
    assert sum(name == "Suzuki" for name, _ in pilot) == 20
    assert all(
        item["target_task"] != RUNNER.OFFICIAL_BUCHWALD_PRODUCT
        for name, item in pilot
        if name == "Buchwald_sub4"
    )
    assert sum(name == "Buchwald_sub4" for name, _ in confirmation) == 21
    assert sum(name == "Suzuki" for name, _ in confirmation) == 80


def test_strength_grid_and_variant_names_are_frozen() -> None:
    assert RUNNER.STRENGTHS == (0.0, 0.15, 0.30, 0.45)
    assert [RUNNER.variant_name(value) for value in RUNNER.STRENGTHS] == [
        "uniform",
        "rank_015",
        "rank_030",
        "rank_045",
    ]


def test_selection_gate_requires_both_datasets_and_positive_combined_auc() -> None:
    assert RUNNER.PILOT_GATE["minimum_per_dataset_auc_delta"] == 0.0
    assert RUNNER.PILOT_GATE["minimum_combined_auc_delta_strict"] == 0.0
    assert RUNNER.PILOT_GATE["minimum_initial_delta"] == -0.5
    assert RUNNER.PILOT_GATE["minimum_best_delta"] == -0.5
    assert RUNNER.PILOT_GATE["maximum_t95_delta"] == 0.25
