from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


RUNNER_PATH = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "run_suzuki_reverse_stage_ard.py"
)
SPEC = importlib.util.spec_from_file_location("run_suzuki_reverse_stage_ard", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_reverse_stage_episode_contract_is_frozen_without_confirmation() -> None:
    pilot = RUNNER.episodes_for_stage("pilot")
    check = RUNNER.episodes_for_stage("implementation_check")
    assert len(pilot) == 60
    assert check == pilot[:4]
    assert all(item["budget"] == 5 for item in pilot)
    assert min(item["split_seed"] for item in pilot) >= 710_000
    assert max(item["split_seed"] for item in pilot) < 810_000
    with pytest.raises(ValueError, match="Unknown stage"):
        RUNNER.episodes_for_stage("confirmation")


def test_reverse_stage_schedules_are_exact() -> None:
    schedules = {
        variant: [RUNNER.model_family_for_step(variant, step) for step in range(1, 6)]
        for variant in RUNNER.VARIANTS
    }
    assert schedules["fixed"] == ["fixed"] * 5
    assert schedules["pure_ard"] == ["component_ard"] * 5
    assert schedules[RUNNER.CANDIDATE] == [
        "component_ard",
        "fixed",
        "fixed",
        "fixed",
        "fixed",
    ]


def test_reverse_stage_problem_is_train_only_and_deterministic() -> None:
    episode = RUNNER.make_pilot_episodes()[0]
    config_a, dataset_a, lookup_a = RUNNER.build_problem(episode)
    config_b, dataset_b, lookup_b = RUNNER.build_problem(episode)
    assert config_a.gp.target_transform == "standard"
    assert config_a == config_b
    assert dataset_a.spec.budget == 5
    assert len(dataset_a.initial) == 20
    assert len(dataset_a.candidates) == len(lookup_a) == 9
    assert set(dataset_a.candidates["query_index"]) == set(lookup_a["query_index"])
    pd.testing.assert_frame_equal(dataset_a.initial, dataset_b.initial)
    pd.testing.assert_frame_equal(dataset_a.candidates, dataset_b.candidates)
    pd.testing.assert_frame_equal(lookup_a, lookup_b)


def test_pilot_gate_requires_all_metric_directions_and_runtime_gain() -> None:
    assert RUNNER.PILOT_GATE == {
        "minimum_initial_delta_vs_fixed": 0.0,
        "minimum_best_delta_vs_fixed": 0.0,
        "minimum_auc_delta_vs_fixed": 0.0,
        "maximum_t95_delta_vs_fixed": 0.0,
        "minimum_runtime_reduction_seconds_vs_pure_ard": 0.25,
        "minimum_runtime_reduction_fraction_vs_pure_ard": 0.10,
    }
