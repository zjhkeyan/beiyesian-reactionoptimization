from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


RUNNER_PATH = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "run_suzuki_surrogate_switch_study.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_suzuki_surrogate_switch_study", RUNNER_PATH
)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_surrogate_switch_episode_contract() -> None:
    episodes = RUNNER.make_episodes()
    assert len(episodes["pilot"]) == 20
    assert len(episodes["confirmation"]) == 80
    assert all(item["budget"] == 5 for phase in episodes.values() for item in phase)


def test_surrogate_schedules_are_predeclared() -> None:
    assert [RUNNER.model_family_for_step("fixed", step) for step in range(1, 6)] == [
        "fixed"
    ] * 5
    assert [RUNNER.model_family_for_step("pure_ard", step) for step in range(1, 6)] == [
        "component_ard"
    ] * 5
    assert [
        RUNNER.model_family_for_step("fixed_q1_ard_later", step)
        for step in range(1, 6)
    ] == ["fixed", "component_ard", "component_ard", "component_ard", "component_ard"]


def test_pilot_gate_requires_strict_gain_and_pure_ard_noninferiority() -> None:
    assert RUNNER.PILOT_GATE["strict_gain_metrics"] == [
        "AUC_best_so_far",
        "best_found",
    ]
    assert RUNNER.PILOT_GATE["minimum_auc_delta_vs_pure_ard"] == 0.0
    assert RUNNER.PILOT_GATE["maximum_t95_delta_vs_fixed"] == 0.25


def test_paired_summary_accepts_an_explicit_baseline() -> None:
    rows = []
    variants = (("fixed", 0.0), ("pure_ard", 1.0), (RUNNER.CANDIDATE, 2.0))
    for variant, offset in variants:
        rows.append(
            {
                "phase": "pilot",
                "episode_family": "test",
                "episode": "episode_00",
                "variant": variant,
                **{metric: offset for metric in RUNNER.SUMMARY_METRICS},
            }
        )
    paired = RUNNER.paired_summary(pd.DataFrame(rows), baseline_name="pure_ard")
    candidate = paired[paired["candidate"].eq(RUNNER.CANDIDATE)]
    assert set(candidate["baseline"]) == {"pure_ard"}
    assert set(candidate["mean_delta"]) == {1.0}
