from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


RUNNER_PATH = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "run_suzuki_ard_hybrid_acquisition.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_suzuki_ard_hybrid_acquisition", RUNNER_PATH
)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_phase4_ard_hybrid_episode_contract() -> None:
    episodes = RUNNER.make_pilot_episodes()
    assert len(episodes) == 60
    assert all(item["budget"] == 5 for item in episodes)
    assert min(item["split_seed"] for item in episodes) >= 710_000
    assert max(item["split_seed"] for item in episodes) < 810_000


def test_frozen_phase2_schedules_are_exact() -> None:
    logei = [
        "posterior_mean",
        RUNNER.frozen_hybrid.BASELINE_POLICY,
        RUNNER.frozen_hybrid.BASELINE_POLICY,
        RUNNER.frozen_hybrid.BASELINE_POLICY,
        RUNNER.frozen_hybrid.BASELINE_POLICY,
    ]
    hybrid = [
        "posterior_mean",
        RUNNER.frozen_hybrid.BASELINE_POLICY,
        RUNNER.frozen_hybrid.THOMPSON_POLICY,
        RUNNER.frozen_hybrid.BASELINE_POLICY,
        RUNNER.frozen_hybrid.THOMPSON_POLICY,
    ]
    assert RUNNER.schedule_for_variant(RUNNER.FIXED_LOGEI) == logei
    assert RUNNER.schedule_for_variant(RUNNER.ARD_LOGEI) == logei
    assert RUNNER.schedule_for_variant(RUNNER.FIXED_HYBRID) == hybrid
    assert RUNNER.schedule_for_variant(RUNNER.ARD_HYBRID) == hybrid


def test_each_surrogate_pair_has_identical_query_one_scores() -> None:
    mean = np.array([2.0, 8.0, 3.0])
    std = np.array([5.0, 0.2, 7.0])
    for logei_name, hybrid_name in (
        (RUNNER.FIXED_LOGEI, RUNNER.FIXED_HYBRID),
        (RUNNER.ARD_LOGEI, RUNNER.ARD_HYBRID),
    ):
        values = []
        for variant in (logei_name, hybrid_name):
            scores, component = RUNNER.frozen_hybrid.scheduled_scores(
                RUNNER.POLICY[variant],
                mean,
                std,
                best_f=4.0,
                step=1,
                covariance=None,
                campaign_seed=751_000,
            )
            assert component == "posterior_mean"
            values.append(scores)
        np.testing.assert_array_equal(values[0], values[1])


def test_gate_is_frozen_before_the_pilot() -> None:
    assert RUNNER.PILOT_GATE == {
        "initial_delta_vs_ard_logei_tolerance": 0.0,
        "minimum_best_delta_vs_fixed_logei": 0.0,
        "minimum_auc_delta_vs_fixed_logei": 0.0,
        "maximum_t95_delta_vs_fixed_logei": 0.0,
        "maximum_loss_frequency_increase_vs_ard_logei": 0.01,
    }
