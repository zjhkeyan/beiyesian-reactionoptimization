from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2] / "experiments"
RUNNER_PATH = EXPERIMENT_ROOT / "run_suzuki_hybrid_acquisition.py"
SPEC = importlib.util.spec_from_file_location("run_suzuki_hybrid_acquisition", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_predeclared_hybrid_schedule() -> None:
    assert [
        RUNNER.policy_component(RUNNER.HYBRID_POLICY, step)
        for step in range(1, 6)
    ] == [
        "posterior_mean",
        RUNNER.BASELINE_POLICY,
        RUNNER.THOMPSON_POLICY,
        RUNNER.BASELINE_POLICY,
        RUNNER.THOMPSON_POLICY,
    ]


def test_all_policies_have_identical_first_query_scores() -> None:
    mean = np.array([1.0, 4.0, 2.0])
    std = np.array([10.0, 0.1, 3.0])
    scores = []
    for policy in RUNNER.POLICIES:
        value, component = RUNNER.scheduled_scores(
            policy,
            mean,
            std,
            best_f=2.0,
            step=1,
            covariance=None,
            campaign_seed=71,
        )
        assert component == "posterior_mean"
        scores.append(value)
    for value in scores:
        np.testing.assert_allclose(value, mean)


def test_untouched_episode_range_is_covered_and_unique() -> None:
    frame = pd.read_csv(RUNNER.TRAIN_FILE)
    episodes = RUNNER.make_episodes()
    assert len(episodes) == 100
    assert len({episode["split_seed"] for episode in episodes}) == 100
    assert min(episode["split_seed"] for episode in episodes) >= 310_000
    for episode in episodes:
        initial = np.asarray(episode["initial_indices"], dtype=int)
        candidates = np.asarray(episode["candidate_indices"], dtype=int)
        assert len(initial) == 20
        assert len(candidates) == len(frame) - 20
        assert set(initial).isdisjoint(candidates)
        assert RUNNER.suzuki_proxy._covered(frame, initial, candidates)
        assert episode["budget"] == 5
