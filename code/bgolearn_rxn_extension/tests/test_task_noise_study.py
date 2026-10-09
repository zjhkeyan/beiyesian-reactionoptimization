from __future__ import annotations

import importlib.util
from pathlib import Path


RUNNER_PATH = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "run_task_noise_study.py"
)
SPEC = importlib.util.spec_from_file_location("run_task_noise_study", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_pilot_excludes_official_target_and_confirmation_is_exhaustive() -> None:
    pilot = RUNNER.make_pilot_episodes()
    confirmation = RUNNER.make_confirmation_episodes()
    assert len(pilot) == 8
    assert all(episode["target_task"] != RUNNER.OFFICIAL_TARGET for episode in pilot)
    assert len(confirmation) == 21
    assert {episode["target_task"] for episode in confirmation} == {
        RUNNER.OFFICIAL_TARGET
    }


def test_task_noise_study_preserves_frozen_frontier_baseline() -> None:
    episode = RUNNER.make_pilot_episodes()[0]
    baseline, baseline_dataset, _, baseline_name = RUNNER.build_problem(
        episode, RUNNER.BASELINE
    )
    candidate, candidate_dataset, _, candidate_name = RUNNER.build_problem(
        episode, RUNNER.CANDIDATE
    )
    assert baseline_name == "component_ard_low_rank_task_gp"
    assert candidate_name == "component_ard_low_rank_task_noise_gp"
    assert baseline.gp == candidate.gp
    assert baseline.dataset == candidate.dataset
    assert baseline_dataset.candidate_ids.tolist() == candidate_dataset.candidate_ids.tolist()
    assert baseline.policy.first_query == baseline_name
    assert candidate.policy.first_query == candidate_name
    assert baseline.gp.target_transform == candidate.gp.target_transform == "standard"


def test_twenty_run_view_is_deterministic_and_metric_independent() -> None:
    import pandas as pd

    episodes = RUNNER.make_confirmation_episodes()
    rows = []
    for episode in reversed(episodes):
        for variant in reversed(RUNNER.VARIANTS):
            rows.append(
                {
                    "episode": episode["episode"],
                    "variant": variant,
                    "AUC_best_so_far": 1000.0 if episode is episodes[-1] else 0.0,
                }
            )
    view, omitted = RUNNER.deterministic_twenty_run_view(pd.DataFrame(rows))
    expected = sorted(episode["episode"] for episode in episodes)
    assert sorted(view["episode"].unique()) == expected[:20]
    assert omitted == expected[20]
    assert len(view) == 40
