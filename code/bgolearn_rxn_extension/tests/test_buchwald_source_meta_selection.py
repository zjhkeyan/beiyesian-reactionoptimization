from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


RUNNER_PATH = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "run_buchwald_source_meta_selection.py"
)
SPEC = importlib.util.spec_from_file_location("run_buchwald_source_meta_selection", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


def test_prior_grid_is_predeclared_and_contains_frozen_frontier() -> None:
    assert len(RUNNER.PRIOR_GRID) == 5
    assert RUNNER.PRIOR_GRID[0].name == "frontier_default"
    assert RUNNER.PRIOR_GRID[0].amplitude_prior_scale == 0.60
    assert RUNNER.PRIOR_GRID[0].task_prior_scale == 0.35


def test_source_meta_episodes_exhaust_four_nonofficial_products() -> None:
    episodes = RUNNER.make_source_meta_episodes()
    assert len(episodes) == 84
    assert len({episode["target_task"] for episode in episodes}) == 4
    assert all(episode["target_task"] != RUNNER.OFFICIAL_TARGET for episode in episodes)
    assert all(len(episode["initial_indices"]) == 2 for episode in episodes)
    assert all(len(episode["candidate_indices"]) == 5 for episode in episodes)


def test_source_problem_excludes_every_official_target_row() -> None:
    episode = RUNNER.make_source_meta_episodes()[0]
    _, dataset, lookup, observed_products = RUNNER.build_problem(
        episode, strict_source_isolation=True
    )
    assert RUNNER.OFFICIAL_TARGET not in observed_products
    assert RUNNER.OFFICIAL_TARGET not in set(dataset.initial["Product"].astype(str))
    assert RUNNER.OFFICIAL_TARGET not in set(dataset.candidates["Product"].astype(str))
    assert RUNNER.OFFICIAL_TARGET not in set(lookup["Product"].astype(str))
    assert len(dataset.initial) == 23
    assert len(dataset.candidates) == 5


def test_confirmation_preserves_phase2_observation_topology() -> None:
    episode = RUNNER.make_official_confirmation_episodes()[0]
    _, dataset, _, observed_products = RUNNER.build_problem(
        episode, strict_source_isolation=False
    )
    assert len(observed_products) == 5
    assert len(dataset.initial) == 30
    assert len(dataset.candidates) == 5
