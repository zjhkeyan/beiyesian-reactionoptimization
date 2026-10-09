from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


RUNNER_PATH = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "run_suzuki_holdout_proxy.py"
)
SPEC = importlib.util.spec_from_file_location("run_suzuki_holdout_proxy", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_random_splits_are_deterministic_covered_and_disjoint() -> None:
    first = RUNNER.make_random_episodes()
    second = RUNNER.make_random_episodes()
    frame = pd.read_csv(RUNNER.TRAIN_FILE)

    assert first == second
    assert len(first) == 40
    assert {episode["phase"] for episode in first} == {"screening", "confirmation"}
    assert sum(episode["phase"] == "screening" for episode in first) == 20
    assert sum(episode["phase"] == "confirmation" for episode in first) == 20
    screen_seeds = {e["split_seed"] for e in first if e["phase"] == "screening"}
    confirm_seeds = {e["split_seed"] for e in first if e["phase"] == "confirmation"}
    assert screen_seeds.isdisjoint(confirm_seeds)

    for episode in first:
        initial = episode["initial_indices"]
        candidates = episode["candidate_indices"]
        assert len(initial) == 20
        assert len(candidates) == 9
        assert episode["budget"] == 5
        assert not set(initial).intersection(candidates)
        assert set(initial).union(candidates) == set(range(len(frame)))
        for role in RUNNER.ROLE_COLUMNS:
            assert set(frame.loc[candidates, role]).issubset(frame.loc[initial, role])


def test_block_splits_hold_out_complete_eligible_levels() -> None:
    frame = pd.read_csv(RUNNER.TRAIN_FILE)
    episodes = RUNNER.make_block_episodes()
    assert episodes
    for episode in episodes:
        initial = episode["initial_indices"]
        candidates = episode["candidate_indices"]
        role = episode["held_role"]
        level = episode["held_level"]
        assert episode["phase"] == "block_diagnostic"
        assert len(candidates) >= 2
        assert not set(initial).intersection(candidates)
        assert set(initial).union(candidates) == set(range(len(frame)))
        assert set(frame.loc[candidates, role].astype(str)) == {level}
        assert level not in set(frame.loc[initial, role].astype(str))


def test_problem_preserves_source_ids_without_overlap() -> None:
    episode = RUNNER.make_random_episodes()[0]
    _, dataset, lookup = RUNNER.build_problem(episode, "gp_only")
    candidate_ids = set(dataset.candidates["query_index"].astype(int))

    assert candidate_ids == set(episode["candidate_indices"])
    assert candidate_ids == set(lookup["query_index"].astype(int))
    assert candidate_ids.isdisjoint(episode["initial_indices"])
    assert len(dataset.initial) == 20
    assert len(dataset.candidates) == 9
