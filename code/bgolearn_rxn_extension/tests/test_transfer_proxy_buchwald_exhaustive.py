from __future__ import annotations

import importlib.util
from pathlib import Path


RUNNER_PATH = (
    Path(__file__).resolve().parents[2] / "experiments" / "run_transfer_proxy.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_transfer_proxy_buchwald_exhaustive", RUNNER_PATH
)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_official_buchwald_product_has_all_unique_initial_pairs() -> None:
    episodes = RUNNER.make_buchwald_exhaustive_pair_episodes(1)
    pairs = {tuple(episode["initial_indices"]) for episode in episodes}

    assert len(episodes) == len(pairs) == 21
    assert all(len(episode["initial_indices"]) == 2 for episode in episodes)
    assert all(len(episode["candidate_indices"]) == 5 for episode in episodes)
    assert all(episode["budget"] == 3 for episode in episodes)
    assert all("ethylphenyl" in episode["target_task"] for episode in episodes)
    assert all(
        not set(episode["initial_indices"]).intersection(episode["candidate_indices"])
        for episode in episodes
    )
