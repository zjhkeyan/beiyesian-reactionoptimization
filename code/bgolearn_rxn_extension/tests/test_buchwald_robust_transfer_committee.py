from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


RUNNER_PATH = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "run_buchwald_robust_transfer_committee.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_buchwald_robust_transfer_committee", RUNNER_PATH
)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


def test_committee_grid_is_compact_and_predeclared() -> None:
    assert len(RUNNER.COMMITTEE_GRID) == 3
    assert all(setting.transfer_weight > setting.target_weight for setting in RUNNER.COMMITTEE_GRID)
    assert all(0.0 < setting.learning_rate <= 0.30 for setting in RUNNER.COMMITTEE_GRID)


def test_target_only_wrapper_filters_source_rows() -> None:
    episode = RUNNER.meta.make_source_meta_episodes()[0]
    config, dataset, _, _ = RUNNER.meta.build_problem(
        episode, strict_source_isolation=True
    )
    runner = RUNNER.CampaignRunner(dataset, config, seed=episode["seed"], backend="native")
    RUNNER.install_committee(runner, RUNNER.COMMITTEE_GRID[0])
    target = runner.ensemble.models[RUNNER.TARGET_MODEL]
    target.fit(
        runner.prepared.X_initial,
        dataset.initial[config.dataset.target_column].to_numpy(dtype=float),
    )
    assert target.n_target_ == 2
    assert len(dataset.initial) == 23


def test_committee_preserves_query_one_and_restores_logei() -> None:
    episode = RUNNER.meta.make_source_meta_episodes()[0]
    output = "/tmp/robust-committee-test"
    baseline = RUNNER.run_one((episode, RUNNER.BASELINE, True, output))
    committee = RUNNER.run_one(
        (episode, RUNNER.COMMITTEE_GRID[0].name, True, output)
    )
    assert committee["query_1_index"] == baseline["query_1_index"]
    assert committee["query_1_acquisition"] == baseline["query_1_acquisition"]
    assert committee["query_2_acquisition"] == "stable_log_ei"
    assert np.isclose(
        committee["final_transfer_weight"] + committee["final_target_weight"], 1.0
    )


def test_source_problem_excludes_official_target() -> None:
    episode = RUNNER.meta.make_source_meta_episodes()[0]
    _, dataset, _, observed_products = RUNNER.meta.build_problem(
        episode, strict_source_isolation=True
    )
    assert RUNNER.OFFICIAL_TARGET not in observed_products
    assert RUNNER.OFFICIAL_TARGET not in set(dataset.initial["Product"].astype(str))
