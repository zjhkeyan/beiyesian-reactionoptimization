from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


RUNNER_PATH = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "run_buchwald_pairwise_rank_warmstart.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_buchwald_pairwise_rank_warmstart", RUNNER_PATH
)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


def test_rank_and_blend_grids_are_predeclared() -> None:
    assert len(RUNNER.RANK_GRID) == 6
    assert RUNNER.BLEND_GRID == (0.0, 0.25, 0.50, 1.0, 2.0)
    assert all(setting.l2_c <= 0.01 for setting in RUNNER.RANK_GRID)


def test_ranker_excludes_official_and_held_products() -> None:
    held = RUNNER.source_products()[0]
    ranker = RUNNER.fit_ranker(RUNNER.RANK_GRID[0].name, held)
    assert RUNNER.OFFICIAL_TARGET not in ranker.training_products
    assert held not in ranker.training_products
    assert len(ranker.training_products) == 3


def test_rank_validation_has_every_setting_product_pair() -> None:
    validation = RUNNER.held_product_rank_validation()
    assert len(validation) == len(RUNNER.RANK_GRID) * 4
    assert set(validation["n_pairs"]) == {21}
    assert validation["pairwise_accuracy"].between(0.0, 1.0).all()


def test_query_one_policy_delegates_after_first_query() -> None:
    episode = RUNNER.meta.make_source_meta_episodes()[0]
    config, dataset, lookup, _ = RUNNER.meta.build_problem(
        episode, strict_source_isolation=True
    )
    ranker = RUNNER.fit_ranker(RUNNER.RANK_GRID[0].name, episode["target_task"])
    scores = ranker.score(dataset.candidates[list(config.dataset.feature_columns)])
    row = RUNNER.run_one(
        (episode, RUNNER.RANK_GRID[0].name, 0.5, True, "/tmp/rank-policy-test")
    )
    assert np.isfinite(scores).all()
    assert row["query_1_acquisition"] == "query_one_pairwise_rank_blend"
    assert row["query_2_acquisition"] == "stable_log_ei"
