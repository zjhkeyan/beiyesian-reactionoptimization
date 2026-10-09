from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2] / "experiments"
sys.path.insert(0, str(EXPERIMENT_ROOT))
HELPER_PATH = EXPERIMENT_ROOT / "acquisition_study_helpers.py"
SPEC = importlib.util.spec_from_file_location("acquisition_study_helpers", HELPER_PATH)
assert SPEC is not None and SPEC.loader is not None
HELPERS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPERS)


def test_correlated_thompson_is_deterministic_and_uses_covariance() -> None:
    mean = np.array([0.0, 0.0])
    covariance = np.array([[1.0, 0.95], [0.95, 1.0]])
    first = HELPERS.correlated_thompson_sample(mean, covariance, seed=17)
    second = HELPERS.correlated_thompson_sample(mean, covariance, seed=17)
    other = HELPERS.correlated_thompson_sample(mean, covariance, seed=18)
    np.testing.assert_allclose(first, second)
    assert not np.allclose(first, other)


def test_ucb_schedules_increase_exploration_weight() -> None:
    mean = np.array([1.0, 1.0])
    std = np.array([0.0, 2.0])
    early = HELPERS.acquisition_scores(
        "mean_then_ucb_025_to_150", mean, std, best_f=0.0, step=2, budget=5
    )
    late = HELPERS.acquisition_scores(
        "mean_then_ucb_025_to_150", mean, std, best_f=0.0, step=5, budget=5
    )
    assert late[1] - late[0] > early[1] - early[0]


def test_predeclared_selection_applies_safety_constraints() -> None:
    rows = []
    values = {
        "mean_then_logei": (50.0, 60.0, 3.0, 40.0),
        "posterior_mean": (55.0, 59.0, 3.0, 40.0),
        "mean_then_pi": (52.0, 59.6, 3.2, 40.0),
    }
    for policy, (auc, best, t95, initial) in values.items():
        for metric, mean in (
            ("AUC_best_so_far", auc),
            ("best_found", best),
            ("round_to_95_global_best", t95),
            ("initial_round_found_best", initial),
        ):
            rows.append({"dataset": "demo", "policy": policy, "metric": metric, "mean": mean})
    selected = HELPERS.select_from_screening(pd.DataFrame(rows), "demo")
    assert selected["selected_policy"] == "mean_then_pi"


def test_every_policy_uses_posterior_mean_at_first_query() -> None:
    mean = np.array([3.0, 1.0])
    std = np.array([0.1, 10.0])
    covariance = np.diag(std**2)
    for policy in HELPERS.POLICIES:
        scores = HELPERS.acquisition_scores(
            policy,
            mean,
            std,
            best_f=2.0,
            step=1,
            budget=5,
            covariance=covariance,
            seed=9,
        )
        np.testing.assert_allclose(scores, mean)
