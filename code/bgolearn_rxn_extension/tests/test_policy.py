from __future__ import annotations

import numpy as np

from bgolearn_rxn.config import PolicyConfig
from bgolearn_rxn.policy import MetricAwarePolicy
from bgolearn_rxn.schemas import Posterior


def test_first_query_maximizes_posterior_mean() -> None:
    posterior = Posterior(
        mean=np.array([5.0, 20.0, 10.0]),
        std=np.array([50.0, 1.0, 2.0]),
        component_means={"gp": np.array([5.0, 20.0, 10.0])},
        component_stds={"gp": np.ones(3)},
        weights={"gp": 1.0},
    )
    recommendation = MetricAwarePolicy(PolicyConfig(), seed=100).select(
        posterior,
        np.array([1, 2, 3]),
        best_f=0.0,
        step=1,
        budget=40,
    )
    assert recommendation.query_index == 2
    assert recommendation.acquisition_name == "posterior_mean"


def test_strain_sensor_pure_acquisition_multiplies_novelty_and_disagreement() -> None:
    posterior = Posterior(
        mean=np.array([90.0, 10.0, 20.0]),
        std=np.ones(3),
        component_means={"gp": np.array([90.0, 10.0, 20.0])},
        component_stds={"gp": np.ones(3)},
        weights={"gp": 1.0},
        disagreement=np.array([0.1, 2.0, 1.0]),
    )
    config = PolicyConfig(novelty_disagreement_mode="pure")
    recommendation = MetricAwarePolicy(config, seed=100).select(
        posterior,
        np.array([1, 2, 3]),
        best_f=0.0,
        step=1,
        budget=40,
        novelty_scores=np.array([10.0, 2.0, 3.0]),
    )
    assert recommendation.query_index == 2
    assert recommendation.acquisition_name == "strain_sensor_novelty_disagreement"


def test_strain_sensor_mode_requires_aligned_novelty() -> None:
    posterior = Posterior(mean=np.ones(2), std=np.ones(2))
    config = PolicyConfig(novelty_disagreement_mode="hybrid")
    with np.testing.assert_raises_regex(ValueError, "novelty_scores are required"):
        MetricAwarePolicy(config, seed=100).select(
            posterior,
            np.array([1, 2]),
            best_f=0.0,
            step=2,
            budget=40,
        )


def test_first_query_can_use_one_ensemble_component() -> None:
    posterior = Posterior(
        mean=np.array([10.0, 20.0]),
        std=np.ones(2),
        component_means={"gp": np.array([30.0, 5.0])},
        component_stds={"gp": np.ones(2)},
        weights={"gp": 1.0},
    )
    recommendation = MetricAwarePolicy(
        PolicyConfig(first_query="gp"), seed=100
    ).select(
        posterior,
        np.array([1, 2]),
        best_f=0.0,
        step=1,
        budget=40,
    )
    assert recommendation.query_index == 1
    assert recommendation.acquisition_name == "gp_posterior_mean"
