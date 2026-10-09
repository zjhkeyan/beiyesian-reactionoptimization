from __future__ import annotations

import numpy as np

from bgolearn_rxn.factory import prepare_problem
from bgolearn_rxn.surrogates.anova_gp import StructuredANOVAGP
from bgolearn_rxn.surrogates.rank_weighted_gp import (
    RankWeightedComponentARDLowRankTaskGP,
    RankWeightedStructuredGP,
    empirical_rank_precision,
)


def test_empirical_rank_precision_is_deterministic_monotone_and_mean_one() -> None:
    y = np.array([50.0, 10.0, 50.0, 90.0])
    first = empirical_rank_precision(y, 0.45)
    second = empirical_rank_precision(y, 0.45)
    np.testing.assert_allclose(first, second, atol=0.0, rtol=0.0)
    assert np.isclose(first.mean(), 1.0, atol=1e-14)
    assert first[1] < first[0] == first[2] < first[3]
    np.testing.assert_allclose(empirical_rank_precision(y, 0.0), 1.0)


def test_rank_weighted_noise_hook_aligns_rows_and_reduces_high_y_noise(
    tiny_dataset, tiny_config
) -> None:
    prepared = prepare_problem(tiny_dataset, tiny_config)
    model = RankWeightedStructuredGP(
        prepared.encoder.schema, tiny_config.gp, strength=0.45, seed=11
    )
    y = tiny_dataset.initial[tiny_config.dataset.target_column].to_numpy(float)
    model.fit(prepared.X_initial, y)
    diagnostics = model.rank_weight_diagnostics()
    assert diagnostics["n_observed"] == len(y)
    assert np.isclose(diagnostics["precision_mean"], 1.0, atol=1e-14)
    assert model.effective_training_noise_[np.argmax(y)] < model.effective_training_noise_[np.argmin(y)]


def test_zero_strength_matches_uniform_gp(tiny_dataset, tiny_config) -> None:
    prepared = prepare_problem(tiny_dataset, tiny_config)
    y = tiny_dataset.initial[tiny_config.dataset.target_column].to_numpy(float)
    baseline = StructuredANOVAGP(
        prepared.encoder.schema, tiny_config.gp, seed=17
    ).fit(prepared.X_initial, y)
    weighted = RankWeightedStructuredGP(
        prepared.encoder.schema, tiny_config.gp, strength=0.0, seed=17
    ).fit(prepared.X_initial, y)
    baseline_posterior = baseline.predict(prepared.X_candidates)
    weighted_posterior = weighted.predict(prepared.X_candidates)
    np.testing.assert_allclose(weighted_posterior.mean, baseline_posterior.mean)
    np.testing.assert_allclose(weighted_posterior.std, baseline_posterior.std)


def test_rank_weighted_component_task_gp_remains_psd(tiny_dataset, tiny_config) -> None:
    prepared = prepare_problem(tiny_dataset, tiny_config)
    y = tiny_dataset.initial[tiny_config.dataset.target_column].to_numpy(float)
    model = RankWeightedComponentARDLowRankTaskGP(
        prepared.encoder.schema, tiny_config.gp, strength=0.30, seed=29
    ).fit(prepared.X_initial, y)
    covariance = model.posterior_covariance(prepared.X_candidates)
    np.testing.assert_allclose(covariance, covariance.T, atol=1e-12)
    assert np.min(np.linalg.eigvalsh(covariance)) >= -1e-8
