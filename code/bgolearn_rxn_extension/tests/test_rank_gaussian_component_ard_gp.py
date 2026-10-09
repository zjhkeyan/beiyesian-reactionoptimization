from __future__ import annotations

from dataclasses import replace

import numpy as np

from bgolearn_rxn.factory import build_ensemble, prepare_problem
from bgolearn_rxn.rank_gaussian_transform import EmpiricalRankGaussianTransform
from bgolearn_rxn.surrogates.rank_gaussian_component_ard_gp import (
    RankGaussianComponentARDStructuredGP,
)


MODEL = "rank_gaussian_component_ard_gp"


def _rank_gaussian_config(tiny_config):
    return replace(
        tiny_config,
        ensemble=replace(
            tiny_config.ensemble,
            enabled_models=(MODEL,),
            initial_weights={MODEL: 1.0},
        ),
        policy=replace(tiny_config.policy, first_query=MODEL),
    )


def test_empirical_normal_scores_are_monotone_finite_and_tie_aware() -> None:
    observed = np.asarray([4.0, 4.0, 11.0, 37.0, 89.0])
    transform = EmpiricalRankGaussianTransform().fit(observed)
    grid = np.linspace(0.0, 100.0, 501)
    scores = transform.transform(grid)

    assert np.isfinite(scores).all()
    assert np.all(np.diff(scores) >= 0.0)
    assert transform.transform(np.asarray([4.0, 4.0]))[0] == transform.transform(
        np.asarray([4.0, 4.0])
    )[1]
    np.testing.assert_allclose(
        transform.inverse(transform.transform(np.unique(observed))),
        np.unique(observed),
        atol=1e-12,
    )
    assert np.isfinite(transform.transform(np.asarray([0.0, 100.0]))).all()


def test_inverse_moments_use_stable_bounded_quadrature() -> None:
    transform = EmpiricalRankGaussianTransform(quadrature_order=20).fit(
        np.asarray([2.0, 8.0, 15.0, 28.0, 52.0, 77.0, 96.0])
    )
    mean = np.asarray([-1.2, 0.0, 1.1])
    std = np.asarray([0.2, 0.8, 1.5])
    first = transform.inverse_mean_std(mean, std)
    second = transform.inverse_mean_std(mean, std)

    np.testing.assert_allclose(first[0], second[0], atol=0.0)
    np.testing.assert_allclose(first[1], second[1], atol=0.0)
    assert np.all((first[0] >= 0.0) & (first[0] <= 100.0))
    assert np.all(np.isfinite(first[1]) & (first[1] > 0.0))
    point_mean, point_std = transform.inverse_mean_std(mean, np.zeros_like(mean))
    np.testing.assert_allclose(point_mean, transform.inverse(mean), atol=1e-10)
    assert np.all(point_std >= 1e-12)


def test_rank_gaussian_component_ard_is_factory_opt_in(
    tiny_dataset, tiny_config
) -> None:
    prepared = prepare_problem(tiny_dataset, tiny_config)
    assert MODEL not in build_ensemble(tiny_config, prepared, seed=100).models

    config = _rank_gaussian_config(tiny_config)
    prepared = prepare_problem(tiny_dataset, config)
    model = build_ensemble(config, prepared, seed=100).models[MODEL]
    assert isinstance(model, RankGaussianComponentARDStructuredGP)
    y = tiny_dataset.initial[config.dataset.target_column].to_numpy(dtype=float)
    model.fit(prepared.X_initial, y)
    posterior = model.predict(prepared.X_candidates)
    covariance = model.posterior_covariance(prepared.X_candidates)

    assert model.target_transform.observed_count_ == len(y)
    np.testing.assert_array_equal(model.target_transform.yield_knots_, np.unique(y))
    assert np.isfinite(posterior.mean).all()
    assert np.all(np.isfinite(posterior.std) & (posterior.std > 0.0))
    np.testing.assert_allclose(covariance, covariance.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(covariance), posterior.std**2, rtol=1e-6)


def test_transform_refit_replaces_prior_feedback() -> None:
    transform = EmpiricalRankGaussianTransform().fit(np.asarray([1.0, 2.0, 90.0]))
    transform.fit(np.asarray([20.0, 30.0, 40.0, 40.0]))
    np.testing.assert_array_equal(
        transform.yield_knots_, np.asarray([20.0, 30.0, 40.0])
    )
    assert transform.observed_count_ == 4
