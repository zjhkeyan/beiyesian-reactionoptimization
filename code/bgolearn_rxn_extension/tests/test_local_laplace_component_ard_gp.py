from __future__ import annotations

from dataclasses import replace

import numpy as np

from bgolearn_rxn.factory import build_ensemble, prepare_problem
from bgolearn_rxn.surrogates.local_laplace_component_ard_gp import (
    LocalLaplaceComponentARDStructuredGP,
)


MODEL = "local_laplace_component_ard_gp"


def _config(tiny_config):
    return replace(
        tiny_config,
        gp=replace(
            tiny_config.gp,
            optimize_hyperparameters=True,
            restarts=1,
            maxiter=80,
            refit_interval=1,
            target_transform="standard",
        ),
        ensemble=replace(
            tiny_config.ensemble,
            enabled_models=(MODEL,),
            initial_weights={MODEL: 1.0},
        ),
        policy=replace(tiny_config.policy, first_query=MODEL),
    )


def test_local_laplace_is_factory_opt_in_and_points_are_bounded(
    tiny_dataset, tiny_config
) -> None:
    prepared = prepare_problem(tiny_dataset, tiny_config)
    assert MODEL not in build_ensemble(tiny_config, prepared, seed=100).models

    config = _config(tiny_config)
    prepared = prepare_problem(tiny_dataset, config)
    model = build_ensemble(config, prepared, seed=100).models[MODEL]
    assert isinstance(model, LocalLaplaceComponentARDStructuredGP)
    y = tiny_dataset.initial[config.dataset.target_column].to_numpy(dtype=float)
    model.fit(prepared.X_initial, y)

    assert model.curvature_diagonal_ is not None
    assert np.all(np.isfinite(model.curvature_diagonal_))
    assert np.all(model.curvature_diagonal_ > 0.0)
    assert 1 <= len(model.laplace_points_) <= 2 * model.max_sigma_axes
    np.testing.assert_allclose(model.laplace_weights_.sum(), 1.0)
    bounds = np.asarray(model._bounds(), dtype=float)
    for point in model.laplace_points_:
        assert np.all(point.theta >= bounds[:, 0])
        assert np.all(point.theta <= bounds[:, 1])
    diagnostic = model.laplace_history_[-1]
    assert diagnostic["accepted_point_count"] == len(model.laplace_points_)
    assert np.isclose(
        diagnostic["effective_sample_count"],
        1.0 / np.sum(model.laplace_weights_**2),
    )


def test_local_laplace_total_variance_and_covariance_are_consistent(
    tiny_dataset, tiny_config
) -> None:
    config = _config(tiny_config)
    prepared = prepare_problem(tiny_dataset, config)
    model = build_ensemble(config, prepared, seed=100).models[MODEL]
    y = tiny_dataset.initial[config.dataset.target_column].to_numpy(dtype=float)
    model.fit(prepared.X_initial, y)

    posterior = model.predict(prepared.X_candidates)
    covariance = model.posterior_covariance(prepared.X_candidates)
    assert np.all(np.isfinite(posterior.mean))
    assert np.all(np.isfinite(posterior.std))
    assert np.all(posterior.std > 0.0)
    assert posterior.disagreement is not None
    assert np.all(posterior.disagreement >= 0.0)
    np.testing.assert_allclose(covariance, covariance.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(covariance), posterior.std**2, rtol=1e-6)
    assert np.min(np.linalg.eigvalsh(covariance)) >= -1e-7

    means = np.asarray([[1.0, 3.0], [3.0, 5.0]])
    variances = np.asarray([[2.0, 4.0], [2.0, 4.0]])
    mean, variance = model._total_variance(
        means, variances, np.asarray([0.5, 0.5])
    )
    np.testing.assert_allclose(mean, [2.0, 4.0])
    np.testing.assert_allclose(variance, [3.0, 5.0])
