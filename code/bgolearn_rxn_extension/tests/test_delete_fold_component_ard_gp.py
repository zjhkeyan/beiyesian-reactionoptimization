from __future__ import annotations

from dataclasses import replace

import numpy as np

from bgolearn_rxn.factory import build_ensemble, prepare_problem
from bgolearn_rxn.surrogates.delete_fold_component_ard_gp import (
    DeleteFoldComponentARDEnsemble,
    aggregate_member_moments,
    deterministic_delete_fold_assignments,
)


MODEL = "delete_fold_component_ard_gp"


def _config(tiny_config):
    return replace(
        tiny_config,
        ensemble=replace(
            tiny_config.ensemble,
            enabled_models=(MODEL,),
            initial_weights={MODEL: 1.0},
        ),
        policy=replace(tiny_config.policy, first_query=MODEL),
    )


def test_delete_folds_are_balanced_quantile_spread_and_deterministic() -> None:
    y = np.asarray([4.0, 1.0, 4.0, 9.0, 3.0, 2.0, 8.0, 7.0, 6.0, 5.0, 0.0, 4.0])
    first = deterministic_delete_fold_assignments(y)
    second = deterministic_delete_fold_assignments(y.copy())
    np.testing.assert_array_equal(first, second)
    sizes = np.bincount(first, minlength=5)
    assert sizes.max() - sizes.min() <= 1

    order = np.lexsort((np.arange(len(y)), y))
    np.testing.assert_array_equal(first[order], np.arange(len(y)) % 5)


def test_total_variance_is_within_plus_population_disagreement() -> None:
    means = np.asarray(
        [[1.0, 8.0], [2.0, 9.0], [3.0, 10.0], [4.0, 11.0], [5.0, 12.0]]
    )
    stds = np.full_like(means, 2.0)
    mean, total, within, between = aggregate_member_moments(means, stds)
    np.testing.assert_allclose(mean, [3.0, 10.0])
    np.testing.assert_allclose(within, [4.0, 4.0])
    np.testing.assert_allclose(between, [2.0, 2.0])
    np.testing.assert_allclose(total, within + between)


def test_delete_fold_ensemble_is_factory_opt_in(tiny_dataset, tiny_config) -> None:
    prepared = prepare_problem(tiny_dataset, tiny_config)
    assert MODEL not in build_ensemble(tiny_config, prepared, seed=100).models
    config = _config(tiny_config)
    prepared = prepare_problem(tiny_dataset, config)
    model = build_ensemble(config, prepared, seed=100).models[MODEL]
    assert isinstance(model, DeleteFoldComponentARDEnsemble)
    assert len(model.members) == 5


def test_fit_predict_is_finite_balanced_and_covariance_consistent(
    tiny_dataset, tiny_config
) -> None:
    config = _config(tiny_config)
    prepared = prepare_problem(tiny_dataset, config)
    model = build_ensemble(config, prepared, seed=100).models[MODEL]
    X = np.vstack([prepared.X_initial, prepared.X_initial, prepared.X_initial, prepared.X_initial])
    y = np.asarray([12.0, 91.0, 20.0, 13.0, 88.0, 23.0, 14.0, 85.0, 26.0, 15.0, 82.0, 29.0])
    model.fit(X, y)
    posterior = model.predict(prepared.X_candidates)
    covariance = model.posterior_covariance(prepared.X_candidates)

    assert model.fold_sizes_ == (3, 3, 2, 2, 2)
    assert model.member_training_sizes_ == (9, 9, 10, 10, 10)
    assert np.isfinite(posterior.mean).all()
    assert np.all(np.isfinite(posterior.std) & (posterior.std > 0.0))
    assert model.between_variance_ is not None
    assert np.any(model.between_variance_ > 0.0)
    np.testing.assert_allclose(covariance, covariance.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(covariance), posterior.std**2, rtol=1e-6)
