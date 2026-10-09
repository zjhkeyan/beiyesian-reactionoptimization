from __future__ import annotations

from dataclasses import replace

import numpy as np

from bgolearn_rxn.factory import build_ensemble, prepare_problem
from bgolearn_rxn.surrogates.component_ard_gp import ComponentARDStructuredGP
from bgolearn_rxn.surrogates.incumbent_localized_jackknife_variance_ard_gp import (
    IncumbentLocalizedJackknifeVarianceARDGP,
)


def _observations(prepared):
    X = np.vstack(
        [prepared.X_initial, prepared.X_initial, prepared.X_initial, prepared.X_initial]
    )
    y = np.asarray(
        [12.0, 91.0, 20.0, 13.0, 88.0, 23.0, 14.0, 85.0, 26.0, 15.0, 82.0, 29.0]
    )
    return X, y


def _model(prepared, tiny_config, seed=100):
    return IncumbentLocalizedJackknifeVarianceARDGP(
        prepared.encoder.schema, tiny_config.gp, seed=seed
    )


def test_factory_builds_exact_opt_in_model(tiny_dataset, tiny_config) -> None:
    model_name = IncumbentLocalizedJackknifeVarianceARDGP.name
    config = replace(
        tiny_config,
        ensemble=replace(
            tiny_config.ensemble,
            enabled_models=(model_name,),
            initial_weights={model_name: 1.0},
        ),
        policy=replace(tiny_config.policy, first_query=model_name),
    )
    prepared = prepare_problem(tiny_dataset, config)
    ensemble = build_ensemble(config, prepared, seed=100)

    assert tuple(ensemble.models) == (model_name,)
    assert isinstance(
        ensemble.models[model_name], IncumbentLocalizedJackknifeVarianceARDGP
    )


def test_query_one_is_bit_identical_to_full_component_ard(
    tiny_dataset, tiny_config
) -> None:
    prepared = prepare_problem(tiny_dataset, tiny_config)
    X, y = _observations(prepared)
    model = _model(prepared, tiny_config)
    baseline = ComponentARDStructuredGP(
        prepared.encoder.schema, tiny_config.gp, seed=117
    )

    model.fit(X, y)
    baseline.fit(X, y)
    actual = model.predict(prepared.X_candidates)
    expected = baseline.predict(prepared.X_candidates)

    np.testing.assert_array_equal(actual.mean, expected.mean)
    np.testing.assert_array_equal(actual.std, expected.std)
    np.testing.assert_array_equal(
        model.posterior_covariance(prepared.X_candidates),
        baseline.posterior_covariance(prepared.X_candidates),
    )
    assert model.active_stage_ == "full_component_ard_query_1"
    assert model.fold_assignments_ is None


def test_later_fit_localizes_correction_and_preserves_full_posterior_outside(
    tiny_dataset, tiny_config
) -> None:
    prepared = prepare_problem(tiny_dataset, tiny_config)
    X, y = _observations(prepared)
    model = _model(prepared, tiny_config)
    model.fit(X, y)
    model.fit(X, y)

    posterior = model.predict(prepared.X_candidates)
    full = model.full_data_model.predict(prepared.X_candidates)
    support = model.support_mask_

    assert support is not None
    assert 0 < model.support_size_ < len(support)
    np.testing.assert_array_equal(posterior.mean, full.mean)
    np.testing.assert_array_equal(posterior.std[~support], full.std[~support])
    np.testing.assert_array_equal(model.localized_variance_[~support], 0.0)
    assert np.any(model.localized_variance_[support] > 0.0)
    assert model.localization_diagnostics_["localized_correction_trace"] < model.localization_diagnostics_["raw_correction_trace"]
    assert model.audit_state_["support_size"] == model.support_size_
    assert model.audit_state_["localization"]["finite"]
    assert model.stage_history_ == [
        "full_component_ard_query_1",
        "full_mean_plus_delete_fold_variance",
    ]


def test_localized_covariance_is_psd_and_matches_predictive_diagonal(
    tiny_dataset, tiny_config
) -> None:
    prepared = prepare_problem(tiny_dataset, tiny_config)
    X, y = _observations(prepared)
    model = _model(prepared, tiny_config)
    model.fit(X, y)
    model.fit(X, y)

    posterior = model.predict(prepared.X_candidates)
    covariance = model.posterior_covariance(prepared.X_candidates)
    support = model.support_mask_

    np.testing.assert_allclose(covariance, covariance.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(covariance), posterior.std**2, rtol=1e-12)
    assert np.linalg.eigvalsh(covariance).min() >= -1e-9
    assert np.linalg.eigvalsh(model.localized_correction_).min() >= -1e-10
    assert np.all(model.localized_correction_[~support, :] == 0.0)
    assert np.all(model.localized_correction_[:, ~support] == 0.0)


def test_localized_posterior_is_deterministic(tiny_dataset, tiny_config) -> None:
    prepared = prepare_problem(tiny_dataset, tiny_config)
    X, y = _observations(prepared)
    left = _model(prepared, tiny_config, seed=100)
    right = _model(prepared, tiny_config, seed=100)
    for model in (left, right):
        model.fit(X, y)
        model.fit(X, y)

    left_posterior = left.predict(prepared.X_candidates)
    right_posterior = right.predict(prepared.X_candidates)

    np.testing.assert_array_equal(left.fold_assignments_, right.fold_assignments_)
    np.testing.assert_array_equal(left.support_mask_, right.support_mask_)
    np.testing.assert_array_equal(left_posterior.mean, right_posterior.mean)
    np.testing.assert_array_equal(left_posterior.std, right_posterior.std)
    assert left.audit_state_ == right.audit_state_


def test_large_pool_predict_uses_only_member_gram_not_candidate_covariance(
    tiny_dataset, tiny_config, monkeypatch
) -> None:
    prepared = prepare_problem(tiny_dataset, tiny_config)
    X, y = _observations(prepared)
    model = _model(prepared, tiny_config)
    model.fit(X, y)
    model.fit(X, y)
    large_pool = np.tile(prepared.X_candidates, (1500, 1))

    original_eigvalsh = np.linalg.eigvalsh
    inspected_shapes: list[tuple[int, ...]] = []

    def guarded_eigvalsh(matrix):
        inspected_shapes.append(matrix.shape)
        assert matrix.shape[0] <= 5
        return original_eigvalsh(matrix)

    monkeypatch.setattr(np.linalg, "eigvalsh", guarded_eigvalsh)
    posterior = model.predict(large_pool)

    assert posterior.mean.shape == (6000,)
    assert posterior.std.shape == (6000,)
    assert inspected_shapes == [(5, 5)]
    assert model.disagreement_correction_ is None
    assert model.localized_correction_ is None
    assert model.member_means_.shape == (5, 6000)
    assert model.support_mask_.shape == (6000,)
