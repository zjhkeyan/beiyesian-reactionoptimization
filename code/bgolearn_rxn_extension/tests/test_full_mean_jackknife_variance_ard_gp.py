from __future__ import annotations

from dataclasses import replace

import numpy as np

from bgolearn_rxn.factory import build_ensemble, prepare_problem
from bgolearn_rxn.surrogates.full_mean_jackknife_variance_ard_gp import (
    FullMeanJackknifeVarianceARDGP,
)


MODEL = "full_mean_jackknife_variance_ard_gp"


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


def _observations(prepared):
    X = np.vstack(
        [prepared.X_initial, prepared.X_initial, prepared.X_initial, prepared.X_initial]
    )
    y = np.asarray(
        [12.0, 91.0, 20.0, 13.0, 88.0, 23.0, 14.0, 85.0, 26.0, 15.0, 82.0, 29.0]
    )
    return X, y


def test_factory_is_opt_in_and_first_fit_is_bit_identical_to_component_ard(
    tiny_dataset, tiny_config
) -> None:
    prepared = prepare_problem(tiny_dataset, tiny_config)
    assert MODEL not in build_ensemble(tiny_config, prepared, seed=100).models

    config = _config(tiny_config)
    prepared = prepare_problem(tiny_dataset, config)
    model = build_ensemble(config, prepared, seed=100).models[MODEL]
    assert isinstance(model, FullMeanJackknifeVarianceARDGP)

    baseline_config = replace(
        tiny_config,
        ensemble=replace(
            tiny_config.ensemble,
            enabled_models=("component_ard_gp",),
            initial_weights={"component_ard_gp": 1.0},
        ),
    )
    baseline_prepared = prepare_problem(tiny_dataset, baseline_config)
    baseline = build_ensemble(baseline_config, baseline_prepared, seed=100).models[
        "component_ard_gp"
    ]
    X, y = _observations(prepared)
    model.fit(X, y)
    baseline.fit(X, y)
    actual = model.predict(prepared.X_candidates)
    expected = baseline.predict(baseline_prepared.X_candidates)

    np.testing.assert_array_equal(actual.mean, expected.mean)
    np.testing.assert_array_equal(actual.std, expected.std)
    np.testing.assert_array_equal(
        model.posterior_covariance(prepared.X_candidates),
        baseline.posterior_covariance(baseline_prepared.X_candidates),
    )
    assert model.fold_assignments_ is None
    assert model.active_stage_ == "full_component_ard_query_1"


def test_transition_preserves_full_mean_and_adds_psd_deterministic_variance(
    tiny_dataset, tiny_config
) -> None:
    config = _config(tiny_config)
    prepared = prepare_problem(tiny_dataset, config)
    X, y = _observations(prepared)
    model = build_ensemble(config, prepared, seed=100).models[MODEL]
    model.fit(X, y)
    model.fit(X, y)

    first_assignments = model.fold_assignments_.copy()
    posterior = model.predict(prepared.X_candidates)
    full = model.full_data_model.predict(prepared.X_candidates)
    covariance = model.posterior_covariance(prepared.X_candidates)

    np.testing.assert_array_equal(posterior.mean, full.mean)
    np.testing.assert_allclose(np.diag(covariance), posterior.std**2, rtol=1e-12)
    np.testing.assert_allclose(covariance, covariance.T, atol=1e-12)
    assert np.linalg.eigvalsh(model.disagreement_correction_).min() >= -1e-10
    assert np.all(np.isfinite(model.disagreement_correction_))
    assert np.any(model.disagreement_variance_ > 0.0)
    assert model.disagreement_diagnostics_["finite"]
    assert model.disagreement_diagnostics_["noncollapsed"]

    replica = build_ensemble(config, prepared, seed=100).models[MODEL]
    replica.fit(X, y)
    replica.fit(X, y)
    np.testing.assert_array_equal(replica.fold_assignments_, first_assignments)
    assert model.fold_sizes_ == (3, 3, 2, 2, 2)
    assert model.member_training_sizes_ == (9, 9, 10, 10, 10)
    assert model.audit_state_["fold_assignments"] == first_assignments.tolist()
    assert model.audit_state_["delete_member_count"] == 5


def test_large_pool_predict_never_eigendecomposes_candidate_covariance(
    tiny_dataset, tiny_config, monkeypatch
) -> None:
    config = _config(tiny_config)
    prepared = prepare_problem(tiny_dataset, config)
    X, y = _observations(prepared)
    model = build_ensemble(config, prepared, seed=100).models[MODEL]
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
    assert model.member_means_.shape == (5, 6000)
