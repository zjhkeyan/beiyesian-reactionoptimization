from __future__ import annotations

from dataclasses import replace

import numpy as np

from bgolearn_rxn.factory import build_ensemble, prepare_problem
from bgolearn_rxn.surrogates.stage_split_delete_fold_ard import (
    StageSplitDeleteFoldARD,
)


MODEL = "stage_split_delete_fold_ard_gp"


def _config(tiny_config, model_name: str):
    return replace(
        tiny_config,
        ensemble=replace(
            tiny_config.ensemble,
            enabled_models=(model_name,),
            initial_weights={model_name: 1.0},
        ),
        policy=replace(tiny_config.policy, first_query=model_name),
    )


def _synthetic_training(prepared):
    X = np.vstack(
        [prepared.X_initial, prepared.X_initial, prepared.X_initial, prepared.X_initial]
    )
    y = np.asarray(
        [12.0, 91.0, 20.0, 13.0, 88.0, 23.0, 14.0, 85.0, 26.0, 15.0, 82.0, 29.0]
    )
    return X, y


def test_query_one_is_bit_identical_to_component_ard(tiny_dataset, tiny_config) -> None:
    baseline_config = _config(tiny_config, "component_ard_gp")
    stage_config = _config(tiny_config, MODEL)
    baseline_prepared = prepare_problem(tiny_dataset, baseline_config)
    stage_prepared = prepare_problem(tiny_dataset, stage_config)
    X, y = _synthetic_training(stage_prepared)
    baseline = build_ensemble(baseline_config, baseline_prepared, seed=100).models[
        "component_ard_gp"
    ]
    stage = build_ensemble(stage_config, stage_prepared, seed=100).models[MODEL]
    baseline.fit(X, y)
    stage.fit(X, y)
    baseline_posterior = baseline.predict(stage_prepared.X_candidates)
    stage_posterior = stage.predict(stage_prepared.X_candidates)

    assert stage.active_stage_ == "component_ard_query_1"
    np.testing.assert_array_equal(stage.first_query_model.theta_, baseline.theta_)
    np.testing.assert_array_equal(stage_posterior.mean, baseline_posterior.mean)
    np.testing.assert_array_equal(stage_posterior.std, baseline_posterior.std)


def test_stage_transition_is_fixed_after_first_feedback(tiny_dataset, tiny_config) -> None:
    config = _config(tiny_config, MODEL)
    prepared = prepare_problem(tiny_dataset, config)
    X, y = _synthetic_training(prepared)
    stage = build_ensemble(config, prepared, seed=100).models[MODEL]
    stage.fit(X, y)
    stage.fit(np.vstack([X, prepared.X_candidates[[0]]]), np.append(y, 55.0))

    assert stage.active_stage_ == "delete_fold_logei_queries_2_5"
    assert stage.stage_history_ == [
        "component_ard_query_1",
        "delete_fold_logei_queries_2_5",
    ]
    assert stage.initial_observed_count_ == 12
    assert stage.transition_observed_count_ == 13


def test_stage_split_is_deterministic_and_uncertainty_is_finite(
    tiny_dataset, tiny_config
) -> None:
    config = _config(tiny_config, MODEL)
    prepared = prepare_problem(tiny_dataset, config)
    X, y = _synthetic_training(prepared)
    models = [build_ensemble(config, prepared, seed=101).models[MODEL] for _ in range(2)]
    posteriors = []
    for model in models:
        model.fit(X, y)
        model.fit(np.vstack([X, prepared.X_candidates[[0]]]), np.append(y, 55.0))
        posteriors.append(model.predict(prepared.X_candidates[1:]))
    np.testing.assert_array_equal(posteriors[0].mean, posteriors[1].mean)
    np.testing.assert_array_equal(posteriors[0].std, posteriors[1].std)
    assert np.all(np.isfinite(posteriors[0].std) & (posteriors[0].std > 0.0))


def test_factory_opt_in_and_covariance_interface(tiny_dataset, tiny_config) -> None:
    prepared = prepare_problem(tiny_dataset, tiny_config)
    assert MODEL not in build_ensemble(tiny_config, prepared, seed=100).models
    config = _config(tiny_config, MODEL)
    prepared = prepare_problem(tiny_dataset, config)
    model = build_ensemble(config, prepared, seed=100).models[MODEL]
    assert isinstance(model, StageSplitDeleteFoldARD)
    X, y = _synthetic_training(prepared)
    model.fit(X, y)
    posterior = model.predict(prepared.X_candidates)
    covariance = model.posterior_covariance(prepared.X_candidates)
    np.testing.assert_allclose(covariance, covariance.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(covariance), posterior.std**2, rtol=1e-6)
