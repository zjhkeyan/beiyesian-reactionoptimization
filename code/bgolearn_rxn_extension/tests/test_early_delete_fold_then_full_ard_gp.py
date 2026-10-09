from __future__ import annotations

from dataclasses import replace

import numpy as np

from bgolearn_rxn.factory import build_ensemble, prepare_problem
from bgolearn_rxn.surrogates.early_delete_fold_then_full_ard_gp import (
    EarlyDeleteFoldThenFullARD,
)


MODEL = "early_delete_fold_then_full_ard_gp"


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


def _training(prepared, count: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(count) % len(prepared.X_initial)
    X = prepared.X_initial[indices]
    y = 10.0 + (np.arange(count) * 17 % 79).astype(float)
    return X, y


def test_factory_is_opt_in_and_query_one_is_bit_identical(
    tiny_dataset, tiny_config
) -> None:
    prepared = prepare_problem(tiny_dataset, tiny_config)
    assert MODEL not in build_ensemble(tiny_config, prepared, seed=100).models

    baseline_config = _config(tiny_config, "component_ard_gp")
    schedule_config = _config(tiny_config, MODEL)
    baseline_prepared = prepare_problem(tiny_dataset, baseline_config)
    schedule_prepared = prepare_problem(tiny_dataset, schedule_config)
    X, y = _training(schedule_prepared, 12)
    baseline = build_ensemble(
        baseline_config, baseline_prepared, seed=100
    ).models["component_ard_gp"]
    schedule = build_ensemble(
        schedule_config, schedule_prepared, seed=100
    ).models[MODEL]
    assert isinstance(schedule, EarlyDeleteFoldThenFullARD)
    baseline.fit(X, y)
    schedule.fit(X, y)
    baseline_posterior = baseline.predict(schedule_prepared.X_candidates)
    schedule_posterior = schedule.predict(schedule_prepared.X_candidates)

    assert schedule.active_stage_ == "full_component_ard_query_1"
    np.testing.assert_array_equal(schedule.full_data_model.theta_, baseline.theta_)
    np.testing.assert_array_equal(schedule_posterior.mean, baseline_posterior.mean)
    np.testing.assert_array_equal(schedule_posterior.std, baseline_posterior.std)


def test_exact_count_transition_and_json_safe_audit(tiny_dataset, tiny_config) -> None:
    config = _config(tiny_config, MODEL)
    prepared = prepare_problem(tiny_dataset, config)
    model = build_ensemble(config, prepared, seed=101).models[MODEL]
    for count in (12, 21, 22, 23):
        X, y = _training(prepared, count)
        model.fit(X, y)

    assert model.stage_history_ == [
        "full_component_ard_query_1",
        "delete_fold_component_ard_early",
        "full_component_ard_mature",
        "full_component_ard_mature",
    ]
    assert model.observed_count_history_ == [12, 21, 22, 23]
    assert model.initial_observed_count_ == 12
    assert model.maturity_threshold_observed_count_ == 22
    assert model.mature_transition_observed_count_ == 22
    audit = model.audit_state()
    assert audit["stage_history"] == model.stage_history_
    assert audit["observed_count_history"] == [12, 21, 22, 23]


def test_delete_and_mature_posteriors_are_deterministic_finite_and_covariant(
    tiny_dataset, tiny_config
) -> None:
    config = _config(tiny_config, MODEL)
    prepared = prepare_problem(tiny_dataset, config)
    models = [build_ensemble(config, prepared, seed=102).models[MODEL] for _ in range(2)]
    stage_posteriors = []
    mature_posteriors = []
    mature_covariances = []
    for model in models:
        for count in (12, 21):
            X, y = _training(prepared, count)
            model.fit(X, y)
        stage_posteriors.append(model.predict(prepared.X_candidates))
        X, y = _training(prepared, 22)
        model.fit(X, y)
        mature_posteriors.append(model.predict(prepared.X_candidates))
        mature_covariances.append(model.posterior_covariance(prepared.X_candidates))

    np.testing.assert_array_equal(stage_posteriors[0].mean, stage_posteriors[1].mean)
    np.testing.assert_array_equal(stage_posteriors[0].std, stage_posteriors[1].std)
    np.testing.assert_array_equal(mature_posteriors[0].mean, mature_posteriors[1].mean)
    np.testing.assert_array_equal(mature_posteriors[0].std, mature_posteriors[1].std)
    covariance = mature_covariances[0]
    assert np.isfinite(mature_posteriors[0].mean).all()
    assert np.all(np.isfinite(mature_posteriors[0].std) & (mature_posteriors[0].std > 0.0))
    assert np.isfinite(covariance).all()
    np.testing.assert_allclose(covariance, covariance.T, atol=1e-12)
    np.testing.assert_allclose(
        np.diag(covariance), mature_posteriors[0].std**2, rtol=1e-6
    )
