from __future__ import annotations

from dataclasses import replace

import numpy as np

from bgolearn_rxn.factory import build_ensemble, prepare_problem
from bgolearn_rxn.surrogates.component_ard_gp import (
    ComponentARDLowRankTaskGP,
    ComponentARDStructuredGP,
    TaskNoiseComponentARDLowRankTaskGP,
)


def _ard_config(tiny_config, model_name: str):
    return replace(
        tiny_config,
        gp=replace(
            tiny_config.gp,
            optimize_hyperparameters=True,
            restarts=1,
            maxiter=80,
            refit_interval=1,
        ),
        ensemble=replace(
            tiny_config.ensemble,
            enabled_models=(model_name,),
            initial_weights={model_name: 1.0},
        ),
        policy=replace(tiny_config.policy, first_query=model_name),
    )


def test_component_ard_amplitudes_move_and_kernel_remains_psd(
    tiny_dataset, tiny_config
) -> None:
    config = _ard_config(tiny_config, "component_ard_low_rank_task_gp")
    prepared = prepare_problem(tiny_dataset, config)
    ensemble = build_ensemble(config, prepared, seed=100)
    model = ensemble.models["component_ard_low_rank_task_gp"]
    assert isinstance(model, ComponentARDLowRankTaskGP)
    initial = model._initial_theta()
    y = tiny_dataset.initial[config.dataset.target_column].to_numpy(dtype=float)
    model.fit(prepared.X_initial, y)

    assert np.linalg.norm(
        model.theta_[model._amplitude_start :] - initial[model._amplitude_start :]
    ) > 1e-5
    assert set(model.component_amplitudes_) == {
        "bias",
        "role:A",
        "role:B",
        "pair:A|B",
    }
    assert all(value > 0.0 for value in model.component_amplitudes_.values())
    assert np.isclose(sum(model.component_shares_.values()), 1.0)
    kernel = model._kernel(prepared.X_initial, prepared.X_initial, model.theta_)
    np.testing.assert_allclose(kernel, kernel.T, atol=1e-12)
    assert np.min(np.linalg.eigvalsh(kernel)) >= -1e-9

    posterior = model.predict(prepared.X_candidates)
    covariance = model.posterior_covariance(prepared.X_candidates)
    np.testing.assert_allclose(covariance, covariance.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(covariance), posterior.std**2, rtol=1e-6)
    assert np.min(np.linalg.eigvalsh(covariance)) >= -1e-8


def test_component_ard_models_are_factory_opt_in(tiny_dataset, tiny_config) -> None:
    prepared = prepare_problem(tiny_dataset, tiny_config)
    default = build_ensemble(tiny_config, prepared, seed=100)
    assert "component_ard_gp" not in default.models
    assert "component_ard_low_rank_task_gp" not in default.models
    assert "component_ard_low_rank_task_noise_gp" not in default.models

    config = _ard_config(tiny_config, "component_ard_gp")
    prepared = prepare_problem(tiny_dataset, config)
    ensemble = build_ensemble(config, prepared, seed=100)
    assert isinstance(ensemble.models["component_ard_gp"], ComponentARDStructuredGP)

    task_noise_config = _ard_config(
        tiny_config, "component_ard_low_rank_task_noise_gp"
    )
    prepared = prepare_problem(tiny_dataset, task_noise_config)
    ensemble = build_ensemble(task_noise_config, prepared, seed=100)
    assert isinstance(
        ensemble.models["component_ard_low_rank_task_noise_gp"],
        TaskNoiseComponentARDLowRankTaskGP,
    )


def test_task_noise_identifiability_and_row_alignment(
    tiny_dataset, tiny_config
) -> None:
    config = _ard_config(tiny_config, "component_ard_low_rank_task_noise_gp")
    prepared = prepare_problem(tiny_dataset, config)
    model = build_ensemble(config, prepared, seed=100).models[
        "component_ard_low_rank_task_noise_gp"
    ]
    initial = model._initial_theta()
    bounds = model._bounds()
    assert bounds[model._noise_index] == tuple(
        np.log(value) for value in model.global_noise_bounds
    )
    noises = model.task_noise_vector(initial)
    global_noise = model._unpack(initial)[2]
    assert np.isclose(np.exp(np.mean(np.log(noises))), global_noise, atol=1e-12)
    assert np.isclose(model._task_noise_log_deviations(initial).sum(), 0.0, atol=1e-12)

    contrast_theta = initial.copy()
    contrast_theta[model._task_noise_start] = 0.4
    variances = model._training_noise_variance(
        contrast_theta, prepared.X_initial
    )
    task = prepared.encoder.schema.task_role
    identity = prepared.X_initial[:, prepared.encoder.schema.identity_slice(task)]
    indices = np.argmax(identity, axis=1)
    expected = model.task_noise_vector(contrast_theta)[indices] ** 2
    np.testing.assert_allclose(variances, expected, atol=1e-14)
    assert len(np.unique(np.round(variances, 12))) == 2


def test_task_noise_learns_separation_without_hitting_contrast_bounds(
    tiny_dataset, tiny_config
) -> None:
    config = _ard_config(tiny_config, "component_ard_low_rank_task_noise_gp")
    prepared = prepare_problem(tiny_dataset, config)
    model = build_ensemble(config, prepared, seed=100).models[
        "component_ard_low_rank_task_noise_gp"
    ]
    low_noise = np.repeat(prepared.X_initial[[0]], 8, axis=0)
    high_noise = np.repeat(prepared.X_initial[[1]], 8, axis=0)
    X = np.vstack([low_noise, high_noise])
    y = np.asarray(
        [49.0, 51.0, 48.5, 51.5, 49.2, 50.8, 48.8, 51.2]
        + [46.0, 54.0, 47.0, 53.0, 45.0, 55.0, 48.0, 52.0],
        dtype=float,
    )
    model.fit(X, y)

    learned = np.asarray(list(model.task_noises_.values()), dtype=float)
    assert learned.max() > learned.min() * 1.05
    assert model.task_noise_contrasts_ is not None
    assert np.linalg.norm(model.task_noise_contrasts_) > 1e-4
    assert np.max(np.abs(model.task_noise_contrasts_)) < (
        model.task_noise_contrast_bound - 1e-3
    )
    assert np.isclose(
        np.exp(np.mean(np.log(learned))),
        model._unpack(model.theta_)[2],
        rtol=1e-10,
    )
