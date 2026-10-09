from __future__ import annotations

from dataclasses import replace

import numpy as np

from bgolearn_rxn.factory import build_ensemble, prepare_problem
from bgolearn_rxn.surrogates.low_rank_task_gp import LowRankTaskCovarianceGP


def test_low_rank_task_correlation_is_learned_and_psd(
    tiny_dataset, tiny_config
) -> None:
    config = replace(
        tiny_config,
        gp=replace(
            tiny_config.gp,
            optimize_hyperparameters=True,
            restarts=1,
            maxiter=40,
            refit_interval=1,
        ),
        ensemble=replace(
            tiny_config.ensemble,
            enabled_models=("low_rank_task_gp",),
            initial_weights={"low_rank_task_gp": 1.0},
        ),
        policy=replace(tiny_config.policy, first_query="low_rank_task_gp"),
    )
    prepared = prepare_problem(tiny_dataset, config)
    ensemble = build_ensemble(config, prepared, seed=100)
    model = ensemble.models["low_rank_task_gp"]
    assert isinstance(model, LowRankTaskCovarianceGP)

    initial_theta = model._initial_theta()
    initial_correlation = model.task_correlation(initial_theta)
    y = tiny_dataset.initial[config.dataset.target_column].to_numpy(dtype=float)
    model.fit(prepared.X_initial, y)
    correlation = model.task_correlation_
    assert correlation is not None

    np.testing.assert_allclose(correlation, correlation.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(correlation), 1.0, atol=1e-12)
    assert np.min(np.linalg.eigvalsh(correlation)) >= -1e-10
    assert np.linalg.norm(
        model.theta_[model._condition_parameter_count :]
        - initial_theta[model._condition_parameter_count :]
    ) > 1e-5
    assert np.linalg.norm(correlation - initial_correlation) > 1e-6

    kernel = model._kernel(prepared.X_initial, prepared.X_initial, model.theta_)
    np.testing.assert_allclose(kernel, kernel.T, atol=1e-12)
    assert np.min(np.linalg.eigvalsh(kernel)) >= -1e-9


def test_low_rank_task_gp_is_factory_opt_in(tiny_dataset, tiny_config) -> None:
    default_prepared = prepare_problem(tiny_dataset, tiny_config)
    default_ensemble = build_ensemble(tiny_config, default_prepared, seed=100)
    assert "low_rank_task_gp" not in default_ensemble.models

    config = replace(
        tiny_config,
        ensemble=replace(
            tiny_config.ensemble,
            enabled_models=("low_rank_task_gp",),
            initial_weights={"low_rank_task_gp": 1.0},
        ),
    )
    prepared = prepare_problem(tiny_dataset, config)
    ensemble = build_ensemble(config, prepared, seed=100)
    assert tuple(ensemble.models) == ("low_rank_task_gp",)
