from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np

import bgolearn_rxn.surrogates.posterior_mode_component_ard_gp as mode_module
from bgolearn_rxn.factory import build_ensemble, prepare_problem
from bgolearn_rxn.surrogates.posterior_mode_component_ard_gp import (
    PosteriorModeComponentARDStructuredGP,
)


MODEL = "posterior_mode_component_ard_gp"


def _config(tiny_config, *, optimize: bool, restarts: int = 1):
    return replace(
        tiny_config,
        gp=replace(
            tiny_config.gp,
            optimize_hyperparameters=optimize,
            restarts=restarts,
            maxiter=40,
            refit_interval=1,
        ),
        ensemble=replace(
            tiny_config.ensemble,
            enabled_models=(MODEL,),
            initial_weights={MODEL: 1.0},
        ),
        policy=replace(tiny_config.policy, first_query=MODEL),
    )


def test_factory_mode_is_opt_in(tiny_dataset, tiny_config) -> None:
    prepared = prepare_problem(tiny_dataset, tiny_config)
    default = build_ensemble(tiny_config, prepared, seed=100)
    assert MODEL not in default.models

    config = _config(tiny_config, optimize=False)
    prepared = prepare_problem(tiny_dataset, config)
    model = build_ensemble(config, prepared, seed=100).models[MODEL]
    assert isinstance(model, PosteriorModeComponentARDStructuredGP)


def test_solution_filter_applies_window_deduplication_cap_and_weights(
    tiny_dataset, tiny_config
) -> None:
    config = _config(tiny_config, optimize=False)
    prepared = prepare_problem(tiny_dataset, config)
    model = build_ensemble(config, prepared, seed=100).models[MODEL]
    theta = model._initial_theta()
    solutions = [
        (theta, 10.0, True, "best"),
        (theta + 1e-8, 10.1, True, "duplicate"),
        (theta + 0.1, 10.5, True, "second"),
        (theta - 0.1, 11.0, True, "third"),
        (theta + 0.2, 11.5, True, "capped"),
        (theta - 0.2, 12.1, True, "outside window"),
    ]
    retained, weights = model._select_solutions(solutions)
    assert [item[1] for item in retained] == [10.0, 10.5, 11.0]
    np.testing.assert_allclose(weights.sum(), 1.0)
    assert np.all(np.diff(weights) < 0.0)


def test_total_variance_contains_between_mode_disagreement() -> None:
    means = np.asarray([[0.0, 2.0], [2.0, 4.0]])
    variances = np.asarray([[1.0, 1.0], [1.0, 1.0]])
    weights = np.asarray([0.25, 0.75])
    mean, variance = PosteriorModeComponentARDStructuredGP._mixture_variance(
        means, variances, weights
    )
    np.testing.assert_allclose(mean, [1.5, 3.5])
    np.testing.assert_allclose(variance, [1.75, 1.75])


def test_single_mode_fit_matches_component_contract(tiny_dataset, tiny_config) -> None:
    config = _config(tiny_config, optimize=False)
    prepared = prepare_problem(tiny_dataset, config)
    model = build_ensemble(config, prepared, seed=100).models[MODEL]
    y = tiny_dataset.initial[config.dataset.target_column].to_numpy(dtype=float)
    model.fit(prepared.X_initial, y)

    assert len(model.modes_) == 1
    np.testing.assert_allclose(model.mode_weights_, [1.0])
    posterior = model.predict(prepared.X_candidates)
    covariance = model.posterior_covariance(prepared.X_candidates)
    np.testing.assert_allclose(np.diag(covariance), posterior.std**2, rtol=1e-6)
    assert np.min(np.linalg.eigvalsh(covariance)) >= -1e-8
    assert model.mode_history_[-1]["mode_count"] == 1
    assert model.mode_history_[-1]["optimizer_solution_count"] == 1
    assert model.mode_history_[-1]["optimizer_distances_to_best"] == [0.0]


def test_worse_restarts_preserve_incumbent(monkeypatch, tiny_dataset, tiny_config) -> None:
    config = _config(tiny_config, optimize=True, restarts=3)
    prepared = prepare_problem(tiny_dataset, config)
    model = build_ensemble(config, prepared, seed=100).models[MODEL]
    initial = model._initial_theta()

    def worse_result(*args, x0, **kwargs):
        del args, kwargs
        return SimpleNamespace(
            x=np.asarray(x0, dtype=float),
            fun=1e6,
            success=True,
            message="forced worse restart",
        )

    monkeypatch.setattr(mode_module, "minimize", worse_result)
    y = tiny_dataset.initial[config.dataset.target_column].to_numpy(dtype=float)
    model.fit(prepared.X_initial, y)

    np.testing.assert_allclose(model.theta_, initial)
    assert len(model.modes_) == 1
    assert model.mode_history_[-1]["optimizer_solution_count"] == 3
