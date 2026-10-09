from __future__ import annotations

from dataclasses import replace

import numpy as np

from bgolearn_rxn.factory import build_ensemble, prepare_problem
from bgolearn_rxn.surrogates.component_ard_gp import ComponentARDStructuredGP
from bgolearn_rxn.surrogates.strong_heredity_component_ard_gp import (
    StrongHeredityComponentARDStructuredGP,
)


MODEL = "strong_heredity_component_ard_gp"


def _config(tiny_config):
    return replace(
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
            enabled_models=(MODEL,),
            initial_weights={MODEL: 1.0},
        ),
        policy=replace(tiny_config.policy, first_query=MODEL),
    )


def test_initialization_reproduces_configured_components(
    tiny_dataset, tiny_config
) -> None:
    config = _config(tiny_config)
    prepared = prepare_problem(tiny_dataset, config)
    model = build_ensemble(config, prepared, seed=7).models[MODEL]
    assert isinstance(model, StrongHeredityComponentARDStructuredGP)

    roles, pairs = model._amplitudes(model._initial_theta())
    expected_roles = [model._role_weight(role) for role in model.roles]
    expected_pairs = [
        model.config.initial_pair_scale * model._interaction_weight(pair)
        for pair in model.config.interaction_pairs
    ]
    np.testing.assert_allclose(roles, expected_roles, rtol=1e-12)
    np.testing.assert_allclose(pairs, expected_pairs, rtol=1e-12)
    assert np.isclose(model._amplitude_penalty(model._initial_theta()), 0.0)


def test_pair_amplitude_obeys_strong_heredity_identity(
    tiny_dataset, tiny_config
) -> None:
    config = _config(tiny_config)
    prepared = prepare_problem(tiny_dataset, config)
    model = build_ensemble(config, prepared, seed=8).models[MODEL]
    theta = model._initial_theta()
    theta[model._amplitude_start] += np.log(4.0)
    theta[model._amplitude_start + 1] += np.log(9.0)
    theta[model._global_multiplier_index] += np.log(0.5)
    theta[model._local_multiplier_start] += np.log(2.0)

    roles, pairs = model._amplitudes(theta)
    global_multiplier, locals_ = model._multipliers(theta)
    expected = global_multiplier * locals_[0] * np.sqrt(roles[0] * roles[1])
    assert np.isclose(pairs[0], expected, rtol=1e-12)


def test_factory_is_opt_in_and_baseline_class_is_unchanged(
    tiny_dataset, tiny_config
) -> None:
    prepared = prepare_problem(tiny_dataset, tiny_config)
    default = build_ensemble(tiny_config, prepared, seed=9)
    assert MODEL not in default.models
    assert ComponentARDStructuredGP.amplitude_prior_scale == 0.60

    config = _config(tiny_config)
    prepared = prepare_problem(tiny_dataset, config)
    model = build_ensemble(config, prepared, seed=9).models[MODEL]
    assert isinstance(model, StrongHeredityComponentARDStructuredGP)


def test_fit_records_finite_heredity_diagnostics_and_psd_kernel(
    tiny_dataset, tiny_config
) -> None:
    config = _config(tiny_config)
    prepared = prepare_problem(tiny_dataset, config)
    model = build_ensemble(config, prepared, seed=10).models[MODEL]
    y = tiny_dataset.initial[config.dataset.target_column].to_numpy(dtype=float)
    model.fit(prepared.X_initial, y)

    assert model.global_interaction_multiplier_ is not None
    assert model.global_interaction_multiplier_ > 0.0
    assert all(value > 0.0 for value in model.local_interaction_multipliers_.values())
    assert all(value > 0.0 for value in model.parent_geometric_means_.values())
    for pair, local in model.local_interaction_multipliers_.items():
        expected = (
            model.global_interaction_multiplier_
            * local
            * model.parent_geometric_means_[pair]
        )
        assert np.isclose(model.component_amplitudes_[f"pair:{pair}"], expected)
    kernel = model._kernel(prepared.X_initial, prepared.X_initial, model.theta_)
    np.testing.assert_allclose(kernel, kernel.T, atol=1e-12)
    assert np.min(np.linalg.eigvalsh(kernel)) >= -1e-9
