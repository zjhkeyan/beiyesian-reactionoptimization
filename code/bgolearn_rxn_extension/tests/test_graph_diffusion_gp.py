from __future__ import annotations

from dataclasses import replace

import numpy as np

from bgolearn_rxn.factory import build_ensemble, prepare_problem
from bgolearn_rxn.datasets import DatasetSpec, ReactionDataset
from bgolearn_rxn.surrogates.component_ard_gp import ComponentARDStructuredGP
from bgolearn_rxn.surrogates.graph_diffusion_gp import GraphDiffusionProductGP


MODEL = "graph_diffusion_product_gp"


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


def test_complete_graph_heat_kernel_has_expected_spectrum() -> None:
    q = 4
    same = np.eye(q)
    kernel = GraphDiffusionProductGP._normalized_complete_graph_heat(
        same, category_count=q, spectral_diffusion=1.5
    )
    np.testing.assert_allclose(np.diag(kernel), 1.0, atol=1e-14)
    assert np.all(kernel[np.triu_indices(q, 1)] > 0.0)
    assert np.min(np.linalg.eigvalsh(kernel)) > 0.0


def test_graph_diffusion_is_factory_opt_in_psd_and_finite(
    tiny_dataset, tiny_config
) -> None:
    assert MODEL not in build_ensemble(
        tiny_config, prepare_problem(tiny_dataset, tiny_config), seed=3
    ).models
    config = _config(tiny_config)
    prepared = prepare_problem(tiny_dataset, config)
    model = build_ensemble(config, prepared, seed=7).models[MODEL]
    assert isinstance(model, GraphDiffusionProductGP)
    assert model.roles == ("A", "B")

    y = tiny_dataset.initial[config.dataset.target_column].to_numpy(dtype=float)
    model.fit(prepared.X_initial, y)
    kernel = model._kernel(prepared.X_initial, prepared.X_initial, model.theta_)
    np.testing.assert_allclose(kernel, kernel.T, atol=1e-12)
    assert np.min(np.linalg.eigvalsh(kernel)) >= -1e-10
    posterior = model.predict(prepared.X_candidates)
    covariance = model.posterior_covariance(prepared.X_candidates)
    assert np.isfinite(posterior.mean).all()
    assert np.isfinite(posterior.std).all()
    assert (posterior.std > 0.0).all()
    assert np.isfinite(covariance).all()
    np.testing.assert_allclose(covariance, covariance.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(covariance), posterior.std**2, rtol=1e-6)
    assert set(model.diffusion_scales_) == {"A", "B"}
    assert all(value > 0.0 for value in model.diffusion_scales_.values())


def test_kernel_is_category_permutation_invariant_and_not_additive(
    tiny_dataset, tiny_config
) -> None:
    config = _config(tiny_config)
    prepared = prepare_problem(tiny_dataset, config)
    model = GraphDiffusionProductGP(prepared.encoder.schema, config.gp, seed=11)
    theta = model._initial_theta()
    X = np.vstack([prepared.X_initial, prepared.X_candidates])
    reference = model._kernel(X, X, theta)

    permuted = X.copy()
    for role in model.roles:
        identity = prepared.encoder.schema.identity_slice(role)
        order = np.arange(identity.stop - identity.start)[::-1]
        permuted[:, identity] = permuted[:, identity][:, order]
    np.testing.assert_allclose(model._kernel(permuted, permuted, theta), reference)

    additive = ComponentARDStructuredGP(
        prepared.encoder.schema, config.gp, seed=11
    )
    additive_kernel = additive._kernel(X, X, additive._initial_theta())
    assert not np.allclose(reference, additive_kernel, rtol=1e-8, atol=1e-8)


def test_derived_roles_are_excluded_from_the_product(tiny_dataset, tiny_config) -> None:
    dataset_config = replace(
        tiny_config.dataset,
        condition_roles=("A", "AB", "B"),
        derived_roles={"AB": ("A", "B")},
    )
    config = replace(_config(tiny_config), dataset=dataset_config)
    dataset = ReactionDataset(
        spec=DatasetSpec.from_config(dataset_config),
        initial=tiny_dataset.initial.copy(),
        candidates=tiny_dataset.candidates.copy(),
    )
    prepared = prepare_problem(dataset, config)
    model = build_ensemble(config, prepared, seed=13).models[MODEL]
    assert model.roles == ("A", "B")
    assert "AB" not in model.category_counts
