from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from bgolearn_rxn.datasets import DatasetSpec, ReactionDataset
from bgolearn_rxn.factory import build_ensemble, prepare_problem
from bgolearn_rxn.surrogates.pair_context_gp import PairContextCoregionalizedGP


def _problem(tiny_config):
    dataset_config = replace(
        tiny_config.dataset,
        name="TinySuzuki",
        feature_columns=("Electrophile", "Nucleophile", "Ligand", "Base", "Solvent"),
        task_column=None,
        target_task_value=None,
        condition_roles=(
            "Electrophile",
            "Nucleophile",
            "SubstratePair",
            "Ligand",
            "Base",
            "Solvent",
        ),
        derived_roles={"SubstratePair": ("Electrophile", "Nucleophile")},
    )
    model_name = "pair_context_coregionalized_gp"
    config = replace(
        tiny_config,
        dataset=dataset_config,
        features=replace(
            tiny_config.features, use_name_rules=False, use_rdkit=False
        ),
        gp=replace(
            tiny_config.gp,
            interaction_pairs=(
                ("SubstratePair", "Ligand"),
                ("Electrophile", "Ligand"),
                ("Nucleophile", "Base"),
                ("Base", "Solvent"),
            ),
            role_weights={
                "Electrophile": 0.75,
                "Nucleophile": 0.75,
                "SubstratePair": 1.35,
                "Ligand": 1.0,
                "Base": 1.0,
                "Solvent": 0.75,
            },
            optimize_hyperparameters=False,
            restarts=1,
            maxiter=5,
        ),
        interaction_model=replace(
            tiny_config.interaction_model, interaction_pairs=()
        ),
        ensemble=replace(
            tiny_config.ensemble,
            enabled_models=(model_name,),
            initial_weights={model_name: 1.0},
        ),
        policy=replace(tiny_config.policy, first_query=model_name),
    )
    initial = pd.DataFrame(
        [
            ("e1", "n1", "l1", "b1", "s1", 10.0),
            ("e1", "n2", "l2", "b1", "s1", 35.0),
            ("e2", "n1", "l1", "b2", "s2", 55.0),
            ("e2", "n2", "l2", "b2", "s2", 80.0),
        ],
        columns=(*dataset_config.feature_columns, "Yield"),
    )
    candidates = pd.DataFrame(
        [
            (0, "e1", "n1", "l2", "b2", "s1"),
            (1, "e1", "n2", "l1", "b2", "s2"),
            (2, "e2", "n1", "l2", "b1", "s2"),
            (3, "e2", "n2", "l1", "b1", "s1"),
        ],
        columns=("query_index", *dataset_config.feature_columns),
    )
    dataset = ReactionDataset(
        spec=DatasetSpec.from_config(dataset_config),
        initial=initial,
        candidates=candidates,
    )
    return config, dataset


def test_pair_context_model_is_factory_opt_in_and_psd(tiny_config) -> None:
    config, dataset = _problem(tiny_config)
    prepared = prepare_problem(dataset, config)
    model = build_ensemble(config, prepared, seed=7).models[
        "pair_context_coregionalized_gp"
    ]
    assert isinstance(model, PairContextCoregionalizedGP)
    assert "SubstratePair" not in model.roles
    assert all("SubstratePair" not in pair for pair in model.non_pair_interactions)

    y = dataset.initial["Yield"].to_numpy(dtype=float)
    model.fit(prepared.X_initial, y)
    kernel = model._kernel(prepared.X_initial, prepared.X_initial, model.theta_)
    np.testing.assert_allclose(kernel, kernel.T, atol=1e-12)
    assert np.min(np.linalg.eigvalsh(kernel)) >= -1e-9
    posterior = model.predict(prepared.X_candidates)
    assert np.isfinite(posterior.mean).all()
    assert np.isfinite(posterior.std).all()
    assert (posterior.std > 0.0).all()
    assert "pair_context:constituent_product" in model.component_amplitudes_
    assert "pair_context:exact_identity" in model.component_amplitudes_


def test_pair_residual_is_identifiable_and_constituents_transfer(tiny_config) -> None:
    config, dataset = _problem(tiny_config)
    prepared = prepare_problem(dataset, config)
    model = PairContextCoregionalizedGP(
        prepared.encoder.schema, config.gp, seed=11
    )
    theta = model._initial_theta()
    theta[model._pair_residual_start] = 0.4
    log_deviations = model._pair_log_deviations(theta)
    assert np.isclose(log_deviations.sum(), 0.0, atol=1e-12)
    diagonal = model.pair_diagonal(theta)
    shared = np.exp(theta[model._pair_diagonal_amplitude_index])
    assert np.isclose(np.exp(np.mean(np.log(diagonal))), shared, rtol=1e-12)

    X = np.vstack([prepared.X_initial, prepared.X_candidates])
    pair_indices = model._pair_indices(X)
    electrophile = X[:, prepared.encoder.schema.identity_slice("Electrophile")]
    selected = None
    for left in range(len(X)):
        for right in range(left + 1, len(X)):
            if pair_indices[left] != pair_indices[right] and np.array_equal(
                electrophile[left], electrophile[right]
            ):
                selected = (left, right)
                break
        if selected is not None:
            break
    assert selected is not None
    cross = model._kernel(X[[selected[0]]], X[[selected[1]]], theta)[0, 0]
    assert cross > 0.0
