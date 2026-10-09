from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from bgolearn_rxn.datasets import DatasetSpec, ReactionDataset
from bgolearn_rxn.factory import build_ensemble, prepare_problem
from bgolearn_rxn.surrogates.source_expert import SourceExpertTransferGP


def _source_expert_problem(tiny_config):
    config = replace(
        tiny_config,
        gp=replace(tiny_config.gp, optimize_hyperparameters=False),
        ensemble=replace(
            tiny_config.ensemble,
            enabled_models=("source_expert",),
            initial_weights={"source_expert": 1.0},
        ),
        policy=replace(tiny_config.policy, first_query="source_expert"),
    )
    initial = pd.DataFrame(
        [
            {"Product": "target", "A": "a1", "B": "b1", "Yield": 10.0},
            {"Product": "target", "A": "a2", "B": "b2", "Yield": 12.0},
            {"Product": "source-close", "A": "a1", "B": "b1", "Yield": 10.0},
            {"Product": "source-close", "A": "a2", "B": "b2", "Yield": 12.0},
            {"Product": "source-far", "A": "a1", "B": "b1", "Yield": 90.0},
            {"Product": "source-far", "A": "a2", "B": "b2", "Yield": 92.0},
        ]
    )
    candidates = pd.DataFrame(
        [
            {"query_index": 0, "Product": "target", "A": "a1", "B": "b2"},
            {"query_index": 1, "Product": "target", "A": "a2", "B": "b1"},
        ]
    )
    dataset = ReactionDataset(
        spec=DatasetSpec.from_config(config.dataset),
        initial=initial,
        candidates=candidates,
    )
    return config, dataset


def test_source_expert_gates_sources_and_forms_proper_mixture(tiny_config) -> None:
    config, dataset = _source_expert_problem(tiny_config)
    prepared = prepare_problem(dataset, config)
    ensemble = build_ensemble(config, prepared, seed=100)
    model = ensemble.models["source_expert"]
    assert isinstance(model, SourceExpertTransferGP)

    y = dataset.initial[config.dataset.target_column].to_numpy(dtype=float)
    model.fit(prepared.X_initial, y)
    task_keys = model._task_keys(prepared.X_initial)
    close_key = task_keys[2]
    far_key = task_keys[4]

    assert tuple(ensemble.models) == ("source_expert",)
    assert len(model.expert_weights_) == 2
    assert abs(sum(model.expert_weights_.values()) - 1.0) < 1e-12
    assert min(model.expert_weights_.values()) >= model.weight_floor
    assert model.expert_weights_[close_key] > model.expert_weights_[far_key]

    posterior = model.predict(prepared.X_candidates)
    component = {
        key: expert.predict(prepared.X_candidates)
        for key, expert in model.expert_models_.items()
    }
    weights = np.asarray([model.expert_weights_[key] for key in component])
    means = np.vstack([component[key].mean for key in component])
    variances = np.vstack([component[key].std**2 for key in component])
    expected_mean = np.sum(weights[:, None] * means, axis=0)
    expected_variance = np.sum(
        weights[:, None] * (variances + (means - expected_mean) ** 2), axis=0
    )
    np.testing.assert_allclose(posterior.mean, expected_mean)
    np.testing.assert_allclose(posterior.std**2, expected_variance)
    np.testing.assert_allclose(
        np.diag(model.posterior_covariance(prepared.X_candidates)),
        posterior.std**2,
    )

    source_fit_calls = {
        key: source._fit_calls for key, source in model.source_models_.items()
    }
    model.fit(prepared.X_initial, y)
    assert {
        key: source._fit_calls for key, source in model.source_models_.items()
    } == source_fit_calls
