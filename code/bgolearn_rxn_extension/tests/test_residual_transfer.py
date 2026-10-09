from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bgolearn_rxn.factory import build_ensemble, prepare_problem
from bgolearn_rxn.surrogates.residual_transfer import ResidualTransferGP


def _residual_config(config):
    return replace(
        config,
        ensemble=replace(
            config.ensemble,
            enabled_models=("residual_transfer",),
            initial_weights={"residual_transfer": 1.0},
        ),
    )


def test_residual_transfer_adds_source_and_target_posteriors(
    tiny_dataset, tiny_config
) -> None:
    config = _residual_config(tiny_config)
    prepared = prepare_problem(tiny_dataset, config)
    ensemble = build_ensemble(config, prepared, seed=100)
    model = ensemble.models["residual_transfer"]
    assert isinstance(model, ResidualTransferGP)

    y = tiny_dataset.initial[config.dataset.target_column].to_numpy(dtype=float)
    model.fit(prepared.X_initial, y)
    posterior = model.predict(prepared.X_candidates)
    source = model.source_gp.predict(prepared.X_candidates)
    residual = model.residual_gp.predict(prepared.X_candidates)

    assert model.n_source_ == 2
    assert model.n_target_ == 1
    np.testing.assert_allclose(posterior.mean, source.mean + residual.mean)
    np.testing.assert_allclose(
        posterior.std**2,
        source.std**2 + residual.std**2,
    )
    assert np.isfinite(posterior.mean).all()
    assert np.all(posterior.std > 0.0)


def test_residual_transfer_rejects_mixed_candidate_tasks(
    tiny_dataset, tiny_config
) -> None:
    config = _residual_config(tiny_config)
    prepared = prepare_problem(tiny_dataset, config)
    schema = prepared.encoder.schema
    assert schema is not None and schema.task_role is not None
    mixed_candidates = prepared.X_candidates.copy()
    mixed_candidates[1, schema.identity_slice(schema.task_role)] = prepared.X_initial[
        1, schema.identity_slice(schema.task_role)
    ]
    mixed = replace(prepared, X_candidates=mixed_candidates)

    with pytest.raises(ValueError, match="one target task identity"):
        build_ensemble(config, mixed, seed=100)


def test_safe_transfer_mixture_preserves_configured_initial_weights(
    tiny_dataset, tiny_config
) -> None:
    config = replace(
        tiny_config,
        ensemble=replace(
            tiny_config.ensemble,
            enabled_models=("gp", "residual_transfer"),
            initial_weights={"gp": 0.8, "residual_transfer": 0.2},
            minimum_weight=0.05,
        ),
        policy=replace(tiny_config.policy, first_query="gp"),
    )
    prepared = prepare_problem(tiny_dataset, config)
    ensemble = build_ensemble(config, prepared, seed=100)

    assert tuple(ensemble.models) == ("gp", "residual_transfer")
    assert ensemble.weights == {"gp": 0.8, "residual_transfer": 0.2}
    assert config.policy.first_query == "gp"
