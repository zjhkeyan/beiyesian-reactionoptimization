from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from bgolearn_rxn.datasets import DatasetSpec, load_reaction_dataset
from bgolearn_rxn.factory import build_ensemble, prepare_problem


def test_encoder_is_frozen_when_candidate_pool_shrinks(tiny_dataset, tiny_config) -> None:
    prepared = prepare_problem(tiny_dataset, tiny_config)
    row = tiny_dataset.candidates.iloc[[2]]
    before = prepared.encoder.transform(row)

    # Removing other candidates must not trigger a refit or coordinate drift.
    reduced = tiny_dataset.candidates.iloc[[0, 2, 3]]
    _ = prepared.encoder.transform(reduced)
    after = prepared.encoder.transform(row)
    np.testing.assert_array_equal(before, after)


def test_encoder_handles_unseen_identity_without_shape_change(tiny_dataset, tiny_config) -> None:
    prepared = prepare_problem(tiny_dataset, tiny_config)
    unknown = pd.DataFrame([{"Product": "target", "A": "never-seen", "B": "b1"}])
    encoded = prepared.encoder.transform(unknown)
    assert encoded.shape == (1, prepared.encoder.schema.n_features)
    assert np.isfinite(encoded).all()


def test_fixed_reaction_context_is_encoded_without_changing_candidate_columns(
    tiny_files, tiny_config
) -> None:
    train_path, candidate_path, _ = tiny_files
    config = replace(
        tiny_config,
        dataset=replace(
            tiny_config.dataset,
            fixed_roles={"ProductContext": "official-coupled-product"},
        ),
    )
    dataset = load_reaction_dataset(
        train_path,
        candidate_path,
        DatasetSpec.from_config(config.dataset),
    )
    prepared = prepare_problem(dataset, config)
    schema = prepared.encoder.schema
    assert schema is not None
    assert schema.fixed_roles == {"ProductContext": "official-coupled-product"}
    assert "ProductContext" in schema.roles
    context = prepared.X_candidates[:, schema.identity_slice("ProductContext")]
    np.testing.assert_array_equal(context, np.ones((len(dataset.candidates), 1)))
    assert "ProductContext" not in dataset.candidates.columns


def test_ensemble_can_enable_a_surrogate_subset(tiny_dataset, tiny_config) -> None:
    config = replace(
        tiny_config,
        ensemble=replace(tiny_config.ensemble, enabled_models=("gp", "interaction")),
    )
    prepared = prepare_problem(tiny_dataset, config)
    ensemble = build_ensemble(config, prepared, seed=100)
    assert tuple(ensemble.models) == ("gp", "interaction")
