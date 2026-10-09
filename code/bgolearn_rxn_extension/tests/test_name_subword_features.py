from __future__ import annotations

from dataclasses import replace

import numpy as np

from bgolearn_rxn.datasets import ReactionDataset
from bgolearn_rxn.factory import prepare_problem


def _subword_config(config):
    return replace(
        config,
        features=replace(
            config.features,
            use_name_subwords=True,
            name_subword_scale=0.35,
            name_subword_max_rank=8,
        ),
    )


def test_name_subword_encoder_is_deterministic_and_frozen(
    tiny_dataset, tiny_config
) -> None:
    config = _subword_config(tiny_config)
    first = prepare_problem(tiny_dataset, config)
    second = prepare_problem(tiny_dataset, config)
    np.testing.assert_array_equal(first.reference_X, second.reference_X)
    assert first.encoder.schema == second.encoder.schema

    row = tiny_dataset.candidates.iloc[[2]]
    before = first.encoder.transform(row)
    _ = first.encoder.transform(tiny_dataset.candidates.iloc[[0, 2]])
    after = first.encoder.transform(row)
    np.testing.assert_array_equal(before, after)


def test_name_subword_fit_does_not_use_yield_labels(tiny_dataset, tiny_config) -> None:
    config = _subword_config(tiny_config)
    relabeled_initial = tiny_dataset.initial.copy()
    relabeled_initial["Yield"] = [99.0, 1.0, 47.0]
    relabeled = ReactionDataset(
        spec=tiny_dataset.spec,
        initial=relabeled_initial,
        candidates=tiny_dataset.candidates.copy(),
    )
    original = prepare_problem(tiny_dataset, config)
    changed = prepare_problem(relabeled, config)
    np.testing.assert_array_equal(original.reference_X, changed.reference_X)
    assert original.encoder.schema == changed.encoder.schema


def test_name_subword_schema_preserves_identity_prefixes(
    tiny_dataset, tiny_config
) -> None:
    baseline = prepare_problem(tiny_dataset, tiny_config)
    subword = prepare_problem(tiny_dataset, _subword_config(tiny_config))
    baseline_schema = baseline.encoder.schema
    subword_schema = subword.encoder.schema
    assert baseline_schema is not None and subword_schema is not None
    assert subword_schema.n_features > baseline_schema.n_features

    for role in subword_schema.roles:
        role_slice = subword_schema.role_slices[role]
        identity_slice = subword_schema.identity_slices[role]
        assert identity_slice.start == role_slice.start
        assert identity_slice.width == baseline_schema.identity_slices[role].width
        names = subword_schema.feature_names[role_slice.start : role_slice.stop]
        subword_names = [name for name in names if "::subword::" in name]
        n_unique = len(
            set(
                subword.encoder._prepare(
                    tiny_dataset.initial.loc[:, tiny_dataset.spec.feature_columns]
                )[role]
            ).union(
                subword.encoder._prepare(
                    tiny_dataset.candidates.loc[:, tiny_dataset.spec.feature_columns]
                )[role]
            )
        )
        assert len(subword_names) <= min(8, max(0, n_unique - 1))


def test_name_subwords_are_disabled_by_default(tiny_dataset, tiny_config) -> None:
    prepared = prepare_problem(tiny_dataset, tiny_config)
    schema = prepared.encoder.schema
    assert schema is not None
    assert not tiny_config.features.use_name_subwords
    assert not any("::subword::" in name for name in schema.feature_names)
