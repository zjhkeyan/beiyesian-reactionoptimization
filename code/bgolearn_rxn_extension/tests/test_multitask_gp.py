from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bgolearn_rxn.config import GPConfig
from bgolearn_rxn.factory import prepare_problem
from bgolearn_rxn.surrogates.anova_gp import StructuredANOVAGP


def test_task_descriptors_attenuate_only_cross_task_covariance(
    tiny_dataset, tiny_config
) -> None:
    prepared = prepare_problem(tiny_dataset, tiny_config)
    schema = prepared.encoder.schema
    assert schema is not None
    baseline = StructuredANOVAGP(
        schema,
        GPConfig(optimize_hyperparameters=False, task_descriptor_scale=0.0),
    )
    descriptor = StructuredANOVAGP(
        schema,
        GPConfig(optimize_hyperparameters=False, task_descriptor_scale=4.0),
    )
    X = prepared.X_initial
    base_kernel = baseline._kernel(X, X, baseline._initial_theta())
    descriptor_kernel = descriptor._kernel(X, X, descriptor._initial_theta())
    same_task = baseline._same_task(X, X).astype(bool)
    assert np.allclose(descriptor_kernel[same_task], base_kernel[same_task])
    assert np.all(descriptor_kernel[~same_task] <= base_kernel[~same_task] + 1e-12)
    assert np.any(descriptor_kernel[~same_task] < base_kernel[~same_task] - 1e-8)


def test_negative_task_descriptor_scale_is_rejected(tiny_dataset, tiny_config) -> None:
    prepared = prepare_problem(tiny_dataset, tiny_config)
    schema = prepared.encoder.schema
    assert schema is not None
    model = StructuredANOVAGP(
        schema,
        GPConfig(optimize_hyperparameters=False, task_descriptor_scale=-1.0),
    )

    with pytest.raises(ValueError, match="must be non-negative"):
        model._task_descriptor_similarity(prepared.X_initial, prepared.X_initial)


def test_task_mean_is_shrunk_and_removed_from_gp_residuals(
    tiny_dataset, tiny_config
) -> None:
    prepared = prepare_problem(tiny_dataset, tiny_config)
    schema = prepared.encoder.schema
    assert schema is not None
    model = StructuredANOVAGP(
        schema,
        GPConfig(
            optimize_hyperparameters=False,
            task_mean_prior_count=1.0,
        ),
    )
    X = prepared.X_initial
    transformed = np.array([-1.0, 1.0, 3.0])
    offsets = model._fit_task_offsets(X, transformed)
    keys = model._task_keys(X)
    target_offset = model.task_offsets_[keys[0]]
    auxiliary_offset = model.task_offsets_[keys[1]]
    assert target_offset == -0.5
    assert auxiliary_offset == 4.0 / 3.0
    assert np.allclose(offsets, [-0.5, 4.0 / 3.0, 4.0 / 3.0])
