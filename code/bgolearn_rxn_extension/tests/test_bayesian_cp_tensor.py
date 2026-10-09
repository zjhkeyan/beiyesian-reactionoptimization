from __future__ import annotations

import itertools
from dataclasses import replace

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from bgolearn_rxn.config import InteractionModelConfig
from bgolearn_rxn.datasets import DatasetSpec, ReactionDataset
from bgolearn_rxn.factory import build_ensemble, prepare_problem
from bgolearn_rxn.schemas import FeatureSchema, SliceSpec
from bgolearn_rxn.surrogates.bayesian_cp_tensor import BayesianCPTensorSurrogate


MODEL = "bayesian_cp_tensor"
ROLES = BayesianCPTensorSurrogate.roles


def _tensor_data() -> tuple[FeatureSchema, np.ndarray, np.ndarray]:
    identity = {
        role: SliceSpec(2 * index, 2 * index + 2)
        for index, role in enumerate(ROLES)
    }
    schema = FeatureSchema(
        dataset_name="SyntheticTensor",
        roles=ROLES,
        condition_roles=ROLES,
        task_role=None,
        role_slices=identity,
        identity_slices=identity,
        feature_names=tuple(
            f"{role}={level}" for role in ROLES for level in ("0", "1")
        ),
    )
    combinations = np.asarray(list(itertools.product((0, 1), repeat=5)), dtype=int)
    X = np.zeros((len(combinations), 2 * len(ROLES)), dtype=float)
    for row_index, row in enumerate(combinations):
        for role_index, level in enumerate(row):
            X[row_index, 2 * role_index + level] = 1.0
    signs = np.where(combinations == 0, -1.0, 1.0)
    y = 50.0 + 25.0 * np.prod(signs, axis=1)
    return schema, X, y


def test_high_order_tensor_recovery_beats_additive_control() -> None:
    schema, X, y = _tensor_data()
    model = BayesianCPTensorSurrogate(
        schema, InteractionModelConfig(target_transform="standard"), seed=7
    ).fit(X, y)
    posterior = model.predict(X)
    additive = LinearRegression().fit(X, y).predict(X)
    cp_rmse = float(np.sqrt(np.mean((posterior.mean - y) ** 2)))
    additive_rmse = float(np.sqrt(np.mean((additive - y) ** 2)))

    assert model.rank == 3
    assert cp_rmse < 0.10 * additive_rmse
    assert np.corrcoef(posterior.mean, y)[0, 1] > 0.98
    assert np.all(posterior.std > 0.0)


def test_unseen_category_is_finite_and_more_uncertain() -> None:
    schema, X, y = _tensor_data()
    model = BayesianCPTensorSurrogate(schema, InteractionModelConfig(), seed=11).fit(
        X, y
    )
    known = model.predict(X[:4])
    unknown_X = X[:4].copy()
    unknown_X[:, schema.identity_slice("Electrophile")] = 0.0
    unknown = model.predict(unknown_X)

    assert np.isfinite(unknown.mean).all()
    assert np.isfinite(unknown.std).all()
    assert unknown.std.mean() > known.std.mean()


def test_covariance_interface_and_seeded_determinism() -> None:
    schema, X, y = _tensor_data()
    left = BayesianCPTensorSurrogate(schema, InteractionModelConfig(), seed=19).fit(
        X, y
    )
    right = BayesianCPTensorSurrogate(schema, InteractionModelConfig(), seed=19).fit(
        X, y
    )
    left_posterior = left.predict(X[:8])
    right_posterior = right.predict(X[:8])
    covariance = left.posterior_covariance(X[:8])

    np.testing.assert_allclose(left_posterior.mean, right_posterior.mean)
    np.testing.assert_allclose(left_posterior.std, right_posterior.std)
    np.testing.assert_allclose(covariance, covariance.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(covariance), left_posterior.std**2, rtol=1e-10)
    assert np.min(np.linalg.eigvalsh(covariance)) >= -1e-8
    assert left_posterior.disagreement is not None
    assert np.any(left_posterior.disagreement > 0.0)


def test_factory_model_is_opt_in(tiny_config) -> None:
    rows = list(itertools.product(("0", "1"), repeat=5))
    columns = list(ROLES)
    initial = pd.DataFrame(rows[:24], columns=columns)
    initial["Yield"] = np.linspace(10.0, 90.0, len(initial))
    candidates = pd.DataFrame(rows[24:], columns=columns)
    candidates.insert(0, "query_index", np.arange(len(candidates)))
    dataset_config = replace(
        tiny_config.dataset,
        name="TinyCP",
        feature_columns=ROLES,
        task_column=None,
        target_task_value=None,
        condition_roles=ROLES,
        derived_roles={},
    )
    dataset = ReactionDataset(
        spec=DatasetSpec.from_config(dataset_config),
        initial=initial,
        candidates=candidates,
    )
    default_config = replace(
        tiny_config,
        dataset=dataset_config,
        features=replace(tiny_config.features, use_name_rules=False, use_rdkit=False),
        gp=replace(tiny_config.gp, interaction_pairs=()),
        interaction_model=replace(tiny_config.interaction_model, interaction_pairs=()),
    )
    prepared = prepare_problem(dataset, default_config)
    assert MODEL not in build_ensemble(default_config, prepared, seed=5).models

    config = replace(
        default_config,
        ensemble=replace(
            default_config.ensemble,
            enabled_models=(MODEL,),
            initial_weights={MODEL: 1.0},
        ),
        policy=replace(default_config.policy, first_query=MODEL),
    )
    prepared = prepare_problem(dataset, config)
    model = build_ensemble(config, prepared, seed=5).models[MODEL]
    assert isinstance(model, BayesianCPTensorSurrogate)
    model.fit(prepared.X_initial, initial["Yield"].to_numpy(dtype=float))
    posterior = model.predict(prepared.X_candidates)
    assert np.isfinite(posterior.mean).all()
    assert np.all(posterior.std > 0.0)
