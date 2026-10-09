from __future__ import annotations

from dataclasses import replace

import numpy as np

from bgolearn_rxn.factory import build_ensemble, prepare_problem
from bgolearn_rxn.schemas import Posterior
from bgolearn_rxn.surrogates.component_ard_gp import ComponentARDStructuredGP
from bgolearn_rxn.surrogates.interaction import (
    BayesianLowRankInteractionSurrogate,
)
from bgolearn_rxn.surrogates.query_one_rank_consensus_ard import (
    COMPONENT_ARD,
    INTERACTION,
    TREES,
    QueryOneRankConsensusARD,
    median_rank_consensus,
)
from bgolearn_rxn.surrogates.trees import BootstrapExtraTreesSurrogate


def _real_wrapper(prepared, config, seed: int) -> QueryOneRankConsensusARD:
    schema = prepared.encoder.schema
    assert schema is not None
    return QueryOneRankConsensusARD(
        {
            COMPONENT_ARD: ComponentARDStructuredGP(
                schema=schema, config=config.gp, seed=seed + 17
            ),
            INTERACTION: BayesianLowRankInteractionSurrogate(
                schema=schema,
                config=config.interaction_model,
                reference_X=prepared.reference_X,
                seed=seed + 23,
            ),
            TREES: BootstrapExtraTreesSurrogate(
                config=config.trees, seed=seed + 37
            ),
        }
    )


class _StaticExpert:
    def __init__(self, name: str, column: int, direction: float = 1.0) -> None:
        self.name = name
        self.column = int(column)
        self.direction = float(direction)
        self.fit_calls = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_StaticExpert":
        assert len(X) == len(y)
        self.fit_calls += 1
        return self

    def predict(self, X: np.ndarray) -> Posterior:
        values = self.direction * np.asarray(X, dtype=float)[:, self.column]
        posterior = Posterior(
            mean=values,
            std=np.full(len(values), 0.25, dtype=float),
        )
        posterior.validate(len(values))
        return posterior

    def posterior_covariance(self, X: np.ndarray) -> np.ndarray:
        return np.eye(len(X), dtype=float) * 0.25**2


def _static_wrapper() -> tuple[QueryOneRankConsensusARD, dict[str, _StaticExpert]]:
    experts = {
        COMPONENT_ARD: _StaticExpert(COMPONENT_ARD, 0),
        INTERACTION: _StaticExpert(INTERACTION, 1),
        TREES: _StaticExpert(TREES, 2, direction=-1.0),
    }
    return QueryOneRankConsensusARD(experts), experts


def test_factory_builds_exact_opt_in_expert_graph(tiny_dataset, tiny_config) -> None:
    model_name = QueryOneRankConsensusARD.name
    config = replace(
        tiny_config,
        ensemble=replace(
            tiny_config.ensemble,
            enabled_models=(model_name,),
            initial_weights={model_name: 1.0},
        ),
        policy=replace(tiny_config.policy, first_query=model_name),
    )
    prepared = prepare_problem(tiny_dataset, config)
    ensemble = build_ensemble(config, prepared, seed=100)
    model = ensemble.models[model_name]

    assert tuple(ensemble.models) == (model_name,)
    assert isinstance(model, QueryOneRankConsensusARD)
    assert tuple(model.experts) == (COMPONENT_ARD, INTERACTION, TREES)
    assert isinstance(model.experts[COMPONENT_ARD], ComponentARDStructuredGP)
    assert isinstance(
        model.experts[INTERACTION], BayesianLowRankInteractionSurrogate
    )
    assert isinstance(model.experts[TREES], BootstrapExtraTreesSurrogate)


def test_median_consensus_is_invariant_to_each_expert_scale() -> None:
    raw = {
        COMPONENT_ARD: np.array([8.0, 2.0, 5.0, 5.0, -1.0]),
        INTERACTION: np.array([0.4, 0.9, 0.2, 0.7, 0.1]),
        TREES: np.array([12.0, 10.0, 11.0, 9.0, 8.0]),
    }
    transformed = {
        COMPONENT_ARD: 71.0 + 13.0 * raw[COMPONENT_ARD],
        INTERACTION: np.exp(raw[INTERACTION]),
        TREES: np.power(2.0, raw[TREES]),
    }
    first, first_ranks, first_disagreement = median_rank_consensus(raw)
    second, second_ranks, second_disagreement = median_rank_consensus(transformed)

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first_disagreement, second_disagreement)
    for name in raw:
        np.testing.assert_array_equal(first_ranks[name], second_ranks[name])


def test_query_one_ordering_is_deterministic_and_audited(
    tiny_dataset, tiny_config
) -> None:
    prepared = prepare_problem(tiny_dataset, tiny_config)
    y = tiny_dataset.initial[tiny_config.dataset.target_column].to_numpy(dtype=float)
    wrappers = [_real_wrapper(prepared, tiny_config, seed=211) for _ in range(2)]
    posteriors = []
    for wrapper in wrappers:
        wrapper.fit(prepared.X_initial, y)
        posterior = wrapper.predict(prepared.X_candidates)
        selected = int(np.lexsort((np.arange(len(posterior.mean)), -posterior.mean))[0])
        wrapper.record_query_one_selection(selected)
        posteriors.append(posterior)

    np.testing.assert_array_equal(posteriors[0].mean, posteriors[1].mean)
    np.testing.assert_array_equal(posteriors[0].std, posteriors[1].std)
    first_order = np.lexsort(
        (np.arange(len(posteriors[0].mean)), -posteriors[0].mean)
    )
    second_order = np.lexsort(
        (np.arange(len(posteriors[1].mean)), -posteriors[1].mean)
    )
    np.testing.assert_array_equal(first_order, second_order)
    audit = wrappers[0].audit_state()
    assert audit["aggregation"] == "median_normalized_rank"
    assert audit["active_stage"] == "median_rank_consensus_query_1"
    assert audit["selected_consensus_rank"] is not None
    assert set(audit["selected_expert_ranks"]) == {
        COMPONENT_ARD,
        INTERACTION,
        TREES,
    }


def test_after_feedback_returns_exact_component_ard_posterior(
    tiny_dataset, tiny_config
) -> None:
    prepared = prepare_problem(tiny_dataset, tiny_config)
    schema = prepared.encoder.schema
    assert schema is not None
    y_initial = tiny_dataset.initial[
        tiny_config.dataset.target_column
    ].to_numpy(dtype=float)
    X_feedback = np.vstack([prepared.X_initial, prepared.X_candidates[[0]]])
    y_feedback = np.append(y_initial, 63.0)
    wrapper = _real_wrapper(prepared, tiny_config, seed=307)
    baseline = ComponentARDStructuredGP(
        schema=schema, config=tiny_config.gp, seed=307 + 17
    )

    wrapper.fit(prepared.X_initial, y_initial)
    baseline.fit(prepared.X_initial, y_initial)
    wrapper.fit(X_feedback, y_feedback)
    baseline.fit(X_feedback, y_feedback)
    wrapped = wrapper.predict(prepared.X_candidates)
    expected = baseline.predict(prepared.X_candidates)

    assert wrapper.active_stage_ == "component_ard_after_feedback"
    assert wrapper.stage_history_ == [
        "median_rank_consensus_query_1",
        "component_ard_after_feedback",
    ]
    assert wrapper.feedback_observed_count_ == len(y_feedback)
    np.testing.assert_array_equal(wrapped.mean, expected.mean)
    np.testing.assert_array_equal(wrapped.std, expected.std)
    np.testing.assert_array_equal(wrapper.component_ard.theta_, baseline.theta_)


def test_query_one_uncertainty_is_finite_positive_and_validated() -> None:
    wrapper, _ = _static_wrapper()
    rng = np.random.default_rng(401)
    X_train = rng.normal(size=(29, 3))
    y_train = rng.normal(size=29)
    X_candidates = rng.normal(size=(137, 3))
    wrapper.fit(X_train, y_train)
    posterior = wrapper.predict(X_candidates)

    posterior.validate(len(X_candidates))
    assert np.isfinite(posterior.mean).all()
    assert np.all(np.isfinite(posterior.std) & (posterior.std > 0.0))
    assert posterior.disagreement is not None
    assert np.all(np.isfinite(posterior.disagreement))


def test_5731_candidate_prediction_is_linear_size_and_skips_retired_experts() -> None:
    wrapper, experts = _static_wrapper()
    rng = np.random.default_rng(503)
    X_train = rng.normal(size=(29, 3))
    y_train = rng.normal(size=29)
    X_candidates = rng.normal(size=(5731, 3))
    wrapper.fit(X_train, y_train)
    posterior = wrapper.predict(X_candidates)

    assert posterior.mean.shape == (5731,)
    assert posterior.std.shape == (5731,)
    assert sum(array.nbytes for array in posterior.component_means.values()) < 200_000
    assert all(expert.fit_calls == 1 for expert in experts.values())

    wrapper.fit(
        np.vstack([X_train, X_candidates[[0]]]),
        np.append(y_train, 1.25),
    )
    assert experts[COMPONENT_ARD].fit_calls == 2
    assert experts[INTERACTION].fit_calls == 1
    assert experts[TREES].fit_calls == 1
