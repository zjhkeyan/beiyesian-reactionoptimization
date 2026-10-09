from __future__ import annotations

import numpy as np

from bgolearn_rxn.factory import build_ensemble, prepare_problem
from bgolearn_rxn.schemas import Posterior
from bgolearn_rxn.surrogates.query_one_rank_consensus_ard import (
    COMPONENT_ARD,
    INTERACTION,
    TREES,
    median_rank_consensus,
)
from bgolearn_rxn.surrogates.surprise_gated_rank_consensus_ard import (
    SurpriseGatedRankConsensusARD,
)


class _FakeExpert:
    def __init__(
        self,
        name: str,
        column: int,
        *,
        direction: float = 1.0,
        offset: float = 0.0,
        std: float = 2.0,
    ) -> None:
        self.name = name
        self.column = int(column)
        self.direction = float(direction)
        self.offset = float(offset)
        self.std = float(std)
        self.fit_calls = 0
        self.predict_calls = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_FakeExpert":
        assert np.asarray(X).ndim == 2
        assert len(X) == len(y)
        self.fit_calls += 1
        return self

    def predict(self, X: np.ndarray) -> Posterior:
        self.predict_calls += 1
        values = (
            self.direction * np.asarray(X, dtype=float)[:, self.column]
            + self.offset
        )
        posterior = Posterior(
            mean=values,
            std=np.full(len(values), self.std, dtype=float),
        )
        posterior.validate(len(values))
        return posterior

    def posterior_covariance(self, X: np.ndarray) -> np.ndarray:
        return np.eye(len(X), dtype=float) * self.std**2


def _wrapper() -> tuple[SurpriseGatedRankConsensusARD, dict[str, _FakeExpert]]:
    experts = {
        COMPONENT_ARD: _FakeExpert(COMPONENT_ARD, 0, std=2.0),
        INTERACTION: _FakeExpert(INTERACTION, 1, direction=-1.0, std=3.0),
        TREES: _FakeExpert(TREES, 2, offset=0.5, std=4.0),
    }
    return SurpriseGatedRankConsensusARD(experts), experts


def _data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_initial = np.array(
        [[-10.0, 0.0, 0.0], [-9.0, 1.0, 1.0], [-8.0, 2.0, 2.0]]
    )
    y_initial = np.array([1.0, 2.0, 3.0])
    candidates = np.array(
        [
            [0.0, 3.0, 2.0],
            [1.0, 2.0, 0.0],
            [2.0, 1.0, 3.0],
            [3.0, 0.0, 1.0],
        ]
    )
    return X_initial, y_initial, candidates


def test_query_one_and_nonsurprising_feedback_are_exact_component() -> None:
    wrapper, experts = _wrapper()
    X_initial, y_initial, candidates = _data()
    wrapper.fit(X_initial, y_initial)

    wrapped_initial = wrapper.predict(candidates)
    expected_initial = experts[COMPONENT_ARD].predict(candidates)
    np.testing.assert_array_equal(wrapped_initial.mean, expected_initial.mean)
    np.testing.assert_array_equal(wrapped_initial.std, expected_initial.std)
    assert experts[INTERACTION].fit_calls == 0
    assert experts[TREES].fit_calls == 0

    # Candidate 1 has component prediction 1.0 and std 2.0: z=-0.5.
    X_next = np.vstack([X_initial, candidates[[1]]])
    y_next = np.append(y_initial, 0.0)
    wrapper.fit(X_next, y_next)
    wrapped_next = wrapper.predict(candidates[[0, 2, 3]])
    expected_next = experts[COMPONENT_ARD].predict(candidates[[0, 2, 3]])

    assert wrapper.active_stage_ == "component_ard_nominal"
    assert wrapper.last_standardized_residual_ == -0.5
    assert not wrapper.last_activation_
    np.testing.assert_array_equal(wrapped_next.mean, expected_next.mean)
    np.testing.assert_array_equal(wrapped_next.std, expected_next.std)


def test_severe_negative_surprise_activates_scale_safe_rank_consensus() -> None:
    wrapper, experts = _wrapper()
    X_initial, y_initial, candidates = _data()
    wrapper.fit(X_initial, y_initial)
    component_before = wrapper.predict(candidates)

    # Candidate 2 has component prediction 2.0 and std 2.0: z=-1 exactly.
    wrapper.fit(
        np.vstack([X_initial, candidates[[2]]]),
        np.append(y_initial, 0.0),
    )
    posterior = wrapper.predict(candidates[[0, 1, 3]])
    component = experts[COMPONENT_ARD].predict(candidates[[0, 1, 3]])
    raw = {
        name: experts[name].predict(candidates[[0, 1, 3]]).mean
        for name in (COMPONENT_ARD, INTERACTION, TREES)
    }
    consensus, _, _ = median_rank_consensus(raw)
    expected_mean = np.interp(
        consensus,
        np.linspace(0.0, 1.0, len(consensus)),
        np.sort(component.mean),
    )

    assert component_before.mean[2] == 2.0
    assert wrapper.active_stage_ == "surprise_rank_consensus"
    assert wrapper.last_activation_
    assert wrapper.last_standardized_residual_ == -1.0
    assert experts[INTERACTION].fit_calls == 1
    assert experts[TREES].fit_calls == 1
    np.testing.assert_array_equal(posterior.mean, expected_mean)
    np.testing.assert_array_equal(posterior.std, component.std)
    posterior.validate(3)
    assert np.isfinite(posterior.mean).all()
    assert np.isfinite(posterior.std).all()


def test_controller_returns_to_exact_component_when_surprise_clears() -> None:
    wrapper, experts = _wrapper()
    X_initial, y_initial, candidates = _data()
    wrapper.fit(X_initial, y_initial)
    wrapper.predict(candidates)
    X_active = np.vstack([X_initial, candidates[[2]]])
    y_active = np.append(y_initial, -1.0)
    wrapper.fit(X_active, y_active)
    remaining = candidates[[0, 1, 3]]
    wrapper.predict(remaining)

    # Candidate 0 has mean 0 and std 2: z=+0.25, so consensus retires.
    X_return = np.vstack([X_active, candidates[[0]]])
    y_return = np.append(y_active, 0.5)
    wrapper.fit(X_return, y_return)
    wrapped = wrapper.predict(candidates[[1, 3]])
    expected = experts[COMPONENT_ARD].predict(candidates[[1, 3]])

    assert wrapper.active_stage_ == "component_ard_nominal"
    assert wrapper.activation_history_ == [False, True, False]
    assert wrapper.stage_history_ == [
        "component_ard_query_1",
        "surprise_rank_consensus",
        "component_ard_nominal",
    ]
    assert experts[INTERACTION].fit_calls == 1
    assert experts[TREES].fit_calls == 1
    np.testing.assert_array_equal(wrapped.mean, expected.mean)
    np.testing.assert_array_equal(wrapped.std, expected.std)


def test_feedback_matching_is_deterministic_under_observation_reordering() -> None:
    wrapper, _ = _wrapper()
    X_initial, y_initial, candidates = _data()
    duplicated_pool = np.vstack([candidates[[1]], candidates, candidates[[1]]])
    wrapper.fit(X_initial, y_initial)
    wrapper.predict(duplicated_pool)

    # Insert the feedback row first rather than appending it. Duplicate matches
    # must resolve to the lowest preceding candidate position.
    reordered_X = np.vstack([candidates[[1]], X_initial[[2, 0, 1]]])
    reordered_y = np.array([-2.0, y_initial[2], y_initial[0], y_initial[1]])
    wrapper.fit(reordered_X, reordered_y)
    audit = wrapper.audit_state()

    assert audit["last_matched_candidate_positions"] == [0]
    record = audit["residual_history"][-1]
    assert record["observed_position"] == 0
    assert record["candidate_position"] == 0
    assert record["standardized_residual"] == -1.5
    assert record["activated"] is True


def test_component_covariance_remains_authoritative_when_active() -> None:
    wrapper, experts = _wrapper()
    X_initial, y_initial, candidates = _data()
    wrapper.fit(X_initial, y_initial)
    wrapper.predict(candidates)
    wrapper.fit(
        np.vstack([X_initial, candidates[[3]]]),
        np.append(y_initial, 0.0),
    )
    subset = candidates[[0, 1, 2]]
    covariance = wrapper.posterior_covariance(subset)
    expected = experts[COMPONENT_ARD].posterior_covariance(subset)

    np.testing.assert_array_equal(covariance, expected)
    assert np.isfinite(covariance).all()
    assert wrapper.audit_state()["uncertainty_authority"] == COMPONENT_ARD


def test_5731_candidate_prediction_has_no_quadratic_state() -> None:
    wrapper, _ = _wrapper()
    rng = np.random.default_rng(731)
    X_initial = rng.normal(size=(29, 3)) - 20.0
    y_initial = rng.normal(size=29)
    candidates = rng.normal(size=(5731, 3))
    wrapper.fit(X_initial, y_initial)
    first = wrapper.predict(candidates)
    wrapper.fit(
        np.vstack([X_initial, candidates[[17]]]),
        np.append(y_initial, first.mean[17] - first.std[17] - 0.1),
    )
    posterior = wrapper.predict(candidates)

    posterior.validate(5731)
    assert posterior.mean.shape == (5731,)
    assert posterior.std.shape == (5731,)
    arrays = [
        value
        for value in vars(wrapper).values()
        if isinstance(value, np.ndarray)
    ]
    assert all(array.ndim <= 2 for array in arrays)
    assert all(array.size < 5731 * 10 for array in arrays)
    assert not any(array.shape == (5731, 5731) for array in arrays)


def test_factory_builds_injected_surprise_controller(
    tiny_dataset, tiny_config
) -> None:
    from dataclasses import replace

    model_name = "surprise_gated_rank_consensus_ard_gp"
    config = replace(
        tiny_config,
        ensemble=replace(
            tiny_config.ensemble,
            enabled_models=(model_name,),
            initial_weights={model_name: 1.0},
        ),
    )
    prepared = prepare_problem(tiny_dataset, config)
    ensemble = build_ensemble(config, prepared, seed=809)

    assert tuple(ensemble.models) == (model_name,)
    controller = ensemble.models[model_name]
    assert isinstance(controller, SurpriseGatedRankConsensusARD)
    assert tuple(controller.experts) == (COMPONENT_ARD, INTERACTION, TREES)
