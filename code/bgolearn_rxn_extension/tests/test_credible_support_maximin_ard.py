from __future__ import annotations

import numpy as np

from bgolearn_rxn.schemas import Posterior
from bgolearn_rxn.surrogates.credible_support_maximin_ard import (
    CredibleSupportMaximinARD,
    credible_support,
    maximin_rank_choice,
)
from bgolearn_rxn.surrogates.query_one_rank_consensus_ard import (
    COMPONENT_ARD,
    INTERACTION,
    TREES,
)


class _StaticExpert:
    def __init__(self, mean: np.ndarray, std: np.ndarray | float = 0.5) -> None:
        self.mean = np.asarray(mean, dtype=float)
        self.std = np.broadcast_to(np.asarray(std, dtype=float), self.mean.shape).copy()
        self.fit_calls = 0
        self.predict_calls = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_StaticExpert":
        assert len(X) == len(y)
        self.fit_calls += 1
        return self

    def predict(self, X: np.ndarray) -> Posterior:
        if len(X) != len(self.mean):
            raise ValueError("Static expert candidate count changed.")
        self.predict_calls += 1
        return Posterior(mean=self.mean.copy(), std=self.std.copy())

    def posterior_covariance(self, X: np.ndarray) -> np.ndarray:
        return np.diag(self.std[: len(X)] ** 2)


def _wrapper(
    component: np.ndarray,
    interaction: np.ndarray,
    trees: np.ndarray,
    component_std: np.ndarray | float = 0.5,
) -> tuple[CredibleSupportMaximinARD, dict[str, _StaticExpert]]:
    experts = {
        COMPONENT_ARD: _StaticExpert(component, component_std),
        INTERACTION: _StaticExpert(interaction),
        TREES: _StaticExpert(trees),
    }
    return CredibleSupportMaximinARD(experts), experts


def _fit_predict(wrapper: CredibleSupportMaximinARD, count: int) -> Posterior:
    X_train = np.arange(58, dtype=float).reshape(29, 2)
    y_train = np.linspace(0.0, 1.0, 29)
    wrapper.fit(X_train, y_train)
    return wrapper.predict(np.zeros((count, 2), dtype=float))


def test_component_credible_support_restricts_the_consensus_choice() -> None:
    wrapper, _ = _wrapper(
        component=np.array([10.0, 9.8, 9.6, 7.0, 6.0]),
        interaction=np.array([0.0, 2.0, 1.0, 100.0, 90.0]),
        trees=np.array([0.0, 2.0, 1.0, 100.0, 90.0]),
        component_std=np.full(5, 0.5),
    )
    posterior = _fit_predict(wrapper, 5)

    assert wrapper.support_size_ == 3
    assert wrapper.selected_candidate_position_ in (0, 1, 2)
    assert wrapper.selected_candidate_position_ == 1
    assert int(np.argmax(posterior.mean)) == 1
    assert wrapper.audit_state()["support_rule"] == (
        "component_peak_minus_one_median_component_sd"
    )


def test_maximin_selects_the_best_worst_expert_rank_inside_support() -> None:
    scores = {
        COMPONENT_ARD: np.array([4.0, 3.0, 2.0]),
        INTERACTION: np.array([1.0, 3.0, 2.0]),
        TREES: np.array([3.0, 2.0, 1.0]),
    }
    selected, ranks, worst = maximin_rank_choice(scores, np.arange(3))

    assert selected == 1
    np.testing.assert_array_equal(worst, np.array([0.0, 0.5, 0.0]))
    assert {name: values[1] for name, values in ranks.items()} == {
        COMPONENT_ARD: 0.5,
        INTERACTION: 1.0,
        TREES: 0.5,
    }


def test_choice_is_invariant_to_monotone_expert_transformations() -> None:
    component = np.array([4.0, 3.0, 2.0, 1.0])
    interaction = np.array([1.0, 3.0, 2.0, 4.0])
    trees = np.array([3.0, 2.0, 4.0, 1.0])
    support, _, _ = credible_support(component, np.full(4, 10.0))
    first = maximin_rank_choice(
        {COMPONENT_ARD: component, INTERACTION: interaction, TREES: trees},
        support,
    )
    transformed = maximin_rank_choice(
        {
            COMPONENT_ARD: 11.0 + 7.0 * component,
            INTERACTION: np.exp(interaction),
            TREES: np.power(trees, 3.0),
        },
        support,
    )

    assert first[0] == transformed[0]
    np.testing.assert_array_equal(first[2], transformed[2])
    for name in first[1]:
        np.testing.assert_array_equal(first[1][name], transformed[1][name])


def test_ties_use_component_rank_then_original_candidate_position() -> None:
    scores = {
        COMPONENT_ARD: np.array([1.0, 2.0, 2.0, 0.0]),
        INTERACTION: np.ones(4),
        TREES: np.ones(4),
    }
    selected, _, _ = maximin_rank_choice(scores, np.array([2, 1, 0, 3]))
    assert selected == 1

    all_tied = {name: np.ones(4) for name in scores}
    selected, _, _ = maximin_rank_choice(all_tied, np.array([3, 2, 1, 0]))
    assert selected == 0


def test_feedback_retires_auxiliary_experts_and_returns_exact_component() -> None:
    component = np.array([8.0, 7.0, 6.0])
    wrapper, experts = _wrapper(component, component[::-1], np.ones(3))
    _fit_predict(wrapper, 3)
    X_feedback = np.arange(60, dtype=float).reshape(30, 2)
    y_feedback = np.linspace(0.0, 1.0, 30)
    wrapper.fit(X_feedback, y_feedback)
    posterior = wrapper.predict(np.zeros((3, 2), dtype=float))

    np.testing.assert_array_equal(posterior.mean, component)
    np.testing.assert_array_equal(posterior.std, np.full(3, 0.5))
    assert experts[COMPONENT_ARD].fit_calls == 2
    assert experts[INTERACTION].fit_calls == 1
    assert experts[TREES].fit_calls == 1
    assert wrapper.stage_history_ == [
        "credible_support_maximin_query_1",
        "component_ard_after_feedback",
    ]
    assert wrapper.audit_state()["feedback_observed_count"] == 30


def test_query_one_posterior_is_finite_and_keeps_component_uncertainty() -> None:
    component_std = np.array([0.2, 0.3, 0.4, 0.5])
    wrapper, _ = _wrapper(
        np.array([4.0, 3.0, 2.0, 1.0]),
        np.array([1.0, 3.0, 2.0, 4.0]),
        np.array([3.0, 2.0, 4.0, 1.0]),
        component_std,
    )
    posterior = _fit_predict(wrapper, 4)

    posterior.validate(4)
    assert np.isfinite(posterior.mean).all()
    assert np.isfinite(posterior.std).all()
    np.testing.assert_array_equal(posterior.std, component_std)
    assert int(np.argmax(posterior.mean)) == wrapper.selected_candidate_position_
    audit = wrapper.audit_state()
    assert set(audit["selected_expert_ranks"]) == {
        COMPONENT_ARD,
        INTERACTION,
        TREES,
    }
    assert audit["selected_maximin_rank"] == min(
        audit["selected_expert_ranks"].values()
    )


def test_5731_candidate_prediction_remains_linear_in_pool_size() -> None:
    count = 5731
    base = np.linspace(0.0, 1.0, count)
    wrapper, experts = _wrapper(
        component=base,
        interaction=np.sin(13.0 * base),
        trees=np.cos(17.0 * base),
        component_std=np.full(count, 0.08),
    )
    posterior = _fit_predict(wrapper, count)

    assert posterior.mean.shape == (count,)
    assert posterior.std.shape == (count,)
    assert posterior.covariance is None
    arrays = [posterior.mean, posterior.std, posterior.disagreement]
    arrays.extend(posterior.component_means.values())
    arrays.extend(posterior.component_stds.values())
    assert sum(array.nbytes for array in arrays if array is not None) < 500_000
    assert all(expert.predict_calls == 1 for expert in experts.values())
