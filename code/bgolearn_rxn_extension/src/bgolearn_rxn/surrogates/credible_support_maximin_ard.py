from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

import numpy as np

from ..schemas import Posterior
from .base import ProbabilisticSurrogate
from .query_one_rank_consensus_ard import (
    COMPONENT_ARD,
    INTERACTION,
    REQUIRED_EXPERTS,
    TREES,
    normalized_rank_desirability,
)


def credible_support(
    component_mean: np.ndarray,
    component_std: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Return the fixed one-median-SD support around the component peak."""

    mean = np.asarray(component_mean, dtype=float).reshape(-1)
    std = np.asarray(component_std, dtype=float).reshape(-1)
    if mean.size == 0 or mean.shape != std.shape:
        raise ValueError("Component mean and standard deviation must align and be nonempty.")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
        raise ValueError("Component posterior summaries must be finite.")
    if np.any(std < 0.0):
        raise ValueError("Component posterior standard deviations cannot be negative.")

    peak = float(np.max(mean))
    median_std = float(np.median(std))
    threshold = peak - median_std
    positions = np.flatnonzero(mean >= threshold)
    if positions.size == 0:
        positions = np.asarray([int(np.argmax(mean))], dtype=int)
    return positions, threshold, median_std


def maximin_rank_choice(
    expert_scores: Mapping[str, np.ndarray],
    support_positions: np.ndarray,
) -> tuple[int, dict[str, np.ndarray], np.ndarray]:
    """Select the support member with the strongest worst expert rank.

    Ranks are computed only over the credible support. Larger values are better.
    Ties are resolved by component-ARD rank and then original candidate position.
    """

    missing = [name for name in REQUIRED_EXPERTS if name not in expert_scores]
    extra = [name for name in expert_scores if name not in REQUIRED_EXPERTS]
    if missing or extra:
        raise ValueError(
            f"Maximin consensus requires exactly {REQUIRED_EXPERTS}; "
            f"missing={missing}, extra={extra}."
        )

    support = np.asarray(support_positions, dtype=int).reshape(-1)
    if support.size == 0:
        raise ValueError("Credible support must be nonempty.")
    lengths = {len(np.asarray(expert_scores[name]).reshape(-1)) for name in REQUIRED_EXPERTS}
    if len(lengths) != 1:
        raise ValueError("Expert scores must align on the same candidate pool.")
    candidate_count = lengths.pop()
    if np.any(support < 0) or np.any(support >= candidate_count):
        raise IndexError("Credible-support position is outside the candidate pool.")
    if len(np.unique(support)) != len(support):
        raise ValueError("Credible-support positions must be unique.")

    ranks = {
        name: normalized_rank_desirability(
            np.asarray(expert_scores[name], dtype=float).reshape(-1)[support]
        )
        for name in REQUIRED_EXPERTS
    }
    rank_matrix = np.vstack([ranks[name] for name in REQUIRED_EXPERTS])
    worst_rank = np.min(rank_matrix, axis=0)
    component_rank = ranks[COMPONENT_ARD]
    order = np.lexsort((support, -component_rank, -worst_rank))
    selected_local = int(order[0])
    return int(support[selected_local]), ranks, worst_rank


class CredibleSupportMaximinARD:
    """Conservative three-expert query-one choice, then exact component ARD.

    Component ARD first defines a plausible high-yield support without a fitted
    tuning parameter. The query-one choice maximizes the worst normalized rank
    among component ARD, the interaction model, and trees inside that support.
    Once feedback expands the observed set, the wrapper retires the two auxiliary
    experts and delegates predictions exactly to full-data component ARD.
    """

    name = "credible_support_maximin_ard_gp"
    aggregation = "maximin_normalized_rank"
    support_rule = "component_peak_minus_one_median_component_sd"

    def __init__(self, experts: Mapping[str, ProbabilisticSurrogate]) -> None:
        missing = [name for name in REQUIRED_EXPERTS if name not in experts]
        extra = [name for name in experts if name not in REQUIRED_EXPERTS]
        if missing or extra:
            raise ValueError(
                f"CredibleSupportMaximinARD requires exactly {REQUIRED_EXPERTS}; "
                f"missing={missing}, extra={extra}."
            )
        self.experts = OrderedDict((name, experts[name]) for name in REQUIRED_EXPERTS)
        self.component_ard = self.experts[COMPONENT_ARD]

        self.initial_observed_count_: int | None = None
        self.feedback_observed_count_: int | None = None
        self.active_stage_: str | None = None
        self.stage_history_: list[str] = []
        self.observed_count_history_: list[int] = []
        self.predict_calls_: int = 0

        self.support_size_: int | None = None
        self.support_threshold_: float | None = None
        self.median_component_std_: float | None = None
        self.selected_candidate_position_: int | None = None
        self.selected_maximin_rank_: float | None = None
        self.selected_expert_ranks_: dict[str, float] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CredibleSupportMaximinARD":
        features = np.asarray(X, dtype=float)
        targets = np.asarray(y, dtype=float).reshape(-1)
        if features.ndim != 2 or len(features) != len(targets):
            raise ValueError("X must be 2D and aligned with y.")
        if not np.all(np.isfinite(features)) or not np.all(np.isfinite(targets)):
            raise ValueError("Training features and targets must be finite.")

        observed_count = len(targets)
        if self.initial_observed_count_ is None:
            self.initial_observed_count_ = observed_count
        elif observed_count < self.initial_observed_count_:
            raise ValueError("Observed count cannot fall below the first-fit count.")

        has_feedback = (
            self.feedback_observed_count_ is not None
            or observed_count > self.initial_observed_count_
        )
        if has_feedback:
            if self.feedback_observed_count_ is None:
                self.feedback_observed_count_ = observed_count
            self.component_ard.fit(features, targets)
            self.active_stage_ = "component_ard_after_feedback"
        else:
            for expert in self.experts.values():
                expert.fit(features, targets)
            self.active_stage_ = "credible_support_maximin_query_1"

        self.stage_history_.append(self.active_stage_)
        self.observed_count_history_.append(observed_count)
        return self

    def _require_fit(self) -> None:
        if self.active_stage_ is None:
            raise RuntimeError("CredibleSupportMaximinARD.fit must be called first.")

    @staticmethod
    def _encode_unique_argmax(component_mean: np.ndarray, selected: int) -> np.ndarray:
        encoded = np.asarray(component_mean, dtype=float).copy()
        peak = float(np.max(encoded))
        promoted = np.nextafter(peak, np.inf)
        if np.isfinite(promoted):
            encoded[selected] = promoted
        else:
            for position in np.flatnonzero(encoded == peak):
                if int(position) != selected:
                    encoded[int(position)] = np.nextafter(peak, -np.inf)
            encoded[selected] = peak
        return encoded

    def predict(self, X: np.ndarray) -> Posterior:
        self._require_fit()
        features = np.asarray(X, dtype=float)
        if features.ndim != 2:
            raise ValueError("X must be 2D.")
        if self.active_stage_ == "component_ard_after_feedback":
            self.predict_calls_ += 1
            return self.component_ard.predict(features)

        expert_posteriors = {
            name: expert.predict(features) for name, expert in self.experts.items()
        }
        for posterior in expert_posteriors.values():
            posterior.validate(len(features))
        component = expert_posteriors[COMPONENT_ARD]
        support, threshold, median_std = credible_support(
            component.mean, component.std
        )
        selected, support_ranks, worst_rank = maximin_rank_choice(
            {name: posterior.mean for name, posterior in expert_posteriors.items()},
            support,
        )
        selected_local = int(np.flatnonzero(support == selected)[0])
        rank_matrix = np.vstack(
            [support_ranks[name] for name in REQUIRED_EXPERTS]
        )
        support_disagreement = np.std(rank_matrix, axis=0, ddof=0)
        disagreement = np.zeros(len(features), dtype=float)
        disagreement[support] = support_disagreement

        posterior = Posterior(
            mean=self._encode_unique_argmax(component.mean, selected),
            std=component.std.copy(),
            component_means={
                name: expert_posteriors[name].mean.copy() for name in REQUIRED_EXPERTS
            },
            component_stds={
                name: expert_posteriors[name].std.copy() for name in REQUIRED_EXPERTS
            },
            weights={name: 1.0 / len(REQUIRED_EXPERTS) for name in REQUIRED_EXPERTS},
            disagreement=disagreement,
        )
        posterior.validate(len(features))

        self.predict_calls_ += 1
        self.support_size_ = int(len(support))
        self.support_threshold_ = float(threshold)
        self.median_component_std_ = float(median_std)
        self.selected_candidate_position_ = selected
        self.selected_maximin_rank_ = float(worst_rank[selected_local])
        self.selected_expert_ranks_ = {
            name: float(support_ranks[name][selected_local])
            for name in REQUIRED_EXPERTS
        }
        return posterior

    def posterior_covariance(self, X: np.ndarray) -> np.ndarray:
        """Use component ARD as the covariance authority in both stages."""

        self._require_fit()
        return self.component_ard.posterior_covariance(np.asarray(X, dtype=float))

    def audit_state(self) -> dict[str, Any]:
        return {
            "model": self.name,
            "aggregation": self.aggregation,
            "support_rule": self.support_rule,
            "rank_basis": "credible_support_only",
            "uncertainty_authority": COMPONENT_ARD,
            "initial_observed_count": self.initial_observed_count_,
            "feedback_observed_count": self.feedback_observed_count_,
            "active_stage": self.active_stage_,
            "stage_history": list(self.stage_history_),
            "observed_count_history": list(self.observed_count_history_),
            "predict_calls": self.predict_calls_,
            "support_size": self.support_size_,
            "support_threshold": self.support_threshold_,
            "median_component_std": self.median_component_std_,
            "selected_candidate_position": self.selected_candidate_position_,
            "selected_maximin_rank": self.selected_maximin_rank_,
            "selected_expert_ranks": dict(self.selected_expert_ranks_),
        }
