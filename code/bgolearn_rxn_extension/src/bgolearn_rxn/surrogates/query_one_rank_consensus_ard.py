from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping

import numpy as np
from scipy.stats import rankdata

from ..schemas import Posterior
from .base import ProbabilisticSurrogate


COMPONENT_ARD = "component_ard_gp"
INTERACTION = "interaction"
TREES = "trees"
REQUIRED_EXPERTS = (COMPONENT_ARD, INTERACTION, TREES)


def normalized_rank_desirability(values: np.ndarray) -> np.ndarray:
    """Map finite scores to scale-free [0, 1] desirability ranks.

    Larger expert predictions remain better. Average ranks make ties independent
    of candidate row order and preserve invariance to strictly increasing
    transformations of any expert's predictions.
    """

    scores = np.asarray(values, dtype=float).reshape(-1)
    if scores.size == 0 or not np.all(np.isfinite(scores)):
        raise ValueError("Expert scores must be nonempty and finite.")
    if len(scores) == 1:
        return np.ones(1, dtype=float)
    return (rankdata(scores, method="average") - 1.0) / (len(scores) - 1.0)


def median_rank_consensus(expert_scores: Mapping[str, np.ndarray]) -> tuple[
    np.ndarray, dict[str, np.ndarray], np.ndarray
]:
    """Return the predeclared median-rank consensus and disagreement."""

    missing = [name for name in REQUIRED_EXPERTS if name not in expert_scores]
    extra = [name for name in expert_scores if name not in REQUIRED_EXPERTS]
    if missing or extra:
        raise ValueError(
            f"Rank consensus requires exactly {REQUIRED_EXPERTS}; "
            f"missing={missing}, extra={extra}."
        )
    ranks = {
        name: normalized_rank_desirability(expert_scores[name])
        for name in REQUIRED_EXPERTS
    }
    lengths = {len(values) for values in ranks.values()}
    if len(lengths) != 1:
        raise ValueError("Expert scores must align on the same candidate pool.")
    matrix = np.vstack([ranks[name] for name in REQUIRED_EXPERTS])
    consensus = np.median(matrix, axis=0)
    disagreement = np.std(matrix, axis=0, ddof=0)
    return consensus, ranks, disagreement


class QueryOneRankConsensusARD:
    """Use robust ordinal expert consensus once, then full-data component ARD.

    The three established experts are injected because they intentionally use
    different configuration objects. On the initial fit all experts are fit;
    their candidate means are converted to scale-free ranks and aggregated by
    the predeclared median. Once any feedback row has been added, only the
    component-ARD expert is fit and its posterior is returned unchanged.
    """

    name = "query_one_rank_consensus_ard_gp"
    aggregation = "median_normalized_rank"

    def __init__(self, experts: Mapping[str, ProbabilisticSurrogate]) -> None:
        missing = [name for name in REQUIRED_EXPERTS if name not in experts]
        extra = [name for name in experts if name not in REQUIRED_EXPERTS]
        if missing or extra:
            raise ValueError(
                f"QueryOneRankConsensusARD requires exactly {REQUIRED_EXPERTS}; "
                f"missing={missing}, extra={extra}."
            )
        self.experts = OrderedDict((name, experts[name]) for name in REQUIRED_EXPERTS)
        self.component_ard = self.experts[COMPONENT_ARD]
        self.active_stage_: str | None = None
        self.stage_history_: list[str] = []
        self.observed_count_history_: list[int] = []
        self.initial_observed_count_: int | None = None
        self.feedback_observed_count_: int | None = None
        self.query_one_predict_calls_: int = 0
        self.query_one_consensus_: np.ndarray | None = None
        self.query_one_expert_ranks_: dict[str, np.ndarray] = {}
        self.query_one_disagreement_: np.ndarray | None = None
        self.selected_candidate_position_: int | None = None
        self.selected_consensus_rank_: float | None = None
        self.selected_expert_ranks_: dict[str, float] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QueryOneRankConsensusARD":
        features = np.asarray(X, dtype=float)
        targets = np.asarray(y, dtype=float).reshape(-1)
        if features.ndim != 2 or len(features) != len(targets):
            raise ValueError("X must be 2D and aligned with y.")
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
            self.active_stage_ = "median_rank_consensus_query_1"

        self.stage_history_.append(self.active_stage_)
        self.observed_count_history_.append(observed_count)
        return self

    def _require_fit(self) -> None:
        if self.active_stage_ is None:
            raise RuntimeError("QueryOneRankConsensusARD.fit must be called first.")

    def predict(self, X: np.ndarray) -> Posterior:
        self._require_fit()
        if self.active_stage_ == "component_ard_after_feedback":
            return self.component_ard.predict(X)

        expert_posteriors = {
            name: expert.predict(X) for name, expert in self.experts.items()
        }
        candidate_count = len(np.asarray(X))
        for posterior in expert_posteriors.values():
            posterior.validate(candidate_count)
        consensus, ranks, disagreement = median_rank_consensus(
            {name: posterior.mean for name, posterior in expert_posteriors.items()}
        )
        rank_resolution = 1.0 / max(candidate_count - 1, 1)
        std = np.sqrt(disagreement**2 + rank_resolution**2)
        posterior = Posterior(
            mean=consensus,
            std=std,
            component_means={name: ranks[name] for name in REQUIRED_EXPERTS},
            component_stds={
                name: np.full(candidate_count, rank_resolution, dtype=float)
                for name in REQUIRED_EXPERTS
            },
            weights={name: 1.0 / len(REQUIRED_EXPERTS) for name in REQUIRED_EXPERTS},
            disagreement=disagreement,
        )
        posterior.validate(candidate_count)
        self.query_one_predict_calls_ += 1
        self.query_one_consensus_ = consensus.copy()
        self.query_one_expert_ranks_ = {
            name: values.copy() for name, values in ranks.items()
        }
        self.query_one_disagreement_ = disagreement.copy()
        return posterior

    def posterior_covariance(self, X: np.ndarray) -> np.ndarray:
        self._require_fit()
        if self.active_stage_ == "component_ard_after_feedback":
            return self.component_ard.posterior_covariance(X)
        posterior = self.predict(X)
        return np.diag(posterior.std**2)

    def record_query_one_selection(self, candidate_position: int) -> None:
        """Attach policy selection ranks to the JSON-safe audit state."""

        if self.query_one_consensus_ is None or not self.query_one_expert_ranks_:
            raise RuntimeError("Query-one prediction is required before selection audit.")
        position = int(candidate_position)
        if position < 0 or position >= len(self.query_one_consensus_):
            raise IndexError("Selected candidate position is outside the predicted pool.")
        descending_consensus_ranks = rankdata(
            -self.query_one_consensus_, method="average"
        )
        self.selected_candidate_position_ = position
        self.selected_consensus_rank_ = float(descending_consensus_ranks[position])
        self.selected_expert_ranks_ = {
            name: float(rankdata(-values, method="average")[position])
            for name, values in self.query_one_expert_ranks_.items()
        }

    def audit_state(self) -> dict[str, object]:
        return {
            "aggregation": self.aggregation,
            "required_experts": list(REQUIRED_EXPERTS),
            "initial_observed_count": self.initial_observed_count_,
            "feedback_observed_count": self.feedback_observed_count_,
            "active_stage": self.active_stage_,
            "stage_history": list(self.stage_history_),
            "observed_count_history": list(self.observed_count_history_),
            "query_one_predict_calls": self.query_one_predict_calls_,
            "selected_candidate_position": self.selected_candidate_position_,
            "selected_consensus_rank": self.selected_consensus_rank_,
            "selected_expert_ranks": dict(self.selected_expert_ranks_),
        }
