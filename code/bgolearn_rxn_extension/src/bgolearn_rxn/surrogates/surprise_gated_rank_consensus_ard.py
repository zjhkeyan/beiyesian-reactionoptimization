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
    median_rank_consensus,
)


class SurpriseGatedRankConsensusARD:
    """Route one prediction through ordinal consensus after negative surprise.

    The component-ARD expert is the incumbent and is the only expert used on
    query one. Before each recommendation its candidate posterior is cached.
    Once feedback is appended, the new row is matched to that cache and its
    honest standardized residual is computed. A residual at or below one
    negative posterior standard deviation activates median-rank consensus for
    the next prediction only; every subsequent feedback re-evaluates the gate.

    Consensus ranks are mapped onto the sorted component-ARD predictive means.
    This preserves the robust three-expert ordering while keeping predictions
    on the target scale. Full-data component ARD remains authoritative for
    marginal uncertainty and posterior covariance.
    """

    name = "surprise_gated_rank_consensus_ard_gp"
    aggregation = "median_normalized_rank"
    severe_negative_z = -1.0
    matching_atol = 1e-12

    def __init__(self, experts: Mapping[str, ProbabilisticSurrogate]) -> None:
        missing = [name for name in REQUIRED_EXPERTS if name not in experts]
        extra = [name for name in experts if name not in REQUIRED_EXPERTS]
        if missing or extra:
            raise ValueError(
                f"SurpriseGatedRankConsensusARD requires exactly {REQUIRED_EXPERTS}; "
                f"missing={missing}, extra={extra}."
            )
        self.experts = OrderedDict((name, experts[name]) for name in REQUIRED_EXPERTS)
        self.component_ard = self.experts[COMPONENT_ARD]

        self.fit_calls_: int = 0
        self.predict_calls_: int = 0
        self.initial_observed_count_: int | None = None
        self.active_stage_: str | None = None
        self.stage_history_: list[str] = []
        self.observed_count_history_: list[int] = []
        self.activation_history_: list[bool] = []
        self.residual_history_: list[dict[str, float | int | bool]] = []

        self.last_standardized_residual_: float | None = None
        self.last_activation_: bool = False
        self.last_matched_candidate_positions_: list[int] = []
        self.last_consensus_: np.ndarray | None = None
        self.last_expert_ranks_: dict[str, np.ndarray] = {}
        self.last_rank_disagreement_: np.ndarray | None = None

        self._observed_X: np.ndarray | None = None
        self._candidate_X: np.ndarray | None = None
        self._candidate_component_mean: np.ndarray | None = None
        self._candidate_component_std: np.ndarray | None = None

    @staticmethod
    def _validated_training_arrays(
        X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        features = np.asarray(X, dtype=float)
        targets = np.asarray(y, dtype=float).reshape(-1)
        if features.ndim != 2 or len(features) != len(targets):
            raise ValueError("X must be 2D and aligned with y.")
        if not np.all(np.isfinite(features)) or not np.all(np.isfinite(targets)):
            raise ValueError("Training features and targets must be finite.")
        return features, targets

    @classmethod
    def _row_matches(cls, matrix: np.ndarray, row: np.ndarray) -> np.ndarray:
        return np.flatnonzero(
            np.all(
                np.isclose(matrix, row, rtol=0.0, atol=cls.matching_atol),
                axis=1,
            )
        )

    def _new_observation_indices(self, features: np.ndarray) -> np.ndarray:
        if self._observed_X is None:
            return np.arange(len(features), dtype=int)
        if features.shape[1] != self._observed_X.shape[1]:
            raise ValueError("Feature width changed between sequential fits.")

        # Consume prior observations as a multiset. This is deterministic for
        # duplicate rows and does not rely on append order.
        available = np.ones(len(features), dtype=bool)
        for previous_row in self._observed_X:
            matches = self._row_matches(features, previous_row)
            matches = matches[available[matches]]
            if len(matches) == 0:
                raise ValueError("A sequential fit removed a previously observed row.")
            available[int(matches[0])] = False
        return np.flatnonzero(available)

    def _feedback_residuals(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        new_indices: np.ndarray,
    ) -> list[dict[str, float | int | bool]]:
        if (
            self._candidate_X is None
            or self._candidate_component_mean is None
            or self._candidate_component_std is None
        ):
            raise RuntimeError(
                "A candidate prediction is required before appending feedback."
            )

        used_candidates = np.zeros(len(self._candidate_X), dtype=bool)
        records: list[dict[str, float | int | bool]] = []
        for observed_position in new_indices:
            matches = self._row_matches(
                self._candidate_X, features[int(observed_position)]
            )
            matches = matches[~used_candidates[matches]]
            if len(matches) == 0:
                raise ValueError(
                    "New feedback row was not found in the preceding candidate pool."
                )
            candidate_position = int(matches[0])
            used_candidates[candidate_position] = True
            predicted_mean = float(self._candidate_component_mean[candidate_position])
            predicted_std = max(
                float(self._candidate_component_std[candidate_position]), 1e-12
            )
            observed_y = float(targets[int(observed_position)])
            standardized_residual = (observed_y - predicted_mean) / predicted_std
            records.append(
                {
                    "observed_position": int(observed_position),
                    "candidate_position": candidate_position,
                    "observed_y": observed_y,
                    "predicted_mean": predicted_mean,
                    "predicted_std": predicted_std,
                    "standardized_residual": float(standardized_residual),
                    "activated": bool(
                        standardized_residual <= self.severe_negative_z
                    ),
                }
            )
        return records

    def fit(
        self, X: np.ndarray, y: np.ndarray
    ) -> "SurpriseGatedRankConsensusARD":
        features, targets = self._validated_training_arrays(X, y)
        new_indices = self._new_observation_indices(features)
        self.fit_calls_ += 1

        if self.initial_observed_count_ is None:
            self.initial_observed_count_ = len(targets)
            self.last_activation_ = False
            self.last_standardized_residual_ = None
            self.last_matched_candidate_positions_ = []
            self.active_stage_ = "component_ard_query_1"
        else:
            if len(new_indices) == 0:
                # Re-fitting the same observed set cannot introduce evidence
                # that changes the current gate.
                activation = self.last_activation_
            else:
                records = self._feedback_residuals(
                    features, targets, new_indices
                )
                self.residual_history_.extend(records)
                activation = any(bool(record["activated"]) for record in records)
                most_negative = min(
                    records,
                    key=lambda record: float(record["standardized_residual"]),
                )
                self.last_standardized_residual_ = float(
                    most_negative["standardized_residual"]
                )
                self.last_matched_candidate_positions_ = [
                    int(record["candidate_position"]) for record in records
                ]
            self.last_activation_ = bool(activation)
            self.active_stage_ = (
                "surprise_rank_consensus"
                if self.last_activation_
                else "component_ard_nominal"
            )

        self.component_ard.fit(features, targets)
        if self.last_activation_:
            self.experts[INTERACTION].fit(features, targets)
            self.experts[TREES].fit(features, targets)

        self._observed_X = features.copy()
        self._candidate_X = None
        self._candidate_component_mean = None
        self._candidate_component_std = None
        self.stage_history_.append(str(self.active_stage_))
        self.observed_count_history_.append(len(targets))
        self.activation_history_.append(self.last_activation_)
        return self

    def _require_fit(self) -> None:
        if self.active_stage_ is None:
            raise RuntimeError("SurpriseGatedRankConsensusARD.fit must be called first.")

    def _cache_component_prediction(
        self, X: np.ndarray, posterior: Posterior
    ) -> None:
        features = np.asarray(X, dtype=float)
        posterior.validate(len(features))
        self._candidate_X = features.copy()
        self._candidate_component_mean = posterior.mean.copy()
        self._candidate_component_std = posterior.std.copy()

    @staticmethod
    def _consensus_on_component_scale(
        consensus: np.ndarray, component_mean: np.ndarray
    ) -> np.ndarray:
        if len(consensus) == 1:
            return component_mean.copy()
        quantiles = np.linspace(0.0, 1.0, len(consensus))
        return np.interp(consensus, quantiles, np.sort(component_mean))

    def predict(self, X: np.ndarray) -> Posterior:
        self._require_fit()
        features = np.asarray(X, dtype=float)
        component = self.component_ard.predict(features)
        self._cache_component_prediction(features, component)
        self.predict_calls_ += 1

        if not self.last_activation_:
            return component

        expert_posteriors = {COMPONENT_ARD: component}
        for name in (INTERACTION, TREES):
            posterior = self.experts[name].predict(features)
            posterior.validate(len(features))
            expert_posteriors[name] = posterior
        consensus, ranks, disagreement = median_rank_consensus(
            {name: posterior.mean for name, posterior in expert_posteriors.items()}
        )
        consensus_mean = self._consensus_on_component_scale(
            consensus, component.mean
        )
        posterior = Posterior(
            mean=consensus_mean,
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
        self.last_consensus_ = consensus.copy()
        self.last_expert_ranks_ = {
            name: ranks[name].copy() for name in REQUIRED_EXPERTS
        }
        self.last_rank_disagreement_ = disagreement.copy()
        return posterior

    def posterior_covariance(self, X: np.ndarray) -> np.ndarray:
        """Return component-ARD covariance in both routing states."""

        self._require_fit()
        features = np.asarray(X, dtype=float)
        component = self.component_ard.predict(features)
        self._cache_component_prediction(features, component)
        return self.component_ard.posterior_covariance(features)

    def audit_state(self) -> dict[str, Any]:
        return {
            "model": self.name,
            "aggregation": self.aggregation,
            "severe_negative_z": self.severe_negative_z,
            "uncertainty_authority": COMPONENT_ARD,
            "fit_calls": self.fit_calls_,
            "predict_calls": self.predict_calls_,
            "initial_observed_count": self.initial_observed_count_,
            "active_stage": self.active_stage_,
            "stage_history": list(self.stage_history_),
            "observed_count_history": list(self.observed_count_history_),
            "activation_history": list(self.activation_history_),
            "last_activation": self.last_activation_,
            "last_standardized_residual": self.last_standardized_residual_,
            "last_matched_candidate_positions": list(
                self.last_matched_candidate_positions_
            ),
            "residual_history": [dict(record) for record in self.residual_history_],
            "candidate_cache_size": (
                None if self._candidate_X is None else len(self._candidate_X)
            ),
        }
