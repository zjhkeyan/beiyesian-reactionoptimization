from __future__ import annotations

import numpy as np

from ..config import GPConfig
from ..schemas import FeatureSchema, Posterior
from .component_ard_gp import ComponentARDStructuredGP
from .delete_fold_component_ard_gp import DeleteFoldComponentARDEnsemble


FEEDBACK_BEFORE_FULL_DATA = 10


class EarlyDeleteFoldThenFullARD:
    """Use deletion stabilization only during the first ten feedback rounds."""

    name = "early_delete_fold_then_full_ard_gp"
    feedback_before_full_data = FEEDBACK_BEFORE_FULL_DATA

    def __init__(
        self, schema: FeatureSchema, config: GPConfig, *, seed: int = 0
    ) -> None:
        self.schema = schema
        self.config = config
        self.seed = int(seed)
        self.full_data_model = ComponentARDStructuredGP(
            schema=schema, config=config, seed=self.seed + 17
        )
        self.delete_fold_model = DeleteFoldComponentARDEnsemble(
            schema=schema, config=config, seed=self.seed + 33
        )
        self.active_stage_: str | None = None
        self.stage_history_: list[str] = []
        self.observed_count_history_: list[int] = []
        self.initial_observed_count_: int | None = None
        self.maturity_threshold_observed_count_: int | None = None
        self.mature_transition_observed_count_: int | None = None

    @property
    def active_model(self):
        if self.active_stage_ in {
            "full_component_ard_query_1",
            "full_component_ard_mature",
        }:
            return self.full_data_model
        if self.active_stage_ == "delete_fold_component_ard_early":
            return self.delete_fold_model
        raise RuntimeError("EarlyDeleteFoldThenFullARD.fit must be called first.")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EarlyDeleteFoldThenFullARD":
        features = np.asarray(X, dtype=float)
        targets = np.asarray(y, dtype=float).reshape(-1)
        if features.ndim != 2 or len(features) != len(targets):
            raise ValueError("X must be 2D and aligned with y.")
        if self.initial_observed_count_ is None:
            self.initial_observed_count_ = len(targets)
            self.maturity_threshold_observed_count_ = (
                self.initial_observed_count_ + self.feedback_before_full_data
            )
        elif len(targets) < self.initial_observed_count_:
            raise ValueError("Observed count cannot fall below the first-fit count.")

        if len(targets) == self.initial_observed_count_:
            stage = "full_component_ard_query_1"
            model = self.full_data_model
        elif len(targets) < self.maturity_threshold_observed_count_:
            stage = "delete_fold_component_ard_early"
            model = self.delete_fold_model
        else:
            stage = "full_component_ard_mature"
            model = self.full_data_model
            if self.mature_transition_observed_count_ is None:
                self.mature_transition_observed_count_ = len(targets)

        model.fit(features, targets)
        self.active_stage_ = stage
        self.stage_history_.append(stage)
        self.observed_count_history_.append(len(targets))
        return self

    def predict(self, X: np.ndarray) -> Posterior:
        return self.active_model.predict(X)

    def posterior_covariance(self, X: np.ndarray) -> np.ndarray:
        return self.active_model.posterior_covariance(X)

    def audit_state(self) -> dict[str, object]:
        """Return JSON-safe state for campaign schedule audits."""

        return {
            "initial_observed_count": self.initial_observed_count_,
            "maturity_threshold_observed_count": self.maturity_threshold_observed_count_,
            "mature_transition_observed_count": self.mature_transition_observed_count_,
            "active_stage": self.active_stage_,
            "stage_history": list(self.stage_history_),
            "observed_count_history": list(self.observed_count_history_),
            "delete_fold_member_training_sizes": list(
                self.delete_fold_model.member_training_sizes_
            ),
        }

    @property
    def members(self):
        return self.delete_fold_model.members

    @property
    def fold_sizes_(self) -> tuple[int, ...]:
        return self.delete_fold_model.fold_sizes_

    @property
    def member_training_sizes_(self) -> tuple[int, ...]:
        return self.delete_fold_model.member_training_sizes_

