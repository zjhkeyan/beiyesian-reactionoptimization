from __future__ import annotations

import numpy as np

from ..config import GPConfig
from ..schemas import FeatureSchema, Posterior
from .component_ard_gp import ComponentARDStructuredGP
from .delete_fold_component_ard_gp import DeleteFoldComponentARDEnsemble


class StageSplitDeleteFoldARD:
    """Incumbent component ARD for fit one, then delete-fold ARD thereafter."""

    name = "stage_split_delete_fold_ard_gp"

    def __init__(
        self, schema: FeatureSchema, config: GPConfig, *, seed: int = 0
    ) -> None:
        self.schema = schema
        self.config = config
        self.seed = int(seed)
        self.first_query_model = ComponentARDStructuredGP(
            schema=schema, config=config, seed=self.seed + 17
        )
        self.later_query_model = DeleteFoldComponentARDEnsemble(
            schema=schema, config=config, seed=self.seed + 33
        )
        self.fit_calls_: int = 0
        self.active_stage_: str | None = None
        self.stage_history_: list[str] = []
        self.initial_observed_count_: int | None = None
        self.transition_observed_count_: int | None = None

    @property
    def active_model(self):
        if self.active_stage_ == "component_ard_query_1":
            return self.first_query_model
        if self.active_stage_ == "delete_fold_logei_queries_2_5":
            return self.later_query_model
        raise RuntimeError("StageSplitDeleteFoldARD.fit must be called first.")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "StageSplitDeleteFoldARD":
        targets = np.asarray(y, dtype=float).reshape(-1)
        self.fit_calls_ += 1
        if self.fit_calls_ == 1:
            self.initial_observed_count_ = len(targets)
            self.active_stage_ = "component_ard_query_1"
            self.first_query_model.fit(X, targets)
        else:
            if self.transition_observed_count_ is None:
                self.transition_observed_count_ = len(targets)
            self.active_stage_ = "delete_fold_logei_queries_2_5"
            self.later_query_model.fit(X, targets)
        self.stage_history_.append(self.active_stage_)
        return self

    def predict(self, X: np.ndarray) -> Posterior:
        return self.active_model.predict(X)

    def posterior_covariance(self, X: np.ndarray) -> np.ndarray:
        return self.active_model.posterior_covariance(X)

    @property
    def members(self):
        return self.later_query_model.members

    @property
    def observed_count_(self) -> int:
        return self.later_query_model.observed_count_

    @property
    def fold_sizes_(self) -> tuple[int, ...]:
        return self.later_query_model.fold_sizes_

    @property
    def member_training_sizes_(self) -> tuple[int, ...]:
        return self.later_query_model.member_training_sizes_

    @property
    def within_variance_(self) -> np.ndarray | None:
        return self.later_query_model.within_variance_

    @property
    def between_variance_(self) -> np.ndarray | None:
        return self.later_query_model.between_variance_

    @property
    def total_variance_(self) -> np.ndarray | None:
        return self.later_query_model.total_variance_

