from __future__ import annotations

from typing import Any

import numpy as np

from ..config import GPConfig
from ..schemas import FeatureSchema, Posterior
from .component_ard_gp import ComponentARDStructuredGP
from .delete_fold_component_ard_gp import DeleteFoldComponentARDEnsemble


class FullMeanJackknifeVarianceARDGP:
    """Full-data ARD mean with delete-fold epistemic variance inflation.

    The first fit is exactly the incumbent component ARD model.  On later fits,
    the full-data model remains authoritative for both the predictive mean and
    its posterior covariance.  Delete-fold fits contribute only the positive
    semidefinite covariance of their predictive-mean disagreement.
    """

    name = "full_mean_jackknife_variance_ard_gp"

    def __init__(
        self, schema: FeatureSchema, config: GPConfig, *, seed: int = 0
    ) -> None:
        self.schema = schema
        self.config = config
        self.seed = int(seed)
        # These offsets match the standalone factory registrations exactly.
        self.full_data_model = ComponentARDStructuredGP(
            schema=schema, config=config, seed=self.seed + 17
        )
        self.delete_fold_model = DeleteFoldComponentARDEnsemble(
            schema=schema, config=config, seed=self.seed + 33
        )
        self.fit_calls_: int = 0
        self.active_stage_: str | None = None
        self.stage_history_: list[str] = []
        self.initial_observed_count_: int | None = None
        self.transition_observed_count_: int | None = None
        self.member_means_: np.ndarray | None = None
        self.member_stds_: np.ndarray | None = None
        self.disagreement_correction_: np.ndarray | None = None
        self.disagreement_variance_: np.ndarray | None = None
        self.full_variance_: np.ndarray | None = None
        self.total_variance_: np.ndarray | None = None
        self.disagreement_diagnostics_: dict[str, float | bool] = {}
        self.audit_state_: dict[str, Any] = {}

    @property
    def members(self):
        return self.delete_fold_model.members

    @property
    def fold_assignments_(self) -> np.ndarray | None:
        return self.delete_fold_model.fold_assignments_

    @property
    def fold_sizes_(self) -> tuple[int, ...]:
        return self.delete_fold_model.fold_sizes_

    @property
    def member_training_sizes_(self) -> tuple[int, ...]:
        return self.delete_fold_model.member_training_sizes_

    @property
    def observed_count_(self) -> int:
        return self.delete_fold_model.observed_count_

    def _refresh_audit(self) -> None:
        assignments = self.fold_assignments_
        self.audit_state_ = {
            "model": self.name,
            "fit_calls": self.fit_calls_,
            "active_stage": self.active_stage_,
            "stage_history": list(self.stage_history_),
            "initial_observed_count": self.initial_observed_count_,
            "transition_observed_count": self.transition_observed_count_,
            "delete_member_count": len(self.members),
            "fold_assignments": (
                None if assignments is None else assignments.astype(int).tolist()
            ),
            "fold_sizes": list(self.fold_sizes_),
            "member_training_sizes": list(self.member_training_sizes_),
            "disagreement": dict(self.disagreement_diagnostics_),
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FullMeanJackknifeVarianceARDGP":
        targets = np.asarray(y, dtype=float).reshape(-1)
        self.fit_calls_ += 1
        self.full_data_model.fit(X, targets)
        if self.fit_calls_ == 1:
            self.initial_observed_count_ = len(targets)
            self.active_stage_ = "full_component_ard_query_1"
        else:
            if self.transition_observed_count_ is None:
                self.transition_observed_count_ = len(targets)
            self.delete_fold_model.fit(X, targets)
            self.active_stage_ = "full_mean_plus_delete_fold_variance"
        self.stage_history_.append(self.active_stage_)
        self._refresh_audit()
        return self

    def _require_fit(self) -> None:
        if self.active_stage_ is None:
            raise RuntimeError(f"{type(self).__name__}.fit must be called first.")

    def _delete_disagreement(
        self,
        X: np.ndarray,
        *,
        full_covariance: bool,
    ) -> tuple[np.ndarray | None, np.ndarray]:
        posteriors = [member.predict(X) for member in self.members]
        means = np.vstack([posterior.mean for posterior in posteriors])
        stds = np.vstack([posterior.std for posterior in posteriors])
        center = np.mean(means, axis=0)
        centered = means - center
        variance = np.maximum(np.mean(centered**2, axis=0), 0.0)

        # The candidate-space correction is centered.T @ centered / M.  Its
        # nonzero eigenvalues equal those of this at-most-5x5 member Gram, so
        # diagnostics never need an O(N^2) allocation or O(N^3) eigendecomposition.
        member_gram = centered @ centered.T / len(self.members)
        member_gram = 0.5 * (member_gram + member_gram.T)
        eigenvalues = np.linalg.eigvalsh(member_gram)
        correction = None
        if full_covariance:
            correction = centered.T @ centered / len(self.members)
            correction = 0.5 * (correction + correction.T)

        diagnostics: dict[str, float | bool] = {
            "finite": bool(
                np.isfinite(means).all()
                and np.isfinite(stds).all()
                and np.isfinite(member_gram).all()
                and (correction is None or np.isfinite(correction).all())
            ),
            "minimum_eigenvalue": float(np.min(eigenvalues)),
            "trace": float(np.sum(variance)),
            "mean_variance": float(np.mean(variance)),
            "maximum_variance": float(np.max(variance)),
            "noncollapsed": bool(np.any(variance > 0.0)),
        }
        if not diagnostics["finite"]:
            raise FloatingPointError("Delete-fold disagreement contains non-finite values.")

        self.member_means_ = means
        self.member_stds_ = stds
        self.disagreement_correction_ = correction
        self.disagreement_variance_ = variance
        self.disagreement_diagnostics_ = diagnostics
        self._refresh_audit()
        return correction, variance

    def predict(self, X: np.ndarray) -> Posterior:
        self._require_fit()
        full = self.full_data_model.predict(X)
        if self.fit_calls_ == 1:
            return full

        _, disagreement_variance = self._delete_disagreement(
            X, full_covariance=False
        )
        full_variance = full.std**2
        total_variance = np.maximum(full_variance + disagreement_variance, 1e-12)
        self.full_variance_ = full_variance
        self.total_variance_ = total_variance
        posterior = Posterior(
            mean=full.mean,
            std=np.sqrt(total_variance),
            disagreement=np.sqrt(disagreement_variance),
        )
        posterior.validate(len(X))
        return posterior

    def posterior_covariance(self, X: np.ndarray) -> np.ndarray:
        self._require_fit()
        full_covariance = self.full_data_model.posterior_covariance(X)
        if self.fit_calls_ == 1:
            return full_covariance

        full = self.full_data_model.predict(X)
        correction, disagreement_variance = self._delete_disagreement(
            X, full_covariance=True
        )
        if correction is None:
            raise RuntimeError("Full disagreement covariance was not constructed.")
        covariance = full_covariance + correction
        covariance = 0.5 * (covariance + covariance.T)
        total_variance = np.maximum(full.std**2 + disagreement_variance, 1e-12)
        covariance.flat[:: len(covariance) + 1] = total_variance
        self.full_variance_ = full.std**2
        self.total_variance_ = total_variance
        return covariance
