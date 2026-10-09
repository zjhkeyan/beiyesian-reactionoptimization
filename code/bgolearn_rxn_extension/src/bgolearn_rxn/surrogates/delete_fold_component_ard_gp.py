from __future__ import annotations

import numpy as np

from ..config import GPConfig
from ..schemas import FeatureSchema, Posterior
from .component_ard_gp import ComponentARDStructuredGP


MEMBER_COUNT = 5


def deterministic_delete_fold_assignments(y: np.ndarray) -> np.ndarray:
    """Assign yield-sorted rows round-robin to five deterministic folds."""

    values = np.asarray(y, dtype=float).reshape(-1)
    if len(values) < MEMBER_COUNT:
        raise ValueError(f"Delete-fold ARD requires at least {MEMBER_COUNT} rows.")
    if not np.isfinite(values).all():
        raise ValueError("Delete-fold targets must be finite.")
    indices = np.arange(len(values), dtype=int)
    order = np.lexsort((indices, values))
    assignments = np.empty(len(values), dtype=int)
    assignments[order] = np.arange(len(values), dtype=int) % MEMBER_COUNT
    return assignments


def aggregate_member_moments(
    member_means: np.ndarray,
    member_stds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return equal-weight mean, total variance, within, and between terms."""

    means = np.asarray(member_means, dtype=float)
    stds = np.asarray(member_stds, dtype=float)
    if means.ndim != 2 or stds.shape != means.shape:
        raise ValueError("Member means and standard deviations must be equal 2D arrays.")
    if means.shape[0] != MEMBER_COUNT:
        raise ValueError(f"Expected exactly {MEMBER_COUNT} member predictions.")
    if not np.isfinite(means).all() or not np.isfinite(stds).all():
        raise ValueError("Member predictive moments must be finite.")
    if np.any(stds <= 0.0):
        raise ValueError("Member predictive standard deviations must be positive.")
    mean = np.mean(means, axis=0)
    within = np.mean(stds**2, axis=0)
    between = np.mean((means - mean) ** 2, axis=0)
    total = np.maximum(within + between, 1e-12)
    return mean, total, within, between


class DeleteFoldComponentARDEnsemble:
    """Five yield-balanced delete-fold fits of the incumbent component ARD GP."""

    name = "delete_fold_component_ard_gp"
    member_count = MEMBER_COUNT

    def __init__(
        self, schema: FeatureSchema, config: GPConfig, *, seed: int = 0
    ) -> None:
        self.schema = schema
        self.config = config
        self.seed = int(seed)
        self.members = tuple(
            ComponentARDStructuredGP(
                schema=schema,
                config=config,
                seed=self.seed + 101 * member_index,
            )
            for member_index in range(MEMBER_COUNT)
        )
        self.fold_assignments_: np.ndarray | None = None
        self.fold_sizes_: tuple[int, ...] = ()
        self.member_training_sizes_: tuple[int, ...] = ()
        self.observed_count_: int = 0
        self.member_means_: np.ndarray | None = None
        self.member_stds_: np.ndarray | None = None
        self.within_variance_: np.ndarray | None = None
        self.between_variance_: np.ndarray | None = None
        self.total_variance_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DeleteFoldComponentARDEnsemble":
        features = np.asarray(X, dtype=float)
        targets = np.asarray(y, dtype=float).reshape(-1)
        if features.ndim != 2 or len(features) != len(targets):
            raise ValueError("X must be 2D and aligned with y.")
        assignments = deterministic_delete_fold_assignments(targets)
        fold_sizes = tuple(
            int(np.sum(assignments == member_index))
            for member_index in range(MEMBER_COUNT)
        )
        training_sizes = []
        for member_index, member in enumerate(self.members):
            keep = assignments != member_index
            member.fit(features[keep], targets[keep])
            training_sizes.append(int(np.sum(keep)))
        self.fold_assignments_ = assignments
        self.fold_sizes_ = fold_sizes
        self.member_training_sizes_ = tuple(training_sizes)
        self.observed_count_ = len(targets)
        return self

    def _require_fit(self) -> None:
        if self.fold_assignments_ is None:
            raise RuntimeError("DeleteFoldComponentARDEnsemble.fit must be called first.")

    def predict(self, X: np.ndarray) -> Posterior:
        self._require_fit()
        posteriors = [member.predict(X) for member in self.members]
        means = np.vstack([posterior.mean for posterior in posteriors])
        stds = np.vstack([posterior.std for posterior in posteriors])
        mean, total, within, between = aggregate_member_moments(means, stds)
        self.member_means_ = means
        self.member_stds_ = stds
        self.within_variance_ = within
        self.between_variance_ = between
        self.total_variance_ = total
        posterior = Posterior(
            mean=mean,
            std=np.sqrt(total),
            disagreement=np.sqrt(np.maximum(between, 0.0)),
        )
        posterior.validate(len(X))
        return posterior

    def posterior_covariance(self, X: np.ndarray) -> np.ndarray:
        self._require_fit()
        posteriors = [member.predict(X) for member in self.members]
        means = np.vstack([posterior.mean for posterior in posteriors])
        mixture_mean = np.mean(means, axis=0)
        within = np.mean(
            np.stack([member.posterior_covariance(X) for member in self.members]),
            axis=0,
        )
        centered = means - mixture_mean
        between = centered.T @ centered / MEMBER_COUNT
        covariance = within + between
        covariance = 0.5 * (covariance + covariance.T)
        diagonal = np.maximum(np.diag(covariance), 1e-12)
        covariance.flat[:: len(covariance) + 1] = diagonal
        return covariance

