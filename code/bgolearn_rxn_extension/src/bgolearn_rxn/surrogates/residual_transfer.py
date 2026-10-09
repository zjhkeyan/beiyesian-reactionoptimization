from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..config import GPConfig
from ..schemas import FeatureSchema, Posterior
from .anova_gp import StructuredANOVAGP


class ResidualTransferGP:
    """Additive source-task GP with target-only residual adaptation."""

    name = "residual_transfer"

    def __init__(
        self,
        schema: FeatureSchema,
        config: GPConfig,
        *,
        target_task_identity: np.ndarray,
        seed: int = 0,
    ) -> None:
        if schema.task_role is None:
            raise ValueError("Residual transfer requires a configured task role.")
        identity = np.asarray(target_task_identity, dtype=float).reshape(-1)
        expected_width = schema.identity_slices[schema.task_role].width
        if identity.shape != (expected_width,) or not np.all(np.isfinite(identity)):
            raise ValueError("Target task identity is incompatible with the feature schema.")
        if np.linalg.norm(identity) <= 1e-12:
            raise ValueError("Target task identity must contain a nonzero identity feature.")

        self.schema = schema
        self.target_task_identity = identity
        self.source_gp = StructuredANOVAGP(schema, config, seed=seed + 1)
        residual_schema = replace(schema, task_role=None)
        residual_config = replace(
            config,
            target_transform="standard",
            task_mean_prior_count=None,
        )
        self.residual_gp = StructuredANOVAGP(
            residual_schema,
            residual_config,
            seed=seed + 2,
        )
        self.n_source_: int | None = None
        self.n_target_: int | None = None

    def _target_mask(self, X: np.ndarray) -> np.ndarray:
        task = self.schema.task_role
        if task is None:
            raise RuntimeError("Residual transfer lost its task role.")
        block = np.asarray(X[:, self.schema.identity_slice(task)], dtype=float)
        return np.all(
            np.isclose(block, self.target_task_identity[None, :], atol=1e-12),
            axis=1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ResidualTransferGP":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.ndim != 2 or X.shape[1] != self.schema.n_features or len(X) != len(y):
            raise ValueError("X and y are incompatible with the residual-transfer schema.")

        target_mask = self._target_mask(X)
        source_mask = ~target_mask
        self.n_target_ = int(target_mask.sum())
        self.n_source_ = int(source_mask.sum())
        if self.n_target_ == 0 or self.n_source_ == 0:
            raise ValueError(
                "Residual transfer requires both target-task and auxiliary-task observations."
            )

        self.source_gp.fit(X[source_mask], y[source_mask])
        source_at_target = self.source_gp.predict(X[target_mask])
        residuals = y[target_mask] - source_at_target.mean
        self.residual_gp.fit(X[target_mask], residuals)
        return self

    def _validate_prediction_tasks(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.schema.n_features:
            raise ValueError("Prediction matrix has an incompatible shape.")
        if not self._target_mask(X).all():
            raise ValueError("Residual transfer predicts only the configured target task.")
        return X

    def predict(self, X: np.ndarray) -> Posterior:
        X = self._validate_prediction_tasks(X)
        source = self.source_gp.predict(X)
        residual = self.residual_gp.predict(X)
        posterior = Posterior(
            mean=source.mean + residual.mean,
            std=np.sqrt(np.maximum(source.std**2 + residual.std**2, 1e-12)),
        )
        posterior.validate(len(X))
        return posterior

    def posterior_covariance(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_prediction_tasks(X)
        covariance = (
            self.source_gp.posterior_covariance(X)
            + self.residual_gp.posterior_covariance(X)
        )
        covariance = 0.5 * (covariance + covariance.T)
        diagonal = np.maximum(np.diag(covariance), 1e-12)
        covariance.flat[:: len(covariance) + 1] = diagonal
        return covariance
