from __future__ import annotations

import numpy as np

from ..config import GPConfig
from ..rank_gaussian_transform import EmpiricalRankGaussianTransform
from ..schemas import FeatureSchema
from .component_ard_gp import ComponentARDStructuredGP


class RankGaussianComponentARDStructuredGP(ComponentARDStructuredGP):
    """Component-ARD GP fitted to observed empirical normal scores."""

    name = "rank_gaussian_component_ard_gp"

    def __init__(
        self, schema: FeatureSchema, config: GPConfig, *, seed: int = 0
    ) -> None:
        super().__init__(schema=schema, config=config, seed=seed)
        self.target_transform = EmpiricalRankGaussianTransform(quadrature_order=20)

    def posterior_covariance(self, X: np.ndarray) -> np.ndarray:
        latent_mean, latent_covariance = self._latent_posterior(
            X, full_covariance=True
        )
        latent_std = np.sqrt(np.maximum(np.diag(latent_covariance), 1e-12))
        _, yield_std = self.target_transform.inverse_mean_std(
            latent_mean, latent_std
        )
        denominator = np.outer(latent_std, latent_std)
        correlation = np.divide(
            latent_covariance,
            denominator,
            out=np.eye(len(X), dtype=float),
            where=denominator > 1e-15,
        )
        correlation = 0.5 * (correlation + correlation.T)
        correlation.flat[:: len(correlation) + 1] = 1.0
        covariance = correlation * np.outer(yield_std, yield_std)
        covariance = 0.5 * (covariance + covariance.T)
        covariance.flat[:: len(covariance) + 1] = yield_std**2
        return covariance

