from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from ..config import GPConfig
from ..schemas import FeatureSchema
from .anova_gp import StructuredANOVAGP
from .component_ard_gp import ComponentARDLowRankTaskGP


def empirical_rank_precision(y: np.ndarray, strength: float) -> np.ndarray:
    """Return deterministic mean-one precision weights from observed ranks."""

    values = np.asarray(y, dtype=float).reshape(-1)
    strength = float(strength)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Observed responses must be nonempty and finite.")
    if not np.isfinite(strength) or strength < 0.0:
        raise ValueError("Rank-weight strength must be finite and non-negative.")
    if len(values) == 1 or strength == 0.0:
        return np.ones(len(values), dtype=float)
    percentile = (rankdata(values, method="average") - 1.0) / (len(values) - 1.0)
    log_precision = strength * (percentile - float(np.mean(percentile)))
    precision = np.exp(log_precision)
    precision /= float(np.mean(precision))
    return precision


class _RankWeightedObservationPrecisionMixin:
    """Use observed-response ranks to scale the default training-noise hook."""

    rank_strength: float
    training_precision_: np.ndarray
    effective_training_noise_: np.ndarray

    def _initialize_rank_weighting(self, strength: float) -> None:
        strength = float(strength)
        if not np.isfinite(strength) or strength < 0.0:
            raise ValueError("Rank-weight strength must be finite and non-negative.")
        self.rank_strength = strength
        self.training_precision_ = np.empty(0, dtype=float)
        self.effective_training_noise_ = np.empty(0, dtype=float)
        self._rank_training_X = np.empty((0, 0), dtype=float)

    def _training_noise_variance(
        self, theta: np.ndarray, X: np.ndarray
    ) -> np.ndarray:
        base = super()._training_noise_variance(theta, X)
        X = np.asarray(X, dtype=float)
        if self.training_precision_.shape != (len(X),):
            raise RuntimeError("Rank precision is not aligned with training rows.")
        if self._rank_training_X.shape != X.shape or not np.allclose(
            self._rank_training_X, X, atol=0.0, rtol=0.0
        ):
            raise RuntimeError("Rank precision may only be used on its fitted rows.")
        return np.asarray(base, dtype=float) / self.training_precision_

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        self.training_precision_ = empirical_rank_precision(y, self.rank_strength)
        self._rank_training_X = X.copy()
        super().fit(X, y)
        if self.theta_ is None:
            raise RuntimeError("Rank-weighted GP did not retain fitted parameters.")
        variance = self._training_noise_variance(self.theta_, X)
        scale = float(getattr(self.target_transform, "scale_", 1.0))
        self.effective_training_noise_ = np.sqrt(variance) * scale
        return self

    def rank_weight_diagnostics(self) -> dict[str, float | int]:
        if self.training_precision_.size == 0:
            raise RuntimeError("Rank-weight diagnostics requested before fitting.")
        return {
            "strength": self.rank_strength,
            "n_observed": int(len(self.training_precision_)),
            "precision_mean": float(np.mean(self.training_precision_)),
            "precision_min": float(np.min(self.training_precision_)),
            "precision_max": float(np.max(self.training_precision_)),
            "effective_noise_min": float(np.min(self.effective_training_noise_)),
            "effective_noise_max": float(np.max(self.effective_training_noise_)),
        }


class RankWeightedStructuredGP(
    _RankWeightedObservationPrecisionMixin, StructuredANOVAGP
):
    """Opt-in structured GP with deterministic observed-rank precision."""

    name = "rank_weighted_gp"

    def __init__(
        self,
        schema: FeatureSchema,
        config: GPConfig,
        *,
        strength: float,
        seed: int = 0,
    ) -> None:
        StructuredANOVAGP.__init__(self, schema=schema, config=config, seed=seed)
        self._initialize_rank_weighting(strength)


class RankWeightedComponentARDLowRankTaskGP(
    _RankWeightedObservationPrecisionMixin, ComponentARDLowRankTaskGP
):
    """Opt-in component-ARD task GP with observed-rank precision."""

    name = "rank_weighted_component_ard_low_rank_task_gp"

    def __init__(
        self,
        schema: FeatureSchema,
        config: GPConfig,
        *,
        strength: float,
        seed: int = 0,
    ) -> None:
        ComponentARDLowRankTaskGP.__init__(
            self, schema=schema, config=config, seed=seed
        )
        self._initialize_rank_weighting(strength)
