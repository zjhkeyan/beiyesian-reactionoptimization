from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit, logit


class TargetTransform:
    """Interface for reversible scalar-target transformations."""

    def fit(self, y: np.ndarray) -> "TargetTransform":
        raise NotImplementedError

    def transform(self, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def inverse_mean_std(
        self,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


@dataclass
class StandardTargetTransform(TargetTransform):
    mean_: float = 0.0
    scale_: float = 1.0

    def fit(self, y: np.ndarray) -> "StandardTargetTransform":
        values = np.asarray(y, dtype=float).reshape(-1)
        self.mean_ = float(np.mean(values))
        scale = float(np.std(values, ddof=0))
        self.scale_ = scale if scale > 1e-12 else 1.0
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        return (np.asarray(y, dtype=float) - self.mean_) / self.scale_

    def inverse_mean_std(
        self,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray(mean, dtype=float) * self.scale_ + self.mean_,
            np.maximum(np.asarray(std, dtype=float) * self.scale_, 1e-12),
        )


@dataclass
class LogitYieldTransform(TargetTransform):
    """Logit transform for bounded yields, followed by standardization.

    Predictive standard deviations are mapped back with a delta-method
    approximation. This is conservative enough for model averaging while
    avoiding expensive Monte Carlo at every candidate and round.
    """

    epsilon: float = 0.5
    mean_: float = 0.0
    scale_: float = 1.0

    def _to_logit(self, y: np.ndarray) -> np.ndarray:
        values = np.asarray(y, dtype=float)
        probability = (values + self.epsilon) / (100.0 + 2.0 * self.epsilon)
        probability = np.clip(probability, 1e-8, 1.0 - 1e-8)
        return logit(probability)

    def fit(self, y: np.ndarray) -> "LogitYieldTransform":
        latent = self._to_logit(y)
        self.mean_ = float(np.mean(latent))
        scale = float(np.std(latent, ddof=0))
        self.scale_ = scale if scale > 1e-12 else 1.0
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        return (self._to_logit(y) - self.mean_) / self.scale_

    def inverse_mean_std(
        self,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        latent_mean = np.asarray(mean, dtype=float) * self.scale_ + self.mean_
        latent_std = np.asarray(std, dtype=float) * self.scale_
        probability = expit(latent_mean)
        yield_mean = probability * (100.0 + 2.0 * self.epsilon) - self.epsilon
        derivative = (100.0 + 2.0 * self.epsilon) * probability * (1.0 - probability)
        yield_std = np.maximum(np.abs(derivative) * latent_std, 1e-12)
        return np.clip(yield_mean, 0.0, 100.0), yield_std


def make_target_transform(name: str) -> TargetTransform:
    normalized = name.strip().lower()
    if normalized in {"standard", "raw", "zscore"}:
        return StandardTargetTransform()
    if normalized in {"logit", "logit_yield", "bounded"}:
        return LogitYieldTransform()
    raise ValueError(f"Unknown target transform: {name!r}")
