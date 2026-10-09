from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

from .transforms import TargetTransform


@dataclass
class EmpiricalRankGaussianTransform(TargetTransform):
    """Observed-only empirical normal scores with yield-scale moments.

    Average ranks give tied yields identical latent values. The fitted knots
    define a monotone piecewise-linear bijection over the observed range, with
    linear tails clipped to the physical yield interval. Predictive moments are
    mapped back using predeclared deterministic Gauss-Hermite quadrature.
    """

    quadrature_order: int = 20
    yield_lower: float = 0.0
    yield_upper: float = 100.0
    yield_knots_: np.ndarray = field(
        default_factory=lambda: np.asarray([], dtype=float), init=False
    )
    latent_knots_: np.ndarray = field(
        default_factory=lambda: np.asarray([], dtype=float), init=False
    )
    observed_count_: int = field(default=0, init=False)

    def _require_fit(self) -> None:
        if self.observed_count_ <= 0 or len(self.yield_knots_) == 0:
            raise RuntimeError("EmpiricalRankGaussianTransform must be fitted first.")

    @staticmethod
    def _linear_map(
        values: np.ndarray,
        source: np.ndarray,
        target: np.ndarray,
    ) -> np.ndarray:
        flat = np.asarray(values, dtype=float).reshape(-1)
        if len(source) == 1:
            mapped = np.full_like(flat, target[0], dtype=float)
        else:
            mapped = np.interp(flat, source, target)
            left = flat < source[0]
            right = flat > source[-1]
            left_slope = (target[1] - target[0]) / (source[1] - source[0])
            right_slope = (target[-1] - target[-2]) / (
                source[-1] - source[-2]
            )
            mapped[left] = target[0] + left_slope * (flat[left] - source[0])
            mapped[right] = target[-1] + right_slope * (
                flat[right] - source[-1]
            )
        return mapped.reshape(np.asarray(values).shape)

    def fit(self, y: np.ndarray) -> "EmpiricalRankGaussianTransform":
        values = np.asarray(y, dtype=float).reshape(-1)
        if len(values) == 0:
            raise ValueError("Cannot fit an empirical transform without observations.")
        if not np.isfinite(values).all():
            raise ValueError("Empirical transform observations must be finite.")
        if self.quadrature_order < 2:
            raise ValueError("quadrature_order must be at least 2.")
        if not self.yield_lower < self.yield_upper:
            raise ValueError("yield_lower must be smaller than yield_upper.")

        unique, counts = np.unique(values, return_counts=True)
        cumulative_before = np.cumsum(counts) - counts
        average_ranks = cumulative_before + 0.5 * (counts + 1.0)
        probabilities = (average_ranks - 0.5) / len(values)
        probabilities = np.clip(
            probabilities,
            0.5 / len(values),
            1.0 - 0.5 / len(values),
        )
        self.yield_knots_ = unique.astype(float, copy=True)
        self.latent_knots_ = norm.ppf(probabilities)
        self.observed_count_ = len(values)
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        self._require_fit()
        values = np.asarray(y, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Values to transform must be finite.")
        return self._linear_map(values, self.yield_knots_, self.latent_knots_)

    def inverse(self, latent: np.ndarray) -> np.ndarray:
        self._require_fit()
        values = np.asarray(latent, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Latent values to invert must be finite.")
        mapped = self._linear_map(values, self.latent_knots_, self.yield_knots_)
        return np.clip(mapped, self.yield_lower, self.yield_upper)

    def inverse_mean_std(
        self,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        self._require_fit()
        latent_mean, latent_std = np.broadcast_arrays(
            np.asarray(mean, dtype=float), np.asarray(std, dtype=float)
        )
        if not np.isfinite(latent_mean).all() or not np.isfinite(latent_std).all():
            raise ValueError("Predictive latent moments must be finite.")
        if np.any(latent_std < 0.0):
            raise ValueError("Predictive latent standard deviations must be nonnegative.")

        nodes, weights = np.polynomial.hermite.hermgauss(self.quadrature_order)
        samples = latent_mean[..., None] + np.sqrt(2.0) * latent_std[..., None] * nodes
        yield_samples = self.inverse(samples)
        normalized_weights = weights / np.sqrt(np.pi)
        yield_mean = np.sum(yield_samples * normalized_weights, axis=-1)
        second_moment = np.sum(yield_samples**2 * normalized_weights, axis=-1)
        yield_variance = np.maximum(second_moment - yield_mean**2, 0.0)
        yield_std = np.maximum(np.sqrt(yield_variance), 1e-12)
        return yield_mean, yield_std

