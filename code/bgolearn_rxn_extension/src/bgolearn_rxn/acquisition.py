from __future__ import annotations

import numpy as np
from scipy.special import log_ndtr
from scipy.stats import norm


_LOG_2PI_HALF = 0.5 * np.log(2.0 * np.pi)


def _log1mexp(log_x: np.ndarray) -> np.ndarray:
    """Stable ``log(1 - exp(log_x))`` for ``log_x <= 0``."""

    x = np.minimum(np.asarray(log_x, dtype=float), -np.finfo(float).eps)
    threshold = -np.log(2.0)
    return np.where(
        x < threshold,
        np.log1p(-np.exp(x)),
        np.log(-np.expm1(x)),
    )


def stable_log_expected_improvement(
    mean: np.ndarray,
    std: np.ndarray,
    best_f: float,
    *,
    xi: float = 0.0,
) -> np.ndarray:
    """Numerically stable log expected improvement for maximization.

    For a Gaussian posterior, ``EI = sigma * (phi(z) + z * Phi(z))`` with
    ``z = (mean - best_f - xi) / sigma``. The negative-z branch is evaluated as
    a stable log difference; an asymptotic expression is used in the extreme
    left tail where ordinary EI underflows.
    """

    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    if mean.shape != std.shape:
        raise ValueError("mean and std must have the same shape.")
    sigma = np.maximum(std, 1e-15)
    improvement = mean - float(best_f) - float(xi)
    z = improvement / sigma

    log_phi = -0.5 * z**2 - _LOG_2PI_HALF
    log_phi_cdf = log_ndtr(z)
    log_h = np.empty_like(z)

    positive = z >= 0.0
    zero = positive & np.isclose(z, 0.0, atol=0.0)
    strictly_positive = positive & ~zero
    log_h[zero] = log_phi[zero]
    if np.any(strictly_positive):
        log_second = np.log(z[strictly_positive]) + log_phi_cdf[strictly_positive]
        log_h[strictly_positive] = np.logaddexp(
            log_phi[strictly_positive], log_second
        )

    moderate_negative = (z < 0.0) & (z > -10.0)
    if np.any(moderate_negative):
        log_ratio = (
            np.log(-z[moderate_negative])
            + log_phi_cdf[moderate_negative]
            - log_phi[moderate_negative]
        )
        log_h[moderate_negative] = (
            log_phi[moderate_negative] + _log1mexp(log_ratio)
        )

    extreme_negative = z <= -10.0
    if np.any(extreme_negative):
        a = -z[extreme_negative]
        correction = np.maximum(1.0 - 3.0 / a**2 + 15.0 / a**4, 1e-12)
        log_h[extreme_negative] = (
            log_phi[extreme_negative] - 2.0 * np.log(a) + np.log(correction)
        )

    result = np.log(sigma) + log_h

    nearly_deterministic = std <= 1e-14
    if np.any(nearly_deterministic):
        deterministic_improvement = improvement[nearly_deterministic]
        result[nearly_deterministic] = np.where(
            deterministic_improvement > 0.0,
            np.log(deterministic_improvement),
            -np.inf,
        )
    return result


def expected_improvement(
    mean: np.ndarray,
    std: np.ndarray,
    best_f: float,
    *,
    xi: float = 0.0,
) -> np.ndarray:
    return np.exp(stable_log_expected_improvement(mean, std, best_f, xi=xi))


def probability_of_improvement(
    mean: np.ndarray,
    std: np.ndarray,
    best_f: float,
    *,
    xi: float = 0.0,
) -> np.ndarray:
    sigma = np.maximum(np.asarray(std, dtype=float), 1e-15)
    z = (np.asarray(mean, dtype=float) - float(best_f) - float(xi)) / sigma
    return norm.cdf(z)


def upper_confidence_bound(
    mean: np.ndarray,
    std: np.ndarray,
    *,
    beta: float,
) -> np.ndarray:
    return np.asarray(mean, dtype=float) + float(beta) * np.asarray(std, dtype=float)


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values)
    replacement = np.min(values[finite]) - 10.0
    clean = np.where(finite, values, replacement)
    scale = float(np.std(clean))
    if scale <= 1e-12:
        return np.zeros_like(clean)
    return (clean - float(np.mean(clean))) / scale
