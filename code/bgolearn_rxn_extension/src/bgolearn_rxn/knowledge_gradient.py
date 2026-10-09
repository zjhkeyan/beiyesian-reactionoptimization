from __future__ import annotations

import numpy as np
from numpy.polynomial.hermite import hermgauss


def finite_pool_knowledge_gradient(
    mean: np.ndarray,
    covariance: np.ndarray,
    *,
    observation_variance: float | np.ndarray,
    best_f: float | None = None,
    quadrature_order: int = 32,
) -> np.ndarray:
    """Deterministic one-step knowledge gradient for a finite candidate pool.

    For candidate ``i`` and a standardized future observation ``Z``, the
    updated posterior mean is ``mean + covariance[:, i] / predictive_std_i * Z``.
    The score is the expected increase in the largest posterior mean. When
    ``best_f`` is supplied, the observed incumbent is included as a constant
    terminal alternative.

    The covariance must describe latent function values. Observation variance
    is added only to the sampled candidate's predictive variance.
    """

    mean = np.asarray(mean, dtype=float).reshape(-1)
    covariance = np.asarray(covariance, dtype=float)
    n_candidates = len(mean)
    if n_candidates == 0:
        raise ValueError("Knowledge gradient requires at least one candidate.")
    if covariance.shape != (n_candidates, n_candidates):
        raise ValueError("Covariance must be square and align with mean.")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(covariance)):
        raise FloatingPointError("Knowledge-gradient inputs must be finite.")
    if quadrature_order < 2:
        raise ValueError("quadrature_order must be at least 2.")
    if not np.allclose(covariance, covariance.T, atol=1e-8, rtol=1e-8):
        raise ValueError("Covariance must be symmetric.")
    covariance = 0.5 * (covariance + covariance.T)
    diagonal = np.diag(covariance)
    if np.any(diagonal < -1e-8):
        raise ValueError("Covariance diagonal must be non-negative.")

    noise = np.asarray(observation_variance, dtype=float)
    if noise.ndim == 0:
        noise = np.full(n_candidates, float(noise))
    else:
        noise = noise.reshape(-1)
    if noise.shape != (n_candidates,) or not np.all(np.isfinite(noise)):
        raise ValueError("observation_variance must be scalar or align with mean.")
    if np.any(noise < 0.0):
        raise ValueError("observation_variance must be non-negative.")

    incumbent = -np.inf if best_f is None else float(best_f)
    if not np.isfinite(incumbent) and best_f is not None:
        raise ValueError("best_f must be finite when supplied.")
    current_value = max(incumbent, float(np.max(mean)))
    nodes, weights = hermgauss(int(quadrature_order))
    standard_normal_nodes = np.sqrt(2.0) * nodes
    standard_normal_weights = weights / np.sqrt(np.pi)

    predictive_std = np.sqrt(np.maximum(diagonal + noise, 1e-15))
    scores = np.empty(n_candidates, dtype=float)
    for candidate in range(n_candidates):
        slopes = covariance[:, candidate] / predictive_std[candidate]
        updated = mean[:, None] + slopes[:, None] * standard_normal_nodes[None, :]
        terminal = np.max(updated, axis=0)
        if best_f is not None:
            terminal = np.maximum(terminal, incumbent)
        expected_value = float(standard_normal_weights @ terminal)
        scores[candidate] = max(expected_value - current_value, 0.0)
    return scores
