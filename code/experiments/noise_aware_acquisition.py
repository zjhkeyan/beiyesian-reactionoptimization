"""Deterministic Bgolearn-style noise-aware acquisition formulas."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from bgolearn_rxn.acquisition import stable_log_expected_improvement


BASELINE_POLICY = "mean_then_logei"
POLICIES = (
    BASELINE_POLICY,
    "mean_then_plugin_ei",
    "mean_then_augmented_ei",
    "mean_then_eqi_075",
    "mean_then_reinterpolation_ei",
)
EQI_BETA = 0.75
AEI_ALPHA = 1.0


def _stable_log_ei(mean: np.ndarray, std: np.ndarray, best_f: float, *, xi: float = 0.0) -> np.ndarray:
    # The core routine deliberately computes both np.where branches for exact
    # zero variance. Suppress the harmless log warning while retaining -inf.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return stable_log_expected_improvement(mean, std, best_f, xi=xi)


def observation_noise_std(model) -> float:
    """Return the fitted homoskedastic observation noise on the yield scale."""

    if model.theta_ is None:
        raise RuntimeError("Observation noise requires a fitted GP.")
    transformed_noise = float(model._unpack(model.theta_)[2])
    scale = float(getattr(model.target_transform, "scale_", 1.0))
    return max(transformed_noise * scale, 0.0)


def _validate(
    mean: np.ndarray,
    std: np.ndarray,
    train_mean: np.ndarray,
    train_std: np.ndarray,
    target_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.asarray(mean, dtype=float).reshape(-1)
    std = np.asarray(std, dtype=float).reshape(-1)
    train_mean = np.asarray(train_mean, dtype=float).reshape(-1)
    train_std = np.asarray(train_std, dtype=float).reshape(-1)
    target_mask = np.asarray(target_mask, dtype=bool).reshape(-1)
    if mean.shape != std.shape:
        raise ValueError("Candidate mean and standard deviation must align.")
    if train_mean.shape != train_std.shape or train_mean.shape != target_mask.shape:
        raise ValueError("Training posterior and target mask must align.")
    if not target_mask.any():
        raise ValueError("At least one target-task observation is required.")
    if np.any(std < 0.0) or np.any(train_std < 0.0):
        raise ValueError("Posterior standard deviations must be nonnegative.")
    return mean, std, train_mean, train_std, target_mask


def noise_aware_scores(
    policy: str,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    observed_best: float,
    train_mean: np.ndarray,
    train_std: np.ndarray,
    target_mask: np.ndarray,
    observation_noise: float,
    step: int,
    budget: int,
    reinterpolated_mean: np.ndarray | None = None,
    reinterpolated_std: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float | str | None]]:
    """Return stable log-scale maximization scores and formula metadata.

    The formulas follow Bgolearn's maximization implementation, while using
    stable LogEI arithmetic and deterministic handling of fitted posteriors.
    """

    if policy not in POLICIES:
        raise ValueError(f"Unknown acquisition policy: {policy}")
    if step < 1 or budget < 1 or step > budget:
        raise ValueError("Invalid campaign step or budget.")
    mean, std, train_mean, train_std, target_mask = _validate(
        mean, std, train_mean, train_std, target_mask
    )
    tau = float(observation_noise)
    if not np.isfinite(tau) or tau < 0.0:
        raise ValueError("Observation noise must be finite and nonnegative.")
    if step == 1:
        return mean.copy(), {
            "formula": "posterior_mean",
            "incumbent": None,
            "observation_noise_std": tau,
            "eqi_beta": None,
        }

    if policy == BASELINE_POLICY:
        progress = (step - 1) / (budget - 1) if budget > 1 else 0.0
        xi = 0.05 * (1.0 - progress)
        scores = _stable_log_ei(mean, std, observed_best, xi=xi)
        return scores, {
            "formula": "stable_log_ei_observed_incumbent",
            "incumbent": float(observed_best),
            "observation_noise_std": tau,
            "eqi_beta": None,
        }

    plugin_incumbent = float(np.max(train_mean[target_mask]))
    if policy == "mean_then_plugin_ei":
        scores = _stable_log_ei(mean, std, plugin_incumbent)
        formula = "plugin_ei_smoothed_incumbent"
        incumbent = plugin_incumbent
    elif policy == "mean_then_augmented_ei":
        target_indices = np.flatnonzero(target_mask)
        proposal = train_mean[target_mask] + AEI_ALPHA * train_std[target_mask]
        incumbent = float(train_mean[target_indices[int(np.argmax(proposal))]])
        log_ei = _stable_log_ei(mean, std, incumbent)
        denominator = np.sqrt(std**2 + tau**2)
        ratio = np.divide(tau, denominator, out=np.ones_like(std), where=denominator > 0.0)
        penalty = np.maximum(1.0 - np.clip(ratio, 0.0, 1.0), 0.0)
        with np.errstate(divide="ignore"):
            scores = log_ei + np.log(penalty)
        formula = "augmented_ei_fitted_noise_alpha_1"
    elif policy == "mean_then_eqi_075":
        z_beta = float(norm.ppf(EQI_BETA))
        incumbent = float(np.max(train_mean[target_mask] + z_beta * train_std[target_mask]))
        variance = std**2
        denominator = np.sqrt(variance + tau**2)
        adjusted_mean = mean + z_beta * np.sqrt(
            np.divide(tau**2 * variance, variance + tau**2, out=np.zeros_like(variance), where=(variance + tau**2) > 0.0)
        )
        adjusted_std = np.divide(
            variance,
            denominator,
            out=np.zeros_like(variance),
            where=denominator > 0.0,
        )
        scores = _stable_log_ei(adjusted_mean, adjusted_std, incumbent)
        formula = "expected_quantile_improvement_beta_0.75_fitted_noise"
    else:
        if reinterpolated_mean is None or reinterpolated_std is None:
            raise ValueError("Reinterpolation EI requires its refitted candidate posterior.")
        re_mean = np.asarray(reinterpolated_mean, dtype=float).reshape(-1)
        re_std = np.asarray(reinterpolated_std, dtype=float).reshape(-1)
        if re_mean.shape != mean.shape or re_std.shape != std.shape:
            raise ValueError("Reinterpolated posterior must align with candidates.")
        incumbent = plugin_incumbent
        scores = _stable_log_ei(re_mean, re_std, incumbent)
        formula = "reinterpolation_ei_smoothed_refit"

    return scores, {
        "formula": formula,
        "incumbent": incumbent,
        "observation_noise_std": tau,
        "eqi_beta": EQI_BETA if policy == "mean_then_eqi_075" else None,
    }


def score_diagnostics(scores: np.ndarray) -> dict[str, int | float | bool]:
    """Record whether numerical edge handling could determine a ranking."""

    values = np.asarray(scores, dtype=float).reshape(-1)
    finite = np.isfinite(values)
    if not finite.any():
        return {
            "finite_count": 0,
            "nonfinite_count": len(values),
            "finite_span": 0.0,
            "all_finite_scores_tied": True,
        }
    finite_values = values[finite]
    span = float(np.max(finite_values) - np.min(finite_values))
    return {
        "finite_count": int(finite.sum()),
        "nonfinite_count": int((~finite).sum()),
        "finite_span": span,
        "all_finite_scores_tied": bool(span <= 1e-12),
    }
