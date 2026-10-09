"""Acquisition policies and predeclared selection for train-only studies."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bgolearn_rxn.acquisition import (
    probability_of_improvement,
    stable_log_expected_improvement,
    upper_confidence_bound,
)


POLICIES = (
    "posterior_mean",
    "mean_then_pi",
    "mean_then_logei",
    "mean_then_ucb_010_to_075",
    "mean_then_ucb_025_to_150",
    "mean_then_correlated_thompson",
)
BASELINE_POLICY = "mean_then_logei"


def correlated_thompson_sample(
    mean: np.ndarray,
    covariance: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    """Draw one deterministic joint Gaussian sample with PSD stabilization."""
    mean = np.asarray(mean, dtype=float).reshape(-1)
    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape != (len(mean), len(mean)):
        raise ValueError("Covariance must be square and align with the mean.")
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 1e-10)
    factor = eigenvectors @ np.diag(np.sqrt(eigenvalues))
    draw = np.random.default_rng(int(seed)).standard_normal(len(mean))
    return mean + factor @ draw


def acquisition_scores(
    policy: str,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    best_f: float,
    step: int,
    budget: int,
    covariance: np.ndarray | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Return maximization scores for one predeclared acquisition geometry."""
    mean = np.asarray(mean, dtype=float).reshape(-1)
    std = np.asarray(std, dtype=float).reshape(-1)
    if mean.shape != std.shape or step < 1 or budget < 1 or step > budget:
        raise ValueError("Invalid posterior shapes or campaign step.")
    if policy not in POLICIES:
        raise ValueError(f"Unknown acquisition policy: {policy}")
    if step == 1:
        return mean.copy()
    progress = 0.0 if budget == 1 else (step - 1) / (budget - 1)
    xi = 0.05 * (1.0 - progress)
    if policy == "posterior_mean":
        return mean.copy()
    if policy == "mean_then_pi":
        return probability_of_improvement(mean, std, best_f, xi=xi)
    if policy == "mean_then_logei":
        return stable_log_expected_improvement(mean, std, best_f, xi=xi)
    if policy == "mean_then_ucb_010_to_075":
        return upper_confidence_bound(mean, std, beta=0.10 + 0.65 * progress)
    if policy == "mean_then_ucb_025_to_150":
        return upper_confidence_bound(mean, std, beta=0.25 + 1.25 * progress)
    if policy == "mean_then_correlated_thompson":
        if covariance is None:
            raise ValueError("Correlated Thompson sampling requires covariance.")
        return correlated_thompson_sample(mean, covariance, seed=seed)
    raise ValueError(f"Unknown acquisition policy: {policy}")


def select_from_screening(summary: pd.DataFrame, dataset: str) -> dict[str, object]:
    """Apply the predeclared safety-constrained screen-to-confirm rule.

    Eligible policies must be within 0.25 rounds of baseline penalized t95 and
    within 0.5 yield points of baseline initial-round and best-found metrics.
    Among them, retain policies within 0.1 AUC of the maximum, then choose
    lower t95, higher best-found, and finally the fixed policy order.
    """
    section = summary[summary["dataset"].eq(dataset)]
    duplicate = section.duplicated(["policy", "metric"], keep=False)
    if duplicate.any():
        raise ValueError(
            "Screening selection requires one aggregate row per dataset/policy/metric."
        )
    table = section.pivot(index="policy", columns="metric", values="mean")
    baseline = table.loc[BASELINE_POLICY]
    eligible = table[
        (table["round_to_95_global_best"] <= baseline["round_to_95_global_best"] + 0.25)
        & (table["best_found"] >= baseline["best_found"] - 0.5)
        & (
            table["initial_round_found_best"]
            >= baseline["initial_round_found_best"] - 0.5
        )
    ].copy()
    if eligible.empty:
        selected = BASELINE_POLICY
    else:
        max_auc = float(eligible["AUC_best_so_far"].max())
        finalists = eligible[eligible["AUC_best_so_far"] >= max_auc - 0.1].copy()
        order = {name: index for index, name in enumerate(POLICIES)}
        finalists["policy_order"] = [order[name] for name in finalists.index]
        selected = str(
            finalists.sort_values(
                ["round_to_95_global_best", "best_found", "policy_order"],
                ascending=[True, False, True],
            ).index[0]
        )
    return {
        "dataset": dataset,
        "selected_policy": selected,
        "baseline_policy": BASELINE_POLICY,
        "rule": {
            "max_t95_increase": 0.25,
            "max_best_found_decrease": 0.5,
            "max_initial_round_decrease": 0.5,
            "auc_tie_band": 0.1,
            "tie_break": ["lower_t95", "higher_best_found", "fixed_policy_order"],
        },
    }
