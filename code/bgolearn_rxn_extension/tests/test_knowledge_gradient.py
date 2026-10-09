from __future__ import annotations

import numpy as np
import pytest

from bgolearn_rxn.knowledge_gradient import finite_pool_knowledge_gradient


def test_knowledge_gradient_is_nonnegative_and_deterministic() -> None:
    mean = np.array([0.4, 0.1, -0.2])
    covariance = np.array(
        [
            [1.0, 0.7, 0.1],
            [0.7, 0.8, 0.2],
            [0.1, 0.2, 0.5],
        ]
    )
    first = finite_pool_knowledge_gradient(
        mean, covariance, observation_variance=0.05, best_f=0.3
    )
    second = finite_pool_knowledge_gradient(
        mean, covariance, observation_variance=0.05, best_f=0.3
    )
    np.testing.assert_allclose(first, second, atol=0.0, rtol=0.0)
    assert np.all(np.isfinite(first))
    assert np.all(first >= 0.0)


def test_knowledge_gradient_uses_cross_candidate_covariance() -> None:
    mean = np.array([0.6, 0.5, -0.5])
    diagonal = np.diag([0.2, 0.2, 0.2])
    correlated = diagonal.copy()
    correlated[0, 2] = correlated[2, 0] = -0.19
    independent_scores = finite_pool_knowledge_gradient(
        mean, diagonal, observation_variance=0.01, best_f=0.55
    )
    correlated_scores = finite_pool_knowledge_gradient(
        mean, correlated, observation_variance=0.01, best_f=0.55
    )
    assert not np.allclose(independent_scores, correlated_scores)
    assert correlated_scores[2] > independent_scores[2]


def test_knowledge_gradient_is_permutation_equivariant() -> None:
    mean = np.array([0.2, 0.9, 0.4])
    covariance = np.array(
        [
            [0.8, 0.1, 0.3],
            [0.1, 0.6, 0.2],
            [0.3, 0.2, 0.9],
        ]
    )
    order = np.array([2, 0, 1])
    original = finite_pool_knowledge_gradient(
        mean, covariance, observation_variance=0.03, best_f=0.7
    )
    permuted = finite_pool_knowledge_gradient(
        mean[order],
        covariance[np.ix_(order, order)],
        observation_variance=0.03,
        best_f=0.7,
    )
    np.testing.assert_allclose(permuted, original[order], atol=1e-12)


def test_knowledge_gradient_rejects_invalid_covariance() -> None:
    with pytest.raises(ValueError, match="symmetric"):
        finite_pool_knowledge_gradient(
            np.zeros(2),
            np.array([[1.0, 0.5], [0.0, 1.0]]),
            observation_variance=0.1,
        )
