from __future__ import annotations

import numpy as np
from scipy.stats import norm

from bgolearn_rxn.acquisition import stable_log_expected_improvement


def test_stable_log_ei_matches_direct_formula_in_regular_region() -> None:
    mean = np.array([-1.0, 0.0, 1.0, 3.0])
    std = np.array([0.8, 1.0, 1.2, 0.4])
    best = 0.5
    z = (mean - best) / std
    direct = std * (norm.pdf(z) + z * norm.cdf(z))
    actual = np.exp(stable_log_expected_improvement(mean, std, best))
    np.testing.assert_allclose(actual, direct, rtol=1e-9, atol=1e-14)


def test_stable_log_ei_remains_finite_in_extreme_left_tail() -> None:
    values = stable_log_expected_improvement(
        np.array([-1000.0]), np.array([1.0]), best_f=0.0
    )
    assert np.isfinite(values[0])
    assert values[0] < -1000.0
