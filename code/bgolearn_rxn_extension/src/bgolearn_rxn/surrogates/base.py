from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from ..schemas import Posterior


@runtime_checkable
class ProbabilisticSurrogate(Protocol):
    """Minimal probabilistic surrogate contract used by the extension."""

    name: str

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ProbabilisticSurrogate":
        ...

    def predict(self, X: np.ndarray) -> Posterior:
        ...

    def posterior_covariance(self, X: np.ndarray) -> np.ndarray:
        ...
