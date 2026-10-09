"""Skeleton for replacing one ensemble component."""

from __future__ import annotations

import numpy as np

from bgolearn_rxn.schemas import Posterior


class MyCalibratedReactionModel:
    name = "my_model"

    def __init__(self, *, seed: int = 0) -> None:
        self.seed = int(seed)
        self._is_fit = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MyCalibratedReactionModel":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError("X and y do not align.")
        # Fit only to currently observed labels. Never access the oracle table.
        self._is_fit = True
        return self

    def predict(self, X: np.ndarray) -> Posterior:
        if not self._is_fit:
            raise RuntimeError("fit must be called before predict.")
        X = np.asarray(X, dtype=float)
        # Replace with a calibrated yield-scale posterior.
        mean = np.full(len(X), 50.0)
        std = np.full(len(X), 20.0)
        posterior = Posterior(mean=mean, std=std)
        posterior.validate(len(X))
        return posterior

    def posterior_covariance(self, X: np.ndarray) -> np.ndarray:
        posterior = self.predict(X)
        return np.diag(posterior.std**2)
