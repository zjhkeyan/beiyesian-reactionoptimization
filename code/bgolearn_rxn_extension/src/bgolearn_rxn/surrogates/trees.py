from __future__ import annotations

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from ..config import TreeConfig
from ..schemas import Posterior
from ..transforms import make_target_transform


class BootstrapExtraTreesSurrogate:
    """Non-smooth ensemble hedge with explicit per-tree uncertainty."""

    name = "trees"

    def __init__(self, config: TreeConfig, *, seed: int = 0) -> None:
        self.config = config
        self.seed = int(seed)
        self.target_transform = make_target_transform(config.target_transform)
        self.model: ExtraTreesRegressor | None = None
        self.residual_std_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BootstrapExtraTreesSurrogate":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        self.target_transform.fit(y)
        transformed_y = self.target_transform.transform(y)
        self.model = ExtraTreesRegressor(
            n_estimators=self.config.n_estimators,
            max_features=self.config.max_features,
            min_samples_leaf=self.config.min_samples_leaf,
            max_depth=self.config.max_depth,
            bootstrap=True,
            random_state=self.seed,
            n_jobs=1,
        )
        self.model.fit(X, transformed_y)
        fitted = self.model.predict(X)
        residual = transformed_y - fitted
        self.residual_std_ = float(np.sqrt(np.mean(residual**2)))
        return self

    def predict(self, X: np.ndarray) -> Posterior:
        if self.model is None:
            raise RuntimeError("Tree surrogate must be fitted before prediction.")
        X = np.asarray(X, dtype=float)
        per_tree = np.vstack([tree.predict(X) for tree in self.model.estimators_])
        latent_mean = np.mean(per_tree, axis=0)
        epistemic = np.std(per_tree, axis=0, ddof=0)
        noise_floor = max(
            self.residual_std_ * self.config.residual_noise_fraction,
            1e-3,
        )
        latent_std = np.sqrt(epistemic**2 + noise_floor**2)
        mean, std = self.target_transform.inverse_mean_std(latent_mean, latent_std)
        posterior = Posterior(mean=mean, std=std)
        posterior.validate(len(X))
        return posterior

    def posterior_covariance(self, X: np.ndarray) -> np.ndarray:
        posterior = self.predict(X)
        return np.diag(posterior.std**2)
