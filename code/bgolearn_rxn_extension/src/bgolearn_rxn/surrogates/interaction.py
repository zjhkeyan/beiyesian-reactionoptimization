from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler

from ..config import InteractionModelConfig
from ..schemas import FeatureSchema, Posterior
from ..transforms import make_target_transform


@dataclass
class _RoleProjector:
    role: str
    pca: PCA | None
    mean_: np.ndarray

    def transform(self, block: np.ndarray) -> np.ndarray:
        centered = np.asarray(block, dtype=float) - self.mean_
        if self.pca is None:
            return centered
        return self.pca.transform(np.asarray(block, dtype=float))


class BayesianLowRankInteractionSurrogate:
    """Bayesian linear model over frozen low-rank role interactions.

    Each reaction role is represented by exact identity one-hot features plus a
    PCA projection learned *without labels* from the complete initial design
    (initial observations + unlabeled candidate pool). Selected pairwise outer
    products form a compact tensor/factorization representation. BayesianRidge
    supplies shrinkage and a closed-form predictive uncertainty estimate.

    This is the robust default factorization implementation. A neural
    factorization machine can replace it behind the same interface after it has
    been validated on unrelated proxy tasks.
    """

    name = "interaction"

    def __init__(
        self,
        schema: FeatureSchema,
        config: InteractionModelConfig,
        reference_X: np.ndarray,
        *,
        seed: int = 0,
    ) -> None:
        self.schema = schema
        self.config = config
        self.seed = int(seed)
        self.reference_X = np.asarray(reference_X, dtype=float)
        if self.reference_X.ndim != 2 or self.reference_X.shape[1] != schema.n_features:
            raise ValueError("reference_X has an incompatible shape.")

        for pair in config.interaction_pairs:
            if pair[0] not in schema.roles or pair[1] not in schema.roles:
                raise ValueError(f"Invalid interaction-model pair: {pair}")

        self.target_transform = make_target_transform(config.target_transform)
        self.projectors: dict[str, _RoleProjector] = {}
        self.design_scaler = StandardScaler(with_mean=True, with_std=True)
        self.model: BayesianRidge | None = None
        self._fit_projectors()
        reference_design = self._raw_design(self.reference_X)
        self.design_scaler.fit(reference_design)

    def _role_rank(self, role: str, block: np.ndarray) -> int:
        requested = int(self.config.role_ranks.get(role, self.config.default_rank))
        unique_rows = max(len(np.unique(block, axis=0)), 1)
        return max(1, min(requested, block.shape[1], unique_rows - 1 if unique_rows > 1 else 1))

    def _fit_projectors(self) -> None:
        for role in self.schema.roles:
            block = self.reference_X[:, self.schema.role_slice(role)]
            mean = np.mean(block, axis=0, keepdims=True)
            rank = self._role_rank(role, block)
            if block.shape[1] <= rank or np.allclose(np.var(block, axis=0), 0.0):
                projector = _RoleProjector(role=role, pca=None, mean_=mean)
            else:
                pca = PCA(n_components=rank, svd_solver="full", random_state=self.seed)
                pca.fit(block)
                projector = _RoleProjector(role=role, pca=pca, mean_=mean)
            self.projectors[role] = projector

    def _role_latents(self, X: np.ndarray) -> dict[str, np.ndarray]:
        latents: dict[str, np.ndarray] = {}
        for role in self.schema.roles:
            block = X[:, self.schema.role_slice(role)]
            latents[role] = self.projectors[role].transform(block)
        return latents

    def _raw_design(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        latents = self._role_latents(X)
        blocks: list[np.ndarray] = []

        # Exact identity main effects protect within-grid interpolation, while
        # the latent block carries chemical/name-rule similarity.
        for role in self.schema.roles:
            blocks.append(X[:, self.schema.identity_slice(role)])
            blocks.append(latents[role])

        for role_a, role_b in self.config.interaction_pairs:
            latent_a = latents[role_a]
            latent_b = latents[role_b]
            outer = np.einsum("ni,nj->nij", latent_a, latent_b, optimize=True)
            blocks.append(
                outer.reshape(len(X), -1) * float(self.config.interaction_scale)
            )

        return np.hstack(blocks).astype(float, copy=False)

    def _design(self, X: np.ndarray) -> np.ndarray:
        raw = self._raw_design(X)
        transformed = self.design_scaler.transform(raw)
        return np.nan_to_num(transformed, nan=0.0, posinf=0.0, neginf=0.0)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BayesianLowRankInteractionSurrogate":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError("X must be 2D and align with y.")
        self.target_transform.fit(y)
        transformed_y = self.target_transform.transform(y)
        self.model = BayesianRidge(
            max_iter=500,
            tol=1e-6,
            alpha_1=self.config.alpha_1,
            alpha_2=self.config.alpha_2,
            lambda_1=self.config.lambda_1,
            lambda_2=self.config.lambda_2,
            compute_score=True,
            fit_intercept=True,
        )
        self.model.fit(self._design(X), transformed_y)
        return self

    def predict(self, X: np.ndarray) -> Posterior:
        if self.model is None:
            raise RuntimeError("Interaction surrogate must be fitted before prediction.")
        latent_mean, latent_std = self.model.predict(self._design(X), return_std=True)
        mean, std = self.target_transform.inverse_mean_std(latent_mean, latent_std)
        posterior = Posterior(mean=mean, std=std)
        posterior.validate(len(X))
        return posterior

    def posterior_covariance(self, X: np.ndarray) -> np.ndarray:
        posterior = self.predict(X)
        return np.diag(posterior.std**2)
