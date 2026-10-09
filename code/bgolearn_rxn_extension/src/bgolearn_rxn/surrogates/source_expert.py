from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from ..config import GPConfig
from ..schemas import FeatureSchema, Posterior
from .anova_gp import StructuredANOVAGP


class SourceExpertTransferGP:
    """Mixture of target-plus-one-source GPs gated by target evidence."""

    name = "source_expert"

    def __init__(
        self,
        schema: FeatureSchema,
        config: GPConfig,
        *,
        target_task_identity: np.ndarray,
        seed: int = 0,
        weight_floor: float = 0.05,
        temperature: float = 2.0,
        predictive_std_floor: float = 2.0,
    ) -> None:
        if schema.task_role is None:
            raise ValueError("Source-expert transfer requires a configured task role.")
        identity = np.asarray(target_task_identity, dtype=float).reshape(-1)
        expected_width = schema.identity_slices[schema.task_role].width
        if identity.shape != (expected_width,) or not np.all(np.isfinite(identity)):
            raise ValueError("Target task identity is incompatible with the feature schema.")
        if np.linalg.norm(identity) <= 1e-12:
            raise ValueError("Target task identity must contain a nonzero identity feature.")
        if not 0.0 <= weight_floor < 1.0:
            raise ValueError("weight_floor must lie in [0, 1).")
        if temperature <= 0.0 or predictive_std_floor <= 0.0:
            raise ValueError("Temperature and predictive std floor must be positive.")

        self.schema = schema
        self.config = config
        self.target_task_identity = identity
        self.seed = int(seed)
        self.weight_floor = float(weight_floor)
        self.temperature = float(temperature)
        self.predictive_std_floor = float(predictive_std_floor)
        self.source_models_: dict[tuple[float, ...], StructuredANOVAGP] = {}
        self.expert_models_: dict[tuple[float, ...], StructuredANOVAGP] = {}
        self.expert_weights_: dict[tuple[float, ...], float] = {}
        self.source_log_scores_: dict[tuple[float, ...], float] = {}
        self.n_target_: int | None = None

    def _task_keys(self, X: np.ndarray) -> list[tuple[float, ...]]:
        task = self.schema.task_role
        if task is None:
            raise RuntimeError("Source-expert transfer lost its task role.")
        block = np.asarray(X[:, self.schema.identity_slice(task)], dtype=float)
        return [tuple(row.tolist()) for row in block]

    def _target_mask(self, X: np.ndarray) -> np.ndarray:
        task = self.schema.task_role
        if task is None:
            raise RuntimeError("Source-expert transfer lost its task role.")
        block = np.asarray(X[:, self.schema.identity_slice(task)], dtype=float)
        return np.all(
            np.isclose(block, self.target_task_identity[None, :], atol=1e-12),
            axis=1,
        )

    def _validate_X(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.schema.n_features:
            raise ValueError("Feature matrix is incompatible with the source-expert schema.")
        return X

    def _initialize_sources(
        self,
        X: np.ndarray,
        y: np.ndarray,
        source_keys: tuple[tuple[float, ...], ...],
        keys: list[tuple[float, ...]],
    ) -> None:
        if self.source_models_:
            if tuple(self.source_models_) != source_keys:
                raise ValueError("Auxiliary task identities changed after source initialization.")
            return
        if self.weight_floor * len(source_keys) >= 1.0:
            raise ValueError("weight_floor must be less than 1 / number of source experts.")
        for index, key in enumerate(source_keys):
            source_mask = np.asarray([value == key for value in keys], dtype=bool)
            source_model = StructuredANOVAGP(
                self.schema,
                self.config,
                seed=self.seed + 100 + 2 * index,
            )
            source_model.fit(X[source_mask], y[source_mask])
            self.source_models_[key] = source_model
            self.expert_models_[key] = StructuredANOVAGP(
                self.schema,
                self.config,
                seed=self.seed + 101 + 2 * index,
            )

    def _score_sources(self, X_target: np.ndarray, y_target: np.ndarray) -> None:
        keys = tuple(self.source_models_)
        log_scores = []
        for key in keys:
            posterior = self.source_models_[key].predict(X_target)
            std = np.maximum(posterior.std, self.predictive_std_floor)
            z = (y_target - posterior.mean) / std
            score = float(
                np.sum(-0.5 * z**2 - np.log(std) - 0.5 * np.log(2.0 * np.pi))
            )
            self.source_log_scores_[key] = score
            log_scores.append(score / self.temperature)

        logits = np.asarray(log_scores, dtype=float)
        probabilities = np.exp(logits - logsumexp(logits))
        floor = self.weight_floor
        probabilities = floor + (1.0 - floor * len(keys)) * probabilities
        probabilities /= probabilities.sum()
        self.expert_weights_ = dict(zip(keys, probabilities, strict=True))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SourceExpertTransferGP":
        X = self._validate_X(X)
        y = np.asarray(y, dtype=float).reshape(-1)
        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of rows.")
        target_mask = self._target_mask(X)
        self.n_target_ = int(target_mask.sum())
        if self.n_target_ == 0 or target_mask.all():
            raise ValueError(
                "Source-expert transfer requires target and auxiliary observations."
            )

        keys = self._task_keys(X)
        source_keys = tuple(
            sorted(
                {
                    key
                    for key, is_target in zip(keys, target_mask, strict=True)
                    if not is_target
                }
            )
        )
        self._initialize_sources(X, y, source_keys, keys)
        X_target = X[target_mask]
        y_target = y[target_mask]
        self._score_sources(X_target, y_target)

        for key, expert in self.expert_models_.items():
            source_mask = np.asarray([value == key for value in keys], dtype=bool)
            expert_mask = target_mask | source_mask
            expert.fit(X[expert_mask], y[expert_mask])
        return self

    def _prediction_posteriors(self, X: np.ndarray) -> dict[tuple[float, ...], Posterior]:
        X = self._validate_X(X)
        if not self.expert_weights_:
            raise RuntimeError("SourceExpertTransferGP.fit must be called before prediction.")
        if not self._target_mask(X).all():
            raise ValueError("Source-expert transfer predicts only the configured target task.")
        return {key: model.predict(X) for key, model in self.expert_models_.items()}

    def predict(self, X: np.ndarray) -> Posterior:
        posteriors = self._prediction_posteriors(X)
        keys = tuple(posteriors)
        weights = np.asarray([self.expert_weights_[key] for key in keys], dtype=float)
        means = np.vstack([posteriors[key].mean for key in keys])
        variances = np.vstack([posteriors[key].std**2 for key in keys])
        mean = np.sum(weights[:, None] * means, axis=0)
        within = np.sum(weights[:, None] * variances, axis=0)
        between = np.sum(weights[:, None] * (means - mean) ** 2, axis=0)
        posterior = Posterior(
            mean=mean,
            std=np.sqrt(np.maximum(within + between, 1e-12)),
            disagreement=np.sqrt(np.maximum(between, 0.0)),
        )
        posterior.validate(len(X))
        return posterior

    def posterior_covariance(self, X: np.ndarray) -> np.ndarray:
        posteriors = self._prediction_posteriors(X)
        keys = tuple(posteriors)
        weights = np.asarray([self.expert_weights_[key] for key in keys], dtype=float)
        mixture_mean = np.sum(
            weights[:, None] * np.vstack([posteriors[key].mean for key in keys]),
            axis=0,
        )
        second_moment = np.zeros((len(X), len(X)), dtype=float)
        for weight, key in zip(weights, keys, strict=True):
            mean = posteriors[key].mean
            covariance = self.expert_models_[key].posterior_covariance(X)
            second_moment += weight * (covariance + np.outer(mean, mean))
        covariance = second_moment - np.outer(mixture_mean, mixture_mean)
        covariance = 0.5 * (covariance + covariance.T)
        diagonal = np.maximum(np.diag(covariance), 1e-12)
        covariance.flat[:: len(covariance) + 1] = diagonal
        return covariance
