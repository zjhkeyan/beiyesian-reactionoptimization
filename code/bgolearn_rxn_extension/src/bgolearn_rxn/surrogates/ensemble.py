from __future__ import annotations

from collections import OrderedDict
from typing import Mapping

import numpy as np
from scipy.special import logsumexp

from ..config import EnsembleConfig
from ..schemas import Posterior
from .base import ProbabilisticSurrogate


class PrequentialEnsemble:
    """Online Bayesian model average driven by honest sequential predictions.

    Initial weights are configured before seeing target-test feedback. After a
    recommendation is measured, the weight update uses each component's
    *pre-measurement* predictive density at that point. This is leakage-safe,
    cheap, and better aligned with sequential optimization than in-sample fit.
    """

    def __init__(
        self,
        models: Mapping[str, ProbabilisticSurrogate],
        config: EnsembleConfig,
    ) -> None:
        if not models:
            raise ValueError("At least one surrogate model is required.")
        self.models = OrderedDict(models)
        self.config = config
        self._model_names = tuple(self.models)
        self._initial_weights = self._normalize_initial_weights(config.initial_weights)
        self._weights = dict(self._initial_weights)
        self._cumulative_log_scores = {name: 0.0 for name in self._model_names}

    def _normalize_initial_weights(self, values: Mapping[str, float]) -> dict[str, float]:
        missing = [name for name in self._model_names if name not in values]
        if missing:
            raise ValueError(f"Initial ensemble weights are missing models: {missing}")
        weights = np.asarray([float(values[name]) for name in self._model_names], dtype=float)
        if np.any(weights < 0.0) or not np.any(weights > 0.0):
            raise ValueError("Ensemble weights must be non-negative and not all zero.")
        weights /= np.sum(weights)
        return dict(zip(self._model_names, weights, strict=True))

    @property
    def weights(self) -> dict[str, float]:
        return dict(self._weights)

    def reset_weights(self) -> None:
        self._weights = dict(self._initial_weights)
        self._cumulative_log_scores = {name: 0.0 for name in self._model_names}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PrequentialEnsemble":
        for model in self.models.values():
            model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> Posterior:
        component_posteriors = {
            name: model.predict(X) for name, model in self.models.items()
        }
        weights = np.asarray([self._weights[name] for name in self._model_names])
        means = np.vstack(
            [component_posteriors[name].mean for name in self._model_names]
        )
        variances = np.vstack(
            [component_posteriors[name].std**2 for name in self._model_names]
        )

        mixture_mean = np.sum(weights[:, None] * means, axis=0)
        within = np.sum(weights[:, None] * variances, axis=0)
        between = np.sum(weights[:, None] * (means - mixture_mean) ** 2, axis=0)
        mixture_std = np.sqrt(np.maximum(within + between, 1e-12))
        disagreement = np.sqrt(np.maximum(between, 0.0))

        posterior = Posterior(
            mean=mixture_mean,
            std=mixture_std,
            component_means={
                name: component_posteriors[name].mean for name in self._model_names
            },
            component_stds={
                name: component_posteriors[name].std for name in self._model_names
            },
            weights=self.weights,
            disagreement=disagreement,
        )
        posterior.validate(len(X))
        return posterior

    def posterior_covariance(self, X: np.ndarray) -> np.ndarray:
        posteriors = {name: self.models[name].predict(X) for name in self._model_names}
        weights = np.asarray([self._weights[name] for name in self._model_names])
        means = np.vstack([posteriors[name].mean for name in self._model_names])
        mixture_mean = np.sum(weights[:, None] * means, axis=0)

        second_moment = np.zeros((len(X), len(X)), dtype=float)
        for weight, name in zip(weights, self._model_names, strict=True):
            covariance = self.models[name].posterior_covariance(X)
            mean = posteriors[name].mean
            second_moment += weight * (covariance + np.outer(mean, mean))
        covariance = second_moment - np.outer(mixture_mean, mixture_mean)
        covariance = 0.5 * (covariance + covariance.T)
        diagonal = np.maximum(np.diag(covariance), 1e-12)
        covariance.flat[:: len(covariance) + 1] = diagonal
        return covariance

    def update_prequential(
        self,
        observed_y: float,
        component_means: Mapping[str, float],
        component_stds: Mapping[str, float],
    ) -> dict[str, float]:
        """Update weights from one prediction made before observing ``observed_y``."""

        floor = float(self.config.predictive_std_floor)
        for name in self._model_names:
            mean = float(component_means[name])
            std = max(float(component_stds[name]), floor)
            z = (float(observed_y) - mean) / std
            log_density = -0.5 * z**2 - np.log(std) - 0.5 * np.log(2.0 * np.pi)
            self._cumulative_log_scores[name] += float(log_density)

        log_prior = np.log(
            np.asarray([self._initial_weights[name] for name in self._model_names])
        )
        cumulative = np.asarray(
            [self._cumulative_log_scores[name] for name in self._model_names]
        )
        logits = log_prior + float(self.config.prequential_learning_rate) * cumulative
        probabilities = np.exp(logits - logsumexp(logits))

        minimum = float(self.config.minimum_weight)
        n_models = len(self._model_names)
        if minimum * n_models >= 1.0:
            raise ValueError("minimum_weight must be less than 1 / number of models.")
        probabilities = minimum + (1.0 - n_models * minimum) * probabilities
        probabilities /= probabilities.sum()
        self._weights = dict(zip(self._model_names, probabilities, strict=True))
        return self.weights

    def state_dict(self) -> dict[str, object]:
        return {
            "weights": self.weights,
            "initial_weights": dict(self._initial_weights),
            "cumulative_log_scores": dict(self._cumulative_log_scores),
        }
