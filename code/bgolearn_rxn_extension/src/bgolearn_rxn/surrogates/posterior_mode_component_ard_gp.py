from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.optimize import minimize

from ..schemas import Posterior
from ..transforms import LogitYieldTransform, StandardTargetTransform
from .component_ard_gp import ComponentARDStructuredGP


@dataclass(frozen=True)
class _PosteriorMode:
    theta: np.ndarray
    objective: float
    success: bool
    message: str
    cholesky: np.ndarray
    alpha: np.ndarray


class PosteriorModeComponentARDStructuredGP(ComponentARDStructuredGP):
    """Component-ARD GP integrating credible MAP restart modes.

    The fixed objective window is measured on the regularized negative log
    posterior, so a two-unit window retains only modes with comparable support.
    Temperature one preserves that objective scale. Three retained modes bound
    prediction cost while allowing more than a binary compromise.
    """

    name = "posterior_mode_component_ard_gp"
    objective_window = 2.0
    objective_temperature = 1.0
    max_modes = 3
    duplicate_relative_tolerance = 1e-3

    def __init__(self, schema, config, *, seed: int = 0) -> None:
        super().__init__(schema, config, seed=seed)
        self.modes_: list[_PosteriorMode] = []
        self.mode_weights_: np.ndarray = np.ones(1, dtype=float)
        self.mode_history_: list[dict[str, object]] = []

    def _normalized_theta_distance(
        self, left: np.ndarray, right: np.ndarray
    ) -> float:
        bounds = np.asarray(self._bounds(), dtype=float)
        span = np.maximum(bounds[:, 1] - bounds[:, 0], 1e-12)
        return float(np.max(np.abs(np.asarray(left) - np.asarray(right)) / span))

    def _select_solutions(
        self,
        solutions: list[tuple[np.ndarray, float, bool, str]],
    ) -> tuple[list[tuple[np.ndarray, float, bool, str]], np.ndarray]:
        finite = [item for item in solutions if np.isfinite(item[1])]
        if not finite:
            raise RuntimeError("Posterior-mode optimization produced no finite solution.")
        ordered = sorted(finite, key=lambda item: item[1])
        best = float(ordered[0][1])
        retained: list[tuple[np.ndarray, float, bool, str]] = []
        for item in ordered:
            if float(item[1]) > best + self.objective_window:
                break
            if any(
                self._normalized_theta_distance(item[0], prior[0])
                <= self.duplicate_relative_tolerance
                for prior in retained
            ):
                continue
            retained.append(item)
            if len(retained) == self.max_modes:
                break
        objectives = np.asarray([item[1] for item in retained], dtype=float)
        log_weights = -(objectives - objectives.min()) / self.objective_temperature
        weights = np.exp(log_weights - np.max(log_weights))
        weights /= weights.sum()
        return retained, weights

    def _factor_mode(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.X_train_ is None or self.y_train_transformed_ is None:
            raise RuntimeError("Cannot factor a mode before fit initialization.")
        _, output_scale, _, _, _ = self._unpack(theta)
        base = self._kernel_from_distances(
            self._train_distances,
            self._same_task(self.X_train_, self.X_train_),
            self._task_descriptor_similarity(self.X_train_, self.X_train_),
            theta,
        )
        covariance = output_scale * base
        covariance.flat[:: len(covariance) + 1] += (
            self._training_noise_variance(theta, self.X_train_) + self.config.jitter
        )
        factor, lower = cho_factor(covariance, lower=True, check_finite=False)
        if not lower:
            raise RuntimeError("Expected a lower-triangular Cholesky factor.")
        alpha = cho_solve(
            (factor, True), self.y_train_transformed_, check_finite=False
        )
        return factor, alpha

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PosteriorModeComponentARDStructuredGP":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError("X must be 2D and have the same number of rows as y.")
        if X.shape[1] != self.schema.n_features:
            raise ValueError(
                f"GP expected {self.schema.n_features} features, got {X.shape[1]}."
            )

        self.X_train_ = X
        self.target_transform.fit(y)
        transformed_y = self.target_transform.transform(y)
        task_offsets = self._fit_task_offsets(X, transformed_y)
        self.y_train_transformed_ = transformed_y - task_offsets
        self._train_distances = {
            role: self._role_sqdist(X, X, role) for role in self.roles
        }
        self._fit_calls += 1

        initial = self.theta_.copy() if self.theta_ is not None else self._initial_theta()
        should_optimize = self.config.optimize_hyperparameters and (
            self.theta_ is None
            or (self._fit_calls - 1) % max(self.config.refit_interval, 1) == 0
        )
        initial_value = float(self._negative_log_posterior(initial))
        solutions: list[tuple[np.ndarray, float, bool, str]] = []
        attempted_solutions: list[tuple[np.ndarray, float, bool, str]] = []
        if should_optimize:
            rng = np.random.default_rng(self.seed + self._fit_calls)
            starts = [initial]
            for _ in range(max(self.config.restarts - 1, 0)):
                starts.append(initial + rng.normal(0.0, 0.35, size=len(initial)))
            for start in starts:
                result = minimize(
                    self._negative_log_posterior,
                    x0=start,
                    method="L-BFGS-B",
                    bounds=self._bounds(),
                    options={"maxiter": self.config.maxiter, "ftol": 1e-9},
                )
                if np.isfinite(result.fun):
                    attempted_solutions.append(
                        (
                            np.asarray(result.x, dtype=float),
                            float(result.fun),
                            bool(result.success),
                            str(result.message),
                        )
                    )
            improving = [item for item in attempted_solutions if item[1] < initial_value]
            solutions = attempted_solutions if improving else [
                (initial, initial_value, True, "incumbent parameters")
            ]
        elif self.modes_:
            solutions = [
                (
                    mode.theta.copy(),
                    float(self._negative_log_posterior(mode.theta)),
                    mode.success,
                    "hyperparameters reused between refits",
                )
                for mode in self.modes_
            ]

        if not solutions:
            solutions = [
                (
                    initial,
                    initial_value,
                    True,
                    "initial parameters",
                )
            ]
        retained, weights = self._select_solutions(solutions)
        modes: list[_PosteriorMode] = []
        for theta, objective, success, message in retained:
            factor, alpha = self._factor_mode(theta)
            modes.append(
                _PosteriorMode(
                    theta=theta.copy(),
                    objective=float(objective),
                    success=success,
                    message=message,
                    cholesky=factor,
                    alpha=alpha,
                )
            )

        self.modes_ = modes
        self.mode_weights_ = weights
        best = modes[0]
        self.theta_ = best.theta.copy()
        self.cholesky_ = best.cholesky
        self.alpha_ = best.alpha
        self.optimization_result_ = {
            "objective": best.objective,
            "success": best.success,
            "message": best.message,
            "fit_calls": self._fit_calls,
            "retained_mode_count": len(modes),
        }
        diagnostics = attempted_solutions if should_optimize else solutions
        attempted_objectives = [float(item[1]) for item in diagnostics]
        attempted_distances = [
            self._normalized_theta_distance(best.theta, item[0])
            for item in diagnostics
        ]
        self.mode_history_.append(
            {
                "fit_call": self._fit_calls,
                "optimized": should_optimize,
                "optimizer_solution_count": len(diagnostics),
                "optimizer_objectives": attempted_objectives,
                "optimizer_distances_to_best": attempted_distances,
                "mode_count": len(modes),
                "objectives": [float(mode.objective) for mode in modes],
                "weights": weights.tolist(),
            }
        )
        self._record_component_amplitudes()
        return self

    def _mode_latent_posterior(
        self, X: np.ndarray, mode: _PosteriorMode, *, full_covariance: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.X_train_ is None:
            raise RuntimeError("Posterior-mode GP must be fitted before prediction.")
        _, output_scale, _, _, _ = self._unpack(mode.theta)
        cross = output_scale * self._kernel(X, self.X_train_, mode.theta)
        mean = cross @ mode.alpha + self._task_offsets(X)
        solved = solve_triangular(
            mode.cholesky, cross.T, lower=True, check_finite=False
        )
        if full_covariance:
            prior = output_scale * self._kernel(X, X, mode.theta)
            covariance = prior - solved.T @ solved
            covariance = 0.5 * (covariance + covariance.T)
            covariance.flat[:: len(covariance) + 1] = np.maximum(
                np.diag(covariance), 1e-12
            )
            return mean, covariance
        variance = np.maximum(output_scale - np.sum(solved**2, axis=0), 1e-12)
        return mean, variance

    @staticmethod
    def _mixture_variance(
        means: np.ndarray, variances: np.ndarray, weights: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        mean = np.sum(weights[:, None] * means, axis=0)
        variance = np.sum(weights[:, None] * (variances + means**2), axis=0) - mean**2
        return mean, np.maximum(variance, 1e-12)

    def _yield_mode_marginals(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        means = []
        variances = []
        for mode in self.modes_:
            latent_mean, latent_variance = self._mode_latent_posterior(
                X, mode, full_covariance=False
            )
            mean, std = self.target_transform.inverse_mean_std(
                latent_mean, np.sqrt(latent_variance)
            )
            means.append(mean)
            variances.append(std**2)
        return np.asarray(means), np.asarray(variances)

    def predict(self, X: np.ndarray) -> Posterior:
        self._require_fit()
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.schema.n_features:
            raise ValueError("Prediction matrix has an incompatible shape.")
        means, variances = self._yield_mode_marginals(X)
        mean, variance = self._mixture_variance(means, variances, self.mode_weights_)
        component_means = {
            f"mode_{index}": values for index, values in enumerate(means)
        }
        component_stds = {
            f"mode_{index}": np.sqrt(values)
            for index, values in enumerate(variances)
        }
        posterior = Posterior(
            mean=mean,
            std=np.sqrt(variance),
            component_means=component_means,
            component_stds=component_stds,
            weights={
                f"mode_{index}": float(weight)
                for index, weight in enumerate(self.mode_weights_)
            },
            disagreement=np.sqrt(
                np.maximum(
                    np.sum(self.mode_weights_[:, None] * (means - mean) ** 2, axis=0),
                    0.0,
                )
            ),
        )
        posterior.validate(len(X))
        return posterior

    def posterior_covariance(self, X: np.ndarray) -> np.ndarray:
        self._require_fit()
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.schema.n_features:
            raise ValueError("Prediction matrix has an incompatible shape.")
        means = []
        covariances = []
        for mode in self.modes_:
            latent_mean, latent_covariance = self._mode_latent_posterior(
                X, mode, full_covariance=True
            )
            latent_std = np.sqrt(np.maximum(np.diag(latent_covariance), 1e-12))
            yield_mean, yield_std = self.target_transform.inverse_mean_std(
                latent_mean, latent_std
            )
            if isinstance(self.target_transform, StandardTargetTransform):
                covariance = latent_covariance * self.target_transform.scale_**2
            elif isinstance(self.target_transform, LogitYieldTransform):
                denominator = np.outer(latent_std, latent_std)
                correlation = np.divide(
                    latent_covariance,
                    denominator,
                    out=np.eye(len(X), dtype=float),
                    where=denominator > 1e-15,
                )
                covariance = np.clip(correlation, -1.0, 1.0) * np.outer(
                    yield_std, yield_std
                )
            else:
                raise TypeError("Unsupported target transform for covariance mapping.")
            means.append(yield_mean)
            covariances.append(0.5 * (covariance + covariance.T))
        mean_array = np.asarray(means)
        mixture_mean = np.sum(self.mode_weights_[:, None] * mean_array, axis=0)
        second = np.zeros((len(X), len(X)), dtype=float)
        for weight, mean, covariance in zip(
            self.mode_weights_, mean_array, covariances, strict=True
        ):
            second += weight * (covariance + np.outer(mean, mean))
        mixture = second - np.outer(mixture_mean, mixture_mean)
        mixture = 0.5 * (mixture + mixture.T)
        mixture.flat[:: len(mixture) + 1] = np.maximum(np.diag(mixture), 1e-12)
        return mixture
