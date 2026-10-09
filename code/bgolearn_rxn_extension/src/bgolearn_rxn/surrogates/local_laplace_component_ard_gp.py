from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular

from ..schemas import Posterior
from ..transforms import LogitYieldTransform, StandardTargetTransform
from .component_ard_gp import ComponentARDStructuredGP


@dataclass(frozen=True)
class _LaplacePoint:
    theta: np.ndarray
    objective: float
    axis: int
    direction: int
    cholesky: np.ndarray
    alpha: np.ndarray


class LocalLaplaceComponentARDStructuredGP(ComponentARDStructuredGP):
    """Component-ARD GP with bounded local Laplace sigma integration."""

    name = "local_laplace_component_ard_gp"
    curvature_step_fraction = 0.01
    curvature_floor_normalized = 1.0
    max_sigma_axes = 4
    max_sigma_bound_fraction = 0.08
    sigma_objective_window = 3.0

    def __init__(self, schema, config, *, seed: int = 0) -> None:
        super().__init__(schema, config, seed=seed)
        self.curvature_diagonal_: np.ndarray | None = None
        self.normalized_posterior_std_: np.ndarray | None = None
        self.laplace_points_: list[_LaplacePoint] = []
        self.laplace_weights_: np.ndarray = np.ones(1, dtype=float)
        self.laplace_history_: list[dict[str, object]] = []

    def _factor_point(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.X_train_ is None or self.y_train_transformed_ is None:
            raise RuntimeError("Cannot factor a Laplace point before fit initialization.")
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

    def _diagonal_curvature(
        self, center: np.ndarray, center_objective: float
    ) -> tuple[np.ndarray, np.ndarray, list[float]]:
        bounds = np.asarray(self._bounds(), dtype=float)
        spans = bounds[:, 1] - bounds[:, 0]
        curvature = self.curvature_floor_normalized / np.maximum(spans**2, 1e-12)
        steps = np.zeros_like(center)
        for index, (lower, upper) in enumerate(bounds):
            symmetric_room = min(center[index] - lower, upper - center[index])
            step = min(
                self.curvature_step_fraction * spans[index],
                0.45 * max(symmetric_room, 0.0),
            )
            if step <= 1e-7 * spans[index]:
                continue
            plus = center.copy()
            minus = center.copy()
            plus[index] += step
            minus[index] -= step
            plus_value = float(self._negative_log_posterior(plus))
            minus_value = float(self._negative_log_posterior(minus))
            estimate = (plus_value - 2.0 * center_objective + minus_value) / step**2
            if np.isfinite(estimate):
                curvature[index] = max(estimate, curvature[index])
                steps[index] = step
        normalized_std = 1.0 / (np.sqrt(curvature) * spans)
        return curvature, normalized_std, steps.tolist()

    def _make_sigma_points(
        self,
        center: np.ndarray,
        center_objective: float,
        curvature: np.ndarray,
        normalized_std: np.ndarray,
    ) -> tuple[list[_LaplacePoint], list[int], list[dict[str, object]]]:
        bounds = np.asarray(self._bounds(), dtype=float)
        spans = bounds[:, 1] - bounds[:, 0]
        symmetric_room = np.minimum(center - bounds[:, 0], bounds[:, 1] - center)
        eligible = [
            index
            for index in range(len(center))
            if symmetric_room[index] > 1e-7 * spans[index]
        ]
        ordered = sorted(eligible, key=lambda index: (-normalized_std[index], index))
        selected = ordered[: self.max_sigma_axes]
        radius = np.sqrt(max(len(selected), 1))
        points: list[_LaplacePoint] = []
        attempts: list[dict[str, object]] = []
        for axis in selected:
            displacement = min(
                radius / np.sqrt(curvature[axis]),
                self.max_sigma_bound_fraction * spans[axis],
                0.8 * symmetric_room[axis],
            )
            pair: list[_LaplacePoint] = []
            rejection = None
            for direction in (-1, 1):
                theta = center.copy()
                theta[axis] += direction * displacement
                objective = float(self._negative_log_posterior(theta))
                if not np.isfinite(objective):
                    rejection = "nonfinite_objective"
                    break
                if objective > center_objective + self.sigma_objective_window:
                    rejection = "outside_objective_window"
                    break
                try:
                    factor, alpha = self._factor_point(theta)
                except (np.linalg.LinAlgError, RuntimeError):
                    rejection = "factorization_failed"
                    break
                pair.append(
                    _LaplacePoint(
                        theta=theta,
                        objective=objective,
                        axis=axis,
                        direction=direction,
                        cholesky=factor,
                        alpha=alpha,
                    )
                )
            accepted = rejection is None and len(pair) == 2
            attempts.append(
                {
                    "axis": axis,
                    "normalized_std": float(normalized_std[axis]),
                    "normalized_displacement": float(displacement / spans[axis]),
                    "accepted": accepted,
                    "rejection": rejection,
                    "objectives": [float(item.objective) for item in pair],
                }
            )
            if accepted:
                points.extend(pair)
        return points, selected, attempts

    def fit(
        self, X: np.ndarray, y: np.ndarray
    ) -> "LocalLaplaceComponentARDStructuredGP":
        super().fit(X, y)
        if self.theta_ is None or self.cholesky_ is None or self.alpha_ is None:
            raise RuntimeError("MAP fit did not initialize the local Laplace surrogate.")
        center = self.theta_.copy()
        center_objective = float(self._negative_log_posterior(center))
        curvature, normalized_std, steps = self._diagonal_curvature(
            center, center_objective
        )
        points, selected, attempts = self._make_sigma_points(
            center, center_objective, curvature, normalized_std
        )
        fallback = not points
        if fallback:
            points = [
                _LaplacePoint(
                    theta=center,
                    objective=center_objective,
                    axis=-1,
                    direction=0,
                    cholesky=self.cholesky_,
                    alpha=self.alpha_,
                )
            ]
        weights = np.full(len(points), 1.0 / len(points), dtype=float)
        self.curvature_diagonal_ = curvature
        self.normalized_posterior_std_ = normalized_std
        self.laplace_points_ = points
        self.laplace_weights_ = weights
        self.laplace_history_.append(
            {
                "fit_call": self._fit_calls,
                "center_objective": center_objective,
                "parameter_count": len(center),
                "curvature_diagonal": curvature.tolist(),
                "curvature_steps": steps,
                "normalized_posterior_std": normalized_std.tolist(),
                "selected_axes": selected,
                "attempts": attempts,
                "accepted_point_count": len(points),
                "weights": weights.tolist(),
                "effective_sample_count": float(1.0 / np.sum(weights**2)),
                "fallback_to_map": fallback,
            }
        )
        self.optimization_result_["laplace_accepted_point_count"] = len(points)
        self.optimization_result_["laplace_fallback_to_map"] = fallback
        return self

    def _point_latent_posterior(
        self, X: np.ndarray, point: _LaplacePoint, *, full_covariance: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.X_train_ is None:
            raise RuntimeError("Local Laplace GP must be fitted before prediction.")
        _, output_scale, _, _, _ = self._unpack(point.theta)
        cross = output_scale * self._kernel(X, self.X_train_, point.theta)
        mean = cross @ point.alpha + self._task_offsets(X)
        solved = solve_triangular(
            point.cholesky, cross.T, lower=True, check_finite=False
        )
        if full_covariance:
            prior = output_scale * self._kernel(X, X, point.theta)
            covariance = prior - solved.T @ solved
            covariance = 0.5 * (covariance + covariance.T)
            covariance.flat[:: len(covariance) + 1] = np.maximum(
                np.diag(covariance), 1e-12
            )
            return mean, covariance
        variance = np.maximum(output_scale - np.sum(solved**2, axis=0), 1e-12)
        return mean, variance

    @staticmethod
    def _total_variance(
        means: np.ndarray, variances: np.ndarray, weights: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        mean = np.sum(weights[:, None] * means, axis=0)
        variance = np.sum(weights[:, None] * (variances + means**2), axis=0) - mean**2
        return mean, np.maximum(variance, 1e-12)

    def _yield_point_marginals(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        means = []
        variances = []
        for point in self.laplace_points_:
            latent_mean, latent_variance = self._point_latent_posterior(
                X, point, full_covariance=False
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
        means, variances = self._yield_point_marginals(X)
        mean, variance = self._total_variance(
            means, variances, self.laplace_weights_
        )
        disagreement = np.sqrt(
            np.maximum(
                np.sum(self.laplace_weights_[:, None] * (means - mean) ** 2, axis=0),
                0.0,
            )
        )
        posterior = Posterior(
            mean=mean,
            std=np.sqrt(variance),
            component_means={
                f"sigma_{index}": values for index, values in enumerate(means)
            },
            component_stds={
                f"sigma_{index}": np.sqrt(values)
                for index, values in enumerate(variances)
            },
            weights={
                f"sigma_{index}": float(weight)
                for index, weight in enumerate(self.laplace_weights_)
            },
            disagreement=disagreement,
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
        for point in self.laplace_points_:
            latent_mean, latent_covariance = self._point_latent_posterior(
                X, point, full_covariance=True
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
        mixture_mean = np.sum(self.laplace_weights_[:, None] * mean_array, axis=0)
        second = np.zeros((len(X), len(X)), dtype=float)
        for weight, mean, covariance in zip(
            self.laplace_weights_, mean_array, covariances, strict=True
        ):
            second += weight * (covariance + np.outer(mean, mean))
        mixture = second - np.outer(mixture_mean, mixture_mean)
        mixture = 0.5 * (mixture + mixture.T)
        mixture.flat[:: len(mixture) + 1] = np.maximum(np.diag(mixture), 1e-12)
        return mixture
