from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.spatial.distance import cdist

from ..config import GPConfig
from ..schemas import FeatureSchema, Posterior
from ..transforms import LogitYieldTransform, StandardTargetTransform, make_target_transform


def _pair_key(pair: tuple[str, str]) -> str:
    return "|".join(pair)


@dataclass
class _ParameterLayout:
    roles: tuple[str, ...]
    has_task: bool

    @property
    def n_parameters(self) -> int:
        # role lengthscales + output scale + noise + pair scale + optional task rho
        return len(self.roles) + 3 + int(self.has_task)

    @property
    def output_index(self) -> int:
        return len(self.roles)

    @property
    def noise_index(self) -> int:
        return len(self.roles) + 1

    @property
    def pair_scale_index(self) -> int:
        return len(self.roles) + 2

    @property
    def task_rho_index(self) -> int | None:
        return len(self.roles) + 3 if self.has_task else None


class StructuredANOVAGP:
    """Exact Gaussian process with role-wise main and interaction kernels.

    The covariance is an ANOVA decomposition over reaction roles. Each role
    receives a separate RBF kernel over its frozen identity/rule/molecular
    block. Selected role-pair products encode chemically motivated
    interactions under a shared shrinkage scale. Buchwald product transfer is
    handled by a positive task kernel with learned cross-task correlation.
    """

    name = "gp"

    def __init__(
        self,
        schema: FeatureSchema,
        config: GPConfig,
        *,
        seed: int = 0,
    ) -> None:
        self.schema = schema
        self.config = config
        self.seed = int(seed)
        self.roles = tuple(
            role for role in schema.condition_roles if role != schema.task_role
        )
        missing_roles = [role for role in self.roles if role not in schema.role_slices]
        if missing_roles:
            raise ValueError(f"GP condition roles missing from feature schema: {missing_roles}")
        for pair in config.interaction_pairs:
            if pair[0] not in self.roles or pair[1] not in self.roles:
                raise ValueError(f"Invalid GP interaction pair {pair}; roles={self.roles}")

        self.layout = _ParameterLayout(self.roles, schema.task_role is not None)
        self.target_transform = make_target_transform(config.target_transform)
        self.theta_: np.ndarray | None = None
        self.X_train_: np.ndarray | None = None
        self.y_train_transformed_: np.ndarray | None = None
        self.cholesky_: np.ndarray | None = None
        self.alpha_: np.ndarray | None = None
        self.task_offsets_: dict[tuple[float, ...], float] = {}
        self._fit_calls = 0
        self._train_distances: dict[str, np.ndarray] = {}
        self.optimization_result_: dict[str, float | bool | str] = {}

    def _role_weight(self, role: str) -> float:
        return float(self.config.role_weights.get(role, 1.0))

    def _interaction_weight(self, pair: tuple[str, str]) -> float:
        return float(self.config.interaction_weights.get(_pair_key(pair), 1.0))

    def _role_sqdist(self, X1: np.ndarray, X2: np.ndarray, role: str) -> np.ndarray:
        block1 = np.asarray(X1[:, self.schema.role_slice(role)], dtype=float)
        block2 = np.asarray(X2[:, self.schema.role_slice(role)], dtype=float)
        width = max(block1.shape[1], 1)
        return cdist(block1, block2, metric="sqeuclidean") / width

    def _same_task(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        task = self.schema.task_role
        if task is None:
            return np.ones((len(X1), len(X2)), dtype=float)
        block1 = X1[:, self.schema.identity_slice(task)]
        block2 = X2[:, self.schema.identity_slice(task)]
        distances = cdist(block1, block2, metric="sqeuclidean")
        return np.isclose(distances, 0.0, atol=1e-12).astype(float)

    def _task_keys(self, X: np.ndarray) -> list[tuple[float, ...]]:
        task = self.schema.task_role
        if task is None:
            return [()] * len(X)
        block = np.asarray(X[:, self.schema.identity_slice(task)], dtype=float)
        return [tuple(row.tolist()) for row in block]

    def _fit_task_offsets(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        prior_count = self.config.task_mean_prior_count
        self.task_offsets_ = {}
        if self.schema.task_role is None or prior_count is None:
            return np.zeros(len(y), dtype=float)
        prior_count = float(prior_count)
        if prior_count < 0.0:
            raise ValueError("task_mean_prior_count must be non-negative or null.")
        keys = self._task_keys(X)
        for key in dict.fromkeys(keys):
            mask = np.asarray([value == key for value in keys], dtype=bool)
            count = int(mask.sum())
            group_mean = float(np.mean(y[mask]))
            self.task_offsets_[key] = count / (count + prior_count) * group_mean
        return np.asarray([self.task_offsets_[key] for key in keys], dtype=float)

    def _task_offsets(self, X: np.ndarray) -> np.ndarray:
        if not self.task_offsets_:
            return np.zeros(len(X), dtype=float)
        return np.asarray(
            [self.task_offsets_.get(key, 0.0) for key in self._task_keys(X)],
            dtype=float,
        )

    def _task_descriptor_similarity(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        task = self.schema.task_role
        scale = float(self.config.task_descriptor_scale)
        if scale < 0.0:
            raise ValueError("task_descriptor_scale must be non-negative.")
        if task is None or scale == 0.0:
            return np.ones((len(X1), len(X2)), dtype=float)
        identity_stop = self.schema.identity_slices[task].stop
        role_stop = self.schema.role_slices[task].stop
        if identity_stop >= role_stop:
            return np.ones((len(X1), len(X2)), dtype=float)
        block1 = np.asarray(X1[:, identity_stop:role_stop], dtype=float)
        block2 = np.asarray(X2[:, identity_stop:role_stop], dtype=float)
        width = max(block1.shape[1], 1)
        distances = cdist(block1, block2, metric="sqeuclidean") / width
        return np.exp(-0.5 * scale * distances)

    def _unpack(self, theta: np.ndarray) -> tuple[dict[str, float], float, float, float, float]:
        lengthscales = {
            role: float(np.exp(theta[index])) for index, role in enumerate(self.roles)
        }
        output_scale = float(np.exp(theta[self.layout.output_index]))
        noise = float(np.exp(theta[self.layout.noise_index]))
        pair_scale = float(np.exp(theta[self.layout.pair_scale_index]))
        if self.layout.task_rho_index is None:
            task_rho = 1.0
        else:
            task_rho = float(expit(theta[self.layout.task_rho_index]))
        return lengthscales, output_scale, noise, pair_scale, task_rho

    def _kernel_from_distances(
        self,
        distances: dict[str, np.ndarray],
        same_task: np.ndarray,
        task_similarity: np.ndarray,
        theta: np.ndarray,
    ) -> np.ndarray:
        lengthscales, _, _, pair_scale, task_rho = self._unpack(theta)
        role_kernels = {
            role: np.exp(-0.5 * distances[role] / (lengthscales[role] ** 2))
            for role in self.roles
        }

        bias_weight = 1.0
        kernel = np.full_like(next(iter(role_kernels.values())), bias_weight, dtype=float)
        normalizer = bias_weight

        for role in self.roles:
            weight = max(self._role_weight(role), 0.0)
            kernel += weight * role_kernels[role]
            normalizer += weight

        for pair in self.config.interaction_pairs:
            weight = max(self._interaction_weight(pair), 0.0)
            kernel += pair_scale * weight * role_kernels[pair[0]] * role_kernels[pair[1]]
            normalizer += pair_scale * weight

        kernel /= max(normalizer, 1e-12)
        if self.schema.task_role is not None:
            task_kernel = (
                task_rho * task_similarity
                + (1.0 - task_rho) * same_task
            )
            kernel *= task_kernel
        return kernel

    def _kernel(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
        theta: np.ndarray,
    ) -> np.ndarray:
        distances = {role: self._role_sqdist(X1, X2, role) for role in self.roles}
        return self._kernel_from_distances(
            distances,
            self._same_task(X1, X2),
            self._task_descriptor_similarity(X1, X2),
            theta,
        )

    def _initial_theta(self) -> np.ndarray:
        values = [np.log(self.config.initial_lengthscale)] * len(self.roles)
        values.extend(
            [
                np.log(1.0),
                np.log(self.config.initial_noise),
                np.log(self.config.initial_pair_scale),
            ]
        )
        if self.schema.task_role is not None:
            rho = np.clip(self.config.initial_task_rho, 1e-4, 1.0 - 1e-4)
            values.append(float(logit(rho)))
        return np.asarray(values, dtype=float)

    def _bounds(self) -> list[tuple[float, float]]:
        bounds: list[tuple[float, float]] = [
            (np.log(0.08), np.log(12.0)) for _ in self.roles
        ]
        bounds.extend(
            [
                (np.log(0.05), np.log(20.0)),
                (np.log(1e-4), np.log(0.75)),
                (np.log(1e-4), np.log(4.0)),
            ]
        )
        if self.schema.task_role is not None:
            bounds.append((logit(0.01), logit(0.99)))
        return bounds

    def _prior_penalty(self, theta: np.ndarray) -> float:
        initial = self._initial_theta()
        penalty = 0.0
        # Length scales and output scale receive weak log-normal priors.
        penalty += 0.5 * float(np.sum(((theta[: len(self.roles)] - initial[: len(self.roles)]) / 0.9) ** 2))
        penalty += 0.5 * float(((theta[self.layout.output_index] - 0.0) / 1.0) ** 2)
        penalty += 0.5 * float(((theta[self.layout.noise_index] - np.log(0.05)) / 1.2) ** 2)
        # Interaction shrinkage is deliberate in the tiny-data regime.
        penalty += 0.5 * float(
            ((theta[self.layout.pair_scale_index] - np.log(self.config.initial_pair_scale)) / 0.8) ** 2
        )
        if self.layout.task_rho_index is not None:
            penalty += 0.5 * float(
                ((theta[self.layout.task_rho_index] - logit(self.config.initial_task_rho)) / 1.5) ** 2
            )
        return penalty

    def _training_noise_variance(
        self, theta: np.ndarray, X: np.ndarray
    ) -> np.ndarray:
        """Return observation-noise variance aligned to training rows."""
        noise = self._unpack(theta)[2]
        return np.full(len(X), noise**2, dtype=float)

    def _negative_log_posterior(self, theta: np.ndarray) -> float:
        if self.y_train_transformed_ is None:
            raise RuntimeError("GP objective called before fit initialization.")
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
        try:
            factor, lower = cho_factor(covariance, lower=True, check_finite=False)
            alpha = cho_solve((factor, lower), self.y_train_transformed_, check_finite=False)
        except np.linalg.LinAlgError:
            return 1e20
        log_det_half = float(np.sum(np.log(np.diag(factor))))
        n = len(self.y_train_transformed_)
        nll = 0.5 * float(self.y_train_transformed_ @ alpha)
        nll += log_det_half + 0.5 * n * np.log(2.0 * np.pi)
        return nll + self._prior_penalty(theta)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "StructuredANOVAGP":
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
        should_optimize = (
            self.config.optimize_hyperparameters
            and (self.theta_ is None or (self._fit_calls - 1) % max(self.config.refit_interval, 1) == 0)
        )

        best_theta = initial
        best_value = self._negative_log_posterior(initial)
        best_success = np.isfinite(best_value)
        best_message = "initial parameters"

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
                if np.isfinite(result.fun) and float(result.fun) < best_value:
                    best_theta = np.asarray(result.x, dtype=float)
                    best_value = float(result.fun)
                    best_success = bool(result.success)
                    best_message = str(result.message)

        self.theta_ = best_theta
        _, output_scale, _, _, _ = self._unpack(best_theta)
        base = self._kernel_from_distances(
            self._train_distances,
            self._same_task(X, X),
            self._task_descriptor_similarity(X, X),
            best_theta,
        )
        covariance = output_scale * base
        covariance.flat[:: len(covariance) + 1] += (
            self._training_noise_variance(best_theta, X) + self.config.jitter
        )
        factor, lower = cho_factor(covariance, lower=True, check_finite=False)
        if not lower:
            raise RuntimeError("Expected a lower-triangular Cholesky factor.")
        self.cholesky_ = factor
        self.alpha_ = cho_solve((factor, True), self.y_train_transformed_, check_finite=False)
        self.optimization_result_ = {
            "objective": best_value,
            "success": best_success,
            "message": best_message,
            "fit_calls": self._fit_calls,
        }
        return self

    def _require_fit(self) -> None:
        if any(
            item is None
            for item in (
                self.theta_,
                self.X_train_,
                self.y_train_transformed_,
                self.cholesky_,
                self.alpha_,
            )
        ):
            raise RuntimeError("StructuredANOVAGP.fit must be called before prediction.")

    def _latent_posterior(
        self,
        X: np.ndarray,
        *,
        full_covariance: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        self._require_fit()
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.schema.n_features:
            raise ValueError("Prediction matrix has an incompatible shape.")

        _, output_scale, _, _, _ = self._unpack(self.theta_)
        cross = output_scale * self._kernel(X, self.X_train_, self.theta_)
        mean = cross @ self.alpha_ + self._task_offsets(X)
        solved = solve_triangular(
            self.cholesky_, cross.T, lower=True, check_finite=False
        )

        if full_covariance:
            prior = output_scale * self._kernel(X, X, self.theta_)
            covariance = prior - solved.T @ solved
            covariance = 0.5 * (covariance + covariance.T)
            diagonal = np.maximum(np.diag(covariance), 1e-12)
            covariance.flat[:: len(covariance) + 1] = diagonal
            return mean, covariance

        variance = np.maximum(output_scale - np.sum(solved**2, axis=0), 1e-12)
        return mean, variance

    def predict(self, X: np.ndarray) -> Posterior:
        latent_mean, latent_variance = self._latent_posterior(X, full_covariance=False)
        latent_std = np.sqrt(latent_variance)
        mean, std = self.target_transform.inverse_mean_std(latent_mean, latent_std)
        posterior = Posterior(mean=mean, std=std)
        posterior.validate(len(X))
        return posterior

    def posterior_covariance(self, X: np.ndarray) -> np.ndarray:
        latent_mean, latent_covariance = self._latent_posterior(X, full_covariance=True)
        latent_std = np.sqrt(np.maximum(np.diag(latent_covariance), 1e-12))
        _, yield_std = self.target_transform.inverse_mean_std(latent_mean, latent_std)

        if isinstance(self.target_transform, StandardTargetTransform):
            return latent_covariance * (self.target_transform.scale_**2)

        if isinstance(self.target_transform, LogitYieldTransform):
            denominator = np.outer(latent_std, latent_std)
            correlation = np.divide(
                latent_covariance,
                denominator,
                out=np.eye(len(X), dtype=float),
                where=denominator > 1e-15,
            )
            correlation = np.clip(correlation, -1.0, 1.0)
            covariance = correlation * np.outer(yield_std, yield_std)
            covariance = 0.5 * (covariance + covariance.T)
            covariance.flat[:: len(covariance) + 1] = yield_std**2
            return covariance

        raise TypeError("Unsupported target transform for covariance mapping.")
