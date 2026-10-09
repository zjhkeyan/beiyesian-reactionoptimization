from __future__ import annotations

import numpy as np
from scipy.linalg import helmert
from scipy.special import expit, logit

from ..config import GPConfig
from ..schemas import FeatureSchema
from .anova_gp import StructuredANOVAGP
from .low_rank_task_gp import LowRankTaskCovarianceGP


class _ComponentAmplitudeMixin:
    """Independent positive MAP amplitudes for ANOVA kernel components."""

    amplitude_prior_scale = 0.60

    def _amplitude_centers(self) -> np.ndarray:
        role = [max(self._role_weight(name), 1e-4) for name in self.roles]
        pair = [
            max(self.config.initial_pair_scale * self._interaction_weight(item), 1e-4)
            for item in self.config.interaction_pairs
        ]
        return np.log(np.asarray(role + pair, dtype=float))

    def _amplitudes(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.exp(np.asarray(theta[self._amplitude_start :], dtype=float))
        split = len(self.roles)
        return values[:split], values[split:]

    def _condition_kernel(
        self,
        distances: dict[str, np.ndarray],
        theta: np.ndarray,
    ) -> np.ndarray:
        lengthscales, _, _, _, _ = self._unpack(theta)
        role_kernels = {
            role: np.exp(-0.5 * distances[role] / (lengthscales[role] ** 2))
            for role in self.roles
        }
        role_amplitudes, pair_amplitudes = self._amplitudes(theta)
        kernel = np.ones_like(next(iter(role_kernels.values())), dtype=float)
        normalizer = 1.0
        for amplitude, role in zip(role_amplitudes, self.roles, strict=True):
            kernel += amplitude * role_kernels[role]
            normalizer += float(amplitude)
        for amplitude, pair in zip(
            pair_amplitudes, self.config.interaction_pairs, strict=True
        ):
            kernel += amplitude * role_kernels[pair[0]] * role_kernels[pair[1]]
            normalizer += float(amplitude)
        return kernel / normalizer

    def _amplitude_penalty(self, theta: np.ndarray) -> float:
        log_amplitudes = np.asarray(theta[self._amplitude_start :], dtype=float)
        centers = self._amplitude_centers()
        return 0.5 * float(
            np.sum(((log_amplitudes - centers) / self.amplitude_prior_scale) ** 2)
        )

    def _record_component_amplitudes(self) -> None:
        if self.theta_ is None:
            raise RuntimeError("Cannot record amplitudes before fitting.")
        role_values, pair_values = self._amplitudes(self.theta_)
        raw = {"bias": 1.0}
        raw.update(
            {f"role:{role}": float(value) for role, value in zip(self.roles, role_values, strict=True)}
        )
        raw.update(
            {
                f"pair:{'|'.join(pair)}": float(value)
                for pair, value in zip(
                    self.config.interaction_pairs, pair_values, strict=True
                )
            }
        )
        total = float(sum(raw.values()))
        self.component_amplitudes_ = raw
        self.component_shares_ = {name: value / total for name, value in raw.items()}


class ComponentARDStructuredGP(_ComponentAmplitudeMixin, StructuredANOVAGP):
    """Structured exact GP with independently shrunk component amplitudes."""

    name = "component_ard_gp"

    def __init__(self, schema: FeatureSchema, config: GPConfig, *, seed: int = 0) -> None:
        super().__init__(schema, config, seed=seed)
        self._output_index = len(self.roles)
        self._noise_index = len(self.roles) + 1
        self._task_rho_index = (
            len(self.roles) + 2 if self.schema.task_role is not None else None
        )
        self._amplitude_start = len(self.roles) + 2 + int(
            self.schema.task_role is not None
        )
        self.component_amplitudes_: dict[str, float] = {}
        self.component_shares_: dict[str, float] = {}

    def _unpack(self, theta: np.ndarray):
        lengthscales = {
            role: float(np.exp(theta[index])) for index, role in enumerate(self.roles)
        }
        output_scale = float(np.exp(theta[self._output_index]))
        noise = float(np.exp(theta[self._noise_index]))
        task_rho = (
            1.0
            if self._task_rho_index is None
            else float(expit(theta[self._task_rho_index]))
        )
        return lengthscales, output_scale, noise, 1.0, task_rho

    def _kernel_from_distances(
        self, distances, same_task, task_similarity, theta
    ) -> np.ndarray:
        condition = self._condition_kernel(distances, theta)
        if self.schema.task_role is None:
            return condition
        task_rho = self._unpack(theta)[4]
        return condition * (
            task_rho * task_similarity + (1.0 - task_rho) * same_task
        )

    def _initial_theta(self) -> np.ndarray:
        values = [np.log(self.config.initial_lengthscale)] * len(self.roles)
        values.extend([np.log(1.0), np.log(self.config.initial_noise)])
        if self.schema.task_role is not None:
            rho = np.clip(self.config.initial_task_rho, 1e-4, 1.0 - 1e-4)
            values.append(float(logit(rho)))
        values.extend(self._amplitude_centers().tolist())
        return np.asarray(values, dtype=float)

    def _bounds(self) -> list[tuple[float, float]]:
        bounds = [(np.log(0.08), np.log(12.0)) for _ in self.roles]
        bounds.extend([(np.log(0.05), np.log(20.0)), (np.log(1e-4), np.log(0.75))])
        if self.schema.task_role is not None:
            bounds.append((logit(0.01), logit(0.99)))
        bounds.extend([(np.log(1e-3), np.log(20.0))] * len(self._amplitude_centers()))
        return bounds

    def _prior_penalty(self, theta: np.ndarray) -> float:
        initial = self._initial_theta()
        penalty = 0.5 * float(
            np.sum(((theta[: len(self.roles)] - initial[: len(self.roles)]) / 0.9) ** 2)
        )
        penalty += 0.5 * float(((theta[self._output_index] - 0.0) / 1.0) ** 2)
        penalty += 0.5 * float(
            ((theta[self._noise_index] - np.log(0.05)) / 1.2) ** 2
        )
        if self._task_rho_index is not None:
            penalty += 0.5 * float(
                ((theta[self._task_rho_index] - logit(self.config.initial_task_rho)) / 1.5) ** 2
            )
        return penalty + self._amplitude_penalty(theta)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ComponentARDStructuredGP":
        super().fit(X, y)
        self._record_component_amplitudes()
        return self


class ComponentARDLowRankTaskGP(_ComponentAmplitudeMixin, LowRankTaskCovarianceGP):
    """Rank-2 task GP with independently shrunk condition amplitudes."""

    name = "component_ard_low_rank_task_gp"

    def __init__(self, schema: FeatureSchema, config: GPConfig, *, seed: int = 0) -> None:
        super().__init__(schema, config, seed=seed)
        self._output_index = len(self.roles)
        self._noise_index = len(self.roles) + 1
        self._condition_parameter_count = len(self.roles) + 2
        self._task_parameter_count = self.n_tasks * self.task_rank + self.n_tasks
        self._amplitude_start = self._condition_parameter_count + self._task_parameter_count
        self.component_amplitudes_: dict[str, float] = {}
        self.component_shares_: dict[str, float] = {}

    def _unpack(self, theta: np.ndarray):
        lengthscales = {
            role: float(np.exp(theta[index])) for index, role in enumerate(self.roles)
        }
        return (
            lengthscales,
            float(np.exp(theta[self._output_index])),
            float(np.exp(theta[self._noise_index])),
            1.0,
            1.0,
        )

    def _kernel_from_distances(
        self, distances, same_task, task_similarity, theta
    ) -> np.ndarray:
        del same_task
        condition = self._condition_kernel(distances, theta)
        task_kernel = self.task_correlation(theta).reshape(-1)[
            np.asarray(task_similarity, dtype=int)
        ]
        return condition * task_kernel

    def _initial_theta(self) -> np.ndarray:
        values = [np.log(self.config.initial_lengthscale)] * len(self.roles)
        values.extend([np.log(1.0), np.log(self.config.initial_noise)])
        rho = float(np.clip(self.config.initial_task_rho, 0.01, 0.99))
        loadings = np.zeros((self.n_tasks, self.task_rank), dtype=float)
        loadings[:, 0] = np.sqrt(rho)
        if self.task_rank > 1:
            loadings[:, 1] = np.linspace(-0.05, 0.05, self.n_tasks)
        values.extend(loadings.reshape(-1).tolist())
        values.extend([0.5 * np.log(1.0 - rho)] * self.n_tasks)
        values.extend(self._amplitude_centers().tolist())
        return np.asarray(values, dtype=float)

    def _bounds(self) -> list[tuple[float, float]]:
        bounds = [(np.log(0.08), np.log(12.0)) for _ in self.roles]
        bounds.extend([(np.log(0.05), np.log(20.0)), (np.log(1e-4), np.log(0.75))])
        bounds.extend([(-2.0, 2.0)] * (self.n_tasks * self.task_rank))
        bounds.extend([(np.log(0.05), np.log(2.0))] * self.n_tasks)
        bounds.extend([(np.log(1e-3), np.log(20.0))] * len(self._amplitude_centers()))
        return bounds

    def _prior_penalty(self, theta: np.ndarray) -> float:
        initial = self._initial_theta()
        penalty = 0.5 * float(
            np.sum(((theta[: len(self.roles)] - initial[: len(self.roles)]) / 0.9) ** 2)
        )
        penalty += 0.5 * float(((theta[self._output_index] - 0.0) / 1.0) ** 2)
        penalty += 0.5 * float(
            ((theta[self._noise_index] - np.log(0.05)) / 1.2) ** 2
        )
        loadings, log_diagonal = self._task_parameters(theta)
        rho = float(np.clip(self.config.initial_task_rho, 0.01, 0.99))
        loading_prior = np.zeros_like(loadings)
        loading_prior[:, 0] = np.sqrt(rho)
        diagonal_prior = 0.5 * np.log(1.0 - rho)
        penalty += 0.5 * float(
            np.sum(((loadings - loading_prior) / self.task_prior_scale) ** 2)
        )
        penalty += 0.5 * float(
            np.sum(((log_diagonal - diagonal_prior) / self.task_prior_scale) ** 2)
        )
        return penalty + self._amplitude_penalty(theta)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ComponentARDLowRankTaskGP":
        super().fit(X, y)
        self._record_component_amplitudes()
        return self


class TaskNoiseComponentARDLowRankTaskGP(ComponentARDLowRankTaskGP):
    """Component-ARD task GP with shrinkage-regularized task noise."""

    name = "component_ard_low_rank_task_noise_gp"
    task_noise_prior_scale = 0.35
    task_noise_contrast_bound = 1.25
    global_noise_bounds = (1e-4, 0.75)

    def __init__(self, schema: FeatureSchema, config: GPConfig, *, seed: int = 0) -> None:
        super().__init__(schema, config, seed=seed)
        self._task_noise_start = self._amplitude_start
        self._task_noise_count = self.n_tasks - 1
        self._amplitude_start += self._task_noise_count
        self._task_noise_contrast = helmert(self.n_tasks, full=False).T
        self.task_noises_: dict[str, float] = {}
        self.task_noise_log_deviations_: dict[str, float] = {}
        self.task_noise_contrasts_: np.ndarray | None = None

    def _initial_theta(self) -> np.ndarray:
        baseline = super()._initial_theta()
        return np.insert(
            baseline,
            self._task_noise_start,
            np.zeros(self._task_noise_count, dtype=float),
        )

    def _bounds(self) -> list[tuple[float, float]]:
        baseline = super()._bounds()
        bounds = [
            (-self.task_noise_contrast_bound, self.task_noise_contrast_bound)
        ] * self._task_noise_count
        return (
            baseline[: self._task_noise_start]
            + bounds
            + baseline[self._task_noise_start :]
        )

    def _task_noise_log_deviations(self, theta: np.ndarray) -> np.ndarray:
        contrasts = np.asarray(
            theta[
                self._task_noise_start : self._task_noise_start
                + self._task_noise_count
            ],
            dtype=float,
        )
        return self._task_noise_contrast @ contrasts

    def task_noise_vector(self, theta: np.ndarray | None = None) -> np.ndarray:
        values = self.theta_ if theta is None else np.asarray(theta, dtype=float)
        if values is None:
            raise RuntimeError("Task noise requested before initialization.")
        global_noise = self._unpack(values)[2]
        return global_noise * np.exp(self._task_noise_log_deviations(values))

    def _training_noise_variance(
        self, theta: np.ndarray, X: np.ndarray
    ) -> np.ndarray:
        task = self.schema.task_role
        if task is None:
            raise RuntimeError("Task-conditioned noise requires a task role.")
        identity = np.asarray(X[:, self.schema.identity_slice(task)], dtype=float)
        if np.any(np.max(np.abs(identity), axis=1) <= 1e-12):
            raise ValueError("Task identity block contains an unencoded task.")
        task_indices = np.argmax(identity, axis=1)
        noises = self.task_noise_vector(theta)
        return noises[task_indices] ** 2

    def _prior_penalty(self, theta: np.ndarray) -> float:
        penalty = super()._prior_penalty(theta)
        contrasts = np.asarray(
            theta[
                self._task_noise_start : self._task_noise_start
                + self._task_noise_count
            ],
            dtype=float,
        )
        penalty += 0.5 * float(
            np.sum((contrasts / self.task_noise_prior_scale) ** 2)
        )
        return penalty

    def _record_task_noises(self) -> None:
        if self.theta_ is None:
            raise RuntimeError("Cannot record task noises before fitting.")
        task = self.schema.task_role
        if task is None:
            raise RuntimeError("Task-conditioned noise requires a task role.")
        identity_spec = self.schema.identity_slices[task]
        names = self.schema.feature_names[identity_spec.start : identity_spec.stop]
        noises = self.task_noise_vector(self.theta_)
        deviations = self._task_noise_log_deviations(self.theta_)
        self.task_noises_ = {
            name: float(value) for name, value in zip(names, noises, strict=True)
        }
        self.task_noise_log_deviations_ = {
            name: float(value)
            for name, value in zip(names, deviations, strict=True)
        }
        self.task_noise_contrasts_ = np.asarray(
            self.theta_[
                self._task_noise_start : self._task_noise_start
                + self._task_noise_count
            ],
            dtype=float,
        ).copy()

    def fit(
        self, X: np.ndarray, y: np.ndarray
    ) -> "TaskNoiseComponentARDLowRankTaskGP":
        super().fit(X, y)
        self._record_task_noises()
        return self
