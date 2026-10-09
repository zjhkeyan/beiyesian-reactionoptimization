from __future__ import annotations

import numpy as np

from ..config import GPConfig
from ..schemas import FeatureSchema
from .anova_gp import StructuredANOVAGP


class LowRankTaskCovarianceGP(StructuredANOVAGP):
    """Structured GP with a learned low-rank PSD task correlation matrix."""

    name = "low_rank_task_gp"

    def __init__(
        self,
        schema: FeatureSchema,
        config: GPConfig,
        *,
        seed: int = 0,
        task_rank: int = 2,
        task_prior_scale: float = 0.35,
    ) -> None:
        super().__init__(schema, config, seed=seed)
        task = schema.task_role
        if task is None:
            raise ValueError("Low-rank task covariance requires a configured task role.")
        self.n_tasks = schema.identity_slices[task].width
        self.task_rank = min(int(task_rank), self.n_tasks)
        if self.task_rank <= 0 or task_prior_scale <= 0.0:
            raise ValueError("Task rank and prior scale must be positive.")
        self.task_prior_scale = float(task_prior_scale)
        self._condition_parameter_count = len(self.roles) + 3
        self.task_correlation_: np.ndarray | None = None

    def _unpack(
        self, theta: np.ndarray
    ) -> tuple[dict[str, float], float, float, float, float]:
        lengthscales = {
            role: float(np.exp(theta[index])) for index, role in enumerate(self.roles)
        }
        offset = len(self.roles)
        output_scale = float(np.exp(theta[offset]))
        noise = float(np.exp(theta[offset + 1]))
        pair_scale = float(np.exp(theta[offset + 2]))
        return lengthscales, output_scale, noise, pair_scale, 1.0

    def _task_parameters(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        start = self._condition_parameter_count
        loading_stop = start + self.n_tasks * self.task_rank
        loadings = np.asarray(theta[start:loading_stop], dtype=float).reshape(
            self.n_tasks, self.task_rank
        )
        log_diagonal = np.asarray(
            theta[loading_stop : loading_stop + self.n_tasks], dtype=float
        )
        return loadings, log_diagonal

    def task_correlation(self, theta: np.ndarray | None = None) -> np.ndarray:
        values = self.theta_ if theta is None else np.asarray(theta, dtype=float)
        if values is None:
            raise RuntimeError("Task correlation requested before initialization.")
        loadings, log_diagonal = self._task_parameters(values)
        covariance = loadings @ loadings.T
        covariance.flat[:: self.n_tasks + 1] += np.exp(2.0 * log_diagonal)
        scale = np.sqrt(np.maximum(np.diag(covariance), 1e-12))
        correlation = covariance / np.outer(scale, scale)
        correlation = 0.5 * (correlation + correlation.T)
        correlation.flat[:: self.n_tasks + 1] = 1.0
        return correlation

    def _task_pair_codes(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        task = self.schema.task_role
        if task is None:
            raise RuntimeError("Low-rank task covariance lost its task role.")
        block1 = np.asarray(X1[:, self.schema.identity_slice(task)], dtype=float)
        block2 = np.asarray(X2[:, self.schema.identity_slice(task)], dtype=float)
        if np.any(np.max(np.abs(block1), axis=1) <= 1e-12) or np.any(
            np.max(np.abs(block2), axis=1) <= 1e-12
        ):
            raise ValueError("Task identity block contains an unencoded task.")
        indices1 = np.argmax(block1, axis=1)
        indices2 = np.argmax(block2, axis=1)
        return indices1[:, None] * self.n_tasks + indices2[None, :]

    def _task_descriptor_similarity(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        # The base fit path passes this matrix into _kernel_from_distances. Here
        # it carries task-pair indices so the learned PSD matrix can be applied.
        return self._task_pair_codes(X1, X2)

    def _kernel_from_distances(
        self,
        distances: dict[str, np.ndarray],
        same_task: np.ndarray,
        task_similarity: np.ndarray,
        theta: np.ndarray,
    ) -> np.ndarray:
        del same_task
        lengthscales, _, _, pair_scale, _ = self._unpack(theta)
        role_kernels = {
            role: np.exp(-0.5 * distances[role] / (lengthscales[role] ** 2))
            for role in self.roles
        }
        kernel = np.ones_like(next(iter(role_kernels.values())), dtype=float)
        normalizer = 1.0
        for role in self.roles:
            weight = max(self._role_weight(role), 0.0)
            kernel += weight * role_kernels[role]
            normalizer += weight
        for pair in self.config.interaction_pairs:
            weight = max(self._interaction_weight(pair), 0.0)
            kernel += pair_scale * weight * role_kernels[pair[0]] * role_kernels[pair[1]]
            normalizer += pair_scale * weight
        condition_kernel = kernel / max(normalizer, 1e-12)
        task_kernel = self.task_correlation(theta).reshape(-1)[
            np.asarray(task_similarity, dtype=int)
        ]
        return condition_kernel * task_kernel

    def _initial_theta(self) -> np.ndarray:
        values = [np.log(self.config.initial_lengthscale)] * len(self.roles)
        values.extend(
            [
                np.log(1.0),
                np.log(self.config.initial_noise),
                np.log(self.config.initial_pair_scale),
            ]
        )
        rho = float(np.clip(self.config.initial_task_rho, 0.01, 0.99))
        loadings = np.zeros((self.n_tasks, self.task_rank), dtype=float)
        loadings[:, 0] = np.sqrt(rho)
        if self.task_rank > 1:
            loadings[:, 1] = np.linspace(-0.05, 0.05, self.n_tasks)
        values.extend(loadings.reshape(-1).tolist())
        values.extend([0.5 * np.log(1.0 - rho)] * self.n_tasks)
        return np.asarray(values, dtype=float)

    def _bounds(self) -> list[tuple[float, float]]:
        bounds = [(np.log(0.08), np.log(12.0)) for _ in self.roles]
        bounds.extend(
            [
                (np.log(0.05), np.log(20.0)),
                (np.log(1e-4), np.log(0.75)),
                (np.log(1e-4), np.log(4.0)),
            ]
        )
        bounds.extend([(-2.0, 2.0)] * (self.n_tasks * self.task_rank))
        bounds.extend([(np.log(0.05), np.log(2.0))] * self.n_tasks)
        return bounds

    def _prior_penalty(self, theta: np.ndarray) -> float:
        initial = self._initial_theta()
        offset = len(self.roles)
        penalty = 0.5 * float(
            np.sum(((theta[: len(self.roles)] - initial[: len(self.roles)]) / 0.9) ** 2)
        )
        penalty += 0.5 * float(((theta[offset] - 0.0) / 1.0) ** 2)
        penalty += 0.5 * float(((theta[offset + 1] - np.log(0.05)) / 1.2) ** 2)
        penalty += 0.5 * float(
            ((theta[offset + 2] - np.log(self.config.initial_pair_scale)) / 0.8) ** 2
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
        return penalty

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LowRankTaskCovarianceGP":
        super().fit(X, y)
        self.task_correlation_ = self.task_correlation()
        return self
