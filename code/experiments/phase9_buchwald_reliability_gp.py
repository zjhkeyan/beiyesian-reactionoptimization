"""Experiment-local reliability shrinkage for the Buchwald task covariance."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from bgolearn_rxn.surrogates.component_ard_gp import ComponentARDLowRankTaskGP


class ReliabilityShrunkComponentARDLowRankTaskGP(ComponentARDLowRankTaskGP):
    """Low-rank task GP with source reliability embedded in its PSD kernel.

    A cold target is identified by its smaller observation count.  Each fully
    observed source task is predicted from the other fully observed sources
    with a fixed condition-only kernel.  Its held-out rank correlation becomes
    a reliability multiplier.  The learned task correlation ``R`` is replaced
    by ``D R D + diag(1 - D**2)``.  This retains unit marginal variances and is
    positive semidefinite for every diagonal ``D`` in ``[0, 1]``.
    """

    name = "reliability_shrunk_component_ard_low_rank_task_gp"

    def __init__(
        self,
        schema,
        config,
        *,
        seed: int = 0,
        reliability_floor: float = 0.35,
        reliability_power: float = 1.0,
        ridge: float = 0.10,
    ) -> None:
        super().__init__(schema, config, seed=seed)
        if not 0.0 <= reliability_floor <= 1.0:
            raise ValueError("reliability_floor must lie in [0, 1].")
        if reliability_power <= 0.0 or ridge <= 0.0:
            raise ValueError("reliability_power and ridge must be positive.")
        self.reliability_floor = float(reliability_floor)
        self.reliability_power = float(reliability_power)
        self.ridge = float(ridge)
        self.target_task_index_: int | None = None
        self.source_reliability_: dict[int, float] = {}
        self.task_multipliers_ = np.ones(self.n_tasks, dtype=float)
        self.raw_task_correlation_: np.ndarray | None = None

    def _task_indices(self, X: np.ndarray) -> np.ndarray:
        task = self.schema.task_role
        if task is None:
            raise RuntimeError("Reliability shrinkage requires a task role.")
        block = np.asarray(X[:, self.schema.identity_slice(task)], dtype=float)
        if np.any(np.max(np.abs(block), axis=1) <= 1e-12):
            raise ValueError("Task identity block contains an unencoded task.")
        return np.argmax(block, axis=1)

    @staticmethod
    def _within_task_standardize(y: np.ndarray, tasks: np.ndarray) -> np.ndarray:
        output = np.zeros(len(y), dtype=float)
        for task_index in np.unique(tasks):
            mask = tasks == task_index
            values = np.asarray(y[mask], dtype=float)
            scale = float(np.std(values, ddof=0))
            output[mask] = (values - float(np.mean(values))) / max(scale, 1e-8)
        return output

    def _fixed_condition_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        lengthscale = float(self.config.initial_lengthscale)
        role_kernels = {
            role: np.exp(
                -0.5 * self._role_sqdist(X1, X2, role) / (lengthscale**2)
            )
            for role in self.roles
        }
        kernel = np.ones((len(X1), len(X2)), dtype=float)
        normalizer = 1.0
        for role in self.roles:
            weight = max(self._role_weight(role), 0.0)
            kernel += weight * role_kernels[role]
            normalizer += weight
        pair_scale = float(self.config.initial_pair_scale)
        for pair in self.config.interaction_pairs:
            weight = max(self._interaction_weight(pair), 0.0) * pair_scale
            kernel += weight * role_kernels[pair[0]] * role_kernels[pair[1]]
            normalizer += weight
        return kernel / max(normalizer, 1e-12)

    def _estimate_source_reliability(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[int, dict[int, float], np.ndarray]:
        task_indices = self._task_indices(X)
        counts = np.bincount(task_indices, minlength=self.n_tasks)
        observed = np.flatnonzero(counts > 0)
        if len(observed) < 3:
            raise ValueError("Reliability shrinkage needs at least three observed tasks.")
        minimum = int(np.min(counts[observed]))
        cold = observed[counts[observed] == minimum]
        if len(cold) != 1 or np.any(counts[observed[observed != cold[0]]] <= minimum):
            raise ValueError("A unique cold target must have fewer rows than every source.")
        target = int(cold[0])
        sources = observed[observed != target]
        standardized = self._within_task_standardize(np.asarray(y, dtype=float), task_indices)
        scores: dict[int, float] = {}

        for held_out in sources:
            validation = task_indices == held_out
            training = np.isin(task_indices, sources[sources != held_out])
            if int(training.sum()) < 2 or int(validation.sum()) < 3:
                raise ValueError("Each source needs enough rows for leave-one-source-out scoring.")
            train_kernel = self._fixed_condition_kernel(X[training], X[training])
            cross_kernel = self._fixed_condition_kernel(X[validation], X[training])
            regularized = train_kernel.copy()
            regularized.flat[:: len(regularized) + 1] += self.ridge
            try:
                coefficients = np.linalg.solve(regularized, standardized[training])
            except np.linalg.LinAlgError:
                coefficients = np.linalg.lstsq(
                    regularized, standardized[training], rcond=None
                )[0]
            prediction = cross_kernel @ coefficients
            correlation = float(
                spearmanr(prediction, standardized[validation]).statistic
            )
            if not np.isfinite(correlation):
                correlation = 0.0
            scores[int(held_out)] = float(np.clip(correlation, 0.0, 1.0))

        multipliers = np.ones(self.n_tasks, dtype=float)
        for source, score in scores.items():
            quality = score**self.reliability_power
            multipliers[source] = self.reliability_floor + (
                1.0 - self.reliability_floor
            ) * quality
        return target, scores, multipliers

    def task_correlation(self, theta: np.ndarray | None = None) -> np.ndarray:
        raw = super().task_correlation(theta)
        multipliers = np.asarray(self.task_multipliers_, dtype=float)
        shrunk = multipliers[:, None] * raw * multipliers[None, :]
        shrunk.flat[:: self.n_tasks + 1] += 1.0 - multipliers**2
        shrunk = 0.5 * (shrunk + shrunk.T)
        shrunk.flat[:: self.n_tasks + 1] = 1.0
        return shrunk

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        target, scores, multipliers = self._estimate_source_reliability(X, y)
        self.target_task_index_ = target
        self.source_reliability_ = scores
        self.task_multipliers_ = multipliers
        super().fit(X, y)
        if self.theta_ is None:
            raise RuntimeError("Reliability GP fit did not produce parameters.")
        self.raw_task_correlation_ = super().task_correlation(self.theta_)
        self.task_correlation_ = self.task_correlation(self.theta_)
        return self
