from __future__ import annotations

import numpy as np
from scipy.linalg import helmert
from scipy.spatial.distance import cdist

from ..config import GPConfig
from ..schemas import FeatureSchema
from .component_ard_gp import ComponentARDStructuredGP


class PairContextCoregionalizedGP(ComponentARDStructuredGP):
    """Component-ARD exact GP with an identifiable substrate-pair block.

    Electrophile and nucleophile main effects are retained exactly once.  The
    ordinary derived ``SubstratePair`` RBF is replaced by their product and an
    exact-pair diagonal.  Pair-specific diagonal residuals use zero-sum
    Helmert contrasts, so the shared exact-pair strength is the geometric mean
    diagonal and cannot be exchanged with an unconstrained residual intercept.
    """

    name = "pair_context_coregionalized_gp"
    pair_role = "SubstratePair"
    constituent_roles = ("Electrophile", "Nucleophile")
    residual_prior_scale = 0.30
    residual_contrast_bound = 1.25

    def __init__(
        self,
        schema: FeatureSchema,
        config: GPConfig,
        *,
        seed: int = 0,
    ) -> None:
        sources = tuple(schema.derived_roles.get(self.pair_role, ()))
        if sources != self.constituent_roles:
            raise ValueError(
                f"{self.name} requires {self.pair_role} derived from "
                f"{self.constituent_roles}, got {sources or None}."
            )
        if schema.task_role is not None:
            raise ValueError(f"{self.name} is a single-task Suzuki surrogate.")
        super().__init__(schema, config, seed=seed)

        # The derived pair RBF is deliberately absent: its information is a
        # deterministic duplicate of the two constituent blocks.
        self.roles = tuple(role for role in self.roles if role != self.pair_role)
        self.non_pair_interactions = tuple(
            pair for pair in config.interaction_pairs if self.pair_role not in pair
        )
        self.n_pairs = schema.identity_slices[self.pair_role].width
        if self.n_pairs < 2:
            raise ValueError(f"{self.name} requires at least two pair contexts.")
        self._pair_contrast = helmert(self.n_pairs, full=False).T

        self._output_index = len(self.roles)
        self._noise_index = len(self.roles) + 1
        self._amplitude_start = len(self.roles) + 2
        self._ordinary_amplitude_count = len(self.roles) + len(
            self.non_pair_interactions
        )
        self._pair_product_amplitude_index = (
            self._amplitude_start + self._ordinary_amplitude_count
        )
        self._pair_diagonal_amplitude_index = self._pair_product_amplitude_index + 1
        self._amplitude_stop = self._pair_diagonal_amplitude_index + 1
        self._pair_residual_start = self._amplitude_stop
        self._pair_residual_count = self.n_pairs - 1
        self.pair_diagonal_: np.ndarray | None = None
        self.pair_residual_contrasts_: np.ndarray | None = None

    def _amplitude_centers(self) -> np.ndarray:
        role = [max(self._role_weight(name), 1e-4) for name in self.roles]
        interactions = [
            max(self.config.initial_pair_scale * self._interaction_weight(pair), 1e-4)
            for pair in self.non_pair_interactions
        ]
        inherited_pair_strength = max(self._role_weight(self.pair_role), 1e-4)
        pair_product = 0.5 * inherited_pair_strength
        pair_diagonal = 0.5 * inherited_pair_strength
        return np.log(
            np.asarray(role + interactions + [pair_product, pair_diagonal], dtype=float)
        )

    def _amplitudes(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.exp(
            np.asarray(theta[self._amplitude_start : self._amplitude_stop], dtype=float)
        )
        return values[: len(self.roles)], values[len(self.roles) :]

    def _pair_indices(self, X: np.ndarray) -> np.ndarray:
        identity = np.asarray(X[:, self.schema.identity_slice(self.pair_role)], dtype=float)
        if identity.shape[1] != self.n_pairs:
            raise ValueError("SubstratePair identity width changed after initialization.")
        if np.any(np.max(np.abs(identity), axis=1) <= 1e-12):
            raise ValueError("SubstratePair identity block contains an unencoded pair.")
        return np.argmax(identity, axis=1)

    def _pair_log_deviations(self, theta: np.ndarray) -> np.ndarray:
        contrasts = np.asarray(
            theta[
                self._pair_residual_start : self._pair_residual_start
                + self._pair_residual_count
            ],
            dtype=float,
        )
        return self._pair_contrast @ contrasts

    def pair_diagonal(self, theta: np.ndarray | None = None) -> np.ndarray:
        values = self.theta_ if theta is None else np.asarray(theta, dtype=float)
        if values is None:
            raise RuntimeError("Pair diagonal requested before initialization.")
        shared = float(np.exp(values[self._pair_diagonal_amplitude_index]))
        return shared * np.exp(self._pair_log_deviations(values))

    def _condition_kernel_for_rows(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
        theta: np.ndarray,
    ) -> np.ndarray:
        lengthscales = self._unpack(theta)[0]
        distances = {
            role: self._role_sqdist(X1, X2, role) for role in self.roles
        }
        role_kernels = {
            role: np.exp(-0.5 * distances[role] / (lengthscales[role] ** 2))
            for role in self.roles
        }
        role_amplitudes, extra_amplitudes = self._amplitudes(theta)
        kernel = np.ones((len(X1), len(X2)), dtype=float)
        normalizer = 1.0
        for amplitude, role in zip(role_amplitudes, self.roles, strict=True):
            kernel += amplitude * role_kernels[role]
            normalizer += float(amplitude)

        interaction_amplitudes = extra_amplitudes[: len(self.non_pair_interactions)]
        for amplitude, pair in zip(
            interaction_amplitudes, self.non_pair_interactions, strict=True
        ):
            kernel += amplitude * role_kernels[pair[0]] * role_kernels[pair[1]]
            normalizer += float(amplitude)

        electrophile, nucleophile = self.constituent_roles
        product_amplitude = float(extra_amplitudes[-2])
        product = role_kernels[electrophile] * role_kernels[nucleophile]
        kernel += product_amplitude * product
        normalizer += product_amplitude

        left_pair = self._pair_indices(X1)
        right_pair = self._pair_indices(X2)
        same_pair = left_pair[:, None] == right_pair[None, :]
        diagonal = self.pair_diagonal(theta)
        pair_diagonal = same_pair * np.sqrt(
            diagonal[left_pair, None] * diagonal[right_pair][None, :]
        )
        kernel += pair_diagonal
        # The shared diagonal amplitude is the geometric-mean contribution.
        normalizer += float(np.exp(theta[self._pair_diagonal_amplitude_index]))
        return kernel / normalizer

    def _kernel_from_distances(
        self, distances, same_task, task_similarity, theta
    ) -> np.ndarray:
        del distances, same_task, task_similarity
        if self.X_train_ is None:
            raise RuntimeError("Training kernel requested before fit initialization.")
        return self._condition_kernel_for_rows(self.X_train_, self.X_train_, theta)

    def _kernel(self, X1: np.ndarray, X2: np.ndarray, theta: np.ndarray) -> np.ndarray:
        return self._condition_kernel_for_rows(X1, X2, theta)

    def _initial_theta(self) -> np.ndarray:
        values = [np.log(self.config.initial_lengthscale)] * len(self.roles)
        values.extend([np.log(1.0), np.log(self.config.initial_noise)])
        values.extend(self._amplitude_centers().tolist())
        values.extend([0.0] * self._pair_residual_count)
        return np.asarray(values, dtype=float)

    def _bounds(self) -> list[tuple[float, float]]:
        bounds = [(np.log(0.08), np.log(12.0)) for _ in self.roles]
        bounds.extend([(np.log(0.05), np.log(20.0)), (np.log(1e-4), np.log(0.75))])
        bounds.extend([(np.log(1e-3), np.log(20.0))] * len(self._amplitude_centers()))
        bounds.extend(
            [(-self.residual_contrast_bound, self.residual_contrast_bound)]
            * self._pair_residual_count
        )
        return bounds

    def _amplitude_penalty(self, theta: np.ndarray) -> float:
        log_amplitudes = np.asarray(
            theta[self._amplitude_start : self._amplitude_stop], dtype=float
        )
        centers = self._amplitude_centers()
        return 0.5 * float(
            np.sum(((log_amplitudes - centers) / self.amplitude_prior_scale) ** 2)
        )

    def _prior_penalty(self, theta: np.ndarray) -> float:
        initial = self._initial_theta()
        penalty = 0.5 * float(
            np.sum(((theta[: len(self.roles)] - initial[: len(self.roles)]) / 0.9) ** 2)
        )
        penalty += 0.5 * float(((theta[self._output_index] - 0.0) / 1.0) ** 2)
        penalty += 0.5 * float(
            ((theta[self._noise_index] - np.log(0.05)) / 1.2) ** 2
        )
        contrasts = theta[
            self._pair_residual_start : self._pair_residual_start
            + self._pair_residual_count
        ]
        penalty += 0.5 * float(
            np.sum((contrasts / self.residual_prior_scale) ** 2)
        )
        return penalty + self._amplitude_penalty(theta)

    def _record_component_amplitudes(self) -> None:
        if self.theta_ is None:
            raise RuntimeError("Cannot record amplitudes before fitting.")
        role_values, extra_values = self._amplitudes(self.theta_)
        raw = {"bias": 1.0}
        raw.update(
            {
                f"role:{role}": float(value)
                for role, value in zip(self.roles, role_values, strict=True)
            }
        )
        interaction_values = extra_values[: len(self.non_pair_interactions)]
        raw.update(
            {
                f"pair:{'|'.join(pair)}": float(value)
                for pair, value in zip(
                    self.non_pair_interactions, interaction_values, strict=True
                )
            }
        )
        raw["pair_context:constituent_product"] = float(extra_values[-2])
        raw["pair_context:exact_identity"] = float(extra_values[-1])
        total = float(sum(raw.values()))
        self.component_amplitudes_ = raw
        self.component_shares_ = {name: value / total for name, value in raw.items()}
        self.pair_diagonal_ = self.pair_diagonal(self.theta_)
        self.pair_residual_contrasts_ = np.asarray(
            self.theta_[
                self._pair_residual_start : self._pair_residual_start
                + self._pair_residual_count
            ],
            dtype=float,
        ).copy()
