from __future__ import annotations

import numpy as np

from ..config import GPConfig
from ..schemas import FeatureSchema
from .component_ard_gp import ComponentARDStructuredGP


class StrongHeredityComponentARDStructuredGP(ComponentARDStructuredGP):
    """Component-ARD GP with global-local strong-heredity interactions.

    Each interaction amplitude is reconstructed from learned parent role
    amplitudes instead of being independently parameterized::

        pair(a, b) = global * local(a, b) * sqrt(role(a) * role(b))

    The initial values reproduce the configured component weights exactly.
    Separate MAP priors on the global and local multipliers resolve their
    otherwise non-identifiable product while retaining episode-local fitting.
    """

    name = "strong_heredity_component_ard_gp"
    role_prior_scale = 0.60
    global_multiplier_prior_scale = 0.35
    local_multiplier_prior_scale = 0.45

    def __init__(
        self, schema: FeatureSchema, config: GPConfig, *, seed: int = 0
    ) -> None:
        super().__init__(schema, config, seed=seed)
        self._role_amplitude_stop = self._amplitude_start + len(self.roles)
        self._global_multiplier_index = self._role_amplitude_stop
        self._local_multiplier_start = self._global_multiplier_index + 1
        self.global_interaction_multiplier_: float | None = None
        self.local_interaction_multipliers_: dict[str, float] = {}
        self.parent_geometric_means_: dict[str, float] = {}

    def _role_amplitude_centers(self) -> np.ndarray:
        return np.log(
            np.asarray(
                [max(self._role_weight(role), 1e-4) for role in self.roles],
                dtype=float,
            )
        )

    def _local_multiplier_centers(self) -> np.ndarray:
        role_centers = np.exp(self._role_amplitude_centers())
        role_index = {role: index for index, role in enumerate(self.roles)}
        centers = []
        for pair in self.config.interaction_pairs:
            configured = max(
                self.config.initial_pair_scale * self._interaction_weight(pair),
                1e-4,
            )
            parent = np.sqrt(
                role_centers[role_index[pair[0]]]
                * role_centers[role_index[pair[1]]]
            )
            centers.append(np.log(configured / max(parent, 1e-12)))
        return np.asarray(centers, dtype=float)

    def _amplitude_centers(self) -> np.ndarray:
        return np.concatenate(
            (
                self._role_amplitude_centers(),
                np.asarray([0.0]),
                self._local_multiplier_centers(),
            )
        )

    def _multipliers(self, theta: np.ndarray) -> tuple[float, np.ndarray]:
        values = np.asarray(theta, dtype=float)
        global_multiplier = float(np.exp(values[self._global_multiplier_index]))
        local_multipliers = np.exp(values[self._local_multiplier_start :])
        return global_multiplier, local_multipliers

    def _amplitudes(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(theta, dtype=float)
        role_amplitudes = np.exp(
            values[self._amplitude_start : self._role_amplitude_stop]
        )
        global_multiplier, local_multipliers = self._multipliers(values)
        role_index = {role: index for index, role in enumerate(self.roles)}
        pair_amplitudes = np.asarray(
            [
                global_multiplier
                * local_multiplier
                * np.sqrt(
                    role_amplitudes[role_index[pair[0]]]
                    * role_amplitudes[role_index[pair[1]]]
                )
                for pair, local_multiplier in zip(
                    self.config.interaction_pairs,
                    local_multipliers,
                    strict=True,
                )
            ],
            dtype=float,
        )
        return role_amplitudes, pair_amplitudes

    def _amplitude_penalty(self, theta: np.ndarray) -> float:
        values = np.asarray(theta, dtype=float)
        role_logs = values[self._amplitude_start : self._role_amplitude_stop]
        global_log = values[self._global_multiplier_index]
        local_logs = values[self._local_multiplier_start :]
        penalty = 0.5 * float(
            np.sum(
                (
                    (role_logs - self._role_amplitude_centers())
                    / self.role_prior_scale
                )
                ** 2
            )
        )
        penalty += 0.5 * float(
            (global_log / self.global_multiplier_prior_scale) ** 2
        )
        penalty += 0.5 * float(
            np.sum(
                (
                    (local_logs - self._local_multiplier_centers())
                    / self.local_multiplier_prior_scale
                )
                ** 2
            )
        )
        return penalty

    def _bounds(self) -> list[tuple[float, float]]:
        condition_bounds = super()._bounds()[: self._amplitude_start]
        role_bounds = [(np.log(1e-3), np.log(20.0))] * len(self.roles)
        global_bounds = [(np.log(0.05), np.log(5.0))]
        local_bounds = [
            (np.log(1e-3), np.log(20.0))
        ] * len(self.config.interaction_pairs)
        return condition_bounds + role_bounds + global_bounds + local_bounds

    def _record_component_amplitudes(self) -> None:
        super()._record_component_amplitudes()
        if self.theta_ is None:
            raise RuntimeError("Cannot record heredity multipliers before fitting.")
        role_amplitudes, _ = self._amplitudes(self.theta_)
        global_multiplier, local_multipliers = self._multipliers(self.theta_)
        role_index = {role: index for index, role in enumerate(self.roles)}
        self.global_interaction_multiplier_ = global_multiplier
        self.local_interaction_multipliers_ = {
            "|".join(pair): float(value)
            for pair, value in zip(
                self.config.interaction_pairs, local_multipliers, strict=True
            )
        }
        self.parent_geometric_means_ = {
            "|".join(pair): float(
                np.sqrt(
                    role_amplitudes[role_index[pair[0]]]
                    * role_amplitudes[role_index[pair[1]]]
                )
            )
            for pair in self.config.interaction_pairs
        }
