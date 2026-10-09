from __future__ import annotations

import numpy as np

from ..config import GPConfig
from ..schemas import FeatureSchema
from .anova_gp import StructuredANOVAGP


class GraphDiffusionProductGP(StructuredANOVAGP):
    """Exact GP on the Cartesian product of categorical role graphs.

    Each raw role is a complete graph.  Its covariance is the complete-graph
    heat kernel ``exp(-beta L)`` normalized to unit diagonal, where the learned
    parameter is the dimensionless spectral diffusion ``q * beta``.  The
    condition covariance is the product over roles, equivalent to diffusion on
    their Cartesian-product graph.
    """

    name = "graph_diffusion_product_gp"
    diffusion_center = 1.5
    diffusion_prior_scale = 0.65
    diffusion_bounds = (0.05, 8.0)
    output_prior_scale = 1.0
    noise_prior_center = 0.05
    noise_prior_scale = 1.2

    def __init__(
        self,
        schema: FeatureSchema,
        config: GPConfig,
        *,
        seed: int = 0,
    ) -> None:
        super().__init__(schema=schema, config=config, seed=seed)
        raw_roles = tuple(
            role
            for role in schema.condition_roles
            if role != schema.task_role and role not in schema.derived_roles
        )
        if not raw_roles:
            raise ValueError("Graph diffusion GP requires at least one raw categorical role.")
        missing = [role for role in raw_roles if role not in schema.identity_slices]
        if missing:
            raise ValueError(f"Graph diffusion identity blocks are missing: {missing}")
        self.roles = raw_roles
        self.category_counts = {
            role: schema.identity_slices[role].width for role in self.roles
        }
        if any(count < 1 for count in self.category_counts.values()):
            raise ValueError("Every graph diffusion role must have a category.")
        self._output_index = len(self.roles)
        self._noise_index = len(self.roles) + 1
        self.diffusion_scales_: dict[str, float] = {}
        self.heat_times_: dict[str, float] = {}
        self.off_diagonal_correlations_: dict[str, float] = {}

    def _unpack(
        self, theta: np.ndarray
    ) -> tuple[dict[str, float], float, float, float, float]:
        diffusion = {
            role: float(np.exp(theta[index]))
            for index, role in enumerate(self.roles)
        }
        output_scale = float(np.exp(theta[self._output_index]))
        noise = float(np.exp(theta[self._noise_index]))
        return diffusion, output_scale, noise, 1.0, 1.0

    def _initial_theta(self) -> np.ndarray:
        values = [np.log(self.diffusion_center)] * len(self.roles)
        values.extend([0.0, np.log(self.config.initial_noise)])
        return np.asarray(values, dtype=float)

    def _bounds(self) -> list[tuple[float, float]]:
        lower, upper = (np.log(value) for value in self.diffusion_bounds)
        bounds = [(lower, upper)] * len(self.roles)
        bounds.extend(
            [(np.log(0.05), np.log(20.0)), (np.log(1e-4), np.log(0.75))]
        )
        return bounds

    def _prior_penalty(self, theta: np.ndarray) -> float:
        log_center = np.log(self.diffusion_center)
        penalty = 0.5 * float(
            np.sum(
                ((theta[: len(self.roles)] - log_center) / self.diffusion_prior_scale)
                ** 2
            )
        )
        penalty += 0.5 * float(
            (theta[self._output_index] / self.output_prior_scale) ** 2
        )
        penalty += 0.5 * float(
            (
                (theta[self._noise_index] - np.log(self.noise_prior_center))
                / self.noise_prior_scale
            )
            ** 2
        )
        return penalty

    @staticmethod
    def _normalized_complete_graph_heat(
        same_category: np.ndarray,
        *,
        category_count: int,
        spectral_diffusion: float,
    ) -> np.ndarray:
        """Return unit-diagonal ``exp(-beta L_Kq)`` with ``q*beta=diffusion``."""

        q = int(category_count)
        if q < 1 or not np.isfinite(spectral_diffusion) or spectral_diffusion <= 0:
            raise ValueError("Complete-graph heat parameters must be finite and positive.")
        decay = float(np.exp(-spectral_diffusion))
        off_diagonal = (1.0 - decay) / (1.0 + (q - 1.0) * decay)
        return off_diagonal + (1.0 - off_diagonal) * np.asarray(
            same_category, dtype=float
        )

    def _same_role_category(
        self, X1: np.ndarray, X2: np.ndarray, role: str
    ) -> np.ndarray:
        block1 = np.asarray(X1[:, self.schema.identity_slice(role)], dtype=float)
        block2 = np.asarray(X2[:, self.schema.identity_slice(role)], dtype=float)
        return np.isclose(block1 @ block2.T, 1.0, atol=1e-12).astype(float)

    def _kernel(
        self, X1: np.ndarray, X2: np.ndarray, theta: np.ndarray
    ) -> np.ndarray:
        diffusion, _, _, _, _ = self._unpack(theta)
        kernel = np.ones((len(X1), len(X2)), dtype=float)
        for role in self.roles:
            kernel *= self._normalized_complete_graph_heat(
                self._same_role_category(X1, X2, role),
                category_count=self.category_counts[role],
                spectral_diffusion=diffusion[role],
            )
        return kernel

    def _kernel_from_distances(
        self,
        distances: dict[str, np.ndarray],
        same_task: np.ndarray,
        task_similarity: np.ndarray,
        theta: np.ndarray,
    ) -> np.ndarray:
        del distances, same_task, task_similarity
        if self.X_train_ is None:
            raise RuntimeError("Graph diffusion training kernel requested before fit.")
        return self._kernel(self.X_train_, self.X_train_, theta)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GraphDiffusionProductGP":
        super().fit(X, y)
        if self.theta_ is None:
            raise RuntimeError("Graph diffusion fit did not produce parameters.")
        diffusion, _, _, _, _ = self._unpack(self.theta_)
        self.diffusion_scales_ = diffusion
        self.heat_times_ = {
            role: diffusion[role] / self.category_counts[role] for role in self.roles
        }
        self.off_diagonal_correlations_ = {
            role: float(
                self._normalized_complete_graph_heat(
                    np.zeros((1, 1), dtype=float),
                    category_count=self.category_counts[role],
                    spectral_diffusion=diffusion[role],
                )[0, 0]
            )
            for role in self.roles
        }
        return self
