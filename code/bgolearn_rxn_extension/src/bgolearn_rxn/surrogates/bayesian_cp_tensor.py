from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import InteractionModelConfig
from ..schemas import FeatureSchema, Posterior
from ..transforms import StandardTargetTransform, make_target_transform


@dataclass(frozen=True)
class _CPMember:
    intercept: float
    additive: tuple[np.ndarray, ...]
    factors: tuple[np.ndarray, ...]
    unknown_additive: np.ndarray
    unknown_factors: np.ndarray
    noise_variance: float


class BayesianCPTensorSurrogate:
    """Shrinkage Bayesian rank-3 CP model over five reaction identities.

    Conditional Gaussian updates learn every categorical factor from observed
    yields. Independent Bayesian-bootstrap chains form a seeded posterior
    ensemble. Additive effects stabilize tiny samples while the CP product is
    the only high-order term.
    """

    name = "bayesian_cp_tensor"
    roles = ("Electrophile", "Nucleophile", "Ligand", "Base", "Solvent")
    rank = 3
    ensemble_size = 16
    sweeps = 16
    additive_precision = 6.0
    factor_precision = 2.0
    intercept_precision = 0.25
    initial_factor_scale = 0.70
    noise_floor = 0.04
    noise_ceiling = 1.50

    def __init__(
        self,
        schema: FeatureSchema,
        config: InteractionModelConfig,
        *,
        seed: int = 0,
    ) -> None:
        self.schema = schema
        self.config = config
        self.seed = int(seed)
        missing = [role for role in self.roles if role not in schema.identity_slices]
        if missing:
            raise ValueError(f"Bayesian CP requires raw categorical roles: {missing}")
        self.level_counts = tuple(schema.identity_slices[role].width for role in self.roles)
        if any(count < 1 for count in self.level_counts):
            raise ValueError("Bayesian CP identity blocks must be nonempty.")
        self.target_transform = make_target_transform(config.target_transform)
        self.members_: list[_CPMember] = []
        self.fit_history_: list[dict[str, object]] = []
        self.last_prediction_diagnostics_: dict[str, object] = {}
        self._fit_calls = 0

    def _category_indices(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.schema.n_features:
            raise ValueError("Bayesian CP received an incompatible feature matrix.")
        columns = []
        for role in self.roles:
            block = np.asarray(X[:, self.schema.identity_slice(role)], dtype=float)
            maximum = np.max(block, axis=1)
            indices = np.argmax(block, axis=1).astype(int)
            indices[maximum <= 1e-12] = -1
            columns.append(indices)
        return np.column_stack(columns)

    @staticmethod
    def _known_take(values: np.ndarray, indices: np.ndarray, unknown) -> np.ndarray:
        output_shape = (len(indices),) + values.shape[1:]
        output = np.broadcast_to(np.asarray(unknown), output_shape).copy()
        known = indices >= 0
        output[known] = values[indices[known]]
        return output

    def _latent_member_mean(self, categories: np.ndarray, member: _CPMember) -> np.ndarray:
        additive = np.full(len(categories), member.intercept, dtype=float)
        product = np.ones((len(categories), self.rank), dtype=float)
        for role_index in range(len(self.roles)):
            indices = categories[:, role_index]
            additive += self._known_take(
                member.additive[role_index],
                indices,
                member.unknown_additive[role_index],
            )
            product *= self._known_take(
                member.factors[role_index],
                indices,
                member.unknown_factors[role_index],
            )
        return np.clip(additive + np.sum(product, axis=1), -5.0, 5.0)

    def _sample_member(
        self, categories: np.ndarray, y: np.ndarray, rng: np.random.Generator
    ) -> _CPMember:
        n_rows = len(y)
        weights = rng.exponential(scale=1.0, size=n_rows)
        weights *= n_rows / max(float(weights.sum()), 1e-12)
        intercept = float(rng.normal(0.0, 0.10))
        additive = [np.zeros(count, dtype=float) for count in self.level_counts]
        factors = [
            rng.normal(0.0, self.initial_factor_scale, size=(count, self.rank))
            for count in self.level_counts
        ]
        noise_variance = 0.35

        for _ in range(self.sweeps):
            additive_prediction = np.full(n_rows, intercept, dtype=float)
            for role_index in range(len(self.roles)):
                additive_prediction += additive[role_index][categories[:, role_index]]
            cp_product = np.ones((n_rows, self.rank), dtype=float)
            for role_index in range(len(self.roles)):
                cp_product *= factors[role_index][categories[:, role_index]]
            cp_prediction = np.sum(cp_product, axis=1)

            residual = y - (additive_prediction - intercept) - cp_prediction
            precision = self.intercept_precision + weights.sum() / noise_variance
            mean = float(np.sum(weights * residual) / noise_variance / precision)
            intercept = float(mean + rng.normal() / np.sqrt(precision))

            for role_index, count in enumerate(self.level_counts):
                other_additive = np.full(n_rows, intercept, dtype=float)
                for other in range(len(self.roles)):
                    if other != role_index:
                        other_additive += additive[other][categories[:, other]]
                response = y - other_additive - cp_prediction
                for level in range(count):
                    mask = categories[:, role_index] == level
                    if not np.any(mask):
                        additive[role_index][level] = rng.normal(
                            0.0, 1.0 / np.sqrt(self.additive_precision)
                        )
                        continue
                    local_weight = weights[mask]
                    local_precision = (
                        self.additive_precision + local_weight.sum() / noise_variance
                    )
                    local_mean = float(
                        np.sum(local_weight * response[mask])
                        / noise_variance
                        / local_precision
                    )
                    additive[role_index][level] = (
                        local_mean + rng.normal() / np.sqrt(local_precision)
                    )

                additive_prediction = np.full(n_rows, intercept, dtype=float)
                for other in range(len(self.roles)):
                    additive_prediction += additive[other][categories[:, other]]

            response = y - additive_prediction
            for role_index, count in enumerate(self.level_counts):
                design = np.ones((n_rows, self.rank), dtype=float)
                for other in range(len(self.roles)):
                    if other != role_index:
                        design *= factors[other][categories[:, other]]
                for level in range(count):
                    mask = categories[:, role_index] == level
                    if not np.any(mask):
                        factors[role_index][level] = rng.normal(
                            0.0,
                            1.0 / np.sqrt(self.factor_precision),
                            size=self.rank,
                        )
                        continue
                    local_design = design[mask]
                    local_weight = weights[mask]
                    weighted_design = local_design * local_weight[:, None]
                    precision_matrix = (
                        self.factor_precision * np.eye(self.rank)
                        + local_design.T @ weighted_design / noise_variance
                    )
                    covariance = np.linalg.inv(precision_matrix)
                    mean = (
                        covariance
                        @ (weighted_design.T @ response[mask])
                        / noise_variance
                    )
                    draw = rng.multivariate_normal(mean, covariance)
                    factors[role_index][level] = np.clip(draw, -3.0, 3.0)

            member = _CPMember(
                intercept=intercept,
                additive=tuple(additive),
                factors=tuple(factors),
                unknown_additive=np.zeros(len(self.roles)),
                unknown_factors=np.zeros((len(self.roles), self.rank)),
                noise_variance=noise_variance,
            )
            fitted = self._latent_member_mean(categories, member)
            residual = y - fitted
            shape = 3.0 + 0.5 * n_rows
            scale = 0.10 + 0.5 * float(np.sum(weights * residual**2))
            noise_variance = float(scale / rng.gamma(shape))
            noise_variance = float(
                np.clip(noise_variance, self.noise_floor, self.noise_ceiling)
            )

        unknown_additive = rng.normal(
            0.0, 1.0 / np.sqrt(self.additive_precision), size=len(self.roles)
        )
        unknown_factors = rng.normal(
            0.0,
            1.0 / np.sqrt(self.factor_precision),
            size=(len(self.roles), self.rank),
        )
        return _CPMember(
            intercept=intercept,
            additive=tuple(array.copy() for array in additive),
            factors=tuple(array.copy() for array in factors),
            unknown_additive=unknown_additive,
            unknown_factors=unknown_factors,
            noise_variance=noise_variance,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BayesianCPTensorSurrogate":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError("X must be 2D and align with y.")
        categories = self._category_indices(X)
        if np.any(categories < 0):
            raise ValueError("Training rows contain an unseen raw categorical identity.")
        self.target_transform.fit(y)
        transformed_y = self.target_transform.transform(y)
        self._fit_calls += 1
        members = []
        for member_index in range(self.ensemble_size):
            rng = np.random.default_rng(
                self.seed + 104729 * self._fit_calls + 7919 * member_index
            )
            members.append(self._sample_member(categories, transformed_y, rng))
        self.members_ = members
        factor_rms = float(
            np.sqrt(
                np.mean(
                    [
                        value**2
                        for member in members
                        for factor in member.factors
                        for value in factor.reshape(-1)
                    ]
                )
            )
        )
        self.fit_history_.append(
            {
                "fit_call": self._fit_calls,
                "n_rows": len(y),
                "rank": self.rank,
                "ensemble_size": self.ensemble_size,
                "factor_rms": factor_rms,
                "noise_std_min": float(
                    np.sqrt(min(member.noise_variance for member in members))
                ),
                "noise_std_max": float(
                    np.sqrt(max(member.noise_variance for member in members))
                ),
            }
        )
        return self

    def _yield_moments(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not self.members_:
            raise RuntimeError("Bayesian CP must be fitted before prediction.")
        categories = self._category_indices(X)
        latent_means = np.asarray(
            [self._latent_member_mean(categories, member) for member in self.members_]
        )
        latent_noises = np.asarray(
            [member.noise_variance for member in self.members_], dtype=float
        )
        yield_means = []
        yield_variances = []
        for mean, noise in zip(latent_means, latent_noises, strict=True):
            yield_mean, yield_std = self.target_transform.inverse_mean_std(
                mean, np.full(len(X), np.sqrt(noise), dtype=float)
            )
            yield_means.append(yield_mean)
            yield_variances.append(yield_std**2)
        means = np.asarray(yield_means)
        within = np.asarray(yield_variances)
        mixture_mean = means.mean(axis=0)
        mixture_variance = np.mean(within + means**2, axis=0) - mixture_mean**2
        return mixture_mean, np.maximum(mixture_variance, 1e-12), means, within

    def predict(self, X: np.ndarray) -> Posterior:
        mean, variance, member_means, member_variances = self._yield_moments(X)
        disagreement = np.std(member_means, axis=0, ddof=0)
        self.last_prediction_diagnostics_ = {
            "n_rows": len(mean),
            "std_min": float(np.sqrt(variance).min()),
            "std_max": float(np.sqrt(variance).max()),
            "disagreement_mean": float(disagreement.mean()),
            "within_std_mean": float(np.sqrt(member_variances.mean(axis=0)).mean()),
        }
        posterior = Posterior(
            mean=mean,
            std=np.sqrt(variance),
            disagreement=disagreement,
        )
        posterior.validate(len(mean))
        return posterior

    def posterior_covariance(self, X: np.ndarray) -> np.ndarray:
        mean, _, member_means, member_variances = self._yield_moments(X)
        centered = member_means - mean
        covariance = centered.T @ centered / len(self.members_)
        covariance.flat[:: len(covariance) + 1] += member_variances.mean(axis=0)
        covariance = 0.5 * (covariance + covariance.T)
        covariance.flat[:: len(covariance) + 1] = np.maximum(
            np.diag(covariance), 1e-12
        )
        return covariance
