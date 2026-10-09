from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SliceSpec:
    """Serializable half-open column slice."""

    start: int
    stop: int

    def as_slice(self) -> slice:
        return slice(self.start, self.stop)

    @property
    def width(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class FeatureSchema:
    """Column layout emitted by :class:`FrozenReactionEncoder`.

    ``role_slices`` include the exact-identity, rule-derived, optional name
    subword, and optional molecular features for each role. ``identity_slices``
    contain only the exact one-hot identity columns and are used for exact
    same-task checks.
    """

    dataset_name: str
    roles: tuple[str, ...]
    condition_roles: tuple[str, ...]
    task_role: str | None
    role_slices: dict[str, SliceSpec]
    identity_slices: dict[str, SliceSpec]
    feature_names: tuple[str, ...]
    candidate_id_column: str = "query_index"
    derived_roles: dict[str, tuple[str, ...]] = field(default_factory=dict)
    fixed_roles: dict[str, str] = field(default_factory=dict)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def role_slice(self, role: str) -> slice:
        return self.role_slices[role].as_slice()

    def identity_slice(self, role: str) -> slice:
        return self.identity_slices[role].as_slice()

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "roles": list(self.roles),
            "condition_roles": list(self.condition_roles),
            "task_role": self.task_role,
            "role_slices": {
                key: {"start": value.start, "stop": value.stop}
                for key, value in self.role_slices.items()
            },
            "identity_slices": {
                key: {"start": value.start, "stop": value.stop}
                for key, value in self.identity_slices.items()
            },
            "feature_names": list(self.feature_names),
            "candidate_id_column": self.candidate_id_column,
            "derived_roles": {key: list(value) for key, value in self.derived_roles.items()},
            "fixed_roles": dict(self.fixed_roles),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeatureSchema":
        return cls(
            dataset_name=str(data["dataset_name"]),
            roles=tuple(data["roles"]),
            condition_roles=tuple(data["condition_roles"]),
            task_role=data.get("task_role"),
            role_slices={
                key: SliceSpec(int(value["start"]), int(value["stop"]))
                for key, value in data["role_slices"].items()
            },
            identity_slices={
                key: SliceSpec(int(value["start"]), int(value["stop"]))
                for key, value in data["identity_slices"].items()
            },
            feature_names=tuple(data["feature_names"]),
            candidate_id_column=str(data.get("candidate_id_column", "query_index")),
            derived_roles={
                key: tuple(value) for key, value in data.get("derived_roles", {}).items()
            },
            fixed_roles={
                str(key): str(value) for key, value in data.get("fixed_roles", {}).items()
            },
        )


@dataclass
class Posterior:
    """Predictive posterior summary on a candidate set."""

    mean: np.ndarray
    std: np.ndarray
    component_means: dict[str, np.ndarray] = field(default_factory=dict)
    component_stds: dict[str, np.ndarray] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    disagreement: np.ndarray | None = None
    covariance: np.ndarray | None = None

    def validate(self, n_rows: int) -> None:
        self.mean = np.asarray(self.mean, dtype=float).reshape(-1)
        self.std = np.asarray(self.std, dtype=float).reshape(-1)
        if self.mean.shape != (n_rows,) or self.std.shape != (n_rows,):
            raise ValueError(
                f"Posterior shape mismatch: mean={self.mean.shape}, "
                f"std={self.std.shape}, expected {(n_rows,)}"
            )
        if not np.all(np.isfinite(self.mean)):
            raise FloatingPointError("Posterior means contain non-finite values.")
        if not np.all(np.isfinite(self.std)):
            raise FloatingPointError("Posterior standard deviations contain non-finite values.")
        self.std = np.maximum(self.std, 1e-12)


@dataclass(frozen=True)
class Recommendation:
    """One selected candidate and the decision diagnostics used to select it."""

    local_index: int
    query_index: int
    acquisition_name: str
    acquisition_value: float
    predicted_mean: float
    predicted_std: float
    component_means: dict[str, float]
    component_stds: dict[str, float]
    model_weights: dict[str, float]
    prior_weight: float
