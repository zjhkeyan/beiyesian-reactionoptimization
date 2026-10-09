from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import DatasetConfig


@dataclass(frozen=True)
class DatasetSpec:
    """Immutable public data contract for one benchmark task."""

    name: str
    feature_columns: tuple[str, ...]
    target_column: str
    task_column: str | None
    target_task_value: str | None
    condition_roles: tuple[str, ...]
    derived_roles: dict[str, tuple[str, ...]]
    fixed_roles: dict[str, str]
    budget: int
    seeds: tuple[int, ...]

    @classmethod
    def from_config(cls, config: DatasetConfig) -> "DatasetSpec":
        return cls(
            name=config.name,
            feature_columns=config.feature_columns,
            target_column=config.target_column,
            task_column=config.task_column,
            target_task_value=config.target_task_value,
            condition_roles=config.condition_roles or config.feature_columns,
            derived_roles=config.derived_roles,
            fixed_roles=config.fixed_roles,
            budget=config.budget,
            seeds=config.seeds,
        )

    @property
    def roles(self) -> tuple[str, ...]:
        roles = tuple(self.feature_columns) + tuple(self.derived_roles) + tuple(self.fixed_roles)
        if len(roles) != len(set(roles)):
            raise ValueError("Feature, derived, and fixed role names must be distinct.")
        return roles

    def add_derived_roles(self, frame: pd.DataFrame) -> pd.DataFrame:
        output = frame.copy()
        for role, source_columns in self.derived_roles.items():
            missing = [column for column in source_columns if column not in output.columns]
            if missing:
                raise ValueError(
                    f"Cannot construct derived role {role!r}; missing columns {missing}."
                )
            output[role] = output.loc[:, list(source_columns)].astype(str).agg(" || ".join, axis=1)
        for role, value in self.fixed_roles.items():
            if role in output.columns:
                raise ValueError(f"Fixed role {role!r} collides with an input column.")
            output[role] = str(value)
        return output

    def target_mask(self, frame: pd.DataFrame) -> np.ndarray:
        if self.task_column is None:
            return np.ones(len(frame), dtype=bool)
        if self.target_task_value is None:
            raise ValueError("task_column is set but target_task_value is missing.")
        if self.task_column in frame.columns:
            task_values = frame[self.task_column]
        elif self.task_column in self.derived_roles or self.task_column in self.fixed_roles:
            task_values = self.add_derived_roles(
                frame.loc[:, list(self.feature_columns)]
            )[self.task_column]
        else:
            raise ValueError(f"Task column {self.task_column!r} is unavailable.")
        return task_values.astype(str).eq(self.target_task_value).to_numpy()


@dataclass
class ReactionDataset:
    spec: DatasetSpec
    initial: pd.DataFrame
    candidates: pd.DataFrame

    def observed_features(self) -> pd.DataFrame:
        return self.spec.add_derived_roles(self.initial.loc[:, self.spec.feature_columns])

    def candidate_features(self) -> pd.DataFrame:
        return self.spec.add_derived_roles(self.candidates.loc[:, self.spec.feature_columns])

    def target_incumbent(self, observed: pd.DataFrame | None = None) -> float:
        frame = self.initial if observed is None else observed
        mask = self.spec.target_mask(frame)
        target_values = frame.loc[mask, self.spec.target_column].to_numpy(dtype=float)
        if target_values.size == 0:
            raise ValueError("No target-task observations are available for the incumbent.")
        return float(np.max(target_values))

    @property
    def candidate_ids(self) -> np.ndarray:
        return self.candidates["query_index"].to_numpy(dtype=int)


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], *, source: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def _condition_hashes(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    def digest(row: pd.Series) -> str:
        payload = "\x1f".join(str(row[column]) for column in columns)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    return frame.apply(digest, axis=1)


def load_reaction_dataset(
    train_path: str | Path,
    candidate_path: str | Path,
    spec: DatasetSpec,
    *,
    candidate_id_column: str = "query_index",
) -> ReactionDataset:
    """Load and audit participant-side train and candidate files.

    This function never reads organizer-side labels. Candidate IDs are copied
    from an existing ``query_index`` column when available; otherwise the
    original CSV row numbers become immutable IDs.
    """

    train = pd.read_csv(train_path)
    candidates = pd.read_csv(candidate_path)

    _require_columns(train, spec.feature_columns + (spec.target_column,), source="train CSV")
    _require_columns(candidates, spec.feature_columns, source="candidate CSV")

    if spec.target_column in candidates.columns:
        raise ValueError(
            f"Candidate file contains forbidden target column {spec.target_column!r}. "
            "Use the unlabeled test_features CSV, never the organizer lookup file."
        )

    train = train.loc[:, list(spec.feature_columns) + [spec.target_column]].copy()

    # Preserve an explicitly supplied immutable ID before subsetting feature
    # columns. Otherwise the original unlabeled CSV row number is the ID.
    if candidate_id_column in candidates.columns:
        query_ids = pd.to_numeric(
            candidates[candidate_id_column], errors="raise"
        ).astype(int).to_numpy(copy=True)
    else:
        query_ids = candidates.index.to_numpy(dtype=int, copy=True)
    candidates = candidates.loc[:, list(spec.feature_columns)].copy()
    candidates.insert(0, "query_index", query_ids)

    if candidates["query_index"].duplicated().any():
        raise ValueError("Candidate query_index values must be unique.")
    if candidates.loc[:, spec.feature_columns].duplicated().any():
        duplicated = int(candidates.loc[:, spec.feature_columns].duplicated().sum())
        raise ValueError(f"Candidate pool contains {duplicated} duplicate conditions.")
    if train.loc[:, spec.feature_columns].duplicated().any():
        duplicated = int(train.loc[:, spec.feature_columns].duplicated().sum())
        raise ValueError(f"Training data contains {duplicated} duplicate conditions.")

    target_train = train.loc[spec.target_mask(train), list(spec.feature_columns)]
    train_hashes = set(_condition_hashes(target_train, spec.feature_columns))
    candidate_hashes = set(_condition_hashes(candidates, spec.feature_columns))
    overlap = train_hashes.intersection(candidate_hashes)
    if overlap:
        raise ValueError(
            f"Target-task train/candidate overlap detected for {len(overlap)} conditions."
        )

    y = train[spec.target_column].to_numpy(dtype=float)
    if not np.all(np.isfinite(y)):
        raise ValueError("Training yields contain NaN or infinite values.")
    if np.any((y < 0.0) | (y > 100.0)):
        raise ValueError("Yield values must lie within [0, 100].")

    if spec.task_column is not None:
        task_is_derived = (
            spec.task_column in spec.derived_roles
            or spec.task_column in spec.fixed_roles
        )
        if not task_is_derived:
            _require_columns(train, (spec.task_column,), source="train CSV")
            _require_columns(candidates, (spec.task_column,), source="candidate CSV")
        if spec.target_task_value is None:
            raise ValueError("A task_column requires target_task_value.")
        candidate_tasks = (
            spec.add_derived_roles(candidates.loc[:, list(spec.feature_columns)])[spec.task_column]
            if task_is_derived
            else candidates[spec.task_column]
        )
        if not candidate_tasks.astype(str).eq(spec.target_task_value).all():
            raise ValueError("Candidate pool contains rows outside the configured target task.")
        if not spec.target_mask(train).any():
            raise ValueError("Training file has no rows for the configured target task.")

    return ReactionDataset(spec=spec, initial=train, candidates=candidates)


def summarize_topology(dataset: ReactionDataset) -> dict[str, object]:
    """Return label-free search-space diagnostics for logs and manifests."""

    features = dataset.candidate_features()
    summary: dict[str, object] = {
        "dataset": dataset.spec.name,
        "n_initial": len(dataset.initial),
        "n_candidates": len(dataset.candidates),
        "roles": {},
    }
    role_summary: dict[str, int] = {}
    for role in dataset.spec.roles:
        role_summary[role] = int(features[role].nunique(dropna=False))
    summary["roles"] = role_summary

    if "SubstratePair" in features.columns:
        pair_counts = features.groupby("SubstratePair", dropna=False).size()
        summary["n_feasible_substrate_pairs"] = int(len(pair_counts))
        summary["candidate_rows_per_pair"] = {
            "min": int(pair_counts.min()),
            "max": int(pair_counts.max()),
        }
    return summary
