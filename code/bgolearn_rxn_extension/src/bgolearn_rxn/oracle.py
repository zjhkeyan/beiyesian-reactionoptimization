from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from .datasets import ReactionDataset


class YieldOracle(Protocol):
    """Minimal interface implemented by a laboratory or offline lookup oracle."""

    def query(self, query_index: int) -> float:
        ...


@dataclass
class OfflineTableOracle:
    """Organizer-side lookup isolated from all participant-side modeling code.

    The labeled table is loaded only by this class. It is never returned to the
    encoder, surrogate, acquisition policy, or hyperparameter logic. The
    feature rows are checked against the public candidate pool before any
    query is allowed.
    """

    dataset: ReactionDataset
    lookup: pd.DataFrame
    allow_repeat_queries: bool = False
    _yield_by_id: dict[int, float] = field(init=False, repr=False)
    _queried: set[int] = field(default_factory=set, init=False, repr=False)

    @classmethod
    def from_csv(
        cls,
        dataset: ReactionDataset,
        lookup_path: str | Path,
        *,
        query_id_column: str = "query_index",
        allow_repeat_queries: bool = False,
    ) -> "OfflineTableOracle":
        lookup = pd.read_csv(lookup_path)
        if dataset.spec.target_column not in lookup.columns:
            raise ValueError(
                f"Oracle lookup is missing {dataset.spec.target_column!r}."
            )
        missing = [
            column
            for column in dataset.spec.feature_columns
            if column not in lookup.columns
        ]
        if missing:
            raise ValueError(f"Oracle lookup is missing feature columns: {missing}")

        if query_id_column in lookup.columns:
            query_ids = pd.to_numeric(lookup[query_id_column], errors="raise").astype(int)
        else:
            query_ids = pd.Series(lookup.index.to_numpy(dtype=int), index=lookup.index)
        lookup = lookup.copy()
        if "query_index" in lookup.columns:
            lookup["query_index"] = query_ids.to_numpy()
        else:
            lookup.insert(0, "query_index", query_ids.to_numpy())
        return cls(
            dataset=dataset,
            lookup=lookup,
            allow_repeat_queries=allow_repeat_queries,
        )

    def __post_init__(self) -> None:
        if self.lookup["query_index"].duplicated().any():
            raise ValueError("Oracle query_index values must be unique.")

        candidate = self.dataset.candidates.set_index("query_index", drop=False)
        lookup = self.lookup.set_index("query_index", drop=False)
        candidate_ids = set(int(value) for value in candidate.index)
        lookup_ids = set(int(value) for value in lookup.index)
        if candidate_ids != lookup_ids:
            missing = sorted(candidate_ids - lookup_ids)[:10]
            extra = sorted(lookup_ids - candidate_ids)[:10]
            raise ValueError(
                "Oracle and public candidate IDs differ. "
                f"Missing in oracle={missing}; extra in oracle={extra}."
            )

        feature_columns = list(self.dataset.spec.feature_columns)
        left = candidate.loc[sorted(candidate_ids), feature_columns].astype(str)
        right = lookup.loc[sorted(candidate_ids), feature_columns].astype(str)
        mismatch = (left.to_numpy() != right.to_numpy()).any(axis=1)
        if mismatch.any():
            bad_ids = np.asarray(sorted(candidate_ids), dtype=int)[mismatch][:10].tolist()
            raise ValueError(
                "Oracle feature rows do not align with the public candidate pool; "
                f"first mismatched IDs: {bad_ids}."
            )

        yields = pd.to_numeric(
            lookup[self.dataset.spec.target_column], errors="raise"
        ).to_numpy(dtype=float)
        if not np.all(np.isfinite(yields)):
            raise ValueError("Oracle yields contain NaN or infinite values.")
        if np.any((yields < 0.0) | (yields > 100.0)):
            raise ValueError("Oracle yields must lie within [0, 100].")
        self._yield_by_id = {
            int(query_id): float(value)
            for query_id, value in zip(
                self.lookup["query_index"].to_numpy(), yields, strict=True
            )
        }

    def query(self, query_index: int) -> float:
        query_index = int(query_index)
        if query_index not in self._yield_by_id:
            raise KeyError(f"Unknown query_index {query_index}.")
        if query_index in self._queried and not self.allow_repeat_queries:
            raise ValueError(f"Repeated oracle query is forbidden: {query_index}.")
        self._queried.add(query_index)
        return self._yield_by_id[query_index]

    @property
    def global_best(self) -> float:
        """Organizer-side diagnostic; never pass this value to the policy."""

        lookup_best = float(max(self._yield_by_id.values()))
        initial_best = self.dataset.target_incumbent()
        return max(lookup_best, initial_best)

    @property
    def queried_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._queried))
