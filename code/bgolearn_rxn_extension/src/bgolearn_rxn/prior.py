from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_candidate_prior(
    path: str | Path,
    candidate_ids: np.ndarray,
    *,
    id_column: str = "query_index",
    score_column: str = "prior_score",
) -> np.ndarray:
    """Load an auditable, label-free prior aligned by immutable candidate ID."""

    frame = pd.read_csv(path)
    if id_column not in frame.columns or score_column not in frame.columns:
        raise ValueError(
            f"Prior CSV must contain {id_column!r} and {score_column!r}."
        )
    if frame[id_column].duplicated().any():
        raise ValueError("Prior CSV contains duplicate candidate IDs.")
    scores = frame.set_index(id_column)[score_column]
    requested = [int(value) for value in np.asarray(candidate_ids).reshape(-1)]
    missing = [value for value in requested if value not in scores.index]
    if missing:
        raise ValueError(f"Prior CSV is missing candidate IDs: {missing[:10]}")
    output = pd.to_numeric(scores.loc[requested], errors="raise").to_numpy(dtype=float)
    if not np.all(np.isfinite(output)):
        raise ValueError("Prior scores contain NaN or infinite values.")
    return output
