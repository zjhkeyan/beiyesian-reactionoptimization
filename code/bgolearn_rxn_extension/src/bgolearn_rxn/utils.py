from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


def seed_everything(seed: int) -> np.random.Generator:
    """Seed Python and NumPy, and return a local generator.

    The package avoids global random draws in model selection and acquisition;
    this helper still seeds global libraries for third-party estimators.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def canonical_json(value: Any) -> str:
    if is_dataclass(value):
        value = asdict(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def stable_hash(value: Any, *, length: int = 16) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()[:length]


def ensure_directory(path: str | Path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def deterministic_argmax(
    scores: np.ndarray,
    candidate_ids: np.ndarray,
    *,
    seed: int,
    atol: float = 1e-12,
) -> int:
    """Return a deterministic, seed-controlled argmax among numerical ties."""

    values = np.asarray(scores, dtype=float).reshape(-1)
    ids = np.asarray(candidate_ids).reshape(-1)
    if values.shape != ids.shape:
        raise ValueError("scores and candidate_ids must have the same shape.")
    if not np.any(np.isfinite(values)):
        raise FloatingPointError("No finite acquisition values are available.")

    best = np.nanmax(values)
    tied = np.flatnonzero(np.isclose(values, best, rtol=0.0, atol=atol))
    if len(tied) == 1:
        return int(tied[0])

    # Sort by immutable candidate ID before the seeded tie draw. This makes the
    # result independent of DataFrame row ordering.
    tied = tied[np.argsort(ids[tied].astype(str), kind="stable")]
    rng = np.random.default_rng(seed)
    return int(rng.choice(tied))
