from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_json(payload: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(_jsonable(payload), indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return target


def save_torch_submission(payload: dict[str, Any], path: str | Path) -> Path:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "PyTorch is required for .pt output. Install `pip install -e '.[submission]'`."
        ) from exc
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_jsonable(payload), target)
    return target


def load_torch_submission(path: str | Path) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("PyTorch is required to read .pt submissions.") from exc
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Submission root must be a dictionary.")
    return payload


def validate_submission_payload(
    payload: dict[str, Any],
    candidates: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    expected_budget: int = 40,
) -> None:
    trajectory = payload.get("trajectory")
    if not isinstance(trajectory, list):
        raise ValueError("Submission must contain a list-valued `trajectory`.")
    if len(trajectory) != expected_budget:
        raise ValueError(
            f"Trajectory has {len(trajectory)} rows; expected {expected_budget}."
        )

    candidate_by_id = candidates.set_index("query_index", drop=False)
    seen: set[int] = set()
    for expected_step, record in enumerate(trajectory, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"Trajectory row {expected_step} is not a dictionary.")
        step = int(record.get("step", -1))
        if step != expected_step:
            raise ValueError(
                f"Trajectory step mismatch at position {expected_step}: got {step}."
            )
        query_index = int(record["query_index"])
        if query_index in seen:
            raise ValueError(f"Duplicate query_index {query_index} in trajectory.")
        seen.add(query_index)
        if query_index not in candidate_by_id.index:
            raise ValueError(f"Invalid query_index {query_index}.")

        condition = record.get("condition")
        if not isinstance(condition, dict):
            raise ValueError(f"Step {step} is missing a condition dictionary.")
        expected = candidate_by_id.loc[query_index, list(feature_columns)]
        for column in feature_columns:
            if str(condition.get(column)) != str(expected[column]):
                raise ValueError(
                    f"Condition mismatch at step {step}, ID {query_index}, "
                    f"column {column!r}."
                )
