from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_trajectory(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["trajectory"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    args = parser.parse_args()

    left = load_trajectory(args.left)
    right = load_trajectory(args.right)
    left_ids = np.array([row["query_index"] for row in left], dtype=int)
    right_ids = np.array([row["query_index"] for row in right], dtype=int)
    first_difference = np.flatnonzero(left_ids != right_ids)
    print(f"same_ids={np.array_equal(left_ids, right_ids)}")
    if first_difference.size:
        step = int(first_difference[0] + 1)
        print(
            f"first_difference_step={step} "
            f"left={left_ids[step-1]} right={right_ids[step-1]}"
        )


if __name__ == "__main__":
    main()
