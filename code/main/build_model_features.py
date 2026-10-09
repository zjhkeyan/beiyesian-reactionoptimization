#!/usr/bin/env python3
"""Build direct model inputs according to each dataset's metadata policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_DATASET_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "datasets.json"


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def build_features(dataset: str, input_path: Path, dataset_root: Path) -> pd.DataFrame:
    registry = load_json(DEFAULT_CONFIG)["datasets"]
    if dataset not in registry:
        raise ValueError(f"Unknown dataset {dataset!r}; choose from {sorted(registry)}")

    dataset_config = registry[dataset]
    metadata_path = dataset_root / dataset_config["directory"] / dataset_config["metadata"]
    metadata = load_json(metadata_path)
    direct_features = metadata["model_feature_policy"]["direct_features"]
    fixed_context = metadata["model_feature_policy"].get("fixed_context_features", {})

    frame = pd.read_csv(input_path)
    missing = [column for column in direct_features if column not in frame.columns]
    if missing:
        raise ValueError(f"Input {input_path} is missing required direct features: {missing}")

    output = frame.loc[:, list(direct_features)].copy()
    for output_column, component_name in fixed_context.items():
        output[output_column] = metadata["fixed_components"][component_name][
            "standardized_name"
        ]
    if "Yield" in frame.columns:
        output["Yield"] = frame["Yield"].to_numpy(copy=True)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["Suzuki", "Buchwald_sub4"])
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = build_features(args.dataset, args.input, args.dataset_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output, index=False)
    print(f"Wrote {len(features)} rows and {len(features.columns)} columns to {args.output}")


if __name__ == "__main__":
    main()
