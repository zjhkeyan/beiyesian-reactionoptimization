#!/usr/bin/env python3
"""Validate option, metadata, and direct-feature contracts for both datasets."""

from __future__ import annotations

import csv
import json
from pathlib import Path


DATASET_ROOT = Path(__file__).resolve().parents[3]
EXPECTED_SUZUKI_PRODUCT = (
    "6-(5-methyl-1-(tetrahydro-2H-pyran-2-yl)-1H-indazol-4-yl)quinoline"
)
EXPECTED_CATALYST = "palladium(2+) diacetate"
FILES = {
    "Suzuki": ("suzuki_train.csv", "suzuki_test_features.csv"),
    "Buchwald_sub4": ("buchwald_sub4_train.csv", "buchwald_sub4_test_features.csv"),
}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def columns(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return next(csv.reader(handle))


def unique_column_values(path: Path) -> dict[str, set[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        values = {column: set() for column in reader.fieldnames or []}
        for row in reader:
            for column, value in row.items():
                values[column].add(value)
    return values


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_dataset(dataset: str) -> None:
    directory = DATASET_ROOT / dataset
    options = load_json(directory / "options.json")
    metadata = load_json(directory / "metadata.json")
    variables = metadata["optimization_variables"]
    fixed = metadata["fixed_components"]
    direct = metadata["model_feature_policy"]["direct_features"]
    fixed_context = metadata["model_feature_policy"].get("fixed_context_features", {})
    train_name, pool_name = FILES[dataset]
    pool_values = unique_column_values(directory / pool_name)

    require(list(options) == variables, f"{dataset}: options must contain optimization variables only")
    require(not set(options).intersection(fixed), f"{dataset}: fixed fields leaked into options")
    for variable, configured_values in options.items():
        require(set(configured_values) == pool_values[variable],
                f"{dataset}: options for {variable} do not match the candidate pool")
    require(set(direct).issubset(columns(directory / train_name)), f"{dataset}: train lacks direct features")
    require(set(direct).issubset(columns(directory / pool_name)), f"{dataset}: pool lacks direct features")
    require("Yield" not in columns(directory / pool_name), f"{dataset}: candidate pool exposes Yield")

    if dataset == "Suzuki":
        policy = metadata["model_feature_policy"]
        require(fixed["Product"]["standardized_name"] == EXPECTED_SUZUKI_PRODUCT,
                "Suzuki: coupled product does not match the specification")
        require(fixed["Product"]["direct_model_feature"],
                "Suzuki: corrected Product must be enabled as fixed model context")
        require(fixed_context == {"ProductContext": "Product"},
                "Suzuki: ProductContext mapping is missing or incorrect")
        require("Product" not in direct,
                "Suzuki: fixed Product must not become a selectable CSV variable")
        require("Catalyst" not in direct, "Suzuki: Catalyst must not be a direct feature")
        require(not policy["reaction_difference_encoding"]["enabled"],
                "Suzuki: product-dependent difference encoding must remain disabled")
    else:
        require(fixed["Catalyst"]["standardized_name"] == EXPECTED_CATALYST,
                "Buchwald_sub4: fixed catalyst metadata is missing or incorrect")

    print(f"PASS {dataset}")


def main() -> None:
    for dataset in FILES:
        validate_dataset(dataset)
    print("All dataset metadata and feature-policy checks passed.")


if __name__ == "__main__":
    main()
