from __future__ import annotations

import pandas as pd
import pytest

from bgolearn_rxn.datasets import DatasetSpec, load_reaction_dataset


def test_buchwald_incumbent_is_target_specific(tiny_dataset) -> None:
    assert tiny_dataset.target_incumbent() == 12.0


def test_candidate_labels_are_rejected(tmp_path, tiny_files, tiny_config) -> None:
    train_path, candidate_path, _ = tiny_files
    candidates = pd.read_csv(candidate_path)
    candidates["Yield"] = 50.0
    bad = tmp_path / "bad.csv"
    candidates.to_csv(bad, index=False)
    with pytest.raises(ValueError, match="forbidden target"):
        load_reaction_dataset(
            train_path,
            bad,
            DatasetSpec.from_config(tiny_config.dataset),
        )


def test_existing_query_ids_are_preserved(tmp_path, tiny_files, tiny_config) -> None:
    train_path, candidate_path, _ = tiny_files
    candidates = pd.read_csv(candidate_path)
    candidates.insert(0, "query_index", [41, 43, 47, 53])
    with_ids = tmp_path / "with_ids.csv"
    candidates.to_csv(with_ids, index=False)
    dataset = load_reaction_dataset(
        train_path,
        with_ids,
        DatasetSpec.from_config(tiny_config.dataset),
    )
    assert dataset.candidate_ids.tolist() == [41, 43, 47, 53]


def test_derived_role_can_define_target_task(tiny_config) -> None:
    config = tiny_config.dataset
    spec = DatasetSpec(
        name=config.name,
        feature_columns=config.feature_columns,
        target_column=config.target_column,
        task_column="Pair",
        target_task_value="a1 || b1",
        condition_roles=("A", "B"),
        derived_roles={"Pair": ("A", "B")},
        fixed_roles={},
        budget=config.budget,
        seeds=config.seeds,
    )
    frame = pd.DataFrame(
        [
            {"Product": "target", "A": "a1", "B": "b1", "Yield": 1.0},
            {"Product": "target", "A": "a1", "B": "b2", "Yield": 2.0},
        ]
    )
    assert spec.target_mask(frame).tolist() == [True, False]
