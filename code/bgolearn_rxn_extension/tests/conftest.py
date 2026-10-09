from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from bgolearn_rxn.config import (
    DatasetConfig,
    EnsembleConfig,
    FeatureConfig,
    GPConfig,
    InteractionModelConfig,
    PolicyConfig,
    RunConfig,
    RuntimeConfig,
    TreeConfig,
)
from bgolearn_rxn.datasets import DatasetSpec, load_reaction_dataset


@pytest.fixture
def tiny_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    train = pd.DataFrame(
        [
            {"Product": "target", "A": "a1", "B": "b1", "Yield": 12.0},
            {"Product": "aux", "A": "a1", "B": "b2", "Yield": 91.0},
            {"Product": "aux", "A": "a2", "B": "b1", "Yield": 20.0},
        ]
    )
    candidates = pd.DataFrame(
        [
            {"Product": "target", "A": "a1", "B": "b2"},
            {"Product": "target", "A": "a2", "B": "b1"},
            {"Product": "target", "A": "a2", "B": "b2"},
            {"Product": "target", "A": "a3", "B": "b1"},
        ]
    )
    oracle = candidates.copy()
    oracle["Yield"] = [55.0, 35.0, 82.0, 10.0]
    train_path = tmp_path / "train.csv"
    candidate_path = tmp_path / "candidates.csv"
    oracle_path = tmp_path / "oracle.csv"
    train.to_csv(train_path, index=False)
    candidates.to_csv(candidate_path, index=False)
    oracle.to_csv(oracle_path, index=False)
    return train_path, candidate_path, oracle_path


@pytest.fixture
def tiny_config() -> RunConfig:
    dataset = DatasetConfig(
        name="Tiny",
        feature_columns=("Product", "A", "B"),
        target_column="Yield",
        task_column="Product",
        target_task_value="target",
        condition_roles=("A", "B"),
        budget=3,
        seeds=(100,),
    )
    return RunConfig(
        dataset=dataset,
        features=FeatureConfig(
            registry_path=None,
            use_name_rules=True,
            use_rdkit=False,
        ),
        gp=GPConfig(
            interaction_pairs=(("A", "B"),),
            optimize_hyperparameters=False,
            restarts=1,
            maxiter=5,
            refit_interval=10,
        ),
        interaction_model=InteractionModelConfig(
            interaction_pairs=(("A", "B"),),
            default_rank=2,
        ),
        trees=TreeConfig(n_estimators=16, min_samples_leaf=1),
        ensemble=EnsembleConfig(
            initial_weights={"gp": 0.5, "interaction": 0.3, "trees": 0.2},
            minimum_weight=0.05,
            predictive_std_floor=1.0,
        ),
        policy=PolicyConfig(
            early_end=1,
            final_start=3,
            challenge_every=0,
        ),
        runtime=RuntimeConfig(backend="native"),
    )


@pytest.fixture
def tiny_dataset(tiny_files, tiny_config):
    train_path, candidate_path, _ = tiny_files
    return load_reaction_dataset(
        train_path,
        candidate_path,
        DatasetSpec.from_config(tiny_config.dataset),
    )
