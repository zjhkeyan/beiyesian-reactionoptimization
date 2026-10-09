from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from bgolearn_rxn.config import load_run_config
from bgolearn_rxn.datasets import DatasetSpec, ReactionDataset
from bgolearn_rxn.factory import build_ensemble, prepare_problem
from bgolearn_rxn.surrogates.component_ard_gp import (
    ComponentARDStructuredGP,
    ComponentARDLowRankTaskGP,
)


CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"


def _synthetic_dataset(config) -> ReactionDataset:
    columns = config.dataset.feature_columns
    rows = [
        {column: f"{column}_{index}" for column in columns}
        for index in range(4)
    ]
    if config.dataset.task_column is not None:
        task = config.dataset.task_column
        rows[0][task] = config.dataset.target_task_value
        rows[1][task] = config.dataset.target_task_value
        rows[2][task] = "auxiliary_product_1"
        rows[3][task] = "auxiliary_product_2"
    initial = pd.DataFrame(rows)
    initial[config.dataset.target_column] = [10.0, 25.0, 45.0, 65.0]
    candidates = pd.DataFrame(
        [
            {column: f"candidate_{column}_{index}" for column in columns}
            for index in range(2)
        ]
    )
    if config.dataset.task_column is not None:
        candidates[config.dataset.task_column] = config.dataset.target_task_value
    candidates.insert(0, "query_index", [0, 1])
    return ReactionDataset(
        spec=DatasetSpec.from_config(config.dataset),
        initial=initial,
        candidates=candidates,
    )


@pytest.mark.parametrize(
    ("filename", "model_name", "model_type"),
    [
        (
            "buchwald_sub4_metric_frontier.yaml",
            "component_ard_low_rank_task_gp",
            ComponentARDLowRankTaskGP,
        ),
        (
            "suzuki_metric_frontier.yaml",
            "component_ard_gp",
            ComponentARDStructuredGP,
        ),
    ],
)
def test_metric_frontier_preset_loads_and_builds_single_model(
    filename, model_name, model_type
) -> None:
    config = load_run_config(CONFIG_ROOT / filename)
    assert tuple(config.ensemble.enabled_models) == (model_name,)
    assert config.ensemble.initial_weights == {model_name: 1.0}
    assert config.policy.first_query == model_name
    assert config.policy.early_end == config.dataset.budget == 40
    assert config.policy.final_start == config.dataset.budget + 1
    assert config.policy.challenge_every == 0

    config = replace(
        config,
        features=replace(config.features, use_rdkit=False),
    )
    prepared = prepare_problem(_synthetic_dataset(config), config)
    ensemble = build_ensemble(config, prepared, seed=100)
    assert tuple(ensemble.models) == (model_name,)
    assert isinstance(ensemble.models[model_name], model_type)
