from __future__ import annotations

import importlib.util
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2] / "experiments"
RUNNER_PATH = EXPERIMENT_ROOT / "run_name_subword_study.py"
SPEC = importlib.util.spec_from_file_location("run_name_subword_study", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_name_subword_study_episode_contract() -> None:
    episodes = RUNNER.make_episodes()
    official = [
        episode
        for episode in episodes["Buchwald_sub4"]
        if episode["phase"] == "official_target_exhaustive"
    ]
    components = [
        episode
        for episode in episodes["Buchwald_sub4"]
        if episode["phase"] == "component_holdout"
    ]
    assert len(official) == 21
    assert {episode["target_task"] for episode in official} == {
        RUNNER.OFFICIAL_BUCHWALD_PRODUCT
    }
    assert len(components) == 11
    assert {episode["held_role"] for episode in components} == {
        "Additive",
        "Ligand",
    }
    assert all(len(episode["candidate_indices"]) >= 3 for episode in components)

    suzuki = episodes["Suzuki"]
    assert len([e for e in suzuki if e["phase"] == "screening"]) == 20
    assert len([e for e in suzuki if e["phase"] == "confirmation"]) == 20


def test_study_changes_only_the_opt_in_feature_block() -> None:
    episode = RUNNER.make_official_buchwald_episodes()[0]
    base_config, base_dataset, _ = RUNNER.build_problem(
        "Buchwald_sub4", episode, "base"
    )
    subword_config, subword_dataset, _ = RUNNER.build_problem(
        "Buchwald_sub4", episode, "name_subword"
    )
    assert not base_config.features.use_name_subwords
    assert subword_config.features.use_name_subwords
    assert base_config.gp == subword_config.gp
    assert base_config.dataset == subword_config.dataset
    assert base_config.ensemble == subword_config.ensemble
    assert tuple(base_config.ensemble.enabled_models) == ("low_rank_task_gp",)
    assert "Yield" not in base_dataset.candidates
    assert "Yield" not in subword_dataset.candidates


def test_component_holdout_removes_the_held_level_from_training() -> None:
    episode = RUNNER.make_buchwald_component_episodes()[0]
    config, dataset, _ = RUNNER.build_problem(
        "Buchwald_sub4", episode, "name_subword"
    )
    role = episode["held_role"]
    level = episode["held_level"]
    assert not dataset.initial[role].astype(str).eq(level).any()
    assert dataset.candidates[role].astype(str).eq(level).all()
    assert config.dataset.task_column is None
    assert role in config.dataset.condition_roles
    assert tuple(config.ensemble.enabled_models) == ("gp",)
