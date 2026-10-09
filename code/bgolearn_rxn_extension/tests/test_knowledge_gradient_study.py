from __future__ import annotations

import importlib.util
from pathlib import Path


RUNNER_PATH = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "run_knowledge_gradient_study.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_knowledge_gradient_study", RUNNER_PATH
)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_knowledge_gradient_study_episode_contract() -> None:
    episodes = RUNNER.make_episodes()
    assert len(episodes["Buchwald_sub4"]) == 21
    assert {
        episode["target_task"] for episode in episodes["Buchwald_sub4"]
    } == {RUNNER.OFFICIAL_BUCHWALD_PRODUCT}
    assert len(episodes["Suzuki"]) == 100
    assert sum(item["phase"] == "pilot" for item in episodes["Suzuki"]) == 20
    assert sum(
        item["phase"] == "confirmation" for item in episodes["Suzuki"]
    ) == 80
    assert all(item["budget"] == 5 for item in episodes["Suzuki"])
    confirmation_ids = {
        item["episode"]
        for item in episodes["Suzuki"]
        if item["phase"] == "confirmation"
    }
    assert {
        f"random_confirmation_{index:02d}" for index in range(20)
    }.issubset(confirmation_ids)


def test_study_uses_frozen_metric_frontier_models() -> None:
    episodes = RUNNER.make_episodes()
    buchwald_config, _, _, buchwald_model = RUNNER.build_problem(
        "Buchwald_sub4", episodes["Buchwald_sub4"][0]
    )
    suzuki_config, _, _, suzuki_model = RUNNER.build_problem(
        "Suzuki", episodes["Suzuki"][0]
    )
    assert buchwald_model == "component_ard_low_rank_task_gp"
    assert tuple(buchwald_config.ensemble.enabled_models) == (buchwald_model,)
    assert suzuki_model == "gp"
    assert tuple(suzuki_config.ensemble.enabled_models) == (suzuki_model,)
    for config, model in (
        (buchwald_config, buchwald_model),
        (suzuki_config, suzuki_model),
    ):
        assert config.policy.first_query == model
        assert config.policy.early_end == config.dataset.budget or config.policy.early_end in {3, 5}
        assert config.policy.challenge_every == 0
        assert config.gp.target_transform == "standard"
