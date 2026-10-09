from __future__ import annotations

import importlib.util
from pathlib import Path

from bgolearn_rxn.transforms import LogitYieldTransform, StandardTargetTransform


RUNNER_PATH = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "run_target_transform_study.py"
)
SPEC = importlib.util.spec_from_file_location("run_target_transform_study", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_official_buchwald_exhaustive_episode_contract() -> None:
    episodes = RUNNER.make_episodes()["Buchwald_sub4"]
    assert len(episodes) == 21
    assert {episode["target_task"] for episode in episodes} == {
        RUNNER.OFFICIAL_BUCHWALD_PRODUCT
    }
    assert all(len(episode["initial_indices"]) == 2 for episode in episodes)
    assert all(len(episode["candidate_indices"]) == 5 for episode in episodes)
    assert all(episode["budget"] == 3 for episode in episodes)


def test_only_gp_target_transform_changes_between_problems() -> None:
    episode = RUNNER.make_episodes()["Buchwald_sub4"][0]
    standard, standard_dataset, _ = RUNNER.build_problem(
        "Buchwald_sub4", episode, "standard"
    )
    logit, logit_dataset, _ = RUNNER.build_problem(
        "Buchwald_sub4", episode, "logit"
    )
    assert standard.gp.target_transform == "standard"
    assert logit.gp.target_transform == "logit"
    assert standard.dataset == logit.dataset
    assert standard.ensemble == logit.ensemble
    assert standard.policy == logit.policy
    assert standard_dataset.candidate_ids.tolist() == logit_dataset.candidate_ids.tolist()


def test_transform_classes_are_existing_implementations() -> None:
    episode = RUNNER.make_episodes()["Suzuki"][0]
    standard, _, _ = RUNNER.build_problem("Suzuki", episode, "standard")
    logit, _, _ = RUNNER.build_problem("Suzuki", episode, "logit")
    assert standard.gp.target_transform == "standard"
    assert logit.gp.target_transform == "logit"
    assert StandardTargetTransform.__name__ == "StandardTargetTransform"
    assert LogitYieldTransform.__name__ == "LogitYieldTransform"
