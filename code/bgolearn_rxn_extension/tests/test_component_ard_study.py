from __future__ import annotations

import importlib.util
from pathlib import Path


RUNNER_PATH = (
    Path(__file__).resolve().parents[2] / "experiments" / "run_component_ard_study.py"
)
SPEC = importlib.util.spec_from_file_location("run_component_ard_study", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_study_uses_paired_official_and_suzuki_episodes() -> None:
    episodes = RUNNER.make_episodes()
    assert len(episodes["Buchwald_sub4"]) == 21
    assert len(episodes["Suzuki"]) == 56
    assert {e["target_task"] for e in episodes["Buchwald_sub4"]} == {
        RUNNER.OFFICIAL_BUCHWALD_PRODUCT
    }


def test_study_preserves_standard_transform_and_incumbent_task_model() -> None:
    episode = RUNNER.make_episodes()["Buchwald_sub4"][0]
    fixed, fixed_dataset, _, fixed_name = RUNNER.build_problem(
        "Buchwald_sub4", episode, "fixed"
    )
    ard, ard_dataset, _, ard_name = RUNNER.build_problem(
        "Buchwald_sub4", episode, "component_ard"
    )
    assert fixed.gp.target_transform == ard.gp.target_transform == "standard"
    assert fixed_name == "low_rank_task_gp"
    assert ard_name == "component_ard_low_rank_task_gp"
    assert fixed.dataset == ard.dataset
    assert fixed_dataset.candidate_ids.tolist() == ard_dataset.candidate_ids.tolist()
