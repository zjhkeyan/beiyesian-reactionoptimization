from __future__ import annotations

import importlib.util
from pathlib import Path


RUNNER_PATH = (
    Path(__file__).resolve().parents[2] / "experiments" / "run_transfer_proxy.py"
)
SPEC = importlib.util.spec_from_file_location("run_transfer_proxy", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_suzuki_episode_mode_preserves_pair_defaults() -> None:
    pair = RUNNER.resolve_variants(["Suzuki"], None)
    role = RUNNER.resolve_variants(
        ["Suzuki"], None, suzuki_episode_mode="role"
    )
    assert pair == {"Suzuki": RUNNER.SUZUKI_PAIR_VARIANTS}
    assert role == {"Suzuki": RUNNER.SUZUKI_ROLE_VARIANTS}


def test_suzuki_role_episode_counts_and_shapes() -> None:
    episodes = RUNNER.make_suzuki_episodes("role")
    family_counts = {}
    for episode in episodes:
        family = episode["episode_family"]
        family_counts[family] = family_counts.get(family, 0) + 1
        assert episode["task_column"] == family
        assert len(episode["initial_indices"]) == 1
        assert len(episode["candidate_indices"]) >= 3
        assert episode["budget"] == min(3, len(episode["candidate_indices"]))
        assert not set(episode["initial_indices"]).intersection(
            episode["candidate_indices"]
        )
    assert family_counts == {"Electrophile": 29, "Nucleophile": 26}
    assert len({episode["episode"] for episode in episodes}) == 55


def test_suzuki_role_problem_has_one_target_initial_and_target_candidates() -> None:
    episode = RUNNER.make_suzuki_episodes("role")[0]
    pooled_config, pooled_dataset, _ = RUNNER.build_problem(
        "Suzuki", "pooled_role_task_gp", episode
    )
    target_config, target_dataset, _ = RUNNER.build_problem(
        "Suzuki", "target_only_gp", episode
    )

    assert pooled_config.dataset.task_column == "Electrophile"
    assert pooled_config.dataset.condition_roles == (
        "Nucleophile",
        "Ligand",
        "Base",
        "Solvent",
    )
    assert pooled_dataset.spec.target_mask(pooled_dataset.initial).sum() == 1
    assert pooled_dataset.spec.target_mask(pooled_dataset.candidates).all()
    assert len(pooled_dataset.initial) > 1
    assert len(target_dataset.initial) == 1
    assert target_config.dataset.task_column == episode["task_column"]
