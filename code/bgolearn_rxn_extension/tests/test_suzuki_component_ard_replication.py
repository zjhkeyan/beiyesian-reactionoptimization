from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


RUNNER_PATH = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "run_suzuki_component_ard_replication.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_suzuki_component_ard_replication", RUNNER_PATH
)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_fresh_replication_episode_contract_is_frozen_and_disjoint() -> None:
    episodes = RUNNER.make_episodes()
    assert len(episodes) == 200
    assert all(item["phase"] == "fresh_replication" for item in episodes)
    assert all(item["budget"] == 5 for item in episodes)
    assert min(item["split_seed"] for item in episodes) >= 610_000
    assert {item["split_seed"] for item in episodes}.isdisjoint(
        RUNNER.PRIOR_SEED_STARTS
    )


def test_replication_changes_only_the_single_gp_family() -> None:
    episode = RUNNER.make_episodes()[0]
    fixed_config, fixed_data, fixed_lookup, fixed_name = RUNNER.build_problem(
        episode, "fixed"
    )
    ard_config, ard_data, ard_lookup, ard_name = RUNNER.build_problem(
        episode, "component_ard"
    )
    assert fixed_name == "gp"
    assert ard_name == "component_ard_gp"
    assert fixed_config.gp == ard_config.gp
    assert fixed_config.features == ard_config.features
    assert fixed_config.policy.early_end == ard_config.policy.early_end == 5
    assert fixed_config.policy.challenge_every == ard_config.policy.challenge_every == 0
    pd.testing.assert_frame_equal(fixed_data.initial, ard_data.initial)
    pd.testing.assert_frame_equal(fixed_data.candidates, ard_data.candidates)
    pd.testing.assert_frame_equal(fixed_lookup, ard_lookup)


def _synthetic_rows(
    *,
    initial_delta: float = 0.1,
    best_delta: float = 0.1,
    auc_delta: float = 1.0,
    t95_delta: float = -1.0,
) -> pd.DataFrame:
    rows = []
    for index in range(200):
        for variant in RUNNER.VARIANTS:
            candidate = variant == RUNNER.CANDIDATE
            rows.append(
                {
                    "episode": f"episode_{index:03d}",
                    "variant": variant,
                    "initial_round_found_best": 10.0
                    + (initial_delta if candidate else 0.0),
                    "best_found": 20.0 + (best_delta if candidate else 0.0),
                    "round_to_95_global_best": 3.0
                    + (t95_delta if candidate else 0.0),
                    "AUC_best_so_far": 15.0 + (auc_delta if candidate else 0.0),
                    "elapsed_seconds": 2.0 if candidate else 1.0,
                }
            )
    return pd.DataFrame(rows)


def _stable_amplitudes() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "mean_share": 0.2,
                "sample_sd_share": 0.01,
                "min_share": 0.1,
                "max_share": 0.3,
            }
        ]
    )


def test_promotion_gate_requires_resolved_auc_and_t95_improvements() -> None:
    passing = RUNNER.promotion_decision(_synthetic_rows(), _stable_amplitudes())
    assert passing["passed"] is True
    auc_unresolved = RUNNER.promotion_decision(
        _synthetic_rows(auc_delta=0.0), _stable_amplitudes()
    )
    assert auc_unresolved["checks"]["auc_ci_excludes_zero"] is False
    t95_unresolved = RUNNER.promotion_decision(
        _synthetic_rows(t95_delta=0.0), _stable_amplitudes()
    )
    assert t95_unresolved["checks"]["t95_ci_excludes_zero"] is False


def test_explicit_twenty_run_view_is_deterministic() -> None:
    rows = []
    for index in range(200):
        for variant in RUNNER.VARIANTS:
            rows.append(
                {
                    "episode": f"random_fresh_replication_{index:02d}",
                    "variant": variant,
                }
            )
    view = RUNNER.explicit_twenty_run(pd.DataFrame(rows))
    assert len(view) == 40
    assert view["episode"].nunique() == 20
    assert view["episode"].max() == "random_fresh_replication_19"
