from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


RUNNER_PATH = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "run_suzuki_pair_context_gp.py"
)
SPEC = importlib.util.spec_from_file_location("run_suzuki_pair_context_gp", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_episode_and_policy_contract_is_frozen() -> None:
    episodes = RUNNER.make_episodes()
    assert len(episodes) == 60
    assert all(910_000 <= item["split_seed"] < 920_000 for item in episodes)
    assert all(item["budget"] == 5 for item in episodes)
    assert len({item["split_seed"] for item in episodes}) == 60

    config, _, _, model_name = RUNNER.build_problem(episodes[0], RUNNER.CANDIDATE)
    assert model_name == "pair_context_coregionalized_gp"
    assert config.ensemble.enabled_models == (model_name,)
    assert config.policy.first_query == model_name
    assert config.policy.early_end == 5
    assert config.policy.final_start == 6
    assert config.policy.challenge_every == 0
    assert config.gp.target_transform == "standard"


def test_gate_requires_random_tail_and_structured_safety() -> None:
    paired = pd.DataFrame(
        [
            {"metric": "initial_round_found_best", "mean_delta": 0.0},
            {"metric": "best_found", "mean_delta": 0.0},
            {"metric": "round_to_95_global_best", "mean_delta": 0.0},
            {"metric": "AUC_best_so_far", "mean_delta": 0.1},
        ]
    )
    tails = pd.DataFrame(
        [
            {
                "variant": RUNNER.BASELINE,
                "auc_10th_percentile": 10.0,
                "cvar_worst_decile_auc": 8.0,
            },
            {
                "variant": RUNNER.CANDIDATE,
                "auc_10th_percentile": 10.0,
                "cvar_worst_decile_auc": 8.0,
            },
        ]
    )
    passing = RUNNER.gate_decision(paired, tails, {"positive": True})
    assert passing["passed"] is True
    failing = RUNNER.gate_decision(paired, tails, {"positive": False})
    assert failing["passed"] is False
    assert failing["decision"] == "stop_before_920"


def test_leave_pair_out_contract_covers_all_sparse_contexts() -> None:
    source = pd.read_csv(RUNNER.TRAIN_FILE)
    assert source.loc[:, RUNNER.PAIR_COLUMNS].drop_duplicates().shape[0] == 13
    assert RUNNER.LPO_PAIR_COUNT == 13
    assert RUNNER.LPO_TRAIN_ROWS == 32
    assert RUNNER.LPO_TEST_ROWS == 8
