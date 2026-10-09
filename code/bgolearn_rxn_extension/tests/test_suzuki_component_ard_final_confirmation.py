from __future__ import annotations

import importlib.util
from pathlib import Path


RUNNER_PATH = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "run_suzuki_component_ard_final_confirmation.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_suzuki_component_ard_final_confirmation", RUNNER_PATH
)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_confirmation_episode_contract_is_exact_and_reserved() -> None:
    episodes = RUNNER.make_episodes()
    seeds = [item["split_seed"] for item in episodes]
    assert len(episodes) == 200
    assert len(set(seeds)) == 200
    assert min(seeds) >= 810_000
    assert max(seeds) < 910_000
    assert all(item["budget"] == 5 for item in episodes)
    assert all(item["phase"] == RUNNER.PHASE for item in episodes)


def test_confirmation_models_and_policy_are_frozen() -> None:
    episode = RUNNER.make_episodes()[0]
    for variant, expected_model in RUNNER.MODEL_NAMES.items():
        config, _, _, model_name = RUNNER.build_problem(episode, variant)
        assert model_name == expected_model
        assert tuple(config.ensemble.enabled_models) == (expected_model,)
        assert config.policy.first_query == expected_model
        assert config.policy.early_end == 5
        assert config.policy.final_start == 6
        assert config.policy.challenge_every == 0
        assert config.gp.target_transform == "standard"
    assert RUNNER.ComponentARDStructuredGP.amplitude_prior_scale == 0.60


def test_confirmation_gate_is_strict_and_predeclared() -> None:
    assert RUNNER.PROMOTION_GATE == {
        "minimum_initial_mean_delta": 0.0,
        "minimum_best_mean_delta": 0.0,
        "minimum_auc_mean_delta_exclusive": 0.0,
        "minimum_auc_ci95_low_exclusive": 0.0,
        "maximum_t95_mean_delta_exclusive": 0.0,
        "maximum_t95_ci95_high_exclusive": 0.0,
        "maximum_runtime_ratio": 2.5,
        "require_finite_noncollapsed_amplitudes": True,
    }


def test_confirmation_cli_has_no_limit_or_tuning_argument() -> None:
    actions = RUNNER.parse_args.__globals__["argparse"].ArgumentParser
    assert actions is not None
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert 'add_argument("--limit"' not in source
    assert 'add_argument("--variant"' not in source
    assert 'add_argument("--prior' not in source
