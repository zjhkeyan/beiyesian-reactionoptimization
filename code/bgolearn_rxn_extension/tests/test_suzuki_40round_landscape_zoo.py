from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from bgolearn_rxn.policy import MetricAwarePolicy
from bgolearn_rxn.schemas import Posterior


RUNNER_PATH = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "run_suzuki_40round_landscape_zoo.py"
)
SPEC = importlib.util.spec_from_file_location("run_suzuki_40round_landscape_zoo", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_frozen_dev_and_test_landscapes_are_disjoint_and_balanced() -> None:
    dev = RUNNER.landscape_specs("dev")
    test = RUNNER.landscape_specs("test")
    assert len(dev) == len(test) == 20
    assert {item["teacher_seed"] for item in dev}.isdisjoint(
        {item["teacher_seed"] for item in test}
    )
    assert {item["campaign_seed"] for item in dev}.isdisjoint(
        {item["campaign_seed"] for item in test}
    )
    for family in RUNNER.TEACHER_FAMILIES:
        assert sum(item["family"] == family for item in dev) == 5
        assert sum(item["family"] == family for item in test) == 5


def test_phase8_registry_is_fresh_disjoint_and_balanced() -> None:
    old_specs = RUNNER.landscape_specs("dev") + RUNNER.landscape_specs("test")
    dev = RUNNER.landscape_specs("phase8_dev")
    held_out = RUNNER.landscape_specs("phase8_test")
    assert len(dev) == len(held_out) == 20
    assert {item["teacher_seed"] for item in old_specs}.isdisjoint(
        {item["teacher_seed"] for item in dev + held_out}
    )
    assert {item["teacher_seed"] for item in dev}.isdisjoint(
        {item["teacher_seed"] for item in held_out}
    )
    assert {item["campaign_seed"] for item in dev}.isdisjoint(
        {item["campaign_seed"] for item in held_out}
    )
    assert {item["family"] for item in dev} == set(RUNNER.TEACHER_FAMILIES)
    assert {item["family"] for item in held_out} == set(
        RUNNER.HELD_OUT_TEACHER_FAMILIES
    )
    for family in RUNNER.TEACHER_FAMILIES:
        assert sum(item["family"] == family for item in dev) == 5
    for family in RUNNER.HELD_OUT_TEACHER_FAMILIES:
        assert sum(item["family"] == family for item in held_out) == 10


def test_phase8_cli_supports_dev_and_sealed_test_splits() -> None:
    for split in ("phase8_dev", "phase8_test"):
        args = RUNNER.parse_args(
            ["--split", split, "--variants", "component_ard", "--jobs", "1"]
        )
        assert args.split == split
        assert args.variants == ["component_ard"]


def test_teacher_generation_is_deterministic_and_label_safe() -> None:
    spec = RUNNER.landscape_specs("phase8_dev", 1)[0]
    first, first_diagnostics = RUNNER.build_landscape(spec)
    second, second_diagnostics = RUNNER.build_landscape(spec)
    np.testing.assert_array_equal(first["Yield"], second["Yield"])
    assert first_diagnostics == second_diagnostics
    assert len(first) == 5731
    assert first["query_index"].is_unique
    assert first["Yield"].between(0.0, 100.0).all()
    assert first["Yield"].std(ddof=1) >= 5.0
    assert RUNNER.TRAIN_FILE not in RUNNER.PROTECTED_FILES
    assert RUNNER.CANDIDATE_FILE not in RUNNER.PROTECTED_FILES


def test_environment_has_meaningful_cross_family_disagreement() -> None:
    diagnostics = RUNNER.environment_diagnostics(
        RUNNER.landscape_specs("phase8_dev", 4)
    )
    assert all(diagnostics["checks"].values())
    assert diagnostics["family_mean_pairwise_rmse_min"] >= 3.0
    assert diagnostics["family_mean_pairwise_correlation_max"] < 0.995


def test_phase8_dev_exact_geometry_and_deterministic_hashes() -> None:
    specs = RUNNER.landscape_specs("phase8_dev")
    first = RUNNER.environment_diagnostics(specs[:4])
    second = RUNNER.environment_diagnostics(specs[:4])
    assert first["landscape_matrix_sha256"] == second["landscape_matrix_sha256"]
    assert all(row["n_initial"] == 29 for row in first["landscapes"])
    assert all(row["n_candidates"] == 5731 for row in first["landscapes"])


def test_variants_preserve_official_geometry_and_policy_horizon() -> None:
    for variant, expected_model in RUNNER.MODEL_NAMES.items():
        config, dataset, model_name = RUNNER.build_problem(variant)
        expected = RUNNER.VARIANT_CONFIGS[variant]
        assert model_name == expected_model
        assert len(dataset.initial) == 29
        assert len(dataset.candidates) == 5731
        assert config.dataset.budget == 40
        assert config.policy.early_end == expected["early_end"]
        assert config.policy.final_start == expected["final_start"]
        assert config.policy.challenge_every == 0
        assert config.policy.first_query == expected["first_query"]
        assert config.policy.final_beta == expected["final_beta"]
        assert config.ensemble.enabled_models == expected["enabled_models"]
        assert config.ensemble.initial_weights == expected["initial_weights"]


def test_new_variants_are_available_to_cli_and_have_exact_model_contracts() -> None:
    assert {
        "full_mean_jackknife",
        "prequential_ard_delete",
        "stage_split_late33",
        "early_delete_then_full",
        "query1_rank_consensus",
        "localized_jackknife",
        "surprise_gated_consensus",
    }.issubset(RUNNER.VARIANTS)
    assert RUNNER.VARIANT_CONFIGS["full_mean_jackknife"]["enabled_models"] == (
        "full_mean_jackknife_variance_ard_gp",
    )
    assert RUNNER.VARIANT_CONFIGS["prequential_ard_delete"]["enabled_models"] == (
        "component_ard_gp",
        "delete_fold_component_ard_gp",
    )
    assert RUNNER.VARIANT_CONFIGS["early_delete_then_full"] == {
        "enabled_models": ("early_delete_fold_then_full_ard_gp",),
        "initial_weights": {"early_delete_fold_then_full_ard_gp": 1.0},
        "first_query": "early_delete_fold_then_full_ard_gp",
        "early_end": 40,
        "final_start": 41,
        "final_beta": 0.15,
    }
    assert RUNNER.VARIANT_CONFIGS["query1_rank_consensus"] == {
        "enabled_models": ("query_one_rank_consensus_ard_gp",),
        "initial_weights": {"query_one_rank_consensus_ard_gp": 1.0},
        "first_query": "query_one_rank_consensus_ard_gp",
        "early_end": 40,
        "final_start": 41,
        "final_beta": 0.15,
    }
    assert RUNNER.VARIANT_CONFIGS["localized_jackknife"] == {
        "enabled_models": ("incumbent_localized_jackknife_variance_ard_gp",),
        "initial_weights": {
            "incumbent_localized_jackknife_variance_ard_gp": 1.0
        },
        "first_query": "incumbent_localized_jackknife_variance_ard_gp",
        "early_end": 40,
        "final_start": 41,
        "final_beta": 0.15,
    }
    assert RUNNER.VARIANT_CONFIGS["surprise_gated_consensus"] == {
        "enabled_models": ("surprise_gated_rank_consensus_ard_gp",),
        "initial_weights": {"surprise_gated_rank_consensus_ard_gp": 1.0},
        "first_query": "surprise_gated_rank_consensus_ard_gp",
        "early_end": 40,
        "final_start": 41,
        "final_beta": 0.15,
    }


def test_phase8_candidate_screen_cli_contract_is_exact() -> None:
    args = RUNNER.parse_args(
        [
            "--split",
            "phase8_dev",
            "--episode-count",
            "4",
            "--variants",
            "component_ard",
            "query1_rank_consensus",
            "localized_jackknife",
            "--jobs",
            "8",
        ]
    )
    assert args.split == "phase8_dev"
    assert args.episode_count == 4
    assert args.variants == [
        "component_ard",
        "query1_rank_consensus",
        "localized_jackknife",
    ]
    assert args.jobs == 8


def test_prequential_first_query_uses_component_ard_mean() -> None:
    config, _, _ = RUNNER.build_problem("prequential_ard_delete")
    posterior = Posterior(
        mean=np.asarray([8.0, 7.0, 6.0]),
        std=np.ones(3),
        component_means={
            "component_ard_gp": np.asarray([1.0, 9.0, 3.0]),
            "delete_fold_component_ard_gp": np.asarray([10.0, 2.0, 1.0]),
        },
        component_stds={
            "component_ard_gp": np.ones(3),
            "delete_fold_component_ard_gp": np.ones(3),
        },
    )
    recommendation = MetricAwarePolicy(config.policy, seed=11).select(
        posterior,
        np.asarray([100, 101, 102]),
        best_f=5.0,
        step=1,
        budget=40,
    )
    assert recommendation.query_index == 101
    assert recommendation.acquisition_name == "component_ard_gp_posterior_mean"


def test_full_mean_and_late33_acquisition_schedules() -> None:
    posterior = Posterior(
        mean=np.asarray([1.0, 3.0, 2.0]),
        std=np.asarray([2.0, 1.0, 4.0]),
    )
    candidate_ids = np.asarray([100, 101, 102])

    full_config, _, _ = RUNNER.build_problem("full_mean_jackknife")
    full_policy = MetricAwarePolicy(full_config.policy, seed=11)
    first = full_policy.select(
        posterior, candidate_ids, best_f=2.0, step=1, budget=40
    )
    second = full_policy.select(
        posterior, candidate_ids, best_f=2.0, step=2, budget=40
    )
    assert first.acquisition_name == "posterior_mean"
    assert second.acquisition_name == "stable_log_ei"

    late_config, _, _ = RUNNER.build_problem("stage_split_late33")
    late_policy = MetricAwarePolicy(late_config.policy, seed=11)
    round32 = late_policy.select(
        posterior, candidate_ids, best_f=2.0, step=32, budget=40
    )
    round33 = late_policy.select(
        posterior, candidate_ids, best_f=2.0, step=33, budget=40
    )
    assert round32.acquisition_name == "stable_log_ei"
    assert round33.acquisition_name == "late_mean_ucb"
