from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


RUNNER_PATH = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "run_suzuki_divergence_atlas.py"
)
SPEC = importlib.util.spec_from_file_location("run_suzuki_divergence_atlas", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_atlas_episode_contract_uses_only_fresh_910_series() -> None:
    episodes = RUNNER.make_episodes()
    seeds = [int(item["split_seed"]) for item in episodes]
    assert len(episodes) == 100
    assert len(set(seeds)) == 100
    assert min(seeds) >= 910_000
    assert max(seeds) < 920_000
    assert all(item["budget"] == 5 for item in episodes)
    assert all(item["phase"] == RUNNER.PHASE for item in episodes)


def test_atlas_models_and_policy_match_current_frontier() -> None:
    episode = RUNNER.make_episodes()[0]
    for variant, model in RUNNER.MODEL_NAMES.items():
        config, _, _, actual = RUNNER.build_problem(episode, variant)
        assert actual == model
        assert tuple(config.ensemble.enabled_models) == (model,)
        assert config.policy.first_query == model
        assert config.policy.early_end == 5
        assert config.policy.final_start == 6
        assert config.policy.challenge_every == 0
        assert config.gp.target_transform == "standard"


def test_calibration_record_and_outcome_classes_are_explicit() -> None:
    record = RUNNER._calibration_record(60.0, 50.0, 10.0)
    assert record["residual"] == 10.0
    assert record["standardized_residual"] == 1.0
    assert record["covered_50"] is False
    assert record["covered_80"] is True
    assert record["covered_90"] is True
    assert record["covered_95"] is True
    assert RUNNER.classify_auc_delta(5.0) == "large_win"
    assert RUNNER.classify_auc_delta(0.1) == "small_win"
    assert RUNNER.classify_auc_delta(0.0) == "tie"
    assert RUNNER.classify_auc_delta(-0.1) == "loss"
    assert RUNNER._jsonable(np.bool_(True)) is True


def test_episode_analysis_locates_first_policy_divergence() -> None:
    metrics = pd.DataFrame(
        [
            {"episode": "e", "variant": "fixed", "AUC_best_so_far": 20.0},
            {"episode": "e", "variant": "component_ard", "AUC_best_so_far": 26.0},
        ]
    )
    rows = []
    for variant, ids in (("fixed", [1, 2, 3, 4, 5]), ("component_ard", [1, 9, 3, 4, 5])):
        for step, query_id in enumerate(ids, 1):
            rows.append(
                {
                    "episode": "e",
                    "variant": variant,
                    "step": step,
                    "split_seed": 910_000,
                    "query_index": query_id,
                    "observed_yield": float(query_id),
                    "substrate_pair": f"pair-{query_id}",
                    "pair_seen_in_initial": False,
                    "acquisition_margin": 1.0,
                    "predicted_std": 2.0,
                    "standardized_residual": 0.5,
                    "component_shares_json": '{"bias": 0.5, "role:Base": 0.5}',
                }
            )
    result = RUNNER.episode_analysis(metrics, pd.DataFrame(rows)).iloc[0]
    assert result["first_divergence_round"] == 2
    assert result["n_differing_rounds"] == 1
    assert result["outcome_class"] == "large_win"
    assert result["ard_divergence_query_index"] == 9
    assert np.isclose(result["ard_share_role:Base"], 0.5)


def test_driver_declares_no_public_or_protected_label_use() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert 'TRAIN_FILE = replication.TRAIN_FILE' in source
    assert 'public_test_labels_used": False' in source
    assert 'public_searchspace_labels_used": False' in source
    assert 'protected_920_or_930_episodes_opened": False' in source
