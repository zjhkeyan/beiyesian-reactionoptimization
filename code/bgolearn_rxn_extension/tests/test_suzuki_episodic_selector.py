from __future__ import annotations

import importlib.util
from pathlib import Path


RUNNER_PATH = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "run_suzuki_episodic_selector.py"
)
SPEC = importlib.util.spec_from_file_location("run_suzuki_episodic_selector", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_partitions_are_exact_disjoint_and_exclude_protected_families() -> None:
    development, validation = RUNNER.make_episode_partitions()
    assert len(development) == 30
    assert len(validation) == 30
    assert not ({item["episode"] for item in development} & {item["episode"] for item in validation})
    seeds = [item["split_seed"] for item in development + validation]
    assert min(seeds) >= 710_000
    assert max(seeds) < 810_000
    assert not any(610_000 <= seed < 710_000 for seed in seeds)


def test_low_confidence_always_falls_back_to_fixed() -> None:
    route, reason = RUNNER.route_from_probability(0.649999)
    assert route == RUNNER.FIXED
    assert reason == "low_confidence_fixed_fallback"
    route, _ = RUNNER.route_from_probability(0.65)
    assert route == RUNNER.PURE_ARD


def test_feature_schema_is_compact_and_prequery_only() -> None:
    assert len(RUNNER.FEATURE_NAMES) == 6
    assert "observed_yield" not in RUNNER.FEATURE_NAMES
    assert "best_found" not in RUNNER.FEATURE_NAMES
    assert "AUC_best_so_far" not in RUNNER.FEATURE_NAMES


def test_selector_artifact_rejects_validation_training_provenance() -> None:
    development, validation = RUNNER.make_episode_partitions()
    artifact = {
        "feature_names": list(RUNNER.FEATURE_NAMES),
        "training_episode_ids": [item["episode"] for item in development],
        "training_split_seeds": [item["split_seed"] for item in development],
        "validation_episode_ids_used_for_training": [validation[0]["episode"]],
        "ard_confidence_threshold": RUNNER.ARD_CONFIDENCE_THRESHOLD,
        "input": {"sha256": RUNNER.file_sha256(RUNNER.TRAIN_FILE)},
        "config": {"sha256": RUNNER.file_sha256(RUNNER.CONFIG_FILE)},
        "public_test_labels_used": False,
    }
    try:
        RUNNER.validate_selector_artifact(artifact, development)
    except ValueError as exc:
        assert "Validation episodes" in str(exc)
    else:
        raise AssertionError("Validation-contaminated selector artifact was accepted.")
