from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bgolearn_rxn.factory import prepare_problem


RUNNER_PATH = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "run_suzuki_phase9_frozen_morgan.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_suzuki_phase9_frozen_morgan", RUNNER_PATH
)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_registry_exactly_covers_public_optimization_options() -> None:
    options = json.loads(RUNNER.BASE.OPTIONS_FILE.read_text(encoding="utf-8"))
    registry = pd.read_csv(RUNNER.REGISTRY_FILE, keep_default_na=False)
    expected = {
        (role, component)
        for role in RUNNER.ATOMIC_ROLES
        for component in options[role]
    }
    observed = set(zip(registry["role"], registry["component"], strict=True))
    assert observed == expected
    assert not registry.duplicated(["role", "component"]).any()
    inferred = registry[registry["validation_status"].eq("derived_unverified")]
    assert len(inferred) == 1
    row = inferred.iloc[0]
    assert (row["role"], row["component"]) == RUNNER.INFERRED_KEY
    assert row["source_precursor_cid"] == "129009568"
    assert row["derivation_rule"]
    assert "Not externally verified" in row["notes"]


def test_primary_encoder_preserves_every_baseline_identity_and_rule_block() -> None:
    config, dataset, _ = RUNNER.BASE.build_problem("component_ard")
    baseline = prepare_problem(dataset, config)
    candidate, audit = RUNNER.build_morgan_prepared(
        dataset, inferred_policy="structure", molecular_scale=0.35
    )
    assert candidate.encoder.schema is not None
    assert baseline.encoder.schema is not None
    for role in candidate.encoder.schema.roles:
        baseline_slice = baseline.encoder.schema.role_slices[role]
        candidate_slice = candidate.encoder.schema.role_slices[role]
        baseline_width = baseline_slice.width
        np.testing.assert_array_equal(
            baseline.X_initial[:, baseline_slice.start : baseline_slice.stop],
            candidate.X_initial[
                :, candidate_slice.start : candidate_slice.start + baseline_width
            ],
        )
        np.testing.assert_array_equal(
            baseline.X_candidates[:, baseline_slice.start : baseline_slice.stop],
            candidate.X_candidates[
                :, candidate_slice.start : candidate_slice.start + baseline_width
            ],
        )
    assert audit["baseline_name_rule_scale"] == config.features.rule_scale == 0.75
    assert sum(audit["active_morgan_widths"].values()) > 0
    assert "ProductContext" not in candidate.encoder.schema.condition_roles
    np.testing.assert_array_equal(
        candidate.X_candidates,
        candidate.encoder.transform(
            dataset.candidates.loc[:, list(dataset.spec.feature_columns)]
        ),
    )


def test_inferred_missing_sensitivity_changes_only_inferred_rows_on_common_coordinates() -> None:
    _, dataset, _ = RUNNER.BASE.build_problem("component_ard")
    structured, _ = RUNNER.build_morgan_prepared(
        dataset, inferred_policy="structure", molecular_scale=0.35
    )
    missing, _ = RUNNER.build_morgan_prepared(
        dataset, inferred_policy="missing_indicator", molecular_scale=0.35
    )
    structured_names = structured.encoder.schema.feature_names
    missing_names = missing.encoder.schema.feature_names
    assert structured_names == missing_names
    missing_lookup = {name: index for index, name in enumerate(missing_names)}
    aligned_missing = missing.X_candidates[
        :, [missing_lookup[name] for name in structured_names]
    ]
    inferred_mask = dataset.candidates["Nucleophile"].astype(str).eq(
        RUNNER.INFERRED_KEY[1]
    ).to_numpy()
    np.testing.assert_array_equal(
        structured.X_candidates[~inferred_mask], aligned_missing[~inferred_mask]
    )
    assert np.any(
        structured.X_candidates[inferred_mask] != aligned_missing[inferred_mask]
    )


def test_train_only_holdouts_cover_components_and_substrate_pairs() -> None:
    episodes = RUNNER.make_train_holdout_episodes()
    assert len(episodes) == 40
    assert {episode["held_role"] for episode in episodes} == {
        *RUNNER.ATOMIC_ROLES,
        "SubstratePair",
    }
    assert any(episode["held_level"] == RUNNER.INFERRED_KEY[1] for episode in episodes)
    source_size = len(pd.read_csv(RUNNER.BASE.TRAIN_FILE))
    for episode in episodes:
        initial = set(episode["initial_indices"])
        candidates = set(episode["candidate_indices"])
        assert initial.isdisjoint(candidates)
        assert initial | candidates == set(range(source_size))
        assert episode["budget"] == min(5, len(candidates))


def test_cli_is_dev_only_and_uses_predeclared_molecular_scale(tmp_path: Path) -> None:
    args = RUNNER.parse_args(["--output-dir", str(tmp_path)])
    assert args.split == "phase8_dev"
    assert args.molecular_scale == 0.35
    with pytest.raises(SystemExit):
        RUNNER.parse_args(
            ["--split", "phase8_test", "--output-dir", str(tmp_path)]
        )


def test_holdout_summary_allows_a_single_candidate_variant() -> None:
    rows = pd.DataFrame(
        [
            {
                "episode_family": "leave_ligand_out",
                "episode": "one",
                "variant": "identity_morgan",
                "initial_round_found_best": 1.0,
                "best_found": 2.0,
                "round_to_95_global_best": 3.0,
                "AUC_best_so_far": 1.5,
                "elapsed_seconds": 0.1,
            }
        ]
    )
    absolute, paired = RUNNER.summarize_holdouts(rows)
    assert len(absolute) == 5
    assert paired.empty
