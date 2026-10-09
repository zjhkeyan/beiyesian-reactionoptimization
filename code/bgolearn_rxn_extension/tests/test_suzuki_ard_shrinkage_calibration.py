from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from bgolearn_rxn.factory import prepare_problem
from bgolearn_rxn.surrogates.component_ard_gp import ComponentARDStructuredGP


RUNNER_PATH = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "run_suzuki_ard_shrinkage_calibration.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_suzuki_ard_shrinkage_calibration", RUNNER_PATH
)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_grid_and_production_default_are_frozen() -> None:
    assert RUNNER.SCALES == (0.25, 0.40, 0.60, 0.90)
    assert RUNNER.DEFAULT_SCALE == 0.60
    assert ComponentARDStructuredGP.amplitude_prior_scale == 0.60
    assert RUNNER.DEVELOPMENT_VARIANTS == (
        "fixed",
        "ard_scale_0p25",
        "ard_scale_0p40",
        "ard_scale_0p60",
        "ard_scale_0p90",
    )


def test_episode_partition_is_exact_disjoint_710k_contract() -> None:
    development, validation = RUNNER.make_episode_partitions()
    assert len(development) == len(validation) == 30
    assert development[0]["episode"].endswith("_00")
    assert development[-1]["episode"].endswith("_29")
    assert validation[0]["episode"].endswith("_30")
    assert validation[-1]["episode"].endswith("_59")
    assert {item["episode"] for item in development}.isdisjoint(
        item["episode"] for item in validation
    )
    seeds = [item["split_seed"] for item in development + validation]
    assert min(seeds) >= 710_000
    assert max(seeds) < 810_000


def test_experiment_subclass_changes_scale_without_class_mutation(
    tiny_dataset, tiny_config
) -> None:
    prepared = prepare_problem(tiny_dataset, tiny_config)
    weak = RUNNER.CalibratedShrinkageComponentARDGP(
        prepared.encoder.schema, tiny_config.gp, 0.90, seed=1
    )
    strong = RUNNER.CalibratedShrinkageComponentARDGP(
        prepared.encoder.schema, tiny_config.gp, 0.25, seed=1
    )
    theta = weak._initial_theta()
    theta[weak._amplitude_start :] += 0.2
    assert strong._amplitude_penalty(theta) > weak._amplitude_penalty(theta)
    assert weak.amplitude_prior_scale == 0.90
    assert strong.amplitude_prior_scale == 0.25
    assert ComponentARDStructuredGP.amplitude_prior_scale == 0.60


def _development_rows(scores: dict[float, tuple[float, float, float]]) -> pd.DataFrame:
    rows = []
    for index in range(30):
        episode = f"random_amplitude_distillation_{index:02d}"
        rows.append(
            {
                "episode": episode,
                "variant": "fixed",
                "initial_round_found_best": 10.0,
                "best_found": 20.0,
                "round_to_95_global_best": 4.0,
                "AUC_best_so_far": 15.0,
                "elapsed_seconds": 1.0,
            }
        )
        for scale, (auc, t95, loss_offset) in scores.items():
            rows.append(
                {
                    "episode": episode,
                    "variant": RUNNER.scale_variant(scale),
                    "initial_round_found_best": 10.0,
                    "best_found": 20.0,
                    "round_to_95_global_best": t95,
                    "AUC_best_so_far": auc + (loss_offset if index == 0 else 0.0),
                    "elapsed_seconds": 2.0,
                }
            )
    return pd.DataFrame(rows)


def test_selection_is_auc_then_t95_then_loss_then_stronger_shrinkage() -> None:
    rows = _development_rows(
        {
            0.25: (16.0, 3.0, 0.0),
            0.40: (16.0, 3.0, 0.0),
            0.60: (15.9, 2.0, 0.0),
            0.90: (15.8, 1.0, 0.0),
        }
    )
    decision = RUNNER.development_selection(rows)
    assert decision["selected_amplitude_prior_scale"] == 0.25
    assert decision["validation_episode_ids_used_for_selection"] == []


def _validation_rows(selected_auc: float = 16.5) -> pd.DataFrame:
    rows = []
    for index in range(30):
        episode = f"random_amplitude_distillation_{index + 30:02d}"
        for variant, initial, best, t95, auc in (
            ("fixed", 10.0, 20.0, 4.0, 15.0),
            ("default_ard", 10.0, 20.5, 3.8, 16.0),
            ("selected_ard", 10.5, 21.0, 3.5, selected_auc),
        ):
            rows.append(
                {
                    "dataset": "Suzuki",
                    "phase": "shrinkage_validation",
                    "episode_family": "random_holdout",
                    "episode": episode,
                    "variant": variant,
                    "initial_round_found_best": initial,
                    "best_found": best,
                    "round_to_95_global_best": t95,
                    "AUC_best_so_far": auc,
                    "elapsed_seconds": 1.0,
                }
            )
    return pd.DataFrame(rows)


def test_validation_gate_requires_strict_default_improvement_and_fixed_safety() -> None:
    passing = RUNNER.validation_gate(_validation_rows(), 0.40)
    assert passing["passed"] is True
    failing = RUNNER.validation_gate(_validation_rows(selected_auc=16.0), 0.40)
    assert failing["checks"]["auc_better_than_default_ard"] is False
    assert failing["passed"] is False


def test_amplitude_diagnostics_are_finite() -> None:
    rows = pd.DataFrame(
        [
            {
                "phase": "development",
                "episode": "episode_00",
                "variant": "ard_scale_0p25",
                "amplitude_prior_scale": 0.25,
                "final_component_amplitudes_json": '{"bias": 1.0, "role:A": 2.0}',
            }
        ]
    )
    diagnostics = RUNNER.amplitude_diagnostics(rows)
    assert diagnostics.loc[0, "n_components"] == 2
    assert np.isfinite(diagnostics.select_dtypes(include=[float])).all().all()
