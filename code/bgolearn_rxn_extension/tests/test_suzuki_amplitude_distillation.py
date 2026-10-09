from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from bgolearn_rxn.factory import prepare_problem
from bgolearn_rxn.surrogates.anova_gp import StructuredANOVAGP


RUNNER_PATH = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "run_suzuki_amplitude_distillation.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_suzuki_amplitude_distillation", RUNNER_PATH
)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_episode_partition_is_frozen_disjoint_and_710k_only() -> None:
    development, validation = RUNNER.make_episode_partitions()
    assert len(development) == len(validation) == 30
    assert development[0]["episode"].endswith("_00")
    assert development[-1]["episode"].endswith("_29")
    assert validation[0]["episode"].endswith("_30")
    assert validation[-1]["episode"].endswith("_59")
    assert {item["episode"] for item in development}.isdisjoint(
        item["episode"] for item in validation
    )
    all_episodes = development + validation
    assert min(item["split_seed"] for item in all_episodes) >= 710_000
    assert max(item["split_seed"] for item in all_episodes) < 810_000


def test_distillation_is_predeclared_geometric_mean_with_no_validation_rows() -> None:
    development = [
        {
            "episode": f"episode_{index:02d}",
            "split_seed": 710_000 + index,
        }
        for index in range(30)
    ]
    rows = []
    for episode_index, episode in enumerate(development):
        for step in range(1, 6):
            for component, base in (("bias", 1.0), ("role:A", 4.0)):
                rows.append(
                    {
                        "episode": episode["episode"],
                        "step": step,
                        "component": component,
                        "amplitude": base * (1.0 if episode_index < 15 else 4.0),
                    }
                )
    weights = RUNNER.distill_weights(pd.DataFrame(rows), development)
    assert weights["method"].startswith("log-space mean")
    assert weights["n_fit_snapshots"] == 150
    assert weights["validation_episode_ids_used_for_learning"] == []
    assert np.isclose(weights["component_amplitudes"]["bias"], 2.0)
    assert np.isclose(weights["component_amplitudes"]["role:A"], 8.0)


def test_frozen_model_has_base_gp_parameter_count_and_fixed_components(
    tiny_dataset, tiny_config
) -> None:
    prepared = prepare_problem(tiny_dataset, tiny_config)
    schema = prepared.encoder.schema
    base = StructuredANOVAGP(schema, tiny_config.gp, seed=1)
    components = {"bias": 1.0}
    components.update({f"role:{role}": 0.7 for role in base.roles})
    components.update(
        {f"pair:{'|'.join(pair)}": 0.2 for pair in tiny_config.gp.interaction_pairs}
    )
    model = RUNNER.FrozenAmplitudeStructuredGP(
        schema, tiny_config.gp, components, seed=1
    )
    assert len(model._initial_theta()) == len(base._initial_theta())
    assert model._bounds()[model.layout.pair_scale_index] == (0.0, 0.0)
    y = tiny_dataset.initial[tiny_config.dataset.target_column].to_numpy(dtype=float)
    model.fit(prepared.X_initial, y)
    assert model.component_amplitudes_ == components
    kernel = model._kernel(prepared.X_initial, prepared.X_initial, model.theta_)
    np.testing.assert_allclose(kernel, kernel.T, atol=1e-12)
    assert np.min(np.linalg.eigvalsh(kernel)) >= -1e-9


def _gate_rows(*, candidate_auc_delta: float = 1.0) -> pd.DataFrame:
    rows = []
    for index in range(30):
        for variant in RUNNER.VARIANTS:
            if variant == "fixed":
                initial, best, t95, auc, runtime = 10.0, 20.0, 4.0, 15.0, 1.0
            elif variant == "pure_ard":
                initial, best, t95 = 10.5, 20.5, 3.5
                auc = 16.0 + float((index % 3) - 1)
                runtime = 2.0
            else:
                initial, best, t95 = 10.5, 20.5, 3.5
                auc = 15.0 + candidate_auc_delta + 0.05 * float(index % 2)
                runtime = 1.2
            rows.append(
                {
                    "episode": f"validation_{index:02d}",
                    "variant": variant,
                    "phase": "amplitude_validation",
                    "initial_round_found_best": initial,
                    "best_found": best,
                    "round_to_95_global_best": t95,
                    "AUC_best_so_far": auc,
                    "elapsed_seconds": runtime,
                }
            )
    return pd.DataFrame(rows)


def test_gate_requires_directions_lower_paired_variance_and_fixed_runtime() -> None:
    passing = RUNNER.pilot_decision(_gate_rows())
    assert passing["passed"] is True
    assert passing["runtime_ratio_distilled_vs_fixed"] == 1.2
    failing = RUNNER.pilot_decision(_gate_rows(candidate_auc_delta=-0.5))
    assert failing["checks"]["auc_noninferior_vs_fixed"] is False
    assert failing["passed"] is False
