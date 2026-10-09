from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2] / "experiments"
sys.path.insert(0, str(EXPERIMENT_ROOT))

from noise_aware_acquisition import noise_aware_scores, score_diagnostics  # noqa: E402
from run_noise_aware_acquisition import (  # noqa: E402
    BASELINE_POLICY,
    confirmation_decision,
    confirmation_policies,
    make_buchwald_confirmation_episodes,
    make_buchwald_pilot_episodes,
    make_suzuki_episodes,
    select_pilot_winner,
)


def _inputs():
    return {
        "mean": np.array([1.0, 2.5, 3.0]),
        "std": np.array([0.2, 0.7, 1.1]),
        "observed_best": 2.0,
        "train_mean": np.array([1.5, 2.2, 9.0]),
        "train_std": np.array([0.1, 0.3, 0.5]),
        "target_mask": np.array([True, True, False]),
        "observation_noise": 0.4,
        "step": 2,
        "budget": 5,
    }


def test_plugin_and_aei_match_direct_bgolearn_formulas():
    values = _inputs()
    plugin, metadata = noise_aware_scores("mean_then_plugin_ei", **values)
    incumbent = 2.2
    z = (values["mean"] - incumbent) / values["std"]
    direct = (values["mean"] - incumbent) * norm.cdf(z) + values["std"] * norm.pdf(z)
    np.testing.assert_allclose(np.exp(plugin), direct, rtol=1e-10, atol=1e-12)
    assert metadata["incumbent"] == incumbent

    aei, _ = noise_aware_scores("mean_then_augmented_ei", **values)
    # The target UCB maximum is the second target row, whose posterior mean is 2.2.
    penalty = 1.0 - 0.4 / np.sqrt(values["std"] ** 2 + 0.4**2)
    np.testing.assert_allclose(np.exp(aei), direct * penalty, rtol=1e-10, atol=1e-12)


def test_eqi_matches_direct_formula_and_handles_zero_variance():
    values = _inputs()
    scores, metadata = noise_aware_scores("mean_then_eqi_075", **values)
    beta_z = norm.ppf(0.75)
    tau = values["observation_noise"]
    variance = values["std"] ** 2
    adjusted_mean = values["mean"] + beta_z * np.sqrt(tau**2 * variance / (tau**2 + variance))
    adjusted_std = variance / np.sqrt(variance + tau**2)
    incumbent = np.max(
        values["train_mean"][values["target_mask"]]
        + beta_z * values["train_std"][values["target_mask"]]
    )
    z = (adjusted_mean - incumbent) / adjusted_std
    direct = (adjusted_mean - incumbent) * norm.cdf(z) + adjusted_std * norm.pdf(z)
    np.testing.assert_allclose(np.exp(scores), direct, rtol=1e-10, atol=1e-12)
    assert metadata["eqi_beta"] == 0.75

    zero = dict(values)
    zero["std"] = np.array([0.0, 0.7, 1.1])
    stable, _ = noise_aware_scores("mean_then_eqi_075", **zero)
    assert np.isneginf(stable[0])
    assert np.isfinite(stable[1:]).all()


def test_reinterpolation_requires_aligned_refit_posterior():
    values = _inputs()
    try:
        noise_aware_scores("mean_then_reinterpolation_ei", **values)
    except ValueError as exc:
        assert "requires" in str(exc)
    else:
        raise AssertionError("Missing reinterpolation posterior was accepted.")
    scores, metadata = noise_aware_scores(
        "mean_then_reinterpolation_ei",
        **values,
        reinterpolated_mean=np.array([1.2, 2.0, 2.7]),
        reinterpolated_std=np.array([0.15, 0.4, 0.8]),
    )
    assert np.isfinite(scores).all()
    assert metadata["formula"] == "reinterpolation_ei_smoothed_refit"


def test_score_diagnostics_detects_formula_ties_and_nonfinite_values():
    diagnostics = score_diagnostics(np.array([-np.inf, 1.0, 1.0]))
    assert diagnostics == {
        "finite_count": 2,
        "nonfinite_count": 1,
        "finite_span": 0.0,
        "all_finite_scores_tied": True,
    }


def test_stage_episode_contracts_are_disjoint_and_fixed_size():
    buchwald_pilot = make_buchwald_pilot_episodes()
    buchwald_confirmation = make_buchwald_confirmation_episodes()
    assert len(buchwald_pilot) == 20
    assert len(buchwald_confirmation) == 21
    assert not {item["target_task"] for item in buchwald_pilot}.intersection(
        {item["target_task"] for item in buchwald_confirmation}
    )
    suzuki_pilot = make_suzuki_episodes("pilot")
    suzuki_confirmation = make_suzuki_episodes("confirmation")
    assert len(suzuki_pilot) == 20
    assert len(suzuki_confirmation) == 80
    assert {item["split_seed"] for item in suzuki_pilot}.isdisjoint(
        {item["split_seed"] for item in suzuki_confirmation}
    )


def test_selection_requires_strict_auc_win_and_confirmation_uses_only_winner():
    import pandas as pd

    rows = []
    policies = (
        BASELINE_POLICY,
        "mean_then_plugin_ei",
        "mean_then_augmented_ei",
        "mean_then_eqi_075",
        "mean_then_reinterpolation_ei",
    )
    for episode in ("a", "b"):
        for policy in policies:
            rows.append(
                {
                    "dataset": "Suzuki",
                    "episode": episode,
                    "policy": policy,
                    "initial_round_found_best": 10.0,
                    "best_found": 20.0,
                    "round_to_95_global_best": 2.0,
                    "AUC_best_so_far": 15.2 if policy == "mean_then_eqi_075" else 15.0,
                    "numerics_clean": True,
                }
            )
    decision = select_pilot_winner(pd.DataFrame(rows), "Suzuki")
    assert decision["passed"] is True
    assert decision["selected_policy"] == "mean_then_eqi_075"
    selection = {"Suzuki": decision}
    assert confirmation_policies(selection, "Suzuki") == (
        BASELINE_POLICY,
        "mean_then_eqi_075",
    )


def test_confirmation_rejects_a_pilot_direction_reversal():
    import pandas as pd

    selection = {
        "Buchwald_sub4": {
            "passed": True,
            "selected_policy": "mean_then_reinterpolation_ei",
        },
        "Suzuki": {"passed": False, "selected_policy": BASELINE_POLICY},
    }
    rows = []
    for episode in ("a", "b"):
        for policy, auc, t95 in (
            (BASELINE_POLICY, 20.0, 2.0),
            ("mean_then_reinterpolation_ei", 19.0, 2.5),
        ):
            rows.append(
                {
                    "dataset": "Buchwald_sub4",
                    "episode": episode,
                    "policy": policy,
                    "initial_round_found_best": 10.0,
                    "best_found": 30.0,
                    "round_to_95_global_best": t95,
                    "AUC_best_so_far": auc,
                    "numerics_clean": True,
                }
            )
    decision = confirmation_decision(pd.DataFrame(rows), selection)
    assert decision["Buchwald_sub4"]["confirmed"] is False
    assert decision["Suzuki"]["reason"] == "not_run_after_failed_pilot_gate"
