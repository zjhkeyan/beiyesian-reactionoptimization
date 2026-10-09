from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from bgolearn_rxn.config import PolicyConfig
from bgolearn_rxn.schemas import Posterior


RUNNER_PATH = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "run_suzuki_homologous_committee.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_suzuki_homologous_committee", RUNNER_PATH
)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def _posterior(ard_weight: float) -> Posterior:
    fixed = np.array([3.0, 0.0])
    ard = np.array([0.0, 4.0])
    fixed_weight = 1.0 - ard_weight
    mean = fixed_weight * fixed + ard_weight * ard
    std = np.ones(2)
    return Posterior(
        mean=mean,
        std=std,
        component_means={RUNNER.MODEL_FIXED: fixed, RUNNER.MODEL_ARD: ard},
        component_stds={RUNNER.MODEL_FIXED: std, RUNNER.MODEL_ARD: std},
        weights={RUNNER.MODEL_FIXED: fixed_weight, RUNNER.MODEL_ARD: ard_weight},
        disagreement=np.array([1.0, 1.0]),
    )


def test_phase4_episode_contract_is_exact_and_confirmation_is_protected() -> None:
    episodes = RUNNER.make_pilot_episodes()
    assert len(episodes) == 60
    assert all(item["budget"] == 5 for item in episodes)
    assert min(item["split_seed"] for item in episodes) >= 710_000
    assert max(item["split_seed"] for item in episodes) < 810_000


def test_stage_variants_freeze_exactly_one_pilot_candidate() -> None:
    assert RUNNER.variants_for_stage("smoke", None) == RUNNER.SMOKE_VARIANTS
    assert RUNNER.variants_for_stage("pilot", RUNNER.GATED) == (
        RUNNER.FIXED,
        RUNNER.PURE_ARD,
        RUNNER.GATED,
    )


def test_disagreement_gate_falls_back_to_fixed_without_evidence() -> None:
    gate = RUNNER.EvidenceDisagreementGate(PolicyConfig(), seed=5)
    recommendation = gate.select(
        _posterior(0.5),
        np.array([10, 11]),
        best_f=0.0,
        step=1,
        budget=5,
    )
    assert recommendation.query_index == 10
    assert gate.last_decision["gate_reason"] == "fixed_fallback"
    assert gate.last_decision["pre_observation_only"] is True


def test_disagreement_gate_accepts_mixture_with_two_to_one_ard_evidence() -> None:
    gate = RUNNER.EvidenceDisagreementGate(PolicyConfig(), seed=5)
    recommendation = gate.select(
        _posterior(0.75),
        np.array([10, 11]),
        best_f=0.0,
        step=1,
        budget=5,
    )
    assert recommendation.query_index == 11
    assert gate.last_decision["gate_reason"] == "ard_prequential_evidence"
