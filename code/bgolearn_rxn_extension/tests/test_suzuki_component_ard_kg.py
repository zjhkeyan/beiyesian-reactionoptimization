from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from bgolearn_rxn.factory import build_ensemble, prepare_problem
from bgolearn_rxn.utils import deterministic_argmax


RUNNER_PATH = Path(__file__).resolve().parents[2] / "experiments/run_suzuki_component_ard_kg.py"
SPEC = importlib.util.spec_from_file_location("run_suzuki_component_ard_kg", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def _fitted_first_problem():
    episode = RUNNER.make_episodes()[0]
    config, dataset, _ = RUNNER.build_problem(episode)
    prepared = prepare_problem(dataset, config)
    model = build_ensemble(config, prepared, seed=int(episode["campaign_seed"])).models[
        RUNNER.MODEL_NAME
    ]
    y = dataset.initial[dataset.spec.target_column].to_numpy(dtype=float)
    model.fit(prepared.X_initial, y)
    posterior = model.predict(prepared.X_candidates)
    best_f = dataset.target_incumbent(dataset.initial)
    return episode, dataset, prepared, model, posterior, best_f


def test_discovery_episode_and_policy_contract() -> None:
    episodes = RUNNER.make_episodes()
    assert len(episodes) == 60
    assert episodes[0]["split_seed"] == 1_010_000
    assert all(len(item["initial_indices"]) == 20 for item in episodes)
    assert all(len(item["candidate_indices"]) == 9 for item in episodes)
    assert all(item["budget"] == 5 for item in episodes)
    config, _, _ = RUNNER.build_problem(episodes[0])
    assert tuple(config.ensemble.enabled_models) == ("component_ard_gp",)
    assert RUNNER.QUADRATURE_ORDER == 32
    assert RUNNER.finite_pool_knowledge_gradient.__module__ == "bgolearn_rxn.knowledge_gradient"


def test_query_one_scores_are_exactly_identical() -> None:
    _, _, prepared, model, posterior, best_f = _fitted_first_problem()
    baseline, baseline_name, _ = RUNNER.policy_scores(
        RUNNER.BASELINE, 1, 5, posterior, model, prepared.X_candidates, best_f
    )
    candidate, candidate_name, _ = RUNNER.policy_scores(
        RUNNER.CANDIDATE, 1, 5, posterior, model, prepared.X_candidates, best_f
    )
    np.testing.assert_array_equal(baseline, candidate)
    assert baseline_name == candidate_name == "posterior_mean"


def test_component_ard_covariance_is_psd_and_matches_marginals() -> None:
    _, _, prepared, model, posterior, _ = _fitted_first_problem()
    covariance = model.posterior_covariance(prepared.X_candidates)
    np.testing.assert_allclose(covariance, covariance.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(covariance), posterior.std**2, rtol=1e-6)
    assert np.min(np.linalg.eigvalsh(covariance)) >= -1e-8
    assert RUNNER.observation_variance(model) > 0.0


def test_kg_policy_is_deterministic_and_tie_breaking_uses_candidate_ids() -> None:
    episode, dataset, prepared, model, posterior, best_f = _fitted_first_problem()
    first, name, first_diagnostic = RUNNER.policy_scores(
        RUNNER.CANDIDATE, 2, 5, posterior, model, prepared.X_candidates, best_f
    )
    second, _, second_diagnostic = RUNNER.policy_scores(
        RUNNER.CANDIDATE, 2, 5, posterior, model, prepared.X_candidates, best_f
    )
    np.testing.assert_allclose(first, second, atol=0.0, rtol=0.0)
    assert name == "finite_pool_one_step_kg"
    assert first_diagnostic["quadrature_order"] == 32
    assert first_diagnostic["covariance_min_eigenvalue"] >= -1e-8
    assert second_diagnostic["kg_score_min"] >= 0.0
    candidate_ids = dataset.candidates["query_index"].to_numpy(dtype=int)
    left = deterministic_argmax(first, candidate_ids, seed=episode["campaign_seed"] + 2)
    right = deterministic_argmax(second, candidate_ids, seed=episode["campaign_seed"] + 2)
    assert left == right
