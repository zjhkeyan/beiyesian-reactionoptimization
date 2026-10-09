from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2] / "experiments"
sys.path.insert(0, str(EXPERIMENT_ROOT))

import run_suzuki_consensus_portfolio as portfolio  # noqa: E402


def test_fixed_top_three_accepts_graph_candidate_at_ard_rank_two() -> None:
    ids = np.array([10, 20, 30, 40, 50])
    decision = portfolio.arbitrate_top_three(
        np.array([5.0, 4.0, 3.0, 2.0, 1.0]),
        np.array([1.0, 9.0, 2.0, 0.0, -1.0]),
        ids,
        seed=17,
    )
    assert portfolio.TOP_K == 3
    assert decision["graph_top_query_index"] == 20
    assert decision["graph_top_ard_rank"] == 2
    assert decision["gate_accepted"] is True
    assert decision["selected_query_index"] == 20
    assert decision["selection_changed_from_ard"] is True


def test_fixed_top_three_rejects_graph_candidate_at_ard_rank_four() -> None:
    ids = np.array([10, 20, 30, 40, 50])
    decision = portfolio.arbitrate_top_three(
        np.array([5.0, 4.0, 3.0, 2.0, 1.0]),
        np.array([1.0, 0.0, 2.0, 9.0, -1.0]),
        ids,
        seed=19,
    )
    assert decision["graph_top_ard_rank"] == 4
    assert decision["gate_accepted"] is False
    assert decision["selected_expert"] == "ard"
    assert decision["selected_query_index"] == 10
    assert decision["selection_changed_from_ard"] is False


def test_consensus_ranking_is_invariant_to_candidate_row_order_with_ties() -> None:
    ids = np.array([101, 202, 303, 404, 505])
    ard = np.array([5.0, 5.0, 4.0, 3.0, 2.0])
    graph = np.array([1.0, 1.0, 7.0, 0.0, -1.0])
    first = portfolio.arbitrate_top_three(ard, graph, ids, seed=23)
    order = np.array([4, 2, 0, 3, 1])
    second = portfolio.arbitrate_top_three(
        ard[order], graph[order], ids[order], seed=23
    )
    for key in (
        "selected_query_index",
        "ard_top_query_index",
        "ard_top_three_query_indices",
        "graph_top_query_index",
        "graph_top_ard_rank",
        "gate_accepted",
    ):
        assert first[key] == second[key]


def test_query_one_schedule_uses_posterior_mean_only() -> None:
    class Posterior:
        mean = np.array([1.0, 3.0, 2.0])
        std = np.array([100.0, 1.0, 1.0])

    scores, name = portfolio.acquisition_scores(Posterior(), None, None, 1)
    np.testing.assert_array_equal(scores, Posterior.mean)
    assert name == "posterior_mean"
