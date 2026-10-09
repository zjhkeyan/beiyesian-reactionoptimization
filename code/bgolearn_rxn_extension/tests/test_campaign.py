from __future__ import annotations

from bgolearn_rxn.campaign import CampaignRunner
from bgolearn_rxn.oracle import OfflineTableOracle
from bgolearn_rxn.serialization import validate_submission_payload


def test_campaign_is_unique_and_serializable(tiny_dataset, tiny_config, tiny_files) -> None:
    _, _, oracle_path = tiny_files
    oracle = OfflineTableOracle.from_csv(tiny_dataset, oracle_path)
    runner = CampaignRunner(tiny_dataset, tiny_config, seed=100)
    result = runner.run(oracle, budget=3)
    payload = result.to_payload()

    ids = [row["query_index"] for row in payload["trajectory"]]
    assert len(ids) == len(set(ids)) == 3
    assert payload["trajectory"][0]["target_incumbent_before_query"] == 12.0
    assert payload["metrics"]["n_queries"] == 3
    validate_submission_payload(
        payload,
        tiny_dataset.candidates,
        feature_columns=tiny_dataset.spec.feature_columns,
        expected_budget=3,
    )


def test_prequential_weights_are_recorded(tiny_dataset, tiny_config, tiny_files) -> None:
    _, _, oracle_path = tiny_files
    oracle = OfflineTableOracle.from_csv(tiny_dataset, oracle_path)
    result = CampaignRunner(tiny_dataset, tiny_config, seed=100).run(oracle, budget=1)
    row = result.trajectory[0]
    assert set(row["model_weights_before_query"]) == {"gp", "interaction", "trees"}
    assert abs(sum(row["model_weights_after_query"].values()) - 1.0) < 1e-12
