"""Minimal ask/tell integration for a laboratory controller.

This example intentionally contains no organizer-side lookup table. Replace the
``execute_experiment`` stub with an instrument or LIMS integration that accepts
an immutable candidate ID and returns one accepted yield measurement.
"""

from __future__ import annotations

from pathlib import Path

from bgolearn_rxn import CampaignRunner, DatasetSpec, load_reaction_dataset
from bgolearn_rxn.config import load_run_config
from bgolearn_rxn.serialization import save_json, save_torch_submission


def execute_experiment(query_index: int) -> float:
    raise NotImplementedError(
        f"Send candidate {query_index} to the laboratory and return Yield."
    )


def main() -> None:
    config = load_run_config("configs/suzuki.yaml")
    dataset = load_reaction_dataset(
        train_path=Path("external_evaluation/Suzuki/suzuki_train.csv"),
        candidate_path=Path("external_evaluation/Suzuki/suzuki_test_features.csv"),
        spec=DatasetSpec.from_config(config.dataset),
    )
    runner = CampaignRunner(dataset, config, seed=100, backend="native")
    session = runner.start_session(budget=40)

    while not session.is_complete:
        recommendation = session.recommend()
        print(
            f"step={session.step} id={recommendation.query_index} "
            f"mean={recommendation.predicted_mean:.2f} "
            f"std={recommendation.predicted_std:.2f} "
            f"policy={recommendation.acquisition_name}"
        )
        measured_yield = execute_experiment(recommendation.query_index)
        record = session.observe(recommendation.query_index, measured_yield)
        print(f"accepted yield={record['observed_yield']:.2f}")

    payload = session.finalize().to_payload()
    save_json(payload, "runs/live_suzuki/seed_100.json")
    save_torch_submission(payload, "runs/live_suzuki/seed_100.pt")


if __name__ == "__main__":
    main()
