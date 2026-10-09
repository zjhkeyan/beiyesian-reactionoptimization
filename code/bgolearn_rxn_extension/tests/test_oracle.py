from __future__ import annotations

import pytest

from bgolearn_rxn.oracle import OfflineTableOracle


def test_oracle_rejects_repeat_queries(tiny_dataset, tiny_files) -> None:
    _, _, oracle_path = tiny_files
    oracle = OfflineTableOracle.from_csv(tiny_dataset, oracle_path)
    first = int(tiny_dataset.candidate_ids[0])
    assert oracle.query(first) == 55.0
    with pytest.raises(ValueError, match="Repeated"):
        oracle.query(first)


def test_oracle_global_best_includes_initial_target(tiny_dataset, tiny_files) -> None:
    _, _, oracle_path = tiny_files
    oracle = OfflineTableOracle.from_csv(tiny_dataset, oracle_path)
    assert oracle.global_best == 82.0
