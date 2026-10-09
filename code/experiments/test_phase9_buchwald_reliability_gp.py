from __future__ import annotations

from dataclasses import replace
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent
sys.path.insert(0, str(CODE_ROOT / "bgolearn_rxn_extension" / "src"))
sys.path.insert(0, str(HERE))

import run_buchwald_source_meta_selection as meta  # noqa: E402
from bgolearn_rxn.campaign import CampaignRunner  # noqa: E402
from phase9_buchwald_reliability_gp import (  # noqa: E402
    ReliabilityShrunkComponentARDLowRankTaskGP,
)


def source_problem():
    episode = meta.make_source_meta_episodes()[0]
    config, dataset, _, products = meta.build_problem(
        episode, strict_source_isolation=True
    )
    runner = CampaignRunner(dataset, config, seed=episode["seed"], backend="native")
    y = dataset.initial[config.dataset.target_column].to_numpy(dtype=float)
    return episode, replace(config.gp, optimize_hyperparameters=False), runner, y, products


def make_model(runner, config, *, floor=0.35):
    return ReliabilityShrunkComponentARDLowRankTaskGP(
        runner.prepared.encoder.schema,
        config,
        seed=50079,
        reliability_floor=floor,
    )


def test_source_firewall_and_target_outcome_invariance():
    _, config, runner, y, products = source_problem()
    assert meta.OFFICIAL_TARGET not in products
    X = runner.prepared.X_initial
    model = make_model(runner, config)
    target, scores, multipliers = model._estimate_source_reliability(X, y)
    tasks = model._task_indices(X)
    changed = y.copy()
    changed[tasks == target] = np.array([-1000.0, 1000.0])
    target_2, scores_2, multipliers_2 = model._estimate_source_reliability(X, changed)
    assert target == target_2
    assert scores == scores_2
    np.testing.assert_allclose(multipliers, multipliers_2, atol=0.0, rtol=0.0)


def test_shrinkage_is_psd_and_preserves_unit_diagonal():
    _, config, runner, y, _ = source_problem()
    model = make_model(runner, config, floor=0.15).fit(runner.prepared.X_initial, y)
    raw = model.raw_task_correlation_
    shrunk = model.task_correlation_
    assert raw is not None and shrunk is not None
    np.testing.assert_allclose(np.diag(shrunk), 1.0, atol=1e-12)
    assert float(np.min(np.linalg.eigvalsh(shrunk))) >= -1e-10
    d = model.task_multipliers_
    expected = d[:, None] * raw * d[None, :]
    expected.flat[:: len(expected) + 1] += 1.0 - d**2
    np.testing.assert_allclose(shrunk, expected, atol=1e-12)
    assert d[model.target_task_index_] == 1.0


def test_reliability_fit_is_deterministic_and_predictive():
    _, config, runner, y, _ = source_problem()
    X = runner.prepared.X_initial
    candidates = runner.prepared.X_candidates
    first = make_model(runner, config).fit(X, y)
    second = make_model(runner, config).fit(X, y)
    assert first.source_reliability_ == second.source_reliability_
    np.testing.assert_allclose(first.task_multipliers_, second.task_multipliers_)
    posterior_1 = first.predict(candidates)
    posterior_2 = second.predict(candidates)
    np.testing.assert_allclose(posterior_1.mean, posterior_2.mean)
    np.testing.assert_allclose(posterior_1.std, posterior_2.std)
    assert np.all(np.isfinite(posterior_1.mean))
    assert np.all(posterior_1.std > 0.0)

