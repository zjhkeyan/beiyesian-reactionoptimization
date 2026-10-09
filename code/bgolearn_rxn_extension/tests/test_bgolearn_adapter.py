from __future__ import annotations

import sys
import types

import numpy as np

from bgolearn_rxn.backend import BgolearnCompatibilityBackend
from bgolearn_rxn.factory import build_ensemble, prepare_problem


def test_stock_bgolearn_bridge_settings_and_predictions(
    monkeypatch, tiny_dataset, tiny_config
) -> None:
    calls: dict[str, object] = {}

    class FakeController:
        def fit(self, **kwargs):
            calls.update(kwargs)
            model = kwargs["Kriging_model"]()
            mean, std = model.fit_pre(
                kwargs["data_matrix"].to_numpy(),
                kwargs["Measured_response"],
                kwargs["virtual_samples"].to_numpy(),
            )
            return types.SimpleNamespace(
                virtual_samples_mean=mean,
                virtual_samples_std=std,
            )

    fake_sampling = types.SimpleNamespace(Bgolearn=FakeController)
    fake_package = types.ModuleType("Bgolearn")
    fake_package.BGOsampling = fake_sampling
    monkeypatch.setitem(sys.modules, "Bgolearn", fake_package)

    prepared = prepare_problem(tiny_dataset, tiny_config)
    ensemble = build_ensemble(tiny_config, prepared, seed=100)
    backend = BgolearnCompatibilityBackend(ensemble=ensemble, seed=100)
    posterior = backend.fit_predict(
        prepared.X_initial,
        tiny_dataset.initial["Yield"].to_numpy(),
        prepared.X_candidates,
    )

    assert posterior.mean.shape == (len(tiny_dataset.candidates),)
    assert calls["Normalize"] is False
    assert calls["Dynamic_W"] is False
    assert calls["CV_test"] is False
