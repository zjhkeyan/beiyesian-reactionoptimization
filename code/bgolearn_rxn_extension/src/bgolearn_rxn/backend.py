from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from .schemas import Posterior
from .surrogates.ensemble import PrequentialEnsemble


class PredictionBackend(Protocol):
    """Fit the persistent ensemble and return its candidate posterior."""

    name: str

    def fit_predict(
        self,
        X_observed: np.ndarray,
        y_observed: np.ndarray,
        X_candidates: np.ndarray,
    ) -> Posterior:
        ...


@dataclass
class NativeBackend:
    """Recommended stateful backend with direct access to all model posteriors."""

    ensemble: PrequentialEnsemble
    name: str = "native"

    def fit_predict(
        self,
        X_observed: np.ndarray,
        y_observed: np.ndarray,
        X_candidates: np.ndarray,
    ) -> Posterior:
        self.ensemble.fit(X_observed, y_observed)
        return self.ensemble.predict(X_candidates)


@dataclass
class BgolearnCompatibilityBackend:
    """Compatibility bridge to stock Bgolearn 3.1.0.

    Bgolearn remains responsible for invoking the custom surrogate on the
    observed and virtual samples. Acquisition is intentionally performed by
    :class:`MetricAwarePolicy`, because the extension requires stable LogEI,
    target-specific incumbents, and ensemble diagnostics not exposed by the
    stock acquisition API.

    Important stock-Bgolearn settings are hard coded here:

    - ``Normalize=False``: coordinates were frozen externally;
    - ``Dynamic_W=False``: do not response-resample tiny reaction datasets;
    - ``CV_test=False``: do not run unstable cross-validation in every round.
    """

    ensemble: PrequentialEnsemble
    seed: int
    name: str = "bgolearn"

    def fit_predict(
        self,
        X_observed: np.ndarray,
        y_observed: np.ndarray,
        X_candidates: np.ndarray,
    ) -> Posterior:
        try:
            from Bgolearn import BGOsampling
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "Bgolearn is not installed. Install the pinned compatibility "
                "extra with `pip install -e '.[bgolearn]'`."
            ) from exc

        shared_ensemble = self.ensemble
        state: dict[str, object] = {}

        class ReactionEnsembleAdapter:
            """No-argument adapter required by Bgolearn's custom-model path."""

            def fit_pre(
                self,
                xtrain: np.ndarray,
                ytrain: np.ndarray,
                xtest: np.ndarray,
            ) -> tuple[np.ndarray, np.ndarray]:
                shared_ensemble.fit(
                    np.asarray(xtrain, dtype=float),
                    np.asarray(ytrain, dtype=float).reshape(-1),
                )
                posterior = shared_ensemble.predict(np.asarray(xtest, dtype=float))
                state["posterior"] = posterior
                return posterior.mean, posterior.std

        observed_frame = pd.DataFrame(
            np.asarray(X_observed, dtype=float),
            columns=[f"x_{index}" for index in range(X_observed.shape[1])],
        )
        candidate_frame = pd.DataFrame(
            np.asarray(X_candidates, dtype=float),
            columns=observed_frame.columns,
        )

        controller = BGOsampling.Bgolearn()
        result = controller.fit(
            data_matrix=observed_frame,
            Measured_response=np.asarray(y_observed, dtype=float).reshape(-1),
            virtual_samples=candidate_frame,
            Mission="Regression",
            Kriging_model=ReactionEnsembleAdapter,
            min_search=False,
            opt_num=1,
            Normalize=False,
            Dynamic_W=False,
            CV_test=False,
            seed=int(self.seed),
        )

        posterior = state.get("posterior")
        if not isinstance(posterior, Posterior):
            # Defensive fallback for future Bgolearn versions that change the
            # custom-model invocation order.
            posterior = shared_ensemble.predict(np.asarray(X_candidates, dtype=float))

        # Verify that stock Bgolearn did not alter the custom predictions.
        returned_mean = np.asarray(result.virtual_samples_mean, dtype=float).reshape(-1)
        returned_std = np.asarray(result.virtual_samples_std, dtype=float).reshape(-1)
        if not np.allclose(returned_mean, posterior.mean, rtol=1e-7, atol=1e-9):
            raise RuntimeError("Bgolearn changed the custom surrogate mean predictions.")
        if not np.allclose(returned_std, posterior.std, rtol=1e-7, atol=1e-9):
            raise RuntimeError("Bgolearn changed the custom surrogate uncertainties.")
        return posterior
