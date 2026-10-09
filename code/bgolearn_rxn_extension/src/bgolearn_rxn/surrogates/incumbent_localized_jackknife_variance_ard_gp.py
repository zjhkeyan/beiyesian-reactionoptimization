from __future__ import annotations

from typing import Any

import numpy as np

from ..config import GPConfig
from ..schemas import FeatureSchema, Posterior
from .full_mean_jackknife_variance_ard_gp import (
    FullMeanJackknifeVarianceARDGP,
)


class IncumbentLocalizedJackknifeVarianceARDGP(
    FullMeanJackknifeVarianceARDGP
):
    """Full component-ARD posterior with basin-local jackknife inflation.

    Query one is exactly the full-data component ARD model.  Later queries use
    delete-fold disagreement only in the high-yield basin identified by the
    current incumbent and the full-data posterior.  The full-data mean and
    variance remain authoritative outside that support.
    """

    name = "incumbent_localized_jackknife_variance_ard_gp"

    def __init__(
        self, schema: FeatureSchema, config: GPConfig, *, seed: int = 0
    ) -> None:
        super().__init__(schema=schema, config=config, seed=seed)
        self.incumbent_: float | None = None
        self.support_mask_: np.ndarray | None = None
        self.support_size_: int = 0
        self.support_fraction_: float = 0.0
        self.support_reference_: float | None = None
        self.support_threshold_: float | None = None
        self.support_scale_: float | None = None
        self.localized_correction_: np.ndarray | None = None
        self.localized_variance_: np.ndarray | None = None
        self.localization_diagnostics_: dict[str, float | int | bool] = {}

    def _refresh_audit(self) -> None:
        super()._refresh_audit()
        self.audit_state_.update(
            {
                "incumbent": self.incumbent_,
                "support_size": self.support_size_,
                "support_fraction": self.support_fraction_,
                "support_reference": self.support_reference_,
                "support_threshold": self.support_threshold_,
                "support_scale": self.support_scale_,
                "localization": dict(self.localization_diagnostics_),
            }
        )

    def fit(
        self, X: np.ndarray, y: np.ndarray
    ) -> "IncumbentLocalizedJackknifeVarianceARDGP":
        targets = np.asarray(y, dtype=float).reshape(-1)
        if len(targets) == 0 or not np.isfinite(targets).all():
            raise ValueError("Localized jackknife targets must be finite and nonempty.")
        self.incumbent_ = float(np.max(targets))
        super().fit(X, targets)
        return self

    def _high_yield_support(self, full: Posterior) -> np.ndarray:
        means = np.asarray(full.mean, dtype=float).reshape(-1)
        stds = np.asarray(full.std, dtype=float).reshape(-1)
        if len(means) == 0:
            raise ValueError("Localized jackknife prediction requires at least one row.")
        if self.incumbent_ is None:
            raise RuntimeError("Incumbent is unavailable before fitting.")
        if not np.isfinite(means).all() or not np.isfinite(stds).all():
            raise FloatingPointError("Full posterior contains non-finite values.")

        # If the observed incumbent is outside the model's predicted range,
        # anchor the basin at the posterior peak instead of returning an empty
        # support.  A median posterior standard deviation gives a deterministic
        # scale that adapts to the current fit and prediction pool.
        posterior_peak = float(np.max(means))
        reference = min(self.incumbent_, posterior_peak)
        scale = max(float(np.median(stds)), 1e-12)
        threshold = reference - scale
        support = means >= threshold
        if not np.any(support):
            support[int(np.argmax(means))] = True

        self.support_mask_ = support
        self.support_size_ = int(np.sum(support))
        self.support_fraction_ = float(np.mean(support))
        self.support_reference_ = reference
        self.support_threshold_ = threshold
        self.support_scale_ = scale
        return support

    def _localized_disagreement(
        self,
        X: np.ndarray,
        full: Posterior,
        *,
        full_covariance: bool,
    ) -> tuple[np.ndarray | None, np.ndarray]:
        raw_correction, raw_variance = self._delete_disagreement(
            X, full_covariance=full_covariance
        )
        support = self._high_yield_support(full)
        localized_variance = raw_variance * support.astype(float)
        localized_correction = None
        if raw_correction is not None:
            localized_correction = raw_correction * support[:, None] * support[None, :]
            localized_correction = 0.5 * (
                localized_correction + localized_correction.T
            )

        diagnostics: dict[str, float | int | bool] = {
            "finite": bool(
                np.isfinite(localized_variance).all()
                and (
                    localized_correction is None
                    or np.isfinite(localized_correction).all()
                )
            ),
            "support_size": self.support_size_,
            "support_fraction": self.support_fraction_,
            "raw_correction_trace": float(np.sum(raw_variance)),
            "localized_correction_trace": float(np.sum(localized_variance)),
            "outside_correction_trace": float(
                np.sum(raw_variance[~support])
            ),
            "noncollapsed": bool(np.any(localized_variance > 0.0)),
        }
        if not diagnostics["finite"]:
            raise FloatingPointError("Localized disagreement contains non-finite values.")

        self.localized_correction_ = localized_correction
        self.localized_variance_ = localized_variance
        self.localization_diagnostics_ = diagnostics
        self._refresh_audit()
        return localized_correction, localized_variance

    def predict(self, X: np.ndarray) -> Posterior:
        self._require_fit()
        full = self.full_data_model.predict(X)
        if self.fit_calls_ == 1:
            return full

        _, localized_variance = self._localized_disagreement(
            X, full, full_covariance=False
        )
        full_variance = full.std**2
        total_variance = np.maximum(full_variance + localized_variance, 1e-12)
        self.full_variance_ = full_variance
        self.total_variance_ = total_variance
        posterior = Posterior(
            mean=full.mean,
            std=np.sqrt(total_variance),
            disagreement=np.sqrt(localized_variance),
        )
        posterior.validate(len(X))
        return posterior

    def posterior_covariance(self, X: np.ndarray) -> np.ndarray:
        self._require_fit()
        full_covariance = self.full_data_model.posterior_covariance(X)
        if self.fit_calls_ == 1:
            return full_covariance

        full = self.full_data_model.predict(X)
        correction, localized_variance = self._localized_disagreement(
            X, full, full_covariance=True
        )
        if correction is None:
            raise RuntimeError("Localized disagreement covariance was not constructed.")
        covariance = full_covariance + correction
        covariance = 0.5 * (covariance + covariance.T)
        total_variance = np.maximum(full.std**2 + localized_variance, 1e-12)
        covariance.flat[:: len(covariance) + 1] = total_variance
        self.full_variance_ = full.std**2
        self.total_variance_ = total_variance
        return covariance
