from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from .backend import BgolearnCompatibilityBackend, NativeBackend, PredictionBackend
from .config import RunConfig
from .datasets import ReactionDataset, summarize_topology
from .factory import PreparedProblem, build_ensemble, prepare_problem
from .metrics import CampaignMetrics, compute_campaign_metrics
from .oracle import OfflineTableOracle, YieldOracle
from .policy import MetricAwarePolicy
from .schemas import Posterior, Recommendation
from .utils import ensure_directory, seed_everything, stable_hash


@dataclass
class CampaignResult:
    dataset: str
    seed: int
    backend: str
    trajectory: list[dict[str, Any]]
    metrics: CampaignMetrics
    manifest: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "dataset": self.dataset,
            "backend": self.backend,
            "trajectory": self.trajectory,
            "metrics": self.metrics.to_dict(),
            "manifest": self.manifest,
        }


class CampaignRunner:
    """Leakage-safe, single-query reaction optimization campaign.

    A runner owns one seed and one independent 40-step campaign. It freezes the
    encoder before the first query, keeps the ensemble's prequential weights
    stateful, removes candidates by immutable ID, and exposes organizer labels
    only through a ``YieldOracle`` after a recommendation has been committed.
    """

    def __init__(
        self,
        dataset: ReactionDataset,
        config: RunConfig,
        *,
        seed: int,
        backend: str | None = None,
        prior_scores: np.ndarray | None = None,
        prepared: PreparedProblem | None = None,
    ) -> None:
        self.dataset = dataset
        self.config = config
        self.seed = int(seed)
        if self.seed not in config.dataset.seeds:
            # Nonstandard seeds are useful for development, but make the
            # deviation visible rather than silently claiming a competition run.
            self.nonstandard_seed = True
        else:
            self.nonstandard_seed = False
        seed_everything(self.seed)

        self.prepared = prepared or prepare_problem(dataset, config)
        self.ensemble = build_ensemble(config, self.prepared, seed=self.seed)
        self.policy = MetricAwarePolicy(config.policy, seed=self.seed)

        backend_name = (backend or config.runtime.backend).strip().lower()
        if backend_name == "native":
            self.backend: PredictionBackend = NativeBackend(self.ensemble)
        elif backend_name == "bgolearn":
            self.backend = BgolearnCompatibilityBackend(
                self.ensemble,
                seed=self.seed,
            )
        else:
            raise ValueError(f"Unknown backend: {backend_name!r}")

        if prior_scores is not None:
            prior = np.asarray(prior_scores, dtype=float).reshape(-1)
            if prior.shape != (len(dataset.candidates),):
                raise ValueError(
                    "prior_scores must align with the full public candidate pool."
                )
            if not np.all(np.isfinite(prior)):
                raise ValueError("prior_scores contain non-finite values.")
            self.prior_scores = prior
        else:
            self.prior_scores = None

    def _manifest(self) -> dict[str, Any]:
        schema = self.prepared.encoder.schema
        return {
            "config_hash": stable_hash(self.config),
            "feature_schema_hash": stable_hash(schema.to_dict() if schema else {}),
            "feature_schema": schema.to_dict() if schema else None,
            "topology": summarize_topology(self.dataset),
            "nonstandard_seed": self.nonstandard_seed,
            "participant_side_only": True,
            "preprocessing_frozen": True,
            "dynamic_weighting": False,
            "bgolearn_internal_normalization": False,
        }

    def _record(
        self,
        step: int,
        recommendation: Recommendation,
        selected_row: pd.Series,
        observed_yield: float,
        best_before: float,
        updated_weights: dict[str, float],
    ) -> dict[str, Any]:
        condition = {
            column: selected_row[column]
            for column in self.dataset.spec.feature_columns
        }
        return {
            "step": int(step),
            "query_index": int(recommendation.query_index),
            "condition": condition,
            "observed_yield": float(observed_yield),
            "actual_yield": float(observed_yield),
            "predicted_yield": float(recommendation.predicted_mean),
            "predicted_std": float(recommendation.predicted_std),
            "target_incumbent_before_query": float(best_before),
            "acquisition_name": recommendation.acquisition_name,
            "acquisition_value": float(recommendation.acquisition_value),
            "component_predictions": {
                name: {
                    "mean": float(recommendation.component_means[name]),
                    "std": float(recommendation.component_stds[name]),
                }
                for name in recommendation.component_means
            },
            "model_weights_before_query": dict(recommendation.model_weights),
            "model_weights_after_query": dict(updated_weights),
            "prior_weight": float(recommendation.prior_weight),
            "seed": self.seed,
        }

    def recommend_once(
        self,
        observed: pd.DataFrame,
        remaining_candidates: pd.DataFrame,
        *,
        remaining_X: np.ndarray,
        step: int,
        prior_scores: np.ndarray | None = None,
    ) -> tuple[Recommendation, Posterior]:
        """Fit and recommend one point without querying an oracle.

        This method is the integration point for a real laboratory agent. The
        caller commits the returned immutable ``query_index``, obtains the
        physical measurement, then updates the observed table before the next
        call. For an offline benchmark, use :meth:`run` to guarantee the
        prequential weight update is performed correctly.
        """

        X_observed = self.prepared.encoder.transform(
            observed.loc[:, list(self.dataset.spec.feature_columns)]
        )
        y_observed = observed[self.dataset.spec.target_column].to_numpy(dtype=float)
        posterior = self.backend.fit_predict(X_observed, y_observed, remaining_X)
        best_f = self.dataset.target_incumbent(observed)
        recommendation = self.policy.select(
            posterior,
            remaining_candidates["query_index"].to_numpy(dtype=int),
            best_f=best_f,
            step=step,
            budget=self.config.dataset.budget,
            prior_scores=prior_scores,
            novelty_scores=np.min(cdist(remaining_X, X_observed), axis=1),
        )
        return recommendation, posterior

    def start_session(
        self,
        *,
        budget: int | None = None,
        score_snapshot_dir: str | Path | None = None,
    ) -> "CampaignSession":
        """Create a stateful session for a laboratory or an offline oracle."""

        return CampaignSession(
            self,
            budget=budget,
            score_snapshot_dir=score_snapshot_dir,
        )

    def run(
        self,
        oracle: YieldOracle,
        *,
        budget: int | None = None,
        score_snapshot_dir: str | Path | None = None,
    ) -> CampaignResult:
        session = self.start_session(
            budget=budget,
            score_snapshot_dir=score_snapshot_dir,
        )
        while not session.is_complete:
            recommendation = session.recommend()
            observed_yield = float(oracle.query(recommendation.query_index))
            session.observe(recommendation.query_index, observed_yield)

        global_best = (
            oracle.global_best if isinstance(oracle, OfflineTableOracle) else None
        )
        return session.finalize(global_best=global_best)


class CampaignSession:
    """Stateful ask/tell session for one independent optimization campaign.

    ``recommend`` commits one candidate in memory and refuses a second ask
    until ``observe`` supplies the matching measurement. This prevents an
    asynchronous integration from accidentally changing the recommendation
    after seeing the result or from observing a different candidate ID.
    """

    def __init__(
        self,
        runner: CampaignRunner,
        *,
        budget: int | None = None,
        score_snapshot_dir: str | Path | None = None,
    ) -> None:
        self.runner = runner
        self.budget = int(budget or runner.config.dataset.budget)
        if self.budget <= 0:
            raise ValueError("budget must be positive.")
        if self.budget > len(runner.dataset.candidates):
            raise ValueError("budget exceeds the candidate pool size.")

        self.observed = runner.dataset.initial.copy()
        self.X_observed = runner.prepared.X_initial.copy()
        self.y_observed = self.observed[
            runner.dataset.spec.target_column
        ].to_numpy(dtype=float)
        self.remaining_rows = np.arange(len(runner.dataset.candidates), dtype=int)
        self.trajectory: list[dict[str, Any]] = []
        self.snapshot_dir = (
            ensure_directory(score_snapshot_dir)
            if score_snapshot_dir is not None
            else None
        )
        self._pending: dict[str, Any] | None = None

    @property
    def step(self) -> int:
        return len(self.trajectory) + 1

    @property
    def is_complete(self) -> bool:
        return len(self.trajectory) >= self.budget

    @property
    def remaining_candidates(self) -> pd.DataFrame:
        return self.runner.dataset.candidates.iloc[self.remaining_rows].copy()

    def recommend(self) -> Recommendation:
        if self.is_complete:
            raise RuntimeError("The campaign budget has been exhausted.")
        if self._pending is not None:
            pending = self._pending["recommendation"]
            raise RuntimeError(
                "A recommendation is already pending observation: "
                f"query_index={pending.query_index}."
            )

        remaining_frame = self.runner.dataset.candidates.iloc[self.remaining_rows]
        X_remaining = self.runner.prepared.X_candidates[self.remaining_rows]
        posterior = self.runner.backend.fit_predict(
            self.X_observed, self.y_observed, X_remaining
        )
        best_before = self.runner.dataset.target_incumbent(self.observed)
        current_prior = (
            None
            if self.runner.prior_scores is None
            else self.runner.prior_scores[self.remaining_rows]
        )
        recommendation = self.runner.policy.select(
            posterior,
            remaining_frame["query_index"].to_numpy(dtype=int),
            best_f=best_before,
            step=self.step,
            budget=self.budget,
            prior_scores=current_prior,
            novelty_scores=np.min(cdist(X_remaining, self.X_observed), axis=1),
        )

        selected_local = int(recommendation.local_index)
        selected_global_row = int(self.remaining_rows[selected_local])
        selected_row = self.runner.dataset.candidates.iloc[selected_global_row]
        if int(selected_row["query_index"]) != recommendation.query_index:
            raise RuntimeError("Candidate ID changed between scoring and commitment.")

        if self.snapshot_dir is not None:
            np.savez_compressed(
                self.snapshot_dir / f"step_{self.step:02d}.npz",
                query_index=remaining_frame["query_index"].to_numpy(dtype=int),
                posterior_mean=posterior.mean,
                posterior_std=posterior.std,
                disagreement=(
                    np.zeros(len(posterior.mean), dtype=float)
                    if posterior.disagreement is None
                    else posterior.disagreement
                ),
            )

        self._pending = {
            "recommendation": recommendation,
            "selected_local": selected_local,
            "selected_global_row": selected_global_row,
            "selected_row": selected_row,
            "best_before": best_before,
        }
        return recommendation

    def observe(self, query_index: int, observed_yield: float) -> dict[str, Any]:
        if self._pending is None:
            raise RuntimeError("No recommendation is pending observation.")
        recommendation: Recommendation = self._pending["recommendation"]
        if int(query_index) != recommendation.query_index:
            raise ValueError(
                f"Observed ID {query_index} does not match committed ID "
                f"{recommendation.query_index}."
            )
        observed_yield = float(observed_yield)
        if not np.isfinite(observed_yield) or not 0.0 <= observed_yield <= 100.0:
            raise ValueError("Observed yield must be finite and lie within [0, 100].")

        updated_weights = self.runner.ensemble.update_prequential(
            observed_y=observed_yield,
            component_means=recommendation.component_means,
            component_stds=recommendation.component_stds,
        )
        selected_row: pd.Series = self._pending["selected_row"]
        record = self.runner._record(
            self.step,
            recommendation,
            selected_row,
            observed_yield,
            float(self._pending["best_before"]),
            updated_weights,
        )
        self.trajectory.append(record)

        new_observation = selected_row.loc[
            list(self.runner.dataset.spec.feature_columns)
        ].to_dict()
        new_observation[self.runner.dataset.spec.target_column] = observed_yield
        self.observed = pd.concat(
            [self.observed, pd.DataFrame([new_observation])],
            ignore_index=True,
        )
        selected_global_row = int(self._pending["selected_global_row"])
        self.X_observed = np.vstack(
            [
                self.X_observed,
                self.runner.prepared.X_candidates[selected_global_row],
            ]
        )
        self.y_observed = np.append(self.y_observed, observed_yield)
        selected_local = int(self._pending["selected_local"])
        self.remaining_rows = np.delete(self.remaining_rows, selected_local)
        self._pending = None
        return record

    def finalize(self, *, global_best: float | None = None) -> CampaignResult:
        if self._pending is not None:
            raise RuntimeError("Cannot finalize while an observation is pending.")
        metrics = compute_campaign_metrics(
            [record["observed_yield"] for record in self.trajectory],
            global_best=global_best,
        )
        manifest = self.runner._manifest()
        manifest["final_ensemble_state"] = self.runner.ensemble.state_dict()
        manifest["budget"] = self.budget
        manifest["completed"] = self.is_complete
        return CampaignResult(
            dataset=self.runner.dataset.spec.name,
            seed=self.runner.seed,
            backend=self.runner.backend.name,
            trajectory=list(self.trajectory),
            metrics=metrics,
            manifest=manifest,
        )
