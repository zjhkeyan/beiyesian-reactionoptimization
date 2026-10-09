from __future__ import annotations

import numpy as np

from .acquisition import stable_log_expected_improvement, upper_confidence_bound, zscore
from .config import PolicyConfig
from .schemas import Posterior, Recommendation
from .utils import deterministic_argmax


class MetricAwarePolicy:
    """Stage-dependent single-query policy aligned with benchmark metrics."""

    def __init__(self, config: PolicyConfig, *, seed: int) -> None:
        self.config = config
        self.seed = int(seed)

    def _xi(self, step: int, budget: int) -> float:
        if budget <= 1:
            return float(self.config.xi_final)
        progress = (step - 1) / (budget - 1)
        return float(
            self.config.xi_initial
            + progress * (self.config.xi_final - self.config.xi_initial)
        )

    def _prior_weight(self, step: int) -> float:
        return float(
            self.config.prior_initial_weight
            * np.exp(-self.config.prior_decay * max(step - 1, 0))
        )

    def _with_prior(
        self,
        scores: np.ndarray,
        prior_scores: np.ndarray | None,
        step: int,
    ) -> tuple[np.ndarray, float]:
        weight = self._prior_weight(step)
        if prior_scores is None or weight <= 0.0:
            return np.asarray(scores, dtype=float), 0.0
        prior = np.asarray(prior_scores, dtype=float).reshape(-1)
        if prior.shape != np.asarray(scores).shape:
            raise ValueError("prior_scores must align with the current candidate pool.")
        return zscore(scores) + weight * zscore(prior), weight

    def select(
        self,
        posterior: Posterior,
        candidate_ids: np.ndarray,
        *,
        best_f: float,
        step: int,
        budget: int,
        prior_scores: np.ndarray | None = None,
        novelty_scores: np.ndarray | None = None,
    ) -> Recommendation:
        posterior.validate(len(candidate_ids))
        candidate_ids = np.asarray(candidate_ids, dtype=int)
        disagreement = (
            np.zeros_like(posterior.mean)
            if posterior.disagreement is None
            else posterior.disagreement
        )
        novelty_mode = self.config.novelty_disagreement_mode.strip().lower()
        if novelty_mode not in {"off", "pure", "hybrid"}:
            raise ValueError(
                "novelty_disagreement_mode must be 'off', 'pure', or 'hybrid'."
            )
        novelty_disagreement: np.ndarray | None = None
        if novelty_mode != "off":
            if novelty_scores is None:
                raise ValueError(
                    "novelty_scores are required when novelty-disagreement acquisition is enabled."
                )
            novelty = np.asarray(novelty_scores, dtype=float).reshape(-1)
            if novelty.shape != posterior.mean.shape:
                raise ValueError("novelty_scores must align with the current candidate pool.")
            if not np.all(np.isfinite(novelty)) or np.any(novelty < 0.0):
                raise ValueError("novelty_scores must be finite and non-negative.")
            novelty_disagreement = novelty * np.maximum(disagreement, 0.0)

        if novelty_mode == "pure":
            raw_scores = novelty_disagreement
            acquisition_name = "strain_sensor_novelty_disagreement"
        elif step == 1:
            first_query = self.config.first_query.strip().lower()
            if first_query == "posterior_mean":
                raw_scores = posterior.mean.copy()
                acquisition_name = "posterior_mean"
            elif first_query in posterior.component_means:
                raw_scores = np.asarray(
                    posterior.component_means[first_query], dtype=float
                ).copy()
                acquisition_name = f"{first_query}_posterior_mean"
            else:
                raise ValueError(
                    "first_query must be 'posterior_mean' or the name of an enabled model."
                )
        elif step >= self.config.final_start:
            raw_scores = upper_confidence_bound(
                posterior.mean,
                posterior.std,
                beta=self.config.final_beta,
            )
            acquisition_name = "late_mean_ucb"
        else:
            raw_scores = stable_log_expected_improvement(
                posterior.mean,
                posterior.std,
                best_f,
                xi=self._xi(step, budget),
            )
            acquisition_name = "stable_log_ei"

            challenge_round = (
                step > self.config.early_end
                and step < self.config.final_start
                and self.config.challenge_every > 0
                and (step - self.config.early_end) % self.config.challenge_every == 0
            )
            if challenge_round:
                base_index = deterministic_argmax(
                    raw_scores,
                    candidate_ids,
                    seed=self.seed + 1000 + step,
                )
                challenge_scores = upper_confidence_bound(
                    posterior.mean,
                    posterior.std,
                    beta=self.config.challenge_beta,
                ) + self.config.challenge_disagreement_weight * zscore(disagreement)
                challenger = deterministic_argmax(
                    challenge_scores,
                    candidate_ids,
                    seed=self.seed + 2000 + step,
                )
                base_lower_bound = (
                    posterior.mean[base_index]
                    - self.config.challenge_gate_sigma * posterior.std[base_index]
                )
                challenger_upper_bound = (
                    posterior.mean[challenger]
                    + self.config.challenge_beta * posterior.std[challenger]
                )
                if challenger_upper_bound >= base_lower_bound:
                    raw_scores = challenge_scores
                    acquisition_name = "gated_model_challenge"

        if novelty_mode == "hybrid" and step > 1:
            weight = float(self.config.novelty_disagreement_weight)
            if weight < 0.0:
                raise ValueError("novelty_disagreement_weight must be non-negative.")
            raw_scores = zscore(raw_scores) + weight * zscore(novelty_disagreement)
            acquisition_name = f"{acquisition_name}_plus_novelty_disagreement"

        combined_scores, prior_weight = self._with_prior(
            raw_scores, prior_scores, step
        )
        local_index = deterministic_argmax(
            combined_scores,
            candidate_ids,
            seed=self.seed + step,
        )

        component_means = {
            name: float(values[local_index])
            for name, values in posterior.component_means.items()
        }
        component_stds = {
            name: float(values[local_index])
            for name, values in posterior.component_stds.items()
        }
        return Recommendation(
            local_index=local_index,
            query_index=int(candidate_ids[local_index]),
            acquisition_name=acquisition_name,
            acquisition_value=float(combined_scores[local_index]),
            predicted_mean=float(posterior.mean[local_index]),
            predicted_std=float(posterior.std[local_index]),
            component_means=component_means,
            component_stds=component_stds,
            model_weights=dict(posterior.weights),
            prior_weight=prior_weight,
        )
