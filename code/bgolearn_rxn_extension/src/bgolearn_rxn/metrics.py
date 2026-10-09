from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
from scipy.stats import t


@dataclass(frozen=True)
class CampaignMetrics:
    initial_round_found_best: float
    best_found: float
    auc_best_so_far: float
    round_to_95_global_best: int | None
    global_best: float | None
    final_simple_regret: float | None
    n_queries: int

    def to_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


def best_so_far_curve(observed_yields: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(observed_yields), dtype=float)
    if values.size == 0:
        raise ValueError("At least one queried yield is required.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Observed yields contain non-finite values.")
    return np.maximum.accumulate(values)


def compute_campaign_metrics(
    observed_yields: Iterable[float],
    *,
    global_best: float | None = None,
    target_fraction: float = 0.95,
) -> CampaignMetrics:
    values = np.asarray(list(observed_yields), dtype=float)
    curve = best_so_far_curve(values)

    t95: int | None = None
    regret: float | None = None
    if global_best is not None:
        global_best = float(global_best)
        threshold = float(target_fraction) * global_best
        hits = np.flatnonzero(curve >= threshold)
        t95 = int(hits[0] + 1) if hits.size else None
        regret = float(max(global_best - curve[-1], 0.0))

    return CampaignMetrics(
        initial_round_found_best=float(curve[0]),
        best_found=float(curve[-1]),
        auc_best_so_far=float(np.mean(curve)),
        round_to_95_global_best=t95,
        global_best=global_best,
        final_simple_regret=regret,
        n_queries=int(values.size),
    )


def summarize_metric(values: Iterable[float], confidence: float = 0.95) -> dict[str, float]:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        raise ValueError("Cannot summarize an empty metric sequence.")
    mean = float(np.mean(array))
    if array.size == 1:
        return {"mean": mean, "std": 0.0, "ci_low": mean, "ci_high": mean}
    std = float(np.std(array, ddof=1))
    sem = std / np.sqrt(array.size)
    critical = float(t.ppf(0.5 + confidence / 2.0, df=array.size - 1))
    return {
        "mean": mean,
        "std": std,
        "ci_low": mean - critical * sem,
        "ci_high": mean + critical * sem,
    }
