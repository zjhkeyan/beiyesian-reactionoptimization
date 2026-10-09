"""Reaction-condition optimization extensions for Bgolearn.

The package separates four concerns:

1. immutable, label-free reaction featurization;
2. probabilistic structured reaction surrogates;
3. metric-aware finite-pool acquisition and campaign control;
4. an optional compatibility bridge to stock Bgolearn 3.1.0.
"""

from .acquisition import stable_log_expected_improvement
from .campaign import CampaignResult, CampaignRunner, CampaignSession
from .config import RunConfig, load_run_config
from .datasets import DatasetSpec, ReactionDataset, load_reaction_dataset
from .features import FrozenReactionEncoder
from .oracle import OfflineTableOracle, YieldOracle
from .policy import MetricAwarePolicy

__all__ = [
    "CampaignResult",
    "CampaignRunner",
    "CampaignSession",
    "DatasetSpec",
    "FrozenReactionEncoder",
    "MetricAwarePolicy",
    "OfflineTableOracle",
    "ReactionDataset",
    "RunConfig",
    "YieldOracle",
    "load_reaction_dataset",
    "load_run_config",
    "stable_log_expected_improvement",
]

__version__ = "0.1.0"
