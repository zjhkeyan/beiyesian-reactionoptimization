from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import RunConfig
from .datasets import ReactionDataset
from .features import FrozenReactionEncoder
from .surrogates.anova_gp import StructuredANOVAGP
from .surrogates.bayesian_cp_tensor import BayesianCPTensorSurrogate
from .surrogates.component_ard_gp import (
    ComponentARDLowRankTaskGP,
    ComponentARDStructuredGP,
    TaskNoiseComponentARDLowRankTaskGP,
)
from .surrogates.credible_support_maximin_ard import CredibleSupportMaximinARD
from .surrogates.delete_fold_component_ard_gp import DeleteFoldComponentARDEnsemble
from .surrogates.early_delete_fold_then_full_ard_gp import (
    EarlyDeleteFoldThenFullARD,
)
from .surrogates.ensemble import PrequentialEnsemble
from .surrogates.graph_diffusion_gp import GraphDiffusionProductGP
from .surrogates.full_mean_jackknife_variance_ard_gp import (
    FullMeanJackknifeVarianceARDGP,
)
from .surrogates.incumbent_localized_jackknife_variance_ard_gp import (
    IncumbentLocalizedJackknifeVarianceARDGP,
)
from .surrogates.interaction import BayesianLowRankInteractionSurrogate
from .surrogates.low_rank_task_gp import LowRankTaskCovarianceGP
from .surrogates.local_laplace_component_ard_gp import (
    LocalLaplaceComponentARDStructuredGP,
)
from .surrogates.pair_context_gp import PairContextCoregionalizedGP
from .surrogates.posterior_mode_component_ard_gp import (
    PosteriorModeComponentARDStructuredGP,
)
from .surrogates.rank_gaussian_component_ard_gp import (
    RankGaussianComponentARDStructuredGP,
)
from .surrogates.query_one_rank_consensus_ard import (
    COMPONENT_ARD,
    INTERACTION,
    TREES,
    QueryOneRankConsensusARD,
)
from .surrogates.residual_transfer import ResidualTransferGP
from .surrogates.source_expert import SourceExpertTransferGP
from .surrogates.stage_split_delete_fold_ard import StageSplitDeleteFoldARD
from .surrogates.strong_heredity_component_ard_gp import (
    StrongHeredityComponentARDStructuredGP,
)
from .surrogates.surprise_gated_rank_consensus_ard import (
    SurpriseGatedRankConsensusARD,
)
from .surrogates.trees import BootstrapExtraTreesSurrogate


@dataclass(frozen=True)
class PreparedProblem:
    """Frozen participant-side representation of one benchmark task."""

    encoder: FrozenReactionEncoder
    X_initial: np.ndarray
    X_candidates: np.ndarray
    reference_X: np.ndarray


def prepare_problem(dataset: ReactionDataset, config: RunConfig) -> PreparedProblem:
    """Fit the label-free encoder once on initial + unlabeled candidate features.

    Fitting on the unlabeled pool is transductive but label-free and therefore
    allowed by the benchmark data contract. The fitted encoder must remain
    frozen for all 40 rounds.
    """

    feature_columns = list(dataset.spec.feature_columns)
    reference_frame = pd.concat(
        [
            dataset.initial.loc[:, feature_columns],
            dataset.candidates.loc[:, feature_columns],
        ],
        axis=0,
        ignore_index=True,
    )
    encoder = FrozenReactionEncoder(dataset.spec, config.features).fit(reference_frame)
    X_initial = encoder.transform(dataset.initial.loc[:, feature_columns])
    X_candidates = encoder.transform(dataset.candidates.loc[:, feature_columns])
    reference_X = encoder.transform(reference_frame)
    return PreparedProblem(
        encoder=encoder,
        X_initial=X_initial,
        X_candidates=X_candidates,
        reference_X=reference_X,
    )


def build_ensemble(
    config: RunConfig,
    prepared: PreparedProblem,
    *,
    seed: int,
) -> PrequentialEnsemble:
    """Construct the recommended reaction-specific model ensemble."""

    schema = prepared.encoder.schema
    if schema is None:
        raise RuntimeError("The reaction encoder has not been fitted.")

    enabled = tuple(config.ensemble.enabled_models)
    component_ard = ComponentARDStructuredGP(
        schema=schema, config=config.gp, seed=seed + 17
    )
    interaction = BayesianLowRankInteractionSurrogate(
        schema=schema,
        config=config.interaction_model,
        reference_X=prepared.reference_X,
        seed=seed + 23,
    )
    trees = BootstrapExtraTreesSurrogate(config=config.trees, seed=seed + 37)
    available = OrderedDict(
        [
            (
                "gp",
                StructuredANOVAGP(schema=schema, config=config.gp, seed=seed + 11),
            ),
            (
                "component_ard_gp",
                component_ard,
            ),
            (
                "posterior_mode_component_ard_gp",
                PosteriorModeComponentARDStructuredGP(
                    schema=schema, config=config.gp, seed=seed + 17
                ),
            ),
            (
                "local_laplace_component_ard_gp",
                LocalLaplaceComponentARDStructuredGP(
                    schema=schema, config=config.gp, seed=seed + 17
                ),
            ),
            (
                "strong_heredity_component_ard_gp",
                StrongHeredityComponentARDStructuredGP(
                    schema=schema, config=config.gp, seed=seed + 19
                ),
            ),
            (
                "rank_gaussian_component_ard_gp",
                RankGaussianComponentARDStructuredGP(
                    schema=schema, config=config.gp, seed=seed + 21
                ),
            ),
            (
                "graph_diffusion_product_gp",
                GraphDiffusionProductGP(
                    schema=schema, config=config.gp, seed=seed + 29
                ),
            ),
            (
                "delete_fold_component_ard_gp",
                DeleteFoldComponentARDEnsemble(
                    schema=schema, config=config.gp, seed=seed + 33
                ),
            ),
            (
                "full_mean_jackknife_variance_ard_gp",
                FullMeanJackknifeVarianceARDGP(
                    schema=schema, config=config.gp, seed=seed
                ),
            ),
            (
                "incumbent_localized_jackknife_variance_ard_gp",
                IncumbentLocalizedJackknifeVarianceARDGP(
                    schema=schema, config=config.gp, seed=seed
                ),
            ),
            (
                "stage_split_delete_fold_ard_gp",
                StageSplitDeleteFoldARD(
                    schema=schema, config=config.gp, seed=seed
                ),
            ),
            (
                "early_delete_fold_then_full_ard_gp",
                EarlyDeleteFoldThenFullARD(
                    schema=schema, config=config.gp, seed=seed
                ),
            ),
            (
                "interaction",
                interaction,
            ),
            (
                "trees",
                trees,
            ),
            (
                "query_one_rank_consensus_ard_gp",
                QueryOneRankConsensusARD(
                    {
                        COMPONENT_ARD: component_ard,
                        INTERACTION: interaction,
                        TREES: trees,
                    }
                ),
            ),
            (
                "credible_support_maximin_ard_gp",
                CredibleSupportMaximinARD(
                    {
                        COMPONENT_ARD: component_ard,
                        INTERACTION: interaction,
                        TREES: trees,
                    }
                ),
            ),
            (
                "surprise_gated_rank_consensus_ard_gp",
                SurpriseGatedRankConsensusARD(
                    {
                        COMPONENT_ARD: component_ard,
                        INTERACTION: interaction,
                        TREES: trees,
                    }
                ),
            ),
        ]
    )
    if "bayesian_cp_tensor" in enabled:
        available["bayesian_cp_tensor"] = BayesianCPTensorSurrogate(
            schema=schema,
            config=config.interaction_model,
            seed=seed + 31,
        )
    if "pair_context_coregionalized_gp" in enabled:
        available["pair_context_coregionalized_gp"] = PairContextCoregionalizedGP(
            schema=schema, config=config.gp, seed=seed + 19
        )
    task_transfer_models = {
        "low_rank_task_gp",
        "component_ard_low_rank_task_gp",
        "component_ard_low_rank_task_noise_gp",
        "residual_transfer",
        "source_expert",
    }.intersection(enabled)
    if task_transfer_models:
        task = schema.task_role
        if task is None:
            raise ValueError("Task-transfer models require dataset.task_column.")
        if len(prepared.X_candidates) == 0:
            raise ValueError("Task-transfer models require a nonempty candidate pool.")
        task_block = np.asarray(
            prepared.X_candidates[:, schema.identity_slice(task)], dtype=float
        )
        target_identity = task_block[0].copy()
        if not np.all(np.isclose(task_block, target_identity[None, :], atol=1e-12)):
            raise ValueError(
                "Task-transfer models require all candidates to share one target task identity."
            )
    if "residual_transfer" in enabled:
        available["residual_transfer"] = ResidualTransferGP(
            schema=schema,
            config=config.gp,
            target_task_identity=target_identity,
            seed=seed + 49,
        )
    if "source_expert" in enabled:
        available["source_expert"] = SourceExpertTransferGP(
            schema=schema,
            config=config.gp,
            target_task_identity=target_identity,
            seed=seed + 61,
        )
    if "low_rank_task_gp" in enabled:
        available["low_rank_task_gp"] = LowRankTaskCovarianceGP(
            schema=schema,
            config=config.gp,
            seed=seed + 73,
        )
    if "component_ard_low_rank_task_gp" in enabled:
        available["component_ard_low_rank_task_gp"] = ComponentARDLowRankTaskGP(
            schema=schema,
            config=config.gp,
            seed=seed + 79,
        )
    if "component_ard_low_rank_task_noise_gp" in enabled:
        available["component_ard_low_rank_task_noise_gp"] = (
            TaskNoiseComponentARDLowRankTaskGP(
                schema=schema,
                config=config.gp,
                seed=seed + 83,
            )
        )
    unknown = sorted(set(enabled) - set(available))
    if unknown:
        raise ValueError(f"Unknown enabled ensemble models: {unknown}")
    if not enabled:
        raise ValueError("At least one ensemble model must be enabled.")
    if len(enabled) != len(set(enabled)):
        raise ValueError("enabled_models must not contain duplicates.")
    models = OrderedDict((name, available[name]) for name in enabled)
    return PrequentialEnsemble(models=models, config=config.ensemble)
