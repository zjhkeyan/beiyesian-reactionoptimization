from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path

from bgolearn_rxn.factory import build_ensemble, prepare_problem
from bgolearn_rxn.surrogates.component_ard_gp import ComponentARDStructuredGP
from bgolearn_rxn.surrogates.credible_support_maximin_ard import (
    CredibleSupportMaximinARD,
)
from bgolearn_rxn.surrogates.interaction import (
    BayesianLowRankInteractionSurrogate,
)
from bgolearn_rxn.surrogates.query_one_rank_consensus_ard import (
    COMPONENT_ARD,
    INTERACTION,
    TREES,
)
from bgolearn_rxn.surrogates.trees import BootstrapExtraTreesSurrogate


RUNNER_PATH = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "run_suzuki_40round_landscape_zoo.py"
)
SPEC = importlib.util.spec_from_file_location(
    "credible_support_maximin_runner", RUNNER_PATH
)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_factory_builds_exact_injected_expert_graph(tiny_dataset, tiny_config) -> None:
    model_name = CredibleSupportMaximinARD.name
    config = replace(
        tiny_config,
        ensemble=replace(
            tiny_config.ensemble,
            enabled_models=(model_name,),
            initial_weights={model_name: 1.0},
        ),
        policy=replace(tiny_config.policy, first_query=model_name),
    )
    prepared = prepare_problem(tiny_dataset, config)
    ensemble = build_ensemble(config, prepared, seed=101)
    model = ensemble.models[model_name]

    assert tuple(ensemble.models) == (model_name,)
    assert isinstance(model, CredibleSupportMaximinARD)
    assert tuple(model.experts) == (COMPONENT_ARD, INTERACTION, TREES)
    assert isinstance(model.experts[COMPONENT_ARD], ComponentARDStructuredGP)
    assert isinstance(
        model.experts[INTERACTION], BayesianLowRankInteractionSurrogate
    )
    assert isinstance(model.experts[TREES], BootstrapExtraTreesSurrogate)


def test_runner_variant_has_exact_query_one_then_logei_contract() -> None:
    variant = "credible_support_maximin"
    model_name = "credible_support_maximin_ard_gp"
    assert variant in RUNNER.VARIANTS
    assert RUNNER.MODEL_NAMES[variant] == model_name
    assert RUNNER.VARIANT_CONFIGS[variant] == {
        "enabled_models": (model_name,),
        "initial_weights": {model_name: 1.0},
        "first_query": model_name,
        "early_end": 40,
        "final_start": 41,
        "final_beta": 0.15,
    }

    config, dataset, selected_model = RUNNER.build_problem(variant)
    assert selected_model == model_name
    assert len(dataset.initial) == 29
    assert len(dataset.candidates) == 5731
    assert config.dataset.budget == 40
    assert config.policy.first_query == model_name
    assert config.policy.early_end == 40
    assert config.policy.final_start == 41
    assert config.policy.challenge_every == 0
    assert config.policy.novelty_disagreement_mode == "off"
