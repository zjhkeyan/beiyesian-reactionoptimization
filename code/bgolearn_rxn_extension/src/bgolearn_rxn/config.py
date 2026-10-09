from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    feature_columns: tuple[str, ...]
    target_column: str = "Yield"
    task_column: str | None = None
    target_task_value: str | None = None
    condition_roles: tuple[str, ...] = ()
    derived_roles: dict[str, tuple[str, ...]] = field(default_factory=dict)
    fixed_roles: dict[str, str] = field(default_factory=dict)
    budget: int = 40
    seeds: tuple[int, ...] = tuple(range(100, 2001, 100))


@dataclass(frozen=True)
class FeatureConfig:
    identity_scale: float = 1.0
    rule_scale: float = 0.75
    molecular_scale: float = 0.35
    name_subword_scale: float = 0.35
    registry_path: str | None = None
    use_name_rules: bool = True
    use_name_subwords: bool = False
    name_subword_max_rank: int = 8
    use_rdkit: bool = True
    morgan_radius: int = 2
    morgan_bits: int = 256
    missing_registry_policy: str = "warn"  # warn | error | zeros


@dataclass(frozen=True)
class GPConfig:
    interaction_pairs: tuple[tuple[str, str], ...] = ()
    role_weights: dict[str, float] = field(default_factory=dict)
    interaction_weights: dict[str, float] = field(default_factory=dict)
    optimize_hyperparameters: bool = True
    restarts: int = 3
    maxiter: int = 150
    refit_interval: int = 1
    initial_lengthscale: float = 1.0
    initial_noise: float = 0.05
    initial_pair_scale: float = 0.25
    initial_task_rho: float = 0.45
    task_descriptor_scale: float = 0.0
    task_mean_prior_count: float | None = None
    jitter: float = 1e-8
    target_transform: str = "standard"


@dataclass(frozen=True)
class InteractionModelConfig:
    interaction_pairs: tuple[tuple[str, str], ...] = ()
    role_ranks: dict[str, int] = field(default_factory=dict)
    default_rank: int = 4
    interaction_scale: float = 0.5
    alpha_1: float = 1e-6
    alpha_2: float = 1e-6
    lambda_1: float = 1e-6
    lambda_2: float = 1e-6
    target_transform: str = "standard"


@dataclass(frozen=True)
class TreeConfig:
    n_estimators: int = 256
    max_features: str | float | int = "sqrt"
    min_samples_leaf: int = 2
    max_depth: int | None = None
    residual_noise_fraction: float = 0.25
    target_transform: str = "standard"


@dataclass(frozen=True)
class EnsembleConfig:
    initial_weights: dict[str, float] = field(
        default_factory=lambda: {"gp": 0.5, "interaction": 0.3, "trees": 0.2}
    )
    prequential_learning_rate: float = 0.2
    minimum_weight: float = 0.08
    predictive_std_floor: float = 1.0
    enabled_models: tuple[str, ...] = ("gp", "interaction", "trees")


@dataclass(frozen=True)
class PolicyConfig:
    first_query: str = "posterior_mean"
    early_end: int = 20
    final_start: int = 33
    challenge_every: int = 5
    challenge_beta: float = 1.5
    challenge_disagreement_weight: float = 0.75
    challenge_gate_sigma: float = 1.0
    final_beta: float = 0.20
    xi_initial: float = 0.01
    xi_final: float = 0.0
    prior_initial_weight: float = 0.0
    prior_decay: float = 0.30
    novelty_disagreement_mode: str = "off"  # off | pure | hybrid
    novelty_disagreement_weight: float = 0.50


@dataclass(frozen=True)
class RuntimeConfig:
    backend: str = "native"  # native | bgolearn
    output_dir: str = "runs"
    save_candidate_scores: bool = False
    strict_reproducibility: bool = True


@dataclass(frozen=True)
class RunConfig:
    dataset: DatasetConfig
    features: FeatureConfig = field(default_factory=FeatureConfig)
    gp: GPConfig = field(default_factory=GPConfig)
    interaction_model: InteractionModelConfig = field(default_factory=InteractionModelConfig)
    trees: TreeConfig = field(default_factory=TreeConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


def _pairs(value: Any) -> tuple[tuple[str, str], ...]:
    if value is None:
        return ()
    pairs: list[tuple[str, str]] = []
    for item in value:
        if len(item) != 2:
            raise ValueError(f"Interaction pair must have length 2, got {item!r}")
        pairs.append((str(item[0]), str(item[1])))
    return tuple(pairs)


def _dataset_config(data: dict[str, Any]) -> DatasetConfig:
    derived = {
        str(key): tuple(str(column) for column in value)
        for key, value in data.get("derived_roles", {}).items()
    }
    return DatasetConfig(
        name=str(data["name"]),
        feature_columns=tuple(str(value) for value in data["feature_columns"]),
        target_column=str(data.get("target_column", "Yield")),
        task_column=data.get("task_column"),
        target_task_value=data.get("target_task_value"),
        condition_roles=tuple(str(value) for value in data.get("condition_roles", [])),
        derived_roles=derived,
        fixed_roles={
            str(key): str(value) for key, value in data.get("fixed_roles", {}).items()
        },
        budget=int(data.get("budget", 40)),
        seeds=tuple(int(value) for value in data.get("seeds", range(100, 2001, 100))),
    )


def load_run_config(path: str | Path) -> RunConfig:
    """Load and validate a YAML run configuration.

    Relative registry paths are resolved against the YAML file, not the
    process working directory.
    """

    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a mapping.")

    gp_raw = dict(raw.get("gp", {}))
    gp_raw["interaction_pairs"] = _pairs(gp_raw.get("interaction_pairs"))

    interaction_raw = dict(raw.get("interaction_model", {}))
    interaction_raw["interaction_pairs"] = _pairs(
        interaction_raw.get("interaction_pairs")
    )

    feature_raw = dict(raw.get("features", {}))
    registry_path = feature_raw.get("registry_path")
    if registry_path:
        registry = Path(str(registry_path))
        if not registry.is_absolute():
            registry = (config_path.parent / registry).resolve()
        feature_raw["registry_path"] = str(registry)

    ensemble_raw = dict(raw.get("ensemble", {}))
    ensemble_raw["enabled_models"] = tuple(
        str(value)
        for value in ensemble_raw.get(
            "enabled_models", ("gp", "interaction", "trees")
        )
    )

    return RunConfig(
        dataset=_dataset_config(raw["dataset"]),
        features=FeatureConfig(**feature_raw),
        gp=GPConfig(**gp_raw),
        interaction_model=InteractionModelConfig(**interaction_raw),
        trees=TreeConfig(**raw.get("trees", {})),
        ensemble=EnsembleConfig(**ensemble_raw),
        policy=PolicyConfig(**raw.get("policy", {})),
        runtime=RuntimeConfig(**raw.get("runtime", {})),
    )
