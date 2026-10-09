#!/usr/bin/env python3
"""Phase 9 Suzuki identity-plus-Morgan development experiment.

This is a frozen molecular representation study, not reaction-pretraining.
Only participant-side training labels, unlabeled candidate identities, and the
provenance-recorded public structure registry are used.  The sealed Phase 8
test split is deliberately unavailable from this command line.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import inchi, rdFingerprintGenerator
from scipy.stats import t
from sklearn.preprocessing import StandardScaler


EXPERIMENT_ROOT = Path(__file__).resolve().parent
CODE_ROOT = EXPERIMENT_ROOT.parent
DATASET_ROOT = CODE_ROOT.parents[1]
EXTENSION_ROOT = CODE_ROOT / "bgolearn_rxn_extension"
sys.path.insert(0, str(EXTENSION_ROOT / "src"))

BASE_PATH = EXPERIMENT_ROOT / "run_suzuki_40round_landscape_zoo.py"
BASE_SPEC = importlib.util.spec_from_file_location("phase8_landscape_zoo", BASE_PATH)
assert BASE_SPEC is not None and BASE_SPEC.loader is not None
BASE = importlib.util.module_from_spec(BASE_SPEC)
BASE_SPEC.loader.exec_module(BASE)

from bgolearn_rxn.campaign import CampaignRunner  # noqa: E402
from bgolearn_rxn.datasets import DatasetSpec, ReactionDataset  # noqa: E402
from bgolearn_rxn.factory import PreparedProblem  # noqa: E402
from bgolearn_rxn.features import NameRuleFeaturizer  # noqa: E402
from bgolearn_rxn.oracle import OfflineTableOracle  # noqa: E402
from bgolearn_rxn.schemas import FeatureSchema, SliceSpec  # noqa: E402


REGISTRY_FILE = EXPERIMENT_ROOT / "phase9_suzuki_structure_registry.csv"
INFERRED_KEY = (
    "Nucleophile",
    "potassium trifluoro(5-methyl-1-(tetrahydro-2H-pyran-2-yl)-"
    "1H-indazol-4-yl)borate",
)
ATOMIC_ROLES = ("Electrophile", "Nucleophile", "Ligand", "Base", "Solvent")
VARIANTS = (
    "component_ard",
    "identity_morgan",
    "identity_morgan_inferred_missing",
)
PRIMARY_VARIANT = "identity_morgan"
CORE_METRICS = BASE.CORE_METRICS
MORGAN_RADIUS = 2
MORGAN_BITS = 256
RULE_SCALE = 0.75


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FrozenIdentityMorganEncoder:
    """Fit-once baseline identity/rule encoder with active Morgan bits appended."""

    def __init__(
        self,
        spec: DatasetSpec,
        registry_path: Path,
        *,
        inferred_policy: str = "structure",
        molecular_scale: float = 1.0,
    ) -> None:
        if inferred_policy not in {"structure", "missing_indicator"}:
            raise ValueError("inferred_policy must be structure or missing_indicator")
        if molecular_scale <= 0.0:
            raise ValueError("molecular_scale must be positive")
        self.spec = spec
        self.registry_path = Path(registry_path)
        self.inferred_policy = inferred_policy
        self.molecular_scale = float(molecular_scale)
        self.schema: FeatureSchema | None = None
        self._levels: dict[str, tuple[str, ...]] = {}
        self._active_bits: dict[str, np.ndarray] = {}
        self._rule_scalers: dict[str, StandardScaler] = {}
        self._rule_featurizer = NameRuleFeaturizer()
        self._fingerprint_cache: dict[tuple[str, str, bool], np.ndarray] = {}
        self._registry = self._load_registry()
        self._generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=MORGAN_RADIUS, fpSize=MORGAN_BITS
        )
        self._is_fit = False

    def _load_registry(self) -> dict[tuple[str, str], dict[str, str]]:
        table = pd.read_csv(self.registry_path, keep_default_na=False)
        required = {
            "role",
            "component",
            "canonical_smiles",
            "inchi_key",
            "validation_status",
            "source_url",
        }
        missing = sorted(required - set(table.columns))
        if missing:
            raise ValueError(f"Registry is missing columns: {missing}")
        if table.duplicated(["role", "component"]).any():
            raise ValueError("Registry contains duplicate role/component keys")
        records: dict[tuple[str, str], dict[str, str]] = {}
        for row in table.to_dict(orient="records"):
            record = {key: str(value).strip() for key, value in row.items()}
            key = (record["role"], record["component"])
            status = record["validation_status"]
            if status == "no_component":
                if record["canonical_smiles"]:
                    raise ValueError(f"No-component registry row has SMILES: {key}")
            else:
                mol = Chem.MolFromSmiles(record["canonical_smiles"])
                if mol is None:
                    raise ValueError(f"RDKit cannot parse registry row: {key}")
                observed_key = inchi.MolToInchiKey(mol)
                if observed_key != record["inchi_key"]:
                    raise ValueError(
                        f"InChIKey mismatch for {key}: {observed_key} != "
                        f"{record['inchi_key']}"
                    )
            records[key] = record
        inferred = records.get(INFERRED_KEY)
        if inferred is None or inferred["validation_status"] != "derived_unverified":
            raise ValueError("The single inferred row must remain explicitly unverified")
        if inferred.get("source_precursor_cid") != "129009568":
            raise ValueError("The inferred row is missing its precursor CID")
        if not inferred.get("derivation_rule"):
            raise ValueError("The inferred row is missing its derivation rule")
        return records

    def _prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        required = list(self.spec.feature_columns)
        missing = [column for column in required if column not in frame]
        if missing:
            raise ValueError(f"Feature frame is missing columns: {missing}")
        prepared = self.spec.add_derived_roles(frame.loc[:, required])
        return prepared.loc[:, list(self.spec.roles)].astype(str)

    def _fingerprint(
        self, role: str, component: str, *, force_inferred_structure: bool = False
    ) -> np.ndarray:
        cache_key = (role, component, force_inferred_structure)
        cached = self._fingerprint_cache.get(cache_key)
        if cached is not None:
            return cached
        if component == "Nothing":
            bits = np.zeros(MORGAN_BITS, dtype=float)
            missing = 0.0
        else:
            key = (role, component)
            if key not in self._registry:
                raise ValueError(f"No strict registry row for {key}")
            record = self._registry[key]
            inferred_missing = (
                key == INFERRED_KEY
                and self.inferred_policy == "missing_indicator"
            )
            if inferred_missing and not force_inferred_structure:
                bits = np.zeros(MORGAN_BITS, dtype=float)
            else:
                mol = Chem.MolFromSmiles(record["canonical_smiles"])
                if mol is None:  # guarded during registry loading
                    raise RuntimeError(f"Registry SMILES became invalid for {key}")
                fingerprint = self._generator.GetFingerprint(mol)
                bits = np.zeros(MORGAN_BITS, dtype=float)
                DataStructs.ConvertToNumpyArray(fingerprint, bits)
            missing = float(inferred_missing)
        # Reserve the same missingness coordinate under both policies so the
        # sensitivity changes values only, never schema width or GP distance
        # normalization.  It is identically zero for the structure policy.
        output = np.concatenate([bits, np.asarray([missing])])
        self._fingerprint_cache[cache_key] = output
        return output

    def _full_molecular_block(
        self,
        prepared: pd.DataFrame,
        role: str,
        *,
        force_inferred_structure: bool = False,
    ) -> np.ndarray:
        if role in ATOMIC_ROLES:
            return np.vstack(
                [
                    self._fingerprint(
                        role,
                        value,
                        force_inferred_structure=force_inferred_structure,
                    )
                    for value in prepared[role]
                ]
            )
        if role == "SubstratePair":
            left = np.vstack(
                [
                    self._fingerprint(
                        "Electrophile",
                        value,
                        force_inferred_structure=force_inferred_structure,
                    )
                    for value in prepared["Electrophile"]
                ]
            )
            right = np.vstack(
                [
                    self._fingerprint(
                        "Nucleophile",
                        value,
                        force_inferred_structure=force_inferred_structure,
                    )
                    for value in prepared["Nucleophile"]
                ]
            )
            return np.hstack([left, right])
        return np.empty((len(prepared), 0), dtype=float)

    def _molecular_feature_name(self, role: str, index: int) -> str:
        unit_width = MORGAN_BITS + 1
        if role in ATOMIC_ROLES:
            if index == MORGAN_BITS:
                return f"{role}::morgan::inferred_missing"
            return f"{role}::morgan::bit_{index:03d}"
        if role == "SubstratePair":
            side = "Electrophile" if index < unit_width else "Nucleophile"
            side_index = index if index < unit_width else index - unit_width
            if side_index == MORGAN_BITS:
                return f"{role}::morgan::{side}::inferred_missing"
            return f"{role}::morgan::{side}::bit_{side_index:03d}"
        raise ValueError(f"Role {role!r} has no molecular features")

    def fit(self, reference_frame: pd.DataFrame) -> "FrozenIdentityMorganEncoder":
        prepared = self._prepare(reference_frame)
        expected = {
            (role, component)
            for role in ATOMIC_ROLES
            for component in set(prepared[role])
        }
        missing = sorted(expected - set(self._registry))
        if missing:
            raise ValueError(f"Registry does not cover reference components: {missing}")

        feature_names: list[str] = []
        role_slices: dict[str, SliceSpec] = {}
        identity_slices: dict[str, SliceSpec] = {}
        cursor = 0
        for role in self.spec.roles:
            levels = tuple(sorted(set(prepared[role])))
            self._levels[role] = levels
            identity_start = cursor
            identity_stop = cursor + len(levels)
            identity_slices[role] = SliceSpec(identity_start, identity_stop)
            feature_names.extend(f"{role}::identity::{level}" for level in levels)

            rules = self._rule_featurizer.transform(role, prepared[role])
            rule_scaler = StandardScaler(with_mean=True, with_std=True).fit(rules)
            self._rule_scalers[role] = rule_scaler
            feature_names.extend(
                f"{role}::rule::{name}"
                for name in self._rule_featurizer.FEATURE_NAMES
            )

            # Freeze active-bit coordinates from the complete structure registry.
            # The sensitivity encoder then changes only the inferred row's values
            # (plus its explicit missing indicator), not the Morgan coordinate set.
            full = self._full_molecular_block(
                prepared, role, force_inferred_structure=True
            )
            active = (
                np.flatnonzero(np.ptp(full, axis=0) > 0.0)
                if full.shape[1]
                else np.asarray([], dtype=int)
            )
            if role == "Nucleophile":
                active = np.unique(np.append(active, MORGAN_BITS))
            elif role == "SubstratePair":
                nucleophile_missing = (MORGAN_BITS + 1) + MORGAN_BITS
                active = np.unique(np.append(active, nucleophile_missing))
            self._active_bits[role] = active
            feature_names.extend(
                self._molecular_feature_name(role, int(index)) for index in active
            )
            role_stop = identity_stop + rules.shape[1] + len(active)
            role_slices[role] = SliceSpec(cursor, role_stop)
            cursor = role_stop

        self.schema = FeatureSchema(
            dataset_name=self.spec.name,
            roles=self.spec.roles,
            condition_roles=self.spec.condition_roles,
            task_role=self.spec.task_column,
            role_slices=role_slices,
            identity_slices=identity_slices,
            feature_names=tuple(feature_names),
            derived_roles=self.spec.derived_roles,
            fixed_roles=self.spec.fixed_roles,
        )
        self._is_fit = True
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not self._is_fit or self.schema is None:
            raise RuntimeError("FrozenIdentityMorganEncoder.fit must be called first")
        prepared = self._prepare(frame)
        blocks: list[np.ndarray] = []
        for role in self.spec.roles:
            levels = self._levels[role]
            index = {level: position for position, level in enumerate(levels)}
            identity = np.zeros((len(prepared), len(levels)), dtype=float)
            for row_index, value in enumerate(prepared[role]):
                if value not in index:
                    raise ValueError(f"Unknown frozen identity for {role}: {value}")
                identity[row_index, index[value]] = 1.0
            rules = self._rule_featurizer.transform(role, prepared[role])
            rules = self._rule_scalers[role].transform(rules) * RULE_SCALE
            molecular = self._full_molecular_block(prepared, role)
            active = self._active_bits[role]
            if len(active):
                molecular = molecular[:, active] * self.molecular_scale
                blocks.append(np.hstack([identity, rules, molecular]))
            else:
                blocks.append(np.hstack([identity, rules]))
        output = np.hstack(blocks)
        if output.shape != (len(prepared), self.schema.n_features):
            raise RuntimeError("Frozen Morgan feature width changed")
        if not np.isfinite(output).all():
            raise FloatingPointError("Frozen Morgan features contain non-finite values")
        return output

    def audit(self) -> dict:
        if self.schema is None:
            raise RuntimeError("Encoder is not fitted")
        return {
            "encoder": "baseline_identity_rules_plus_active_morgan_role_blocks",
            "baseline_name_rule_scale": RULE_SCALE,
            "baseline_name_rule_width_per_role": len(
                self._rule_featurizer.FEATURE_NAMES
            ),
            "morgan_radius": MORGAN_RADIUS,
            "morgan_bits": MORGAN_BITS,
            "molecular_scale": self.molecular_scale,
            "inferred_policy": self.inferred_policy,
            "active_morgan_widths": {
                role: int(len(indices)) for role, indices in self._active_bits.items()
            },
            "identity_widths": {
                role: len(levels) for role, levels in self._levels.items()
            },
            "n_features": self.schema.n_features,
            "fit_reference": "initial plus unlabeled candidates",
            "preprocessing_frozen_before_feedback": True,
            "fixed_product_in_condition_kernel": False,
        }


def build_morgan_prepared(
    dataset: ReactionDataset,
    *,
    inferred_policy: str,
    molecular_scale: float,
) -> tuple[PreparedProblem, dict]:
    columns = list(dataset.spec.feature_columns)
    reference = pd.concat(
        [dataset.initial.loc[:, columns], dataset.candidates.loc[:, columns]],
        ignore_index=True,
    )
    encoder = FrozenIdentityMorganEncoder(
        dataset.spec,
        REGISTRY_FILE,
        inferred_policy=inferred_policy,
        molecular_scale=molecular_scale,
    ).fit(reference)
    prepared = PreparedProblem(
        encoder=encoder,
        X_initial=encoder.transform(dataset.initial.loc[:, columns]),
        X_candidates=encoder.transform(dataset.candidates.loc[:, columns]),
        reference_X=encoder.transform(reference),
    )
    return prepared, encoder.audit()


def _candidate_policy(variant: str) -> str:
    if variant == "identity_morgan":
        return "structure"
    if variant == "identity_morgan_inferred_missing":
        return "missing_indicator"
    raise ValueError(f"Not a molecular candidate variant: {variant}")


def run_landscape_one(job: tuple[str, dict, str, float]) -> dict:
    variant, landscape_spec, output_root_string, molecular_scale = job
    if variant == "component_ard":
        return BASE.run_one((variant, landscape_spec, output_root_string))

    config, dataset, model_name = BASE.build_problem("component_ard")
    prepared, representation = build_morgan_prepared(
        dataset,
        inferred_policy=_candidate_policy(variant),
        molecular_scale=molecular_scale,
    )
    lookup, teacher = BASE.build_landscape(landscape_spec)
    oracle = OfflineTableOracle(dataset=dataset, lookup=lookup)
    runner = CampaignRunner(
        dataset,
        config,
        seed=int(landscape_spec["campaign_seed"]),
        backend="native",
        prepared=prepared,
    )
    started = time.perf_counter()
    result = runner.run(oracle, budget=BASE.BUDGET)
    elapsed = time.perf_counter() - started
    payload = result.to_payload()
    metrics = BASE.campaign_metrics(payload["trajectory"], oracle.global_best, BASE.BUDGET)
    first20 = BASE.campaign_metrics(payload["trajectory"][:20], oracle.global_best, 20)
    queried = [int(row["query_index"]) for row in payload["trajectory"]]
    contract = {
        "split": landscape_spec["split"],
        "landscape": landscape_spec["landscape"],
        "teacher_family": landscape_spec["family"],
        "variant": variant,
        "model": model_name,
        "n_initial": len(dataset.initial),
        "n_candidates": len(dataset.candidates),
        "budget": BASE.BUDGET,
        "all_queries_unique": len(set(queried)) == BASE.BUDGET,
        "all_queries_in_candidate_pool": set(queried).issubset(set(dataset.candidate_ids)),
        "public_test_labels_used": False,
        "public_searchspace_labels_used": False,
        "production_default_changed": False,
        "representation": representation,
        "registry_sha256": file_sha256(REGISTRY_FILE),
        "true_reaction_pretraining": False,
        "pretraining_note": (
            "PericyclicTL provides extraction notebooks but no trainable model or "
            "checkpoint; this experiment is frozen molecular side information only."
        ),
        "teacher": teacher,
    }
    payload["landscape_metrics"] = metrics
    payload["first20_metrics"] = first20
    payload["landscape_contract"] = contract
    path = (
        Path(output_root_string)
        / "trajectories"
        / variant
        / f"{landscape_spec['landscape']}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "dataset": "Suzuki",
        **landscape_spec,
        "variant": variant,
        "model": model_name,
        "n_initial": len(dataset.initial),
        "n_candidates": len(dataset.candidates),
        "budget": BASE.BUDGET,
        "global_best": float(oracle.global_best),
        **metrics,
        "elapsed_seconds": elapsed,
        "query_ids_sha256": BASE.payload_sha256(queried),
        "contract": contract,
        "first20": first20,
    }


def make_train_holdout_episodes() -> list[dict]:
    """Reuse the established train-only cold-level diagnostic geometry."""
    frame = pd.read_csv(BASE.TRAIN_FILE)
    episodes: list[dict] = []
    family_index = 0
    for role in ATOMIC_ROLES:
        values = frame[role].astype(str)
        eligible = sorted(
            level for level, count in values.value_counts().items() if count >= 2
        )
        for level_index, level in enumerate(eligible):
            candidates = np.flatnonzero(values.eq(level).to_numpy())
            initial = np.flatnonzero(~values.eq(level).to_numpy())
            episodes.append(
                {
                    "episode": f"component_{role.lower()}_{level_index:02d}",
                    "episode_family": f"leave_{role.lower()}_out",
                    "held_role": role,
                    "held_level": level,
                    "initial_indices": initial.tolist(),
                    "candidate_indices": candidates.tolist(),
                    "budget": min(5, len(candidates)),
                    "campaign_seed": 910_000 + family_index * 1_000 + level_index,
                }
            )
        family_index += 1

    pair_values = (
        frame["Electrophile"].astype(str)
        + " || "
        + frame["Nucleophile"].astype(str)
    )
    eligible_pairs = sorted(
        level for level, count in pair_values.value_counts().items() if count >= 2
    )
    for level_index, level in enumerate(eligible_pairs):
        candidates = np.flatnonzero(pair_values.eq(level).to_numpy())
        initial = np.flatnonzero(~pair_values.eq(level).to_numpy())
        episodes.append(
            {
                "episode": f"substrate_pair_{level_index:02d}",
                "episode_family": "leave_substrate_pair_out",
                "held_role": "SubstratePair",
                "held_level": level,
                "initial_indices": initial.tolist(),
                "candidate_indices": candidates.tolist(),
                "budget": min(5, len(candidates)),
                "campaign_seed": 916_000 + level_index,
            }
        )
    return episodes


def build_holdout_problem(episode: dict) -> tuple[object, ReactionDataset, pd.DataFrame]:
    source = pd.read_csv(BASE.TRAIN_FILE).reset_index(names="source_index")
    config, _, _ = BASE.build_problem("component_ard")
    budget = int(episode["budget"])
    seed = int(episode["campaign_seed"])
    config = replace(
        config,
        dataset=replace(config.dataset, budget=budget, seeds=(seed,)),
        policy=replace(
            config.policy,
            early_end=budget,
            final_start=budget + 1,
            challenge_every=0,
        ),
    )
    initial_indices = np.asarray(episode["initial_indices"], dtype=int)
    candidate_indices = np.asarray(episode["candidate_indices"], dtype=int)
    features = list(config.dataset.feature_columns)
    initial = source.loc[
        initial_indices, features + [config.dataset.target_column]
    ].copy()
    candidates = source.loc[candidate_indices, features].copy()
    candidates.insert(
        0, "query_index", source.loc[candidate_indices, "source_index"].astype(int)
    )
    lookup = source.loc[
        candidate_indices, features + [config.dataset.target_column]
    ].copy()
    lookup.insert(
        0, "query_index", source.loc[candidate_indices, "source_index"].astype(int)
    )
    dataset = ReactionDataset(
        spec=DatasetSpec.from_config(config.dataset),
        initial=initial,
        candidates=candidates,
    )
    return config, dataset, lookup


def run_holdout_one(job: tuple[str, dict, str, float]) -> dict:
    variant, episode, output_root_string, molecular_scale = job
    config, dataset, lookup = build_holdout_problem(episode)
    prepared = None
    representation = {"encoder": "phase8_component_ard_baseline"}
    if variant != "component_ard":
        prepared, representation = build_morgan_prepared(
            dataset,
            inferred_policy=_candidate_policy(variant),
            molecular_scale=molecular_scale,
        )
    oracle = OfflineTableOracle(dataset=dataset, lookup=lookup)
    runner = CampaignRunner(
        dataset,
        config,
        seed=int(episode["campaign_seed"]),
        backend="native",
        prepared=prepared,
    )
    started = time.perf_counter()
    result = runner.run(oracle, budget=int(episode["budget"]))
    elapsed = time.perf_counter() - started
    payload = result.to_payload()
    metrics = BASE.campaign_metrics(
        payload["trajectory"], float(lookup["Yield"].max()), int(episode["budget"])
    )
    contract = {
        "train_label_only": True,
        "mechanism_diagnostic_only": True,
        "not_a_promotion_metric": True,
        "synthetic_proxy_limitation": (
            "This evaluates cold-level interpolation within 29 historical labels and "
            "does not establish that Phase 8 synthetic teachers correlate with chemistry."
        ),
        "episode": episode,
        "representation": representation,
        "public_test_labels_used": False,
    }
    payload["holdout_metrics"] = metrics
    payload["holdout_contract"] = contract
    path = (
        Path(output_root_string)
        / "holdout_diagnostic"
        / "trajectories"
        / variant
        / f"{episode['episode']}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "episode": episode["episode"],
        "episode_family": episode["episode_family"],
        "held_role": episode["held_role"],
        "held_level": episode["held_level"],
        "variant": variant,
        "budget": int(episode["budget"]),
        "n_initial": len(dataset.initial),
        "n_candidates": len(dataset.candidates),
        **metrics,
        "elapsed_seconds": elapsed,
    }


def _mean_ci(values: pd.Series) -> tuple[float, float, float, float]:
    array = values.to_numpy(dtype=float)
    mean = float(array.mean())
    if len(array) == 1:
        return mean, 0.0, mean, mean
    sd = float(array.std(ddof=1))
    half = float(t.ppf(0.975, len(array) - 1) * sd / math.sqrt(len(array)))
    return mean, sd, mean - half, mean + half


def summarize_holdouts(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    absolute: list[dict] = []
    paired: list[dict] = []
    for (family, variant), group in rows.groupby(
        ["episode_family", "variant"], sort=True
    ):
        for metric in CORE_METRICS + ("elapsed_seconds",):
            mean, sd, low, high = _mean_ci(group[metric])
            absolute.append(
                {
                    "episode_family": family,
                    "variant": variant,
                    "metric": metric,
                    "n": len(group),
                    "mean": mean,
                    "sample_sd": sd,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    for family, section in rows.groupby("episode_family", sort=True):
        baseline = section[section["variant"].eq("component_ard")].set_index("episode")
        if baseline.empty:
            continue
        for variant in sorted(set(section["variant"]) - {"component_ard"}):
            candidate = section[section["variant"].eq(variant)].set_index("episode")
            common = baseline.index.intersection(candidate.index)
            if common.empty:
                continue
            for metric in CORE_METRICS + ("elapsed_seconds",):
                delta = candidate.loc[common, metric] - baseline.loc[common, metric]
                mean, sd, low, high = _mean_ci(delta)
                paired.append(
                    {
                        "episode_family": family,
                        "candidate": variant,
                        "baseline": "component_ard",
                        "metric": metric,
                        "n_pairs": len(common),
                        "mean_delta": mean,
                        "sample_sd_delta": sd,
                        "ci95_low": low,
                        "ci95_high": high,
                    }
                )
    return pd.DataFrame(absolute), pd.DataFrame(paired)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("phase8_dev",), default="phase8_dev")
    parser.add_argument("--episode-count", type=int, default=20)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--molecular-scale", type=float, default=0.35)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-holdout-diagnostic", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.jobs <= 0:
        raise ValueError("--jobs must be positive")
    if args.episode_count <= 0 or args.episode_count > 20:
        raise ValueError("--episode-count must be in [1, 20]")
    count = min(2, args.episode_count) if args.smoke else args.episode_count
    specs = BASE.landscape_specs(args.split, count)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    environment = BASE.environment_diagnostics(specs)
    manifest = {
        "phase": "phase9_frozen_identity_morgan",
        "split": args.split,
        "smoke": bool(args.smoke),
        "variants": args.variants,
        "landscape_count": len(specs),
        "balanced_full_panel": count == 20,
        "official_geometry": {"n_initial": 29, "n_candidates": 5731, "budget": 40},
        "molecular_representation": {
            "kind": "baseline identity and standardized name rules plus active Morgan role blocks",
            "baseline_name_rule_scale": RULE_SCALE,
            "radius": MORGAN_RADIUS,
            "bits_before_label_free_active_bit_filter": MORGAN_BITS,
            "molecular_scale": args.molecular_scale,
            "frozen_before_feedback": True,
            "product_feature_used": False,
            "true_reaction_pretraining": False,
        },
        "registry": {
            "path": str(REGISTRY_FILE.relative_to(DATASET_ROOT)),
            "sha256": file_sha256(REGISTRY_FILE),
            "inferred_rows": 1,
            "inferred_row_externally_verified": False,
            "sensitivity_variant": "identity_morgan_inferred_missing",
        },
        "allowed_inputs": {
            str(path.relative_to(DATASET_ROOT)): file_sha256(path)
            for path in (
                BASE.TRAIN_FILE,
                BASE.CANDIDATE_FILE,
                BASE.OPTIONS_FILE,
                BASE.METADATA_FILE,
                BASE.CONFIG_FILE,
                REGISTRY_FILE,
            )
        },
        "protected_inputs": {
            str(path.relative_to(DATASET_ROOT)): {"opened": False, "hashed": False}
            for path in BASE.PROTECTED_FILES
        },
        "public_test_labels_used": False,
        "public_searchspace_labels_used": False,
        "production_default_changed": False,
        "environment_diagnostics": environment,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "rdkit": rdBase.rdkitVersion,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    jobs = [
        (variant, spec, str(output), args.molecular_scale)
        for spec in specs
        for variant in args.variants
    ]
    results: list[dict] = []
    wall_started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=min(args.jobs, len(jobs))) as executor:
        futures = {executor.submit(run_landscape_one, job): job for job in jobs}
        for future in as_completed(futures):
            row = future.result()
            results.append(row)
            print(
                f"landscape {row['variant']} {row['landscape']}: "
                f"AUC={row['AUC_best_so_far']:.3f}",
                flush=True,
            )
    landscape_wall_seconds = time.perf_counter() - wall_started
    contracts = [row.pop("contract") for row in results]
    first20 = [dict(row, **row.pop("first20")) for row in results]
    rows = pd.DataFrame(results).sort_values(["landscape", "variant"])
    rows.to_csv(output / "per_landscape.csv", index=False)
    pd.DataFrame(first20).sort_values(["landscape", "variant"]).to_csv(
        output / "first20_per_landscape.csv", index=False
    )
    BASE.absolute_summary(rows).to_csv(output / "summary.csv", index=False)
    BASE.absolute_summary(rows, group_columns=("family", "variant")).to_csv(
        output / "per_teacher_summary.csv", index=False
    )
    paired = (
        BASE.paired_summary(rows, "component_ard")
        if "component_ard" in args.variants and len(args.variants) > 1
        else pd.DataFrame()
    )
    paired.to_csv(output / "paired_vs_component_ard.csv", index=False)
    BASE.tail_summary(rows).to_csv(output / "tail_summary.csv", index=False)

    holdout_rows = pd.DataFrame()
    if not args.skip_holdout_diagnostic:
        episodes = make_train_holdout_episodes()
        if args.smoke:
            pair = next(e for e in episodes if e["held_role"] == "SubstratePair")
            inferred = next(e for e in episodes if e["held_level"] == INFERRED_KEY[1])
            episodes = [inferred, pair]
        diagnostic_jobs = [
            (variant, episode, str(output), args.molecular_scale)
            for episode in episodes
            for variant in args.variants
        ]
        diagnostic_results: list[dict] = []
        with ProcessPoolExecutor(
            max_workers=min(args.jobs, len(diagnostic_jobs))
        ) as executor:
            futures = {
                executor.submit(run_holdout_one, job): job for job in diagnostic_jobs
            }
            for future in as_completed(futures):
                row = future.result()
                diagnostic_results.append(row)
                print(
                    f"holdout {row['variant']} {row['episode']}: "
                    f"AUC={row['AUC_best_so_far']:.3f}",
                    flush=True,
                )
        holdout_rows = pd.DataFrame(diagnostic_results).sort_values(
            ["episode", "variant"]
        )
        diagnostic_root = output / "holdout_diagnostic"
        holdout_rows.to_csv(diagnostic_root / "per_episode.csv", index=False)
        absolute, holdout_paired = summarize_holdouts(holdout_rows)
        absolute.to_csv(diagnostic_root / "summary.csv", index=False)
        holdout_paired.to_csv(
            diagnostic_root / "paired_vs_component_ard.csv", index=False
        )

    query_checks = [
        contract["all_queries_unique"] and contract["all_queries_in_candidate_pool"]
        for contract in contracts
    ]
    audit = {
        "passed": bool(
            len(rows) == len(specs) * len(args.variants)
            and rows["n_initial"].eq(29).all()
            and rows["n_candidates"].eq(5731).all()
            and rows["budget"].eq(40).all()
            and all(query_checks)
            and all(environment["checks"].values())
        ),
        "result_rows": len(rows),
        "trajectory_files": len(list((output / "trajectories").glob("*/*.json"))),
        "unique_landscapes": int(rows["landscape"].nunique()),
        "query_contracts_passed": int(sum(query_checks)),
        "landscape_wall_seconds": landscape_wall_seconds,
        "holdout_result_rows": len(holdout_rows),
        "protected_inputs_opened": False,
        "production_default_changed": False,
        "true_reaction_pretraining": False,
    }
    if not audit["passed"]:
        raise RuntimeError(f"Phase 9 campaign audit failed: {audit}")
    (output / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    score_rows = BASE.absolute_summary(rows)
    primary = score_rows[
        score_rows["variant"].eq(PRIMARY_VARIANT)
        & score_rows["metric"].eq("AUC_best_so_far")
    ]
    score = None if primary.empty else float(primary.iloc[0]["mean"])
    (output / "score.json").write_text(
        json.dumps(
            {
                "score": score,
                "metric": "absolute mean AUC_best_so_far",
                "variant": PRIMARY_VARIANT,
                "split": args.split,
                "n_landscapes": len(specs),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"output": str(output), "score": score, "audit": audit}, indent=2
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
