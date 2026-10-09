from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .chemistry import ChemicalRegistry
from .config import FeatureConfig
from .datasets import DatasetSpec
from .schemas import FeatureSchema, SliceSpec


class NameRuleFeaturizer:
    """Label-free mechanistic and lexical flags extracted from component names.

    These rules are deliberately conservative. They do not claim to replace
    verified molecular structures; they provide an immediately runnable bridge
    between exact one-hot identities and a curated structure registry.
    """

    FEATURE_NAMES = (
        "name_length",
        "count_methyl",
        "count_phenyl",
        "count_methoxy",
        "count_tert_butyl",
        "is_nothing",
        "has_bromo",
        "has_chloro",
        "has_iodo",
        "has_triflate",
        "has_boronic_acid",
        "has_boronate_ester",
        "has_trifluoroborate",
        "has_phosphane",
        "has_bisphosphine_cue",
        "has_ferrocene_or_iron",
        "has_xanthene",
        "has_adamantyl",
        "has_cyclohexyl",
        "has_oxazole",
        "has_benzoxazole",
        "has_isoxazole_carboxylate",
        "has_furan",
        "has_thiophene",
        "has_pyrrole",
        "has_guanidine",
        "has_phosphazene",
        "has_hydroxide",
        "has_carbonate",
        "has_phosphate",
        "has_fluoride",
        "has_alkoxide",
        "has_trialkylamine",
        "solvent_amide",
        "solvent_nitrile",
        "solvent_alcohol",
        "solvent_ether",
        "contains_heteroaryl_cue",
        "contains_fluorinated_cue",
    )

    @staticmethod
    def _contains(text: str, *patterns: str) -> float:
        return float(any(pattern in text for pattern in patterns))

    def transform(self, role: str, values: pd.Series) -> np.ndarray:
        rows: list[list[float]] = []
        for raw in values.astype(str):
            text = raw.strip().lower()
            rows.append(
                [
                    float(len(text)),
                    float(text.count("methyl")),
                    float(text.count("phenyl")),
                    float(text.count("methoxy")),
                    float(text.count("tert-butyl") + text.count("ditert-butyl") + text.count("tritert-butyl")),
                    float(text == "nothing"),
                    self._contains(text, "bromo", "bromide"),
                    self._contains(text, "chloro", "chloride"),
                    self._contains(text, "iodo", "iodide"),
                    self._contains(text, "trifluoromethanesulfonate", "triflate"),
                    self._contains(text, "boronic acid"),
                    self._contains(text, "dioxaborolan", "pinacol", "boronate"),
                    self._contains(text, "trifluoro(", "trifluorobor", "bf3"),
                    self._contains(text, "phosphane", "phosphino", "phosphanyl"),
                    self._contains(text, "bis(diphenylphosph", "bis(ditert-butyl", "diphosph"),
                    self._contains(text, "ferrocene", "iron(2+)", ";iron"),
                    self._contains(text, "xanthen", "xanthene"),
                    self._contains(text, "adamantyl"),
                    self._contains(text, "cyclohexyl"),
                    self._contains(text, "oxazole", "ccon", "ccno"),
                    self._contains(text, "benzoxazole"),
                    float("oxazole" in text and "carboxylate" in text),
                    self._contains(text, "furan"),
                    self._contains(text, "thiophen"),
                    self._contains(text, "pyrrol"),
                    self._contains(text, "guanidine", "guanid"),
                    self._contains(text, "phosphaz", "lambda5-phosph"),
                    self._contains(text, "hydroxide"),
                    self._contains(text, "carbonate", "hydrogen carbonate"),
                    self._contains(text, "phosphate"),
                    self._contains(text, "fluoride"),
                    self._contains(text, "olate", "alkoxide"),
                    self._contains(text, "triethylamine", "n,n-diethylethanamine"),
                    self._contains(text, "dimethylformamide", "formamide"),
                    self._contains(text, "acetonitrile", "nitrile"),
                    self._contains(text, "methanol", "ethanol", "alcohol"),
                    self._contains(text, "oxolane", "tetrahydrofuran", "ether"),
                    self._contains(text, "pyridin", "quinolin", "indazol", "furan", "thiophen", "pyrrol"),
                    self._contains(text, "trifluoro", "difluoro", "fluoro"),
                ]
            )
        return np.asarray(rows, dtype=float)


@dataclass
class _RoleState:
    encoder: OneHotEncoder
    rule_scaler: StandardScaler | None
    subword_vectorizer: TfidfVectorizer | None
    subword_reducer: TruncatedSVD | None
    subword_scaler: StandardScaler | None
    molecular_scaler: StandardScaler | None
    identity_names: tuple[str, ...]
    rule_names: tuple[str, ...]
    subword_names: tuple[str, ...]
    molecular_names: tuple[str, ...]


class FrozenReactionEncoder:
    """Fit-once, transform-many reaction featurizer.

    Fit this encoder on the union of participant-side initial features and the
    *unlabeled* candidate pool. Do not refit it after removing queried rows.
    That invariant prevents round-dependent coordinates and is one of the key
    differences from using Bgolearn's per-call ``Normalize=True`` behavior.
    """

    def __init__(
        self,
        spec: DatasetSpec,
        config: FeatureConfig,
    ) -> None:
        self.spec = spec
        self.config = config
        self.rule_featurizer = NameRuleFeaturizer()
        self.registry = ChemicalRegistry(
            config.registry_path,
            use_rdkit=config.use_rdkit,
            morgan_radius=config.morgan_radius,
            morgan_bits=config.morgan_bits,
            missing_policy=config.missing_registry_policy,
        )
        self._states: dict[str, _RoleState] = {}
        self.schema: FeatureSchema | None = None
        self._is_fit = False

    def _prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        required = list(self.spec.feature_columns)
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"Feature frame is missing columns: {missing}")
        prepared = self.spec.add_derived_roles(frame.loc[:, required])
        return prepared.loc[:, list(self.spec.roles)].astype(str)

    @staticmethod
    def _fit_scaler(matrix: np.ndarray) -> StandardScaler | None:
        if matrix.shape[1] == 0:
            return None
        scaler = StandardScaler(with_mean=True, with_std=True)
        scaler.fit(matrix)
        return scaler

    @staticmethod
    def _apply_scaler(matrix: np.ndarray, scaler: StandardScaler | None) -> np.ndarray:
        if scaler is None:
            return matrix
        output = scaler.transform(matrix)
        return np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)

    def _fit_name_subwords(
        self, role: str, values: pd.Series
    ) -> tuple[
        TfidfVectorizer | None,
        TruncatedSVD | None,
        StandardScaler | None,
        tuple[str, ...],
    ]:
        if not self.config.use_name_subwords:
            return None, None, None, ()
        if self.config.name_subword_max_rank < 1:
            raise ValueError("name_subword_max_rank must be positive when enabled.")
        unique_names = sorted(set(values.astype(str)))
        if len(unique_names) < 2:
            return None, None, None, ()
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            lowercase=True,
            norm="l2",
            sublinear_tf=True,
            dtype=np.float64,
        )
        unique_matrix = vectorizer.fit_transform(unique_names)
        rank = min(
            int(self.config.name_subword_max_rank),
            len(unique_names) - 1,
            unique_matrix.shape[1] - 1,
        )
        if rank < 1:
            return None, None, None, ()
        reducer = TruncatedSVD(
            n_components=rank,
            algorithm="randomized",
            n_iter=7,
            random_state=0,
        )
        reducer.fit(unique_matrix)
        embedded = reducer.transform(vectorizer.transform(values.astype(str)))
        scaler = self._fit_scaler(embedded)
        names = tuple(f"{role}::subword::svd_{index:02d}" for index in range(rank))
        return vectorizer, reducer, scaler, names

    def fit(self, reference_frame: pd.DataFrame) -> "FrozenReactionEncoder":
        prepared = self._prepare(reference_frame)
        feature_names: list[str] = []
        role_slices: dict[str, SliceSpec] = {}
        identity_slices: dict[str, SliceSpec] = {}
        cursor = 0

        for role in self.spec.roles:
            values = prepared[[role]]
            encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
                dtype=np.float64,
            )
            encoder.fit(values)
            identity_names = tuple(
                f"{role}::identity::{name}"
                for name in encoder.get_feature_names_out([role])
            )

            if self.config.use_name_rules:
                rules = self.rule_featurizer.transform(role, prepared[role])
                rule_scaler = self._fit_scaler(rules)
                rule_names = tuple(
                    f"{role}::rule::{name}" for name in self.rule_featurizer.FEATURE_NAMES
                )
            else:
                rule_scaler = None
                rule_names = ()

            (
                subword_vectorizer,
                subword_reducer,
                subword_scaler,
                subword_names,
            ) = self._fit_name_subwords(role, prepared[role])

            molecular = self.registry.transform(role, prepared[role])
            molecular_scaler = self._fit_scaler(molecular)
            molecular_names = tuple(
                f"{role}::molecular::{name}" for name in self.registry.feature_names
            )

            identity_start = cursor
            identity_stop = cursor + len(identity_names)
            identity_slices[role] = SliceSpec(identity_start, identity_stop)
            role_stop = (
                identity_stop
                + len(rule_names)
                + len(subword_names)
                + len(molecular_names)
            )
            role_slices[role] = SliceSpec(cursor, role_stop)
            cursor = role_stop

            feature_names.extend(identity_names)
            feature_names.extend(rule_names)
            feature_names.extend(subword_names)
            feature_names.extend(molecular_names)
            self._states[role] = _RoleState(
                encoder=encoder,
                rule_scaler=rule_scaler,
                subword_vectorizer=subword_vectorizer,
                subword_reducer=subword_reducer,
                subword_scaler=subword_scaler,
                molecular_scaler=molecular_scaler,
                identity_names=identity_names,
                rule_names=rule_names,
                subword_names=subword_names,
                molecular_names=molecular_names,
            )

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
            raise RuntimeError("FrozenReactionEncoder.fit must be called first.")
        prepared = self._prepare(frame)
        blocks: list[np.ndarray] = []

        for role in self.spec.roles:
            state = self._states[role]
            identity = state.encoder.transform(prepared[[role]]) * self.config.identity_scale
            role_blocks = [identity]

            if self.config.use_name_rules:
                rules = self.rule_featurizer.transform(role, prepared[role])
                rules = self._apply_scaler(rules, state.rule_scaler)
                role_blocks.append(rules * self.config.rule_scale)

            if state.subword_vectorizer is not None and state.subword_reducer is not None:
                subwords = state.subword_reducer.transform(
                    state.subword_vectorizer.transform(prepared[role].astype(str))
                )
                subwords = self._apply_scaler(subwords, state.subword_scaler)
                role_blocks.append(subwords * self.config.name_subword_scale)

            molecular = self.registry.transform(role, prepared[role])
            molecular = self._apply_scaler(molecular, state.molecular_scaler)
            role_blocks.append(molecular * self.config.molecular_scale)
            blocks.append(np.hstack(role_blocks))

        output = np.hstack(blocks).astype(np.float64, copy=False)
        if output.shape[1] != self.schema.n_features:
            raise RuntimeError(
                f"Encoder width changed: got {output.shape[1]}, expected {self.schema.n_features}."
            )
        if not np.all(np.isfinite(output)):
            raise FloatingPointError("Encoded design contains non-finite values.")
        return output

    def fit_transform(self, reference_frame: pd.DataFrame) -> np.ndarray:
        return self.fit(reference_frame).transform(reference_frame)

    def feature_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.schema is None:
            raise RuntimeError("Encoder is not fitted.")
        return pd.DataFrame(self.transform(frame), columns=self.schema.feature_names, index=frame.index)

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output)
        return output

    @classmethod
    def load(cls, path: str | Path) -> "FrozenReactionEncoder":
        loaded = joblib.load(path)
        if not isinstance(loaded, cls):
            raise TypeError(f"Artifact at {path} is not a FrozenReactionEncoder.")
        return loaded
