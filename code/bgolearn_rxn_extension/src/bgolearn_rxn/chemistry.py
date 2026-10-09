from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RegistryRecord:
    role: str
    component: str
    canonical_smiles: str | None


class ChemicalRegistry:
    """Verified molecular side-information keyed by ``(role, component)``.

    The benchmark distributes names rather than guaranteed machine-readable
    structures. This class therefore refuses to infer structures from names.
    Users provide a reviewed registry with canonical SMILES; unresolved rows
    receive explicit missingness features rather than fabricated molecules.
    """

    REQUIRED_COLUMNS = ("role", "component", "canonical_smiles")

    def __init__(
        self,
        path: str | Path | None,
        *,
        use_rdkit: bool = True,
        morgan_radius: int = 2,
        morgan_bits: int = 256,
        missing_policy: str = "warn",
    ) -> None:
        if missing_policy not in {"warn", "error", "zeros"}:
            raise ValueError("missing_policy must be one of: warn, error, zeros")
        self.path = None if path is None else Path(path)
        self.use_rdkit = use_rdkit
        self.morgan_radius = int(morgan_radius)
        self.morgan_bits = int(morgan_bits)
        self.missing_policy = missing_policy
        self._records: dict[tuple[str, str], RegistryRecord] = {}
        self._warned: set[tuple[str, str]] = set()
        self._rdkit_available = False
        self._load()

    @property
    def enabled(self) -> bool:
        return bool(self._records) and self.use_rdkit and self._rdkit_available

    @property
    def feature_names(self) -> tuple[str, ...]:
        if self.path is None:
            return ()
        names = [f"morgan_{i:03d}" for i in range(self.morgan_bits)]
        names.extend(
            [
                "mol_wt",
                "mol_logp",
                "tpsa",
                "hbd",
                "hba",
                "ring_count",
                "fraction_csp3",
                "heavy_atom_count",
                "formal_charge",
                "registry_missing",
            ]
        )
        return tuple(names)

    def _load(self) -> None:
        if self.path is None:
            return
        if not self.path.exists():
            raise FileNotFoundError(f"Chemical registry not found: {self.path}")

        table = pd.read_csv(self.path, keep_default_na=False)
        missing_columns = [column for column in self.REQUIRED_COLUMNS if column not in table]
        if missing_columns:
            raise ValueError(f"Registry is missing required columns: {missing_columns}")

        duplicated = table.duplicated(subset=["role", "component"], keep=False)
        if duplicated.any():
            examples = table.loc[duplicated, ["role", "component"]].head().to_dict("records")
            raise ValueError(f"Registry has duplicate role/component keys, e.g. {examples}")

        for row in table.itertuples(index=False):
            role = str(getattr(row, "role")).strip()
            component = str(getattr(row, "component")).strip()
            smiles_raw = str(getattr(row, "canonical_smiles")).strip()
            smiles = smiles_raw or None
            self._records[(role, component)] = RegistryRecord(role, component, smiles)

        if self.use_rdkit:
            try:
                import rdkit  # noqa: F401

                self._rdkit_available = True
            except ImportError:
                warnings.warn(
                    "RDKit is unavailable; molecular registry rows will be represented "
                    "only by registry_missing indicators.",
                    RuntimeWarning,
                    stacklevel=2,
                )

    def _lookup(self, role: str, component: str) -> RegistryRecord | None:
        direct = self._records.get((role, component))
        if direct is not None:
            return direct
        # A reviewed global record can be declared with role='*'.
        return self._records.get(("*", component))

    def _missing_vector(self) -> np.ndarray:
        vector = np.zeros(len(self.feature_names), dtype=float)
        vector[-1] = 1.0
        return vector

    def _handle_missing(self, role: str, component: str, reason: str) -> np.ndarray:
        key = (role, component)
        if self.missing_policy == "error":
            raise ValueError(f"Missing molecular registry data for {key}: {reason}")
        if self.missing_policy == "warn" and key not in self._warned:
            warnings.warn(
                f"Missing molecular registry data for role={role!r}, "
                f"component={component!r}: {reason}",
                RuntimeWarning,
                stacklevel=3,
            )
            self._warned.add(key)
        return self._missing_vector()

    def _featurize_smiles(self, smiles: str) -> np.ndarray:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import Descriptors, Lipinski, rdFingerprintGenerator

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("RDKit could not parse canonical_smiles")

        generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.morgan_radius,
            fpSize=self.morgan_bits,
        )
        fingerprint = generator.GetFingerprint(mol)
        bits = np.zeros(self.morgan_bits, dtype=float)
        DataStructs.ConvertToNumpyArray(fingerprint, bits)

        formal_charge = float(sum(atom.GetFormalCharge() for atom in mol.GetAtoms()))
        descriptors = np.array(
            [
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.TPSA(mol),
                float(Lipinski.NumHDonors(mol)),
                float(Lipinski.NumHAcceptors(mol)),
                float(Lipinski.RingCount(mol)),
                Descriptors.FractionCSP3(mol),
                float(Descriptors.HeavyAtomCount(mol)),
                formal_charge,
                0.0,
            ],
            dtype=float,
        )
        return np.concatenate([bits, descriptors])

    def transform(self, role: str, components: pd.Series) -> np.ndarray:
        if self.path is None:
            return np.empty((len(components), 0), dtype=float)
        rows: list[np.ndarray] = []
        for raw_component in components.astype(str):
            component = raw_component.strip()
            if component.lower() == "nothing":
                rows.append(self._missing_vector())
                continue
            record = self._lookup(role, component)
            if record is None:
                rows.append(self._handle_missing(role, component, "no reviewed registry row"))
                continue
            if not record.canonical_smiles:
                rows.append(self._handle_missing(role, component, "canonical_smiles is blank"))
                continue
            if not self.use_rdkit or not self._rdkit_available:
                rows.append(self._handle_missing(role, component, "RDKit features are disabled"))
                continue
            try:
                rows.append(self._featurize_smiles(record.canonical_smiles))
            except Exception as exc:  # registry validation should fail explicitly when strict
                rows.append(self._handle_missing(role, component, str(exc)))
        if not rows:
            return np.empty((0, len(self.feature_names)), dtype=float)
        return np.vstack(rows)


def build_registry_template(
    frames: list[pd.DataFrame],
    roles: tuple[str, ...],
    output_path: str | Path,
) -> Path:
    """Write a deduplicated structure-registry template for manual curation."""

    records: list[dict[str, str]] = []
    for role in roles:
        values: set[str] = set()
        for frame in frames:
            if role in frame.columns:
                values.update(frame[role].dropna().astype(str).tolist())
        for component in sorted(values):
            records.append(
                {
                    "role": role,
                    "component": component,
                    "canonical_smiles": "",
                    "inchi_key": "",
                    "source": "",
                    "review_status": "unreviewed",
                    "notes": "",
                }
            )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output, index=False)
    return output
