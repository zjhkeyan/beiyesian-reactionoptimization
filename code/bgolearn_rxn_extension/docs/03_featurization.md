# 3. Featurization

## Principle

Reaction components are unordered chemical identities with distinct mechanistic roles. They should not be represented as arbitrary consecutive integers in a Euclidean RBF kernel.

`FrozenReactionEncoder` constructs a separate feature block for every role:

```text
role block = exact identity + conservative name rules + optional name subwords + optional reviewed molecular features
```

The encoder concatenates role blocks but preserves their column slices in `FeatureSchema`, allowing models to apply role-specific kernels and interactions.

## Fit-once protocol

```python
reference = concat(initial_features, unlabeled_candidate_features)
encoder.fit(reference)
X_initial = encoder.transform(initial_features)
X_candidates = encoder.transform(candidate_features)
```

Do not call `fit` again after a query.

## Exact identity block

A dense one-hot encoder with `handle_unknown="ignore"` represents exact category identity. It is essential for within-grid interpolation and for preserving distinctions that fingerprints may blur.

Identity features are scaled by `identity_scale`, default `1.0`.

## Conservative name-rule block

The package provides immediately runnable, label-free flags such as:

- chloride, bromide, iodide, triflate;
- boronic acid, boronate ester, trifluoroborate;
- phosphine, bisphosphine, ferrocene, xanthene, adamantyl;
- hydroxide, carbonate, phosphate, fluoride, alkoxide;
- amide, nitrile, alcohol, and ether solvent cues;
- oxazole, benzoxazole, carboxylate, furan, thiophene, and pyrrole cues.

These features are deliberately modest. They bridge exact one-hot categories and a fully curated molecular registry; they are not claimed to be complete chemistry descriptors.

Rule features are standardized once on the reference pool and scaled by `rule_scale`.

## Optional name-subword block

The disabled-by-default name-subword block learns lexical coordinates without
using yields. It fits one `char_wb` TF-IDF model per role on sorted unique names
from the initial observations and unlabeled candidate pool, using character
3-5 grams. A deterministic compact SVD retains at most eight coordinates and
is further capped by the number of unique levels and vocabulary width. The
coordinates are standardized on the same frozen reference pool and scaled by
`name_subword_scale`.

This block captures systematic fragments not enumerated by the fixed name
rules while retaining exact identity as the prefix of each role block. Like all
encoder state, its vocabulary, projection, and scaler are fitted only once and
are never updated during sequential optimization.

## Reviewed molecular block

`ChemicalRegistry` is keyed by `(role, component)` and accepts canonical SMILES only from a reviewed CSV.

With RDKit installed, it computes:

- Morgan fingerprints;
- molecular weight;
- logP;
- topological polar surface area;
- hydrogen-bond donors and acceptors;
- ring count;
- fraction sp3;
- heavy-atom count;
- formal charge;
- explicit registry-missing indicator.

Molecular descriptors are standardized once. Fingerprints currently enter the same frozen block; a more advanced extension can give fingerprint columns a dedicated Tanimoto kernel.

## Why structures are not inferred automatically

Name-to-structure services can return:

- the wrong salt;
- a different tautomer;
- an incorrect stereoisomer;
- an unresolved common name;
- a plausible but wrong structure for long ligand names.

For a small benchmark registry, manual review is inexpensive and safer than silent resolution errors.

## Registry workflow

Generate a template:

```bash
bgolearn-rxn make-registry \
  --config configs/suzuki.yaml \
  --train suzuki_train.csv \
  --candidates suzuki_test_features.csv \
  --output registry_templates/suzuki_registry.csv
```

Curate the following fields:

| Field | Meaning |
|---|---|
| `role` | Reaction role or `*` for a global identity |
| `component` | Exact benchmark string |
| `canonical_smiles` | Reviewed canonical structure |
| `inchi_key` | Cross-check identifier |
| `source` | Supplier, SI, database, or manual derivation |
| `review_status` | Use `reviewed` only after verification |
| `notes` | Salt, protonation, tautomer, or ambiguity notes |

For final runs, set:

```yaml
missing_registry_policy: error
```

## `Nothing`

`Nothing` is not a molecule. It receives:

- zero molecular fingerprint and descriptors;
- an explicit absence/missing indicator;
- an exact identity category;
- a name-rule `is_nothing` flag.

## Derived roles

Suzuki adds `SubstratePair` because the candidate topology contains only selected electrophile–nucleophile pairs. The pair role supports strong within-pair interpolation.

The model also retains constituent Electrophile and Nucleophile roles, which are needed to generalize to an unobserved pair.

## Buchwald representation

Recommended roles:

- `Product` — task identity, not a condition role;
- `Reactant2` — exact aryl halide plus leaving-group and molecular cues;
- `Ligand` — exact identity plus ligand-family/structure information;
- `Additive` — exact identity plus heterocycle/substituent structure;
- `Base` — exact identity plus base-family cues.

The additive molecular block is likely to be especially valuable because initial target coverage is sparse.

## Suzuki representation

Recommended roles:

- `Electrophile` — scaffold and leaving group;
- `Nucleophile` — scaffold and boron form;
- `SubstratePair` — exact feasible-pair residual;
- `Ligand`;
- `Base`;
- `Solvent`.

## Extending the encoder

A new role featurizer should satisfy:

1. no yield access;
2. deterministic output;
3. fixed column order;
4. transform support for all candidate rows;
5. explicit missingness;
6. serialization support;
7. unit tests proving coordinate stability.

Good additions include:

- KRAKEN-style phosphorus-ligand descriptors;
- solvent scales;
- quantum or semiempirical descriptors calculated consistently;
- frozen reaction-language embeddings;
- DRFP/RXNFP fingerprints;
- role-specific graph embeddings.

Add them as separate blocks so ablations can determine their contribution.
