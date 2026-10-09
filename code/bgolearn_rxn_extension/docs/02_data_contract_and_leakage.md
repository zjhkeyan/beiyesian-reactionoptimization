# 2. Data Contract and Leakage Controls

## Participant-side inputs

The optimizer may read:

- the initial labeled training CSV;
- the unlabeled test-feature candidate CSV;
- `options.json` for names and fixed metadata;
- externally curated, label-free chemical structures or descriptors;
- an audited, label-free external prior.

The optimizer may not read the test `Yield` column except through the one-query-at-a-time oracle boundary.

## Buchwald contract

The recommended configuration expects:

```text
Product, Reactant2, Ligand, Additive, Base, Yield
```

The target product is:

```text
N-(4-ethylphenyl)-4-methylaniline
```

The candidate file must contain the same feature columns without `Yield`.

The training rows from the four auxiliary products are legitimate training observations. They are not target-task incumbents and must not be confused with target feedback.

## Suzuki contract

The recommended configuration expects:

```text
Electrophile, Nucleophile, Ligand, Base, Solvent, Yield
```

The candidate file contains the same feature columns without `Yield`.

The code derives:

```text
SubstratePair = Electrophile + " || " + Nucleophile
```

This is a model feature, not a generated feasibility rule. Only rows in the candidate CSV can be selected.

## Immutable candidate IDs

When the public candidate CSV has no `query_index`, IDs are the original zero-based row numbers. These IDs remain attached to candidates even after filtering or reordering.

When an explicit `query_index` is present, it must be numeric and unique.

The ID is never used as a model feature.

## Audit checks

`load_reaction_dataset` performs the following checks:

1. Required columns exist.
2. Candidate `Yield` is absent.
3. Candidate IDs are unique.
4. Candidate conditions are unique.
5. Training conditions are unique.
6. Target-task training conditions do not overlap candidates.
7. Training yields are finite and in `[0, 100]`.
8. Buchwald candidate rows all belong to the configured target product.

`OfflineTableOracle` additionally checks:

1. Labeled lookup IDs exactly match public candidate IDs.
2. Every feature value matches the public candidate row at that ID.
3. Lookup yields are finite and in `[0, 100]`.
4. A candidate cannot be queried twice.

## Metadata caveats

The benchmark documentation should be treated as the authoritative reaction description. In the supplied materials:

- the Buchwald README lists palladium diacetate as fixed, while its options file does not expose a catalyst key;
- the earlier Suzuki product naming error has been corrected in `metadata.json`;
- names mix systematic names, common naming conventions, salts, and several SMILES-like strings.

Consequences:

- fixed Product may be included as constant reaction context but must not be
  treated as a selectable Suzuki variable;
- cross-dataset reaction embeddings should use the corrected coupled product,
  `6-(5-methyl-1-(tetrahydro-2H-pyran-2-yl)-1H-indazol-4-yl)quinoline`, only
  after its standardized structure has been reviewed;
- the Buchwald catalyst should be added to external reaction metadata;
- automated name-to-structure resolution must be manually verified.

## What counts as leakage

The following are leakage even when performed indirectly:

- fitting a featurizer to test yields;
- selecting molecular descriptors because they perform well on the full table;
- choosing interactions after inspecting test-label heat maps;
- selecting ensemble weights from completed public trajectories;
- asking an LLM for the known best conditions on the named dataset;
- retrieving the original source table and using its yields;
- using candidate row order as a proxy for source-table chemistry;
- tuning a prior-strength schedule against test outcomes.

## Safe transductive operations

The following operations use no labels and are safe under the stated data contract:

- one-hot category discovery from the complete unlabeled pool;
- standardization fitted on initial + candidate features;
- PCA fitted on feature blocks without yields;
- candidate topology and feasible-pair analysis;
- descriptor computation from reviewed molecular structures;
- diversity calculations over candidate features.

## Row-order precautions

The source tables retain original row order. The extension therefore:

- never encodes row number;
- never encodes option-list position;
- uses IDs only for lookup and deterministic tie-breaking;
- recommends that organizers shuffle hidden candidate files independently.

## Separation of development and evaluation

A defensible development workflow uses:

- held-out Buchwald products;
- held-out additive families;
- held-out Suzuki substrate pairs;
- unrelated reaction datasets;
- synthetic categorical landscapes with known low-rank interactions;
- prequential calibration checks.

The public lookup oracle should be used to produce final benchmark trajectories, not to iterate on model design.
