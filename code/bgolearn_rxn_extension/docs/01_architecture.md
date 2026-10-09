# 1. Architecture

## Objective

The extension is designed for a finite, discrete reaction candidate pool with one queried experiment per round. Its primary objective is not generic supervised prediction; it is to maximize early and final best-so-far yield under a 40-query budget while preserving reproducibility and avoiding test-label leakage.

The implementation separates six responsibilities:

1. **Data contract** — load only participant-side files and preserve immutable IDs.
2. **Representation** — fit a label-free reaction encoder once.
3. **Surrogates** — maintain several probabilistic hypotheses about the landscape.
4. **Combination** — update model weights from honest sequential predictions.
5. **Policy** — select one remaining candidate according to the benchmark stage.
6. **Oracle boundary** — reveal exactly one yield only after an ID is committed.

## Nonnegotiable invariants

### Invariant A: test labels never enter model construction

The only object allowed to read the labeled test CSV is `OfflineTableOracle`. No other class accepts a lookup path.

```text
train.csv ───────────────► encoder/model/policy
unlabeled candidates.csv ► encoder/model/policy
labeled test.csv ────────► OfflineTableOracle only
```

### Invariant B: the feature map is frozen

`FrozenReactionEncoder.fit` is called once on:

```text
initial participant features + unlabeled candidate features
```

It is not refitted after candidates are queried and removed. This preserves a stable geometry across all rounds.

### Invariant C: acquisition uses immutable IDs

Every candidate receives `query_index` from its public CSV row number unless a valid explicit ID is supplied. Tie-breaking is performed after sorting tied points by immutable ID, making a recommendation independent of current DataFrame row order.

### Invariant D: the Buchwald incumbent is task-specific

The pooled 35-row Buchwald training set contains five products. All rows may inform the surrogate, but expected improvement must compare against the best observed yield for the target product only.

### Invariant E: candidates are never synthesized from option lists

The supplied candidate CSV defines feasibility. This matters because:

- the Buchwald table has fewer rows than its nominal full Cartesian product;
- the Suzuki table contains only a subset of electrophile–nucleophile pairs.

### Invariant F: ensemble weights are updated prequentially

A component model is rewarded or penalized using the prediction it made before the new yield was known. In-sample residuals are not used to decide sequential ensemble weights.

## Object graph

```text
RunConfig
  ├── DatasetConfig
  ├── FeatureConfig
  ├── GPConfig
  ├── InteractionModelConfig
  ├── TreeConfig
  ├── EnsembleConfig
  ├── PolicyConfig
  └── RuntimeConfig

DatasetSpec ─► ReactionDataset
                  │
                  ▼
           prepare_problem(...)
                  │
                  ├── FrozenReactionEncoder
                  ├── X_initial
                  ├── X_candidates
                  └── reference_X

build_ensemble(...)
  ├── StructuredANOVAGP
  ├── BayesianLowRankInteractionSurrogate
  └── BootstrapExtraTreesSurrogate
            │
            ▼
     PrequentialEnsemble
            │
            ▼
 NativeBackend or BgolearnCompatibilityBackend
            │
            ▼
      MetricAwarePolicy
            │
            ▼
       CampaignRunner
            │
            ▼
         YieldOracle
```

## One round in detail

At round \(t\):

1. Fit each surrogate to all currently observed rows.
2. Predict every remaining candidate.
3. Form mixture mean, mixture variance, and between-model disagreement.
4. Compute the target-task incumbent.
5. Generate acquisition scores according to the current campaign stage.
6. Optionally add a decaying, standardized external prior.
7. Select a unique immutable candidate ID.
8. Commit the ID.
9. Query exactly one yield.
10. Update ensemble weights from pre-query component predictions.
11. Append the selected condition and yield to observations.
12. Remove the selected ID from the pool.
13. Persist a complete decision record.

## Why a persistent runner matters

`CampaignRunner` owns the prequential ensemble state. Reconstructing it every round would reset cumulative model evidence. A production integration should either:

- keep one runner process alive; or
- serialize and restore `ensemble.state_dict()` plus all observed rows and remaining IDs.

## Native and Bgolearn modes

### Native mode

The preferred mode uses the extension directly. It preserves model objects, diagnostics, and posterior interfaces.

### Bgolearn compatibility mode

The compatibility mode passes frozen numeric matrices into Bgolearn's custom surrogate path. It exists to integrate with existing Bgolearn workflows and logs. It intentionally does not delegate acquisition ranking to stock Bgolearn.

## Failure policy

The code fails loudly on:

- forbidden labels in candidate files;
- duplicate conditions;
- train/candidate overlap;
- nonfinite predictions;
- nonunique IDs;
- candidate/oracle row mismatch;
- repeated oracle queries;
- incomplete structure registries in strict mode;
- malformed submission trajectories.

Silent fallbacks are avoided because hidden inconsistencies can produce deceptively strong offline results.
