# 12. Production Checklist

Use this checklist before generating final benchmark trajectories or connecting the optimizer to a laboratory.

## Data

- [ ] Training path points to the participant initial CSV.
- [ ] Candidate path points to the unlabeled `test_features` CSV.
- [ ] Oracle path is isolated and passed only to `OfflineTableOracle`.
- [ ] Candidate audit reports the expected row count.
- [ ] No candidate `Yield` column is present.
- [ ] Query IDs are unique and immutable.
- [ ] Candidate conditions are unique.
- [ ] No target-task train/candidate overlap exists.
- [ ] All yields lie in `[0, 100]`.
- [ ] Candidate row order is not encoded.

## Benchmark semantics

- [ ] Buchwald target product string is exact.
- [ ] Buchwald incumbent is target-specific.
- [ ] All 35 Buchwald training rows are available to the multitask model.
- [ ] Suzuki `SubstratePair` is derived from constituents.
- [ ] Suzuki acquisition is restricted to supplied feasible rows.
- [ ] Fixed Product is omitted from within-Suzuki feature ranking.
- [ ] Catalyst and corrected product metadata are used for cross-dataset embeddings.

## Features

- [ ] Encoder is fitted once on initial + unlabeled pool.
- [ ] Encoder is never refitted after a query.
- [ ] `Normalize=False` is used in Bgolearn compatibility mode.
- [ ] Structure registry is reviewed and hashed.
- [ ] `missing_registry_policy=error` is used for final molecular runs.
- [ ] RDKit and descriptor versions are recorded.
- [ ] `Nothing` is represented as absence, not a molecule.
- [ ] All external embeddings are frozen or strongly constrained.

## Models

- [ ] GP roles and interaction pairs match the dataset preset.
- [ ] GP hyperparameters warm-start across rounds.
- [ ] Hyperparameter restarts and refit interval are fixed before evaluation.
- [ ] Factorization ranks are selected on proxy tasks.
- [ ] Tree model has a limited initial weight.
- [ ] Predictive standard deviations are finite and positive.
- [ ] Prequential weight floor is below `1 / n_models`.
- [ ] Model-weight updates use pre-query predictions.

## Acquisition

- [ ] Query 1 uses posterior mean.
- [ ] Stable LogEI is used in middle rounds.
- [ ] `best_f` is computed from currently observed target rows only.
- [ ] Challenge gates compare quantities in yield units.
- [ ] Final rounds use reduced exploration.
- [ ] Prior scores align by query ID.
- [ ] Prior weight decays rapidly.
- [ ] Tie-breaking uses immutable IDs and a local seed.
- [ ] A queried candidate is removed before the next recommendation.

## External knowledge

- [ ] No LLM prompt names the public source dataset.
- [ ] No prompt asks for known optimum or exact benchmark yield.
- [ ] Model identifier, date, prompts, and raw outputs are archived.
- [ ] Prior CSV was fixed before campaign execution.
- [ ] Prior creation did not read lookup labels.
- [ ] A no-external baseline is retained.
- [ ] Public-table contamination risk is disclosed.

## Reproducibility

- [ ] Python and package versions are recorded.
- [ ] Git commit is recorded.
- [ ] Config and feature-schema hashes are present.
- [ ] Structure registry and prior hashes are recorded.
- [ ] All 20 configured seeds are run independently.
- [ ] No feedback is shared between seeds.
- [ ] CPU/GPU deterministic settings are documented.
- [ ] Thread counts are controlled during parallel seed execution.

## Output

- [ ] Exactly 20 `.pt` files are present.
- [ ] Filenames match `seed_100.pt` through `seed_2000.pt`.
- [ ] Every trajectory has exactly 40 rows.
- [ ] Steps are numbered 1–40.
- [ ] Every run has 40 unique query IDs.
- [ ] Every condition matches its candidate row.
- [ ] Observed yield is recorded after each query.
- [ ] Human-readable JSON mirrors each `.pt` file.
- [ ] `bgolearn-rxn validate` passes for every file.

## Scientific review

- [ ] Required ablations have been run on permitted proxy tasks.
- [ ] Calibration is reported alongside optimization metrics.
- [ ] Performance is reported at queries 1, 5, 10, 20, and 40.
- [ ] Confidence intervals use paired runs where appropriate.
- [ ] Runtime and external-model cost are disclosed.
- [ ] No claim of universal SOTA is made from public tables alone.

## Laboratory-specific additions

- [ ] Candidate reservation is atomic.
- [ ] Instrument worklist preserves query ID.
- [ ] Reagent stock and feasibility are rechecked before execution.
- [ ] Failed and censored runs have predefined handling.
- [ ] QC status is separate from measured yield.
- [ ] Replicates and batch identifiers are stored.
- [ ] Model update occurs only after accepted measurement.
- [ ] Safety and waste constraints can veto acquisition.
