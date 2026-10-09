# Validation Report

Date: 2026-07-23

## Environment

The native implementation was exercised with:

- Python 3.13 in the artifact runtime;
- NumPy 2.3.5;
- pandas 2.2.3;
- SciPy 1.17.0;
- scikit-learn 1.8.0;
- RDKit 2025.09.4 available, but no curated registry enabled in the baseline presets;
- PyTorch 2.10.0 for `.pt` serialization.

## Automated tests

Command:

```bash
PYTHONPATH=src pytest -q
```

Result:

```text
18 passed
```

Coverage includes stable LogEI, participant/oracle separation, target-specific incumbents, immutable IDs, frozen features, complete no-repeat campaigns, prequential weights, submission validation, and an interface-level stock-Bgolearn bridge test.

## Full-budget native smoke campaigns

These runs are execution checks, not model-selection evidence and not a claim of state of the art. The public lookup labels were used only through `OfflineTableOracle`.

### Buchwald_sub4, seed 100, 40 queries

```text
initial_round_found_best: 74.97052076
best_found:              82.71053034
AUC_best_so_far:         77.07021716025
t95:                     37
global_best:             86.59757822
elapsed:                  16.55 seconds
peak RSS:                 about 389 MB
```

The generated `.pt` trajectory passed `bgolearn-rxn validate` with 40 unique, condition-aligned candidate IDs.

## Current-environment reproduction note

The same presets were rerun on 2026-07-28 under Python 3.12.4, NumPy 1.26.4,
and scikit-learn 1.4.2. The trajectories executed correctly but did not exactly
match the reference metrics above, which were generated with Python 3.13,
NumPy 2.3.5, and scikit-learn 1.8.0. This confirms that the reference lock file
must be enforced for exact trajectory reproduction; the YAML
`strict_reproducibility` flag does not itself install or verify dependencies.

### Suzuki, seed 100, 40 queries

```text
initial_round_found_best: 92.97
best_found:              98.69
AUC_best_so_far:         97.42075
t95:                     2
global_best:             99.9
elapsed:                  28.43 seconds
peak RSS:                 about 423 MB
```

The generated `.pt` trajectory passed `bgolearn-rxn validate` with 40 unique, condition-aligned candidate IDs.

## Bgolearn compatibility status

The compatibility bridge was originally validated with a mocked implementation
of the stock custom-model API. On 2026-07-28 it was additionally exercised
against the supplied local Bgolearn source at commit `03a2bc6`, six commits
after tag `V3.1.0`.

A two-query Suzuki campaign was run through both `native` and `bgolearn`
backends. The candidate IDs were identical at both steps, both metric payloads
matched, and both generated `.pt` files passed submission validation.

Stock Bgolearn writes a full candidate-prediction CSV and prints its banner on
each refit. The two-step compatibility smoke produced an approximately 11 MB
prediction file, so the native backend remains recommended for full campaigns.

The interface-level test remains available with:

```bash
PYTHONPATH=src pytest tests/test_bgolearn_adapter.py
```

Use `scripts/compare_runs.py` when repeating parity tests after a Bgolearn
version change.

## Remaining validation before a leaderboard submission

- Run all 20 seeds from a frozen commit and environment.
- Validate every `.pt` file.
- Run the required leakage and candidate-order invariance tests in the competition harness.
- Curate and verify the molecular registry before enabling RDKit features.
- Evaluate model and prior ablations on permitted proxy or hidden-development tasks, not the public lookup labels.
