# 7. Running the Benchmarks

## Directory assumptions

The package does not bundle benchmark labels. A typical local layout is:

```text
project/
├── bgolearn_rxn_extension/
└── external_evaluation/
    ├── Buchwald_sub4/
    │   ├── buchwald_sub4_train.csv
    │   ├── buchwald_sub4_test_features.csv
    │   └── buchwald_sub4_test.csv       # offline oracle only
    └── Suzuki/
        ├── suzuki_train.csv
        ├── suzuki_test_features.csv
        └── suzuki_test.csv              # offline oracle only
```

## Step 1: install and test

```bash
cd bgolearn_rxn_extension
python -m pip install -e '.[chem,submission,dev]'
PYTHONPATH=src pytest
```

Use the `bgolearn` extra only when testing the compatibility backend.

## Step 2: audit public inputs

```bash
bgolearn-rxn audit \
  --config configs/buchwald_sub4.yaml \
  --train ../external_evaluation/Buchwald_sub4/buchwald_sub4_train.csv \
  --candidates ../external_evaluation/Buchwald_sub4/buchwald_sub4_test_features.csv
```

Inspect:

- initial count;
- candidate count;
- unique level count per role;
- target incumbent;
- feasible substrate-pair count for Suzuki.

Do not proceed when the audit disagrees with the benchmark documentation.

## Step 3: smoke test

Run two or three steps before launching all seeds:

```bash
bgolearn-rxn simulate \
  --config configs/buchwald_sub4.yaml \
  --train ../external_evaluation/Buchwald_sub4/buchwald_sub4_train.csv \
  --candidates ../external_evaluation/Buchwald_sub4/buchwald_sub4_test_features.csv \
  --oracle ../external_evaluation/Buchwald_sub4/buchwald_sub4_test.csv \
  --seed 100 \
  --budget 2 \
  --backend native \
  --output-dir smoke/buchwald
```

Confirm:

- the first target incumbent equals the best of the seven target rows, not the pooled maximum;
- exactly two distinct IDs were queried;
- output conditions match candidate rows;
- model weights before and after each query sum to one;
- the encoder schema hash is present.

## Step 4: run one complete seed

```bash
bgolearn-rxn simulate \
  --config configs/suzuki.yaml \
  --train ../external_evaluation/Suzuki/suzuki_train.csv \
  --candidates ../external_evaluation/Suzuki/suzuki_test_features.csv \
  --oracle ../external_evaluation/Suzuki/suzuki_test.csv \
  --seed 100 \
  --budget 40 \
  --backend native \
  --output-dir runs/suzuki
```

Optionally save all posterior snapshots:

```bash
... --save-candidate-scores
```

This is useful for debugging but produces larger artifacts and must not be used to tune against test outcomes.

## Step 5: run all seeds

```bash
bgolearn-rxn run-all \
  --config configs/suzuki.yaml \
  --train ../external_evaluation/Suzuki/suzuki_train.csv \
  --candidates ../external_evaluation/Suzuki/suzuki_test_features.csv \
  --oracle ../external_evaluation/Suzuki/suzuki_test.csv \
  --backend native \
  --output-dir runs/suzuki
```

`summary.json` reports:

- mean, standard deviation, and t-based 95% interval for first-query yield;
- the same for final best yield and best-so-far AUC;
- fraction of runs reaching 95% of the organizer-side global best;
- conditional t95 summary among successful runs.

## Step 6: validate every submission

```bash
for f in runs/suzuki/seed_*.pt; do
  bgolearn-rxn validate \
    --config configs/suzuki.yaml \
    --train ../external_evaluation/Suzuki/suzuki_train.csv \
    --candidates ../external_evaluation/Suzuki/suzuki_test_features.csv \
    --submission "$f"
done
```

Also verify that exactly the configured 20 filenames are present.

## Output record

A trajectory row contains:

```json
{
  "step": 1,
  "query_index": 906,
  "condition": {
    "Electrophile": "...",
    "Nucleophile": "...",
    "Ligand": "...",
    "Base": "...",
    "Solvent": "..."
  },
  "observed_yield": 92.97,
  "predicted_yield": 71.3,
  "predicted_std": 18.2,
  "target_incumbent_before_query": 71.51,
  "acquisition_name": "posterior_mean",
  "acquisition_value": 71.3,
  "component_predictions": {
    "gp": {"mean": 74.1, "std": 12.0},
    "interaction": {"mean": 72.5, "std": 18.8},
    "trees": {"mean": 63.0, "std": 21.4}
  },
  "model_weights_before_query": {
    "gp": 0.4,
    "interaction": 0.4,
    "trees": 0.2
  },
  "model_weights_after_query": {
    "gp": 0.44,
    "interaction": 0.39,
    "trees": 0.17
  },
  "prior_weight": 0.0,
  "seed": 100
}
```

The richer fields are optional for the competition but important for reproducibility and diagnosis.

## Runtime guidance

The exact GP is inexpensive at 35–75 rows, but hyperparameter optimization across 20 × 40 fits can still dominate runtime. Recommended controls:

- warm-start GP hyperparameters;
- use `refit_interval: 2` or `3`;
- keep 2–4 restarts;
- run seeds in separate processes;
- set tree `n_jobs=1` inside each process to avoid oversubscription;
- use native BLAS thread limits when parallelizing seeds.

Example:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

Do not parallelize rounds within a campaign: each query depends on prior feedback.

## Re-running a seed

A repeated run with the same:

- source files;
- config;
- package version;
- structure registry;
- prior CSV;
- Python/scikit-learn/RDKit stack;
- seed;

should reproduce the trajectory. Compare `config_hash`, `feature_schema_hash`, and queried IDs before comparing floating-point diagnostics.

## Using the Bgolearn backend

After installing Bgolearn 3.1.0:

```bash
... --backend bgolearn
```

Run one seed first and compare its trajectory with the native backend. The custom predictions should match. Tiny floating-point differences may change exact ties; query IDs should otherwise agree.
