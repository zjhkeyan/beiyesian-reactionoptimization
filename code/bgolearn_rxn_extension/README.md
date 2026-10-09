# Bgolearn-RxN

A reference implementation of a **best-practice reaction-condition optimization extension for Bgolearn**, designed around the public Buchwald–Hartwig `Buchwald_sub4` and Suzuki–Miyaura benchmarks.

The package keeps Bgolearn as an optional finite-pool campaign controller, while replacing the parts that are most important for sparse chemical optimization:

- immutable, role-aware reaction featurization;
- target-aware multitask transfer for Buchwald;
- constrained substrate-pair/tensor structure for Suzuki;
- structured Gaussian-process and low-rank interaction surrogates;
- calibrated ensemble uncertainty;
- numerically stable Log Expected Improvement;
- metric-aware query scheduling;
- strict separation of participant-side inputs and organizer-side labels;
- reproducible `.pt` trajectory generation and validation.

This is an engineering reference and research starting point—not a claim that one fixed configuration is universally optimal. The hidden benchmark should decide which ablations generalize.

## Contents

- [`configs/buchwald_sub4.yaml`](configs/buchwald_sub4.yaml): recommended public Buchwald preset.
- [`configs/suzuki.yaml`](configs/suzuki.yaml): recommended public Suzuki preset.
- [`configs/buchwald_sub4_metric_frontier.yaml`](configs/buchwald_sub4_metric_frontier.yaml): opt-in Buchwald component-ARD transfer preset selected on train-only official-target proxies.
- [`configs/suzuki_metric_frontier.yaml`](configs/suzuki_metric_frontier.yaml): opt-in Suzuki single-GP preset selected on train-only random and block holdouts.
- [`src/bgolearn_rxn`](src/bgolearn_rxn): implementation.
- [`docs`](docs): design rationale, leakage controls, model details, and extension guidance.
- [`tests`](tests): unit and end-to-end tests.
- [`registry_templates`](registry_templates): review-first molecular structure registries.
- [`scripts`](scripts): benchmark launch examples.

## Design in one diagram

```text
participant train.csv + unlabeled test_features.csv
                     │
                     ▼
       strict data audit + immutable query IDs
                     │
                     ▼
  fit-once role-aware encoder on train + unlabeled pool
       │ exact identity  │ name rules │ reviewed SMILES
       └─────────────────┴────────────┴────────────────┐
                                                       ▼
      ┌────────────────────────────────────────────────────────┐
      │ persistent probabilistic ensemble                      │
      │  1. structured ANOVA chemistry GP                      │
      │  2. Bayesian low-rank interaction/tensor model         │
      │  3. bootstrapped Extra Trees misspecification hedge    │
      └────────────────────────────────────────────────────────┘
                             │
                             ▼
           mixture mean + within-model uncertainty
                     + between-model disagreement
                             │
                             ▼
        query 1: posterior mean
        middle: stable full-pool LogEI
        periodic: gated model-challenge query
        final: low-exploration mean/UCB
                             │
                             ▼
         commit immutable query_index before oracle call
                             │
                             ▼
     laboratory / isolated offline lookup returns one Yield
                             │
                             ▼
 prequential model-weight update, append observation, repeat
```

## Why this is an extension rather than a stock Bgolearn recipe

Bgolearn is valuable as a practical Bayesian-optimization workflow and supports custom regression models. Its stock defaults, however, are general-purpose. This project makes several reaction-specific choices:

1. **No round-dependent normalization.** Features are fitted once and frozen. The stock bridge always calls Bgolearn with `Normalize=False`.
2. **No response-weighted resampling.** The bridge always uses `Dynamic_W=False`.
3. **No per-round cross-validation.** The bridge uses `CV_test=False`; calibration is evaluated separately.
4. **No integer geometry for unordered reagents.** Categories are represented by exact identities and role-specific chemistry blocks.
5. **No generic isotropic RBF as the sole model.** The GP uses role-wise kernels and shrunk interaction products.
6. **Stable external LogEI.** The implementation evaluates LogEI directly instead of taking `log(EI + constant)`.
7. **Target-specific Buchwald incumbent.** Auxiliary products inform the surrogate, but their yields do not define improvement for the target product.
8. **No independent-normal pseudo-Thompson sampling.** The default policy uses LogEI and model disagreement; full correlated posterior sampling is exposed as an extension point.

The optional stock bridge is intentionally narrow: Bgolearn invokes the custom surrogate, while `bgolearn-rxn` retains responsibility for frozen features, posterior combination, acquisition, ID management, and logging.

## Benchmark assumptions encoded here

### Buchwald_sub4

- Input features: `Product`, `Reactant2`, `Ligand`, `Additive`, `Base`.
- Initial labels: 35 pooled observations from five products.
- Target task: `N-(4-ethylphenyl)-4-methylaniline`.
- The improvement incumbent uses only target-product observations.
- Primary modeling emphasis: low-rank multitask transfer and additive similarity.

### Suzuki

- Input features: `Electrophile`, `Nucleophile`, `Ligand`, `Base`, `Solvent`.
- Fixed product metadata: `6-(5-methyl-1-(tetrahydro-2H-pyran-2-yl)-1H-indazol-4-yl)quinoline`.
- The fixed product is included as `ProductContext` while remaining outside the
  selectable candidate columns.
- The encoder derives `SubstratePair = Electrophile || Nucleophile`.
- Only candidate rows supplied in `suzuki_test_features.csv` are feasible.
- Primary modeling emphasis: pair identity plus constituent chemistry and low-rank pair–condition interactions.

Never generate a Cartesian product from `options.json`; acquire only over the supplied candidate CSV.

The similarly named boronate compound remains a valid nucleophile in the
candidate pool. Correcting the fixed product does not rename or remove that
nucleophile.

The two `metric_frontier` presets are frozen deployment candidates selected
without public test labels. They use posterior mean for query 1 and stable
LogEI for the remainder of the 40-query budget. The Buchwald preset enables
only `component_ard_low_rank_task_gp`; the Suzuki preset enables only
`component_ard_gp`, promoted after a sealed 200-episode train-only confirmation
passed the predeclared AUC, t95, safety, runtime, and amplitude gates. They are
opt-in and do not replace the general public presets above.

## Installation

Python 3.10 or newer is required.

```bash
cd bgolearn_rxn_extension
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[all,dev]'
```

For the recommended native backend without stock Bgolearn:

```bash
python -m pip install -e '.[chem,submission,dev]'
```

For the stock compatibility bridge, install the pinned Bgolearn release:

```bash
python -m pip install -e '.[bgolearn,chem,submission,dev]'
```

Bgolearn is pinned to `3.1.0` because the bridge depends on its custom `fit_pre` behavior. Re-run the adapter tests before changing that version.

## Quick audit

```bash
bgolearn-rxn audit \
  --config configs/buchwald_sub4.yaml \
  --train /path/to/buchwald_sub4_train.csv \
  --candidates /path/to/buchwald_sub4_test_features.csv
```

```bash
bgolearn-rxn audit \
  --config configs/suzuki.yaml \
  --train /path/to/suzuki_train.csv \
  --candidates /path/to/suzuki_test_features.csv
```

The audit rejects:

- a candidate file containing `Yield`;
- missing feature columns;
- duplicate candidate IDs or conditions;
- target-task train/candidate overlap;
- nonfinite or out-of-range training yields;
- Buchwald candidates outside the configured target product.

## Offline dry-lab simulation

The labeled test file is passed only to `OfflineTableOracle`. It is never passed to the encoder, models, policy, or prior.

### One Buchwald run

```bash
bgolearn-rxn simulate \
  --config configs/buchwald_sub4.yaml \
  --train /path/to/buchwald_sub4_train.csv \
  --candidates /path/to/buchwald_sub4_test_features.csv \
  --oracle /path/to/buchwald_sub4_test.csv \
  --seed 100 \
  --budget 40 \
  --backend native \
  --output-dir runs/buchwald_sub4
```

### One Suzuki run

```bash
bgolearn-rxn simulate \
  --config configs/suzuki.yaml \
  --train /path/to/suzuki_train.csv \
  --candidates /path/to/suzuki_test_features.csv \
  --oracle /path/to/suzuki_test.csv \
  --seed 100 \
  --budget 40 \
  --backend native \
  --output-dir runs/suzuki
```

### All 20 configured seeds

```bash
bgolearn-rxn run-all \
  --config configs/buchwald_sub4.yaml \
  --train /path/to/buchwald_sub4_train.csv \
  --candidates /path/to/buchwald_sub4_test_features.csv \
  --oracle /path/to/buchwald_sub4_test.csv \
  --backend native \
  --output-dir runs/buchwald_sub4
```

The command writes:

```text
runs/buchwald_sub4/
├── seed_100.json
├── seed_100.pt
├── ...
├── seed_2000.json
├── seed_2000.pt
└── summary.json
```

The JSON is human-auditable; the `.pt` file matches the benchmark submission convention.

## Validate a trajectory

```bash
bgolearn-rxn validate \
  --config configs/buchwald_sub4.yaml \
  --train /path/to/buchwald_sub4_train.csv \
  --candidates /path/to/buchwald_sub4_test_features.csv \
  --submission runs/buchwald_sub4/seed_100.pt
```

Validation checks exactly 40 ordered steps, valid and unique IDs, and exact agreement between every recorded condition and its candidate row.

## Recommended backend versus stock Bgolearn bridge

### Recommended

```bash
--backend native
```

This keeps the ensemble state in memory and exposes component predictions, model disagreement, and posterior covariance methods directly.

### Compatibility demonstration

```bash
--backend bgolearn
```

The bridge calls stock Bgolearn with:

```python
controller.fit(
    data_matrix=frozen_observed_dataframe,
    Measured_response=y_observed,
    virtual_samples=frozen_candidate_dataframe,
    Mission="Regression",
    Kriging_model=ReactionEnsembleAdapter,
    min_search=False,
    opt_num=1,
    Normalize=False,
    Dynamic_W=False,
    CV_test=False,
    seed=seed,
)
```

It then verifies that the means and standard deviations returned by Bgolearn equal the custom ensemble predictions. Candidate ranking is performed outside Bgolearn by the metric-aware policy.

## Molecular registry workflow

Names are not silently converted into structures. Create a registry template:

```bash
bgolearn-rxn make-registry \
  --config configs/buchwald_sub4.yaml \
  --train /path/to/buchwald_sub4_train.csv \
  --candidates /path/to/buchwald_sub4_test_features.csv \
  --output registry_templates/buchwald_registry.csv
```

Then manually curate and review:

```csv
role,component,canonical_smiles,inchi_key,source,review_status,notes
Ligand,...,CC...,ABC...,vendor/SI,reviewed,...
```

Only after review should the config use the registry:

```yaml
features:
  registry_path: ../registry_templates/buchwald_registry.csv
  missing_registry_policy: error
```

`missing_registry_policy: error` is recommended for final external-track runs. It prevents a partially curated registry from silently changing the representation.

For `Nothing`, the encoder emits a zero molecular vector and an explicit missing/absence indicator.

## Optional external prior

An external chemistry or LLM prior is accepted only as a candidate-ID-aligned CSV:

```csv
query_index,prior_score
0,0.18
1,-0.42
...
```

Run with:

```bash
... --prior /path/to/audited_prior.csv
```

The combined score is:

```text
z(acquisition) + lambda(step) * z(prior_score)
```

with exponentially decaying `lambda(step)`. Keep the prior weak, log its provenance, redact benchmark names from LLM prompts, and never generate it from public or hidden yields.

## Real laboratory integration

Use the stateful ask/tell session so a recommendation is committed before its result is accepted:

```python
session = runner.start_session(budget=40)

while not session.is_complete:
    recommendation = session.recommend()
    measured_yield = laboratory.execute(recommendation.query_index)
    record = session.observe(
        recommendation.query_index,
        measured_yield,
    )

result = session.finalize()
```

`recommend()` refuses a second ask while a result is pending, and `observe()` rejects any ID other than the committed candidate. Preserve one session for the complete campaign so prequential ensemble weights remain stateful.

## Core models

### Structured ANOVA GP

For reaction roles \(r\), the kernel is approximately:

\[
k(x,x') = \sum_r w_r k_r(x_r,x'_r)
+ \lambda_{\mathrm{pair}}\sum_{(r,s)}w_{rs}k_r(x_r,x'_r)k_s(x_s,x'_s).
\]

Buchwald multiplies this by a learned task kernel and adds target-specific behavior through exact task identity. Every role has its own length scale; interaction strength is globally shrunk.

### Bayesian low-rank interaction model

Each role is projected label-free from the frozen reference design. Selected outer products form a compact interaction design, and Bayesian ridge regression supplies shrinkage and predictive uncertainty.

### Tree hedge

Bootstrapped Extra Trees captures discontinuities and higher-order exceptions. Its weight is deliberately limited at initialization because tree uncertainty is less reliable at very small sample sizes.

### Prequential ensemble

After observing a queried yield, each model receives a log predictive score computed from the prediction made **before** the measurement. Model weights update from cumulative sequential evidence with a nonzero floor. This avoids fitting ensemble weights on the candidate labels in advance.

## Acquisition schedule

| Rounds | Policy | Purpose |
|---|---|---|
| 1 | Posterior mean | Maximize expected first-query yield |
| 2–20 | Stable full-pool LogEI | Fast improvement and trajectory AUC |
| 21–32 | LogEI plus gated disagreement/UCB challenges | Correct shared model misspecification |
| 33–40 | Low-exploration mean/UCB | Concentrate on final simple regret |

The exact boundaries are configurable.

## Development rules

Do not use public test yields to choose:

- feature representations;
- molecular registry entries;
- kernel interaction lists;
- model weights;
- hyperparameters;
- prior prompts;
- acquisition schedules.

Use product-held-out and reagent-family-held-out diagnostics for Buchwald, substrate-pair-held-out diagnostics for Suzuki, unrelated reaction datasets, and synthetic structured categorical landscapes.

See [`docs/09_validation_and_ablation.md`](docs/09_validation_and_ablation.md).

## Test suite

```bash
PYTHONPATH=src pytest
```

The tests cover:

- stable LogEI numerics;
- target-specific incumbents;
- label rejection;
- immutable IDs;
- frozen coordinates;
- complete no-repeat campaigns;
- prequential weights;
- submission validation;
- an interface-level stock-Bgolearn compatibility test.

## Validation status

The native backend passed the packaged test suite and complete 40-query seed-100 execution checks on both uploaded benchmarks. See [`VALIDATION_REPORT.md`](VALIDATION_REPORT.md) for metrics, runtime, and the explicit limitation that the stock Bgolearn package itself could not be installed in this isolated build environment.

## Known limitations

- The molecular registry templates are intentionally uncurated; canonical structures must be reviewed by a chemist.
- The low-rank interaction model is a conservative Bayesian linear factorization, not an end-to-end neural factorization machine.
- The tree covariance is diagonal; it is used as an ensemble hedge, not for correlated Thompson sampling.
- The structured GP uses MAP hyperparameters. A fully Bayesian GP may improve robustness at additional cost.
- The stock Bgolearn bridge relies on the 3.1.0 custom-model interface and should be retested against later releases.
- Public-table performance is not evidence of hidden out-of-distribution generalization.

## Documentation map

1. [Architecture](docs/01_architecture.md)
2. [Data contract and leakage controls](docs/02_data_contract_and_leakage.md)
3. [Featurization](docs/03_featurization.md)
4. [Surrogates and uncertainty](docs/04_surrogates_and_uncertainty.md)
5. [Acquisition policy](docs/05_acquisition_policy.md)
6. [Bgolearn integration](docs/06_bgolearn_integration.md)
7. [Running the benchmarks](docs/07_running_benchmarks.md)
8. [External knowledge and LLM priors](docs/08_external_knowledge.md)
9. [Validation and ablation](docs/09_validation_and_ablation.md)
10. [API and extension points](docs/10_api_and_extension_points.md)
11. [Research basis](docs/11_research_basis.md)
12. [Production checklist](docs/12_production_checklist.md)
13. [Configuration reference](docs/13_configuration_reference.md)

## License

MIT. See [`LICENSE`](LICENSE).
