# 10. API and Extension Points

## Configuration

Load a YAML preset:

```python
from bgolearn_rxn.config import load_run_config

config = load_run_config("configs/buchwald_sub4.yaml")
```

Relative molecular-registry paths are resolved against the YAML file location.

## Dataset loading

```python
from bgolearn_rxn.datasets import DatasetSpec, load_reaction_dataset

spec = DatasetSpec.from_config(config.dataset)
dataset = load_reaction_dataset(
    "buchwald_sub4_train.csv",
    "buchwald_sub4_test_features.csv",
    spec,
)
```

Important properties:

```python
dataset.initial
dataset.candidates
dataset.candidate_ids
dataset.target_incumbent()
dataset.observed_features()
dataset.candidate_features()
```

## Preparing the frozen design

```python
from bgolearn_rxn.factory import prepare_problem

prepared = prepare_problem(dataset, config)
```

Outputs:

```python
prepared.encoder
prepared.X_initial
prepared.X_candidates
prepared.reference_X
```

`reference_X` contains all initial and unlabeled candidate feature rows and is used only for label-free projection/scaling.

## Building the model ensemble

```python
from bgolearn_rxn.factory import build_ensemble

ensemble = build_ensemble(config, prepared, seed=100)
ensemble.fit(prepared.X_initial, y_initial)
posterior = ensemble.predict(prepared.X_candidates)
```

`Posterior` contains:

```python
posterior.mean
posterior.std
posterior.component_means
posterior.component_stds
posterior.weights
posterior.disagreement
```

## Adding a surrogate

A component must implement:

```python
class ProbabilisticSurrogate(Protocol):
    name: str

    def fit(self, X: ndarray, y: ndarray): ...
    def predict(self, X: ndarray) -> Posterior: ...
    def posterior_covariance(self, X: ndarray) -> ndarray: ...
```

Then add it to the ordered mapping passed to `PrequentialEnsemble` and add an initial weight under the same name.

### Requirements

- `predict` must return yield-scale means and standard deviations.
- Standard deviations must be finite and nonnegative.
- The model must not access candidate labels.
- Randomness must be controlled by an explicit seed.
- Repeated `fit` calls should support growing observed datasets.

## Adding a chemistry kernel

The current GP uses RBF kernels over role blocks. A role-specific Tanimoto kernel can be introduced by:

1. recording fingerprint-only slices in `FeatureSchema`;
2. implementing `k_tanimoto(X1, X2)`;
3. selecting the kernel by role or feature block;
4. retaining exact identity and descriptor kernels in a positive weighted sum;
5. testing positive semidefiniteness numerically.

A mixed role kernel could be:

\[
k_r = \rho_\text{id}k_\text{id}
+ \rho_\text{fp}k_\text{Tanimoto}
+ \rho_\text{desc}k_\text{Matérn}.
\]

Constrain all coefficients to be nonnegative.

## Adding a fully Bayesian GP

Replace MAP hyperparameters with samples or a variational posterior. Recommended interface behavior:

- integrate hyperparameter uncertainty into candidate mean and variance;
- cache samples between rounds;
- warm-start from the previous posterior;
- cap computation per round;
- expose posterior samples for non-Gaussian LogEI.

Do not let hyperparameter sampling dominate the 20 × 40 campaign runtime without demonstrated benefit.

## Adding correlated Thompson sampling

Implement:

```python
def sample_posterior(self, X, n_samples, seed):
    mean = self.predict(X).mean
    covariance = self.posterior_covariance(X)
    return multivariate_normal_draws(mean, covariance, n_samples, seed)
```

For Suzuki, shortlist candidates first to avoid forming a full dense covariance matrix.

## Adding a new acquisition policy

A policy should accept:

```python
select(
    posterior,
    candidate_ids,
    best_f,
    step,
    budget,
    prior_scores=None,
) -> Recommendation
```

`Recommendation` must contain:

- local index in the current pool;
- immutable query ID;
- acquisition name/value;
- mixture mean/std;
- component means/stds;
- model weights;
- prior weight.

Do not let the policy query an oracle.

## Adding a laboratory oracle

Implement only:

```python
class LabOracle:
    def query(self, query_index: int) -> float:
        # Translate ID to instrument worklist, execute or retrieve result,
        # validate the measurement, and return one yield.
        ...
```

For asynchronous laboratories, use a reservation/commit state before execution so the same point cannot be selected twice.

## Adding cost or constraints

Extend candidate metadata with label-free columns such as:

- reagent cost;
- hazard category;
- stock availability;
- plate/well assignment;
- expected turnaround time.

Possible policies:

\[
\frac{\mathrm{EI}(x)}{\mathrm{cost}(x)^\alpha},
\]

or constrained acquisition:

\[
\mathrm{EI}(x)\prod_jP(c_j(x)\le 0).
\]

Keep objective prediction and constraint prediction in separate models.

## Batch acquisition

For a real 96-well workflow:

1. generate a high-value shortlist;
2. select a diverse batch using greedy fantasies, local penalization, or qLogEI;
3. enforce plate and reagent constraints;
4. return all batch IDs atomically;
5. update after batch results arrive.

The public benchmark is `q=1`, so batch logic is intentionally absent from the default runner.

## Serialization extension

The default output saves JSON and PyTorch `.pt`. A database-backed deployment should store:

- campaign ID;
- candidate registry version;
- encoder artifact hash;
- config hash;
- model versions;
- every pre-query posterior;
- query commitment timestamp;
- experimental execution metadata;
- raw measurement and QC status;
- final accepted yield.

## Backward compatibility

When altering feature schema or serialization:

- increment the package version;
- include a schema version in the manifest;
- retain migration scripts;
- invalidate incompatible cached models;
- rerun submission validation.
