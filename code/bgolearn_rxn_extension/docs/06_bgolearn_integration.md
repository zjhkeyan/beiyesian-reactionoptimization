# 6. Bgolearn Integration

## Supported version

The compatibility layer targets Bgolearn `3.1.0`. The version is pinned in the optional `bgolearn` dependency group.

## Division of responsibility

### Bgolearn handles

- invocation of a custom regression surrogate;
- compatibility with existing Bgolearn campaign scripts;
- return of virtual-sample means and standard deviations.

### Bgolearn-RxN handles

- data audits;
- frozen feature construction;
- multitask and reaction-specific surrogate structure;
- component posterior combination;
- stable LogEI;
- target-specific incumbent;
- candidate IDs and no-repeat logic;
- prequential model weights;
- trajectory serialization.

## Why `Normalize=False`

Stock Bgolearn's normalization path fits a `MinMaxScaler` to the current virtual samples. Once a selected candidate is removed, the remaining minima or maxima can change, causing the same reaction to move in feature space.

Bgolearn-RxN fits and freezes its own representation. The bridge therefore refuses internal normalization.

## Why `Dynamic_W=False`

Dynamic response weighting resamples high-response observations. In a seven-point target task, duplicated high-yield rows can cause overconfident extrapolation and alter the likelihood interpretation.

The acquisition function, not the training likelihood, should express a preference for improvement.

## Why `CV_test=False`

Repeated leave-one-out or k-fold validation during every round is:

- statistically unstable at tiny sample sizes;
- computationally expensive;
- not needed for making the next recommendation;
- awkward for a stateful model ensemble.

Run calibration diagnostics in a separate development pipeline.

## Adapter implementation

The stock custom-model path expects a no-argument class with:

```python
fit_pre(xtrain, ytrain, xtest) -> (mean, std)
```

`BgolearnCompatibilityBackend` creates a closure-backed adapter:

```python
shared_ensemble = persistent_ensemble

class ReactionEnsembleAdapter:
    def fit_pre(self, xtrain, ytrain, xtest):
        shared_ensemble.fit(xtrain, ytrain)
        posterior = shared_ensemble.predict(xtest)
        return posterior.mean, posterior.std
```

The shared object preserves prequential weights across rounds even if Bgolearn instantiates the adapter class internally.

## Bridge call

```python
controller = BGOsampling.Bgolearn()
result = controller.fit(
    data_matrix=observed_frame,
    Measured_response=y_observed,
    virtual_samples=candidate_frame,
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

The bridge checks:

```python
allclose(result.virtual_samples_mean, posterior.mean)
allclose(result.virtual_samples_std, posterior.std)
```

Any transformation or API drift raises an error.

## Why acquisition remains external

The extension needs:

- stable LogEI rather than `log(EI + epsilon)`;
- a target-product incumbent for Buchwald;
- model disagreement diagnostics;
- a stage-dependent policy;
- candidate-ID-aligned external priors;
- exact deterministic tie-breaking.

These are clearer and more testable in the extension policy.

## Recommended future patch to Bgolearn

A more general custom-surrogate API would accept a configured instance or factory:

```python
class SurrogateProtocol(Protocol):
    def fit(self, X, y) -> None: ...
    def predict(self, X) -> tuple[mean, std]: ...
    def posterior_covariance(self, X) -> ndarray: ...
    def sample_posterior(self, X, n_samples, seed) -> ndarray: ...
```

Then a Bgolearn call could accept:

```python
controller.fit(
    ...,
    surrogate=reaction_ensemble,
    acquisition=metric_aware_policy,
)
```

Benefits:

- no no-argument constructor constraint;
- persistent model caches;
- true correlated posterior sampling;
- custom posterior diagnostics;
- simpler typed tests;
- easier multi-output and multitask support.

## Compatibility test

`tests/test_bgolearn_adapter.py` supplies a fake implementation of the stock API and verifies:

- the adapter is invoked;
- mean and standard deviation are returned unchanged;
- `Normalize`, `Dynamic_W`, and `CV_test` are all false.

Before production use, run an additional integration test against the actual pinned Bgolearn package in the deployment environment.
