# 4. Surrogates and Uncertainty

## Why an ensemble

With 7 target Buchwald observations or 29 Suzuki observations, the landscape structure cannot be identified reliably in advance. Several hypotheses are plausible:

- smooth chemical similarity;
- dominant categorical main effects;
- low-rank pair interactions;
- abrupt exceptions and higher-order rules;
- cross-product transfer with target-specific departures.

The extension therefore maintains three deliberately different probabilistic models.

## 4.1 Structured ANOVA Gaussian process

### Kernel

For non-task roles \(r\), the GP uses:

\[
k_\text{condition}(x,x') =
\frac{1 + \sum_r w_r k_r(x_r,x'_r)
+ \lambda_p\sum_{(r,s)\in\mathcal I} w_{rs}k_r(x_r,x'_r)k_s(x_s,x'_s)}
{1 + \sum_r w_r + \lambda_p\sum_{(r,s)\in\mathcal I}w_{rs}}.
\]

Each role kernel is an RBF over that role's frozen block:

\[
k_r(x_r,x'_r)=
\exp\left[-\frac{\lVert z_r-z'_r\rVert^2}{2\ell_r^2 d_r}\right].
\]

The division by block width \(d_r\) keeps distances from being dominated solely by a wider role representation.

### Interaction shrinkage

All configured pair products share a learned positive `pair_scale`. Pair-specific fixed weights encode a conservative prior about which interactions deserve attention. This is less data-hungry than giving every interaction its own unrestricted variance.

### Buchwald task transfer

For task identities \(t,t'\):

\[
k_\text{task}(t,t') = \rho + (1-\rho)\mathbf{1}[t=t'].
\]

The complete kernel is:

\[
k((t,x),(t',x')) = k_\text{condition}(x,x')k_\text{task}(t,t').
\]

- \(\rho\approx 1\) means nearly complete transfer.
- \(\rho\approx 0\) means auxiliary products transfer little.

The learned value is regularized and bounded. A future extension can replace this scalar coregionalization with a rank-1 or rank-2 task matrix, but a fully free task covariance is difficult to estimate from seven rows per product.

### Hyperparameter fitting

The GP optimizes a regularized negative log marginal likelihood over:

- one length scale per condition role;
- output scale;
- observation-noise nugget;
- shared pair-interaction scale;
- task correlation when applicable.

It warm-starts from the previous round and only reoptimizes according to `refit_interval`.

### Predictive covariance

`posterior_covariance` returns the full GP candidate covariance. Full covariance should be requested only for a shortlist or a small candidate subset; a dense 5,731-by-5,731 matrix is unnecessary for ordinary LogEI.

## 4.2 Bayesian low-rank interaction surrogate

### Label-free role projection

Each role block is centered and projected with PCA fitted on the initial-plus-unlabeled reference design. The rank is role-specific and cannot exceed the number of distinct reference rows.

### Design

The regression design contains:

- exact identity main effects;
- low-rank role projections;
- selected outer products between projected role vectors.

For two role embeddings \(u_r\in\mathbb R^{p}\) and \(u_s\in\mathbb R^q\), the interaction block is:

\[
\operatorname{vec}(u_ru_s^\top).
\]

A Bayesian ridge posterior provides coefficient shrinkage and predictive variance.

### Why it is useful

The model can express many pair interactions with far fewer parameters than an unrestricted one-hot pair table. It is particularly well matched to Suzuki's substrate-pair × ligand × base × solvent tensor.

### Limitations

The current implementation uses a diagonal predictive covariance when queried through the common interface. It is a robust baseline, not a complete Bayesian tensor-process implementation. Suitable replacements include:

- probabilistic CP or Tucker decomposition;
- Bayesian factorization machines;
- variational latent-factor GPs;
- neural factorization machines with Laplace or ensemble uncertainty.

A replacement should retain the same `fit`, `predict`, and `posterior_covariance` interface.

## 4.3 Bootstrapped Extra Trees

Extra Trees capture discontinuities and higher-order exceptions without a smoothness assumption.

Uncertainty is estimated from:

- variation across bootstrapped trees;
- a residual-noise floor estimated on observed data.

This uncertainty is not treated as a definitive Bayesian posterior. The model starts with only 20% ensemble weight and serves as a misspecification hedge.

## 4.4 Source-expert transfer GP

The opt-in `source_expert` surrogate isolates task transfer by fitting one GP
for each target-plus-auxiliary-task pair. Before fitting those experts, a
source-only GP scores all currently observed target rows. Gaussian predictive
log densities are divided by a fixed temperature of 2.0 and normalized, with
a 5% minimum weight per expert and a 2-yield-unit predictive standard-deviation
floor. These conservative constants avoid eliminating a source from only two
target observations while still suppressing grossly incompatible tasks.

Expert means and variances are combined as a proper mixture, including the
between-expert disagreement term. The source-only fits are cached; weights and
target-plus-source experts update after each new target observation. This model
is disabled unless explicitly selected.

## 4.5 Target transformations

### Standardized raw yield

The default standardizes raw yield each round. It is simple and preserves the original objective scale after inverse transformation.

### Bounded logit yield

An optional transformation maps yields to a softened probability before applying a logit. Predictive standard deviations are mapped back with a delta-method approximation.

A strong future ensemble can include both raw-yield and logit-yield versions of the GP and interaction model. That increases diversity around 0% and 100% boundaries.

## 4.6 Mixture uncertainty

For component weights \(w_m\), means \(\mu_m\), and variances \(\sigma_m^2\):

\[
\mu = \sum_mw_m\mu_m,
\]

\[
\sigma^2 = \sum_mw_m\left[\sigma_m^2+(\mu_m-\mu)^2\right].
\]

The second term is between-model disagreement. It is essential when model-form uncertainty is larger than within-model posterior uncertainty.

## 4.7 Prequential weighting

Before querying candidate \(x_t\), each model records \((\mu_{m,t},\sigma_{m,t})\). After observing \(y_t\), the component receives Gaussian log predictive density:

\[
\log p_m(y_t) =
-\frac12\left(\frac{y_t-\mu_{m,t}}{\max(\sigma_{m,t},\sigma_\min)}\right)^2
-\log\max(\sigma_{m,t},\sigma_\min)
-\frac12\log(2\pi).
\]

Cumulative scores update the weights with a learning rate and a minimum weight floor.

The floor prevents a model from being eliminated after one unlucky early prediction. The predictive-standard-deviation floor prevents an overconfident component from dominating the log score numerically.

## 4.8 Calibration diagnostics

On allowed proxy tasks, report:

- negative log predictive density;
- interval coverage at 50%, 80%, and 95%;
- mean interval width;
- standardized residual distribution;
- calibration by campaign round;
- calibration by reaction role or substrate pair;
- ensemble-weight trajectories.

Optimization quality and uncertainty calibration should be evaluated together. An excellent mean predictor with poor uncertainty can still make poor sequential decisions.
