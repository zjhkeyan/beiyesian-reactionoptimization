# 5. Acquisition Policy

## Goals implied by the benchmark

The benchmark rewards:

- a strong first recommendation;
- high best-so-far trajectory AUC;
- early arrival near the global optimum;
- a high final best yield;
- robustness across 20 independent runs.

One stationary acquisition rule is not necessarily optimal for all four goals.

## Query 1: posterior mean

The first query is selected by:

\[
x_1=\arg\max_x\mu(x).
\]

This directly maximizes expected first-query yield. Pure UCB or maximum variance is misaligned with the explicit first-query score.

The model can still express caution through the ensemble: extreme predictions unsupported by the GP, factorization model, and tree hedge will often have lower mixture mean and high disagreement.

## Stable Log Expected Improvement

For maximization under a Gaussian posterior:

\[
\mathrm{EI}(x)=\sigma(x)\left[\phi(z)+z\Phi(z)\right],
\qquad
z=\frac{\mu(x)-f_\max-\xi}{\sigma(x)}.
\]

The implementation returns \(\log\mathrm{EI}\) using stable log-CDF and log-difference expressions. It does not compute EI first and then add an arbitrary floor.

For a finite candidate set, maximizing LogEI produces the same ordering as EI in exact arithmetic while retaining numerical distinctions in extreme tails.

## Incumbent definition

### Suzuki

Use the best yield among all currently observed rows.

### Buchwald

Use the best yield among rows whose `Product` equals the configured target product. Auxiliary-product yields do not define improvement for the target reaction.

## Exploration parameter

The implementation linearly interpolates `xi` from `xi_initial` to `xi_final`. The defaults use a small early value and approach zero near the end.

Because yields are modeled in original percentage units after inverse transformation, `xi` should be interpreted as yield percentage points.

## Gated model-challenge rounds

During the middle campaign, every `challenge_every` rounds:

1. identify the normal LogEI recommendation;
2. form a challenger score from UCB plus standardized ensemble disagreement;
3. find the highest challenger;
4. accept the challenger only when its upper confidence bound is at least competitive with the LogEI point's lower confidence bound.

The gate avoids spending a query on an uncertain point whose plausible upside is clearly inferior.

## Final rounds

From `final_start` onward, the default uses:

\[
a(x)=\mu(x)+\beta_\text{final}\sigma(x),
\]

with small \(\beta_\text{final}\). This concentrates on simple regret while retaining a limited ability to correct residual uncertainty.

A valid alternative is probability of improvement or posterior mean alone. Choose among them using allowed proxy benchmarks, not public test labels.

## Optional external prior

A candidate prior enters as:

\[
\tilde a_t(x)=z(a_t(x))+\lambda_0e^{-\gamma(t-1)}z(s_\text{prior}(x)).
\]

Both terms are standardized over the current remaining pool. This prevents arbitrary prior units from determining the acquisition scale.

The prior should be strongest on query 1 and largely gone after experimental feedback accumulates.

## Tie-breaking

Numerical ties are handled by:

1. selecting all values equal to the maximum within absolute tolerance;
2. sorting them by immutable candidate ID;
3. drawing one with a local generator seeded by campaign seed and step.

This makes the result reproducible and independent of DataFrame order.

## Why full-pool scoring

The largest pool has only 5,731 candidates. Every candidate can be evaluated directly each round. Continuous relaxation, evolutionary acquisition optimization, and random candidate subsampling add unnecessary approximation error.

## Why the current policy does not use nonmyopic rollout by default

Knowledge gradient or short-horizon fantasy rollout can improve simple regret, but they add:

- sensitivity to posterior calibration;
- substantial implementation complexity;
- nested model fitting or posterior updates;
- more hyperparameters that may overfit public tables.

Representation and uncertainty quality are higher-priority improvements. A rollout should be added only after the simpler policy is validated and only over a small shortlist.

## Correlated Thompson sampling extension

A correct GP Thompson sample requires a joint draw:

\[
f(X_\text{pool})\sim\mathcal N(\mu,K).
\]

Do not independently sample each candidate from its marginal normal distribution. For a large pool:

1. shortlist candidates by UCB or LogEI;
2. request covariance on the shortlist;
3. draw a joint sample with jittered Cholesky or low-rank root decomposition;
4. select its maximum.

The default disagreement challenger avoids the cost of full-pool covariance.
