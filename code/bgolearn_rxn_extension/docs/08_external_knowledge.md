# 8. External Knowledge and LLM Priors

## External-track objective

External knowledge should improve cold-start decisions without turning the benchmark into a lookup of a famous public yield table.

The recommended hierarchy is:

1. reviewed molecular structures;
2. generic physicochemical descriptors;
3. frozen chemistry embeddings;
4. literature-derived mechanistic classes;
5. weak LLM preference scores;
6. experimental feedback, which must rapidly dominate all priors.

## Structure and descriptor provenance

Every structure registry row should record:

- source URL, supporting information, or vendor page;
- salt and protonation assumptions;
- tautomer or protecting-group interpretation;
- reviewer and review date;
- canonicalization tool version.

Compute descriptors with one fixed software version and parameters. Hash the registry and descriptor artifact.

## Frozen embeddings

Potential representation sources include:

- molecular fingerprints;
- DRFP or RXNFP reaction fingerprints;
- chemistry language-model embeddings;
- graph-neural-network embeddings;
- ligand-specific descriptor databases.

At this sample size, freeze the large encoder. Learn at most:

- role-specific scaling;
- a small linear projection;
- GP kernel length scales;
- a low-rank adapter selected on unrelated proxy tasks.

End-to-end fine-tuning from seven target observations is high variance and risks memorizing the public table.

## LLM preference prior

### Safer query format

Ask for pairwise or coarse ordinal judgments, not exact yields. Redact:

- dataset name;
- source paper;
- row indices;
- known benchmark terminology that identifies the table.

A prompt can describe generic roles and request a comparison:

```text
For a palladium-catalyzed aryl amination with the listed substrate,
which of conditions A and B is more likely to give a productive
reaction? Return A, B, or uncertain, plus a confidence from 0 to 1.
Do not estimate a numerical yield.
```

Aggregate judgments with a Bradley–Terry or Thurstone preference model to obtain one candidate utility score.

### Required audit fields

Store:

- model provider and exact model identifier;
- access date;
- system and user prompts;
- temperature and sampling settings;
- raw responses;
- parser version;
- candidate pairs shown;
- utility-model code and seed;
- final prior CSV hash.

### Decay

Set the prior weight so that it can break an early model tie but cannot dominate clear experimental evidence. Suggested starting ranges:

- Buchwald: `0.25–0.50` standardized units;
- Suzuki: `0.20–0.40` standardized units;
- exponential decay `0.25–0.40` per query.

These are priors for proxy validation, not public-test-tuned constants.

## Contamination safeguards

The benchmark source tables are public and widely reused. Therefore:

- do not ask a model to recall the Ahneman or Perera dataset;
- do not retrieve repositories containing complete yields;
- do not use a checkpoint fine-tuned specifically on these labeled tables unless the competition explicitly permits it and a separate contaminated track is declared;
- do not expose target test yields in demonstrations or few-shot prompts;
- do not choose prompts after observing completed public trajectories;
- retain a no-external-data baseline.

## Prior file contract

```csv
query_index,prior_score
0,-0.12
1,0.48
2,0.03
```

Requirements:

- one unique row per public candidate ID;
- finite numeric scores;
- no yield-derived columns;
- fixed before the campaign starts;
- same prior across seeds unless stochastic prior generation is part of a preregistered method.

## Learned embedding kernel extension

A GOLLuM-like extension can replace static embedding distances with a small learned projection optimized jointly with the GP marginal likelihood:

\[
k(x,x')=k_\theta(g_\phi(e(x)),g_\phi(e(x'))).
\]

To keep this defensible:

- freeze the base LLM or chemistry encoder;
- constrain \(g_\phi\) to a low-rank linear or shallow network;
- regularize strongly toward identity;
- select architecture on unrelated reaction tasks;
- keep an exact-identity kernel in parallel;
- report calibration, not just discovery rate.

## BO-ICL as an ensemble component

An in-context model capable of returning predictive uncertainty can be added as a fourth component if it implements:

```python
fit(X, y)
predict(X) -> Posterior(mean, std)
```

However, natural-language representations should not replace immutable candidate identities. Cache every prediction for reproducibility and cost accounting.

## External literature priors

Mechanistic literature can support broad features such as:

- leaving-group class;
- ligand denticity and steric/electronic family;
- base type;
- solvent donor/protic class;
- organoboron form;
- additive heterocycle class.

It should not be used to hand-code the known optimum of the named benchmark.

## Recommended track reporting

Report separate results for:

1. categorical/no-external model;
2. reviewed structures and deterministic descriptors;
3. frozen pretrained embeddings;
4. LLM preference prior;
5. full external ensemble.

This decomposition shows whether gains come from chemical generalization or benchmark memorization.
