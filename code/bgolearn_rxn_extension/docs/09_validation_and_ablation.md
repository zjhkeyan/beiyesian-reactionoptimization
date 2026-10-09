# 9. Validation and Ablation

## Principle

A public offline table can make a highly specialized method look excellent. Model selection should therefore emphasize transfer diagnostics that do not expose target candidate yields.

## Buchwald validation

### Leave-one-product-out

For each of the five products:

1. designate it as a pseudo-target;
2. retain only its permitted small initial subset;
3. use other products as auxiliary training data;
4. evaluate predictions or sequential optimization on labels not used for fitting.

When only the public 35 training rows are available, use nested leave-one-row-out diagnostics with caution; sample size is tiny.

### Additive-family holdout

Group additives by reviewed structure or conservative family rules and hold out complete families. Compare:

- one-hot identity only;
- name rules;
- Morgan/Tanimoto features;
- frozen learned embeddings.

This tests whether external chemistry helps genuinely unseen additives.

### Negative-transfer test

Compare target-only and multitask models. Track whether auxiliary products improve:

- predictive log density;
- query-1 ranking;
- early regret;
- calibration.

A task kernel should shrink toward weak transfer when auxiliary products are misleading.

## Suzuki validation

### Substrate-pair holdout

Hold out complete feasible electrophile–nucleophile pairs. Evaluate:

- pair-identity-only model;
- constituent Electrophile/Nucleophile model;
- combined constituent plus pair residual;
- molecular descriptors.

Pair identity should excel within observed pairs but cannot predict an unseen pair. Constituent features should carry the out-of-pair signal.

### Role-level holdout

Where data allow, hold out:

- one ligand;
- one base;
- one solvent;
- one organoboron form;
- one leaving-group class.

This is a demanding test of representation quality.

## Sequential proxy evaluation

For each proxy task:

- use multiple initial designs;
- use paired seeds across algorithms;
- evaluate at query checkpoints 1, 5, 10, 20, and 40;
- report best-so-far AUC and final simple regret;
- report confidence intervals across runs;
- preserve the exact same candidate pool and initial set for paired comparisons.

## Required ablation ladder

| Stage | Description |
|---|---|
| 0 | Random no-replacement sampling |
| 1 | Stock Bgolearn RBF GP + stock EI |
| 2 | Frozen exact identities + structured categorical GP |
| 3 | Add shrunk reaction-role interactions |
| 4A | Add Buchwald task kernel |
| 4B | Add Suzuki substrate-pair role |
| 5 | Add Bayesian low-rank interaction model |
| 6 | Add tree hedge and mixture disagreement |
| 7 | Add stable LogEI and staged policy |
| 8 | Add reviewed molecular structures |
| 9 | Add frozen pretrained embeddings |
| 10 | Add decaying LLM preference prior |

Each stage should beat or justify its complexity against the previous stage on allowed development tasks.

## Hyperparameter policy

Classify hyperparameters into:

### Fixed engineering constants

Examples:

- numerical jitter;
- stable-LogEI tail threshold;
- candidate-ID tie tolerance.

### Dataset-topology presets

Examples:

- interaction list;
- initial ensemble weights;
- role ranks.

These must be based on role structure and proxy evidence, not public test labels.

### Learned online parameters

Examples:

- GP length scales;
- task correlation;
- noise nugget;
- Bayesian ridge precision;
- prequential model weights.

They may use only currently observed labels.

## Calibration tests

At every proxy round, store predictions before observing the selected label. Aggregate:

- standardized residuals;
- log predictive density;
- interval coverage;
- predicted standard deviation versus absolute error;
- calibration by acquisition type;
- calibration by seen/unseen role level.

A prequential log score is more honest than random cross-validation on adaptively acquired data.

## Robustness tests

Perturb:

- initial-design selection;
- GP length-scale initialization;
- model random seed;
- descriptor subsets;
- target transform;
- prior strength;
- candidate row ordering.

The queried IDs must be invariant to candidate row permutation except for true numerical ties resolved by immutable IDs.

## Leakage tests

Automate checks that:

- no model module imports or opens the oracle path;
- candidate files with `Yield` are rejected;
- feature schema is identical before and after queries;
- output decisions are unchanged when oracle labels for unqueried rows are permuted;
- candidate row order does not change predictions or selections;
- prior generation does not read labeled lookup files.

The oracle-label permutation test is especially important: until a row is queried, its value must have no effect on decisions.

## Statistical comparison

Use paired differences across the same seeds and initial designs. Report:

- mean paired difference;
- bootstrap or t-based confidence interval;
- win/tie/loss count;
- effect at early and late checkpoints;
- runtime and external-model cost.

Avoid declaring superiority from one seed or one public table.
