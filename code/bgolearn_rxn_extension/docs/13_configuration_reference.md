# 13. Configuration Reference

The YAML schema maps directly to frozen dataclasses in `bgolearn_rxn.config`. Unknown keys cause constructor errors rather than being ignored.

## `dataset`

| Key | Type | Meaning |
|---|---|---|
| `name` | string | Manifest and output dataset name |
| `feature_columns` | list[str] | Exact participant CSV columns used as inputs |
| `target_column` | string | Objective column, normally `Yield` |
| `task_column` | string or null | Multitask identity column; `Product` for Buchwald |
| `target_task_value` | string or null | Task whose yield is optimized and defines the incumbent |
| `condition_roles` | list[str] | Roles included in the GP condition kernel |
| `derived_roles` | mapping | New role to source-column list, e.g. `SubstratePair` |
| `fixed_roles` | mapping | Constant reaction-context roles appended to every row |
| `budget` | int | Maximum number of sequential queries |
| `seeds` | list[int] | Accepted benchmark seeds |

### Important distinction

`feature_columns` are real CSV columns. `condition_roles` may also include a role created through `derived_roles`. A `fixed_roles` value is encoded for every row but is not a selectable candidate dimension.

For Buchwald, `Product` is a feature/task identity but not a condition role. For Suzuki, `SubstratePair` is a derived condition role and `ProductContext` is the officially corrected coupled product shared by every candidate; neither is an input CSV column. A fixed role can support transfer across reactions, but a constant value cannot by itself rank candidates within one task.

## `features`

| Key | Default | Meaning |
|---|---:|---|
| `identity_scale` | 1.0 | Multiplier on exact one-hot identity |
| `rule_scale` | 0.75 | Multiplier on standardized name-rule features |
| `molecular_scale` | 0.35 | Multiplier on standardized registry/RDKit features |
| `name_subword_scale` | 0.35 | Multiplier on standardized name-subword SVD features |
| `registry_path` | null | Reviewed structure CSV; relative to config file |
| `use_name_rules` | true | Enable conservative lexical/mechanistic features |
| `use_name_subwords` | false | Enable label-free per-role IUPAC name subword features |
| `name_subword_max_rank` | 8 | Maximum deterministic SVD rank per role |
| `use_rdkit` | true | Generate molecular fingerprints/descriptors |
| `morgan_radius` | 2 | Morgan fingerprint radius |
| `morgan_bits` | 256 | Fingerprint length |
| `missing_registry_policy` | warn | `warn`, `error`, or `zeros` |

Use `registry_path: null` for the no-external baseline. Use `error` only after every required registry row is reviewed.

The opt-in name-subword block fits once on the initial observations plus the
unlabeled candidate names. For each role it uses `char_wb` TF-IDF 3-5 grams on
sorted unique names, then deterministic compact SVD with rank capped by 8,
the number of unique levels minus one, and vocabulary width minus one. The
result is standardized on the frozen reference frame and is never refitted
during a campaign. Yield labels are not accepted by or supplied to this fit.

## `gp`

| Key | Meaning |
|---|---|
| `interaction_pairs` | Role pairs whose kernels are multiplied |
| `role_weights` | Fixed nonnegative main-effect prior weights |
| `interaction_weights` | Fixed relative weights keyed as `RoleA|RoleB` |
| `optimize_hyperparameters` | Enable MAP marginal-likelihood fitting |
| `restarts` | Number of L-BFGS-B starts at a refit round |
| `maxiter` | Maximum optimizer iterations per start |
| `refit_interval` | Rounds between hyperparameter reoptimization |
| `initial_lengthscale` | Initial role-kernel length scale |
| `initial_noise` | Initial latent observation-noise standard deviation |
| `initial_pair_scale` | Initial shared interaction shrinkage scale |
| `initial_task_rho` | Initial cross-task correlation for Buchwald |
| `task_descriptor_scale` | Nonnegative attenuation of cross-task correlation by frozen task descriptors; 0 uses exchangeable tasks |
| `task_mean_prior_count` | Optional nonnegative pseudo-count shrinking task-specific latent intercepts toward the global mean |
| `jitter` | Cholesky numerical stabilization |
| `target_transform` | `standard` or `logit` |

### Interaction key ordering

The key must use the same order as the configured pair:

```yaml
interaction_pairs:
  - [Base, Solvent]
interaction_weights:
  "Base|Solvent": 1.0
```

## `interaction_model`

| Key | Meaning |
|---|---|
| `interaction_pairs` | Selected projected outer products |
| `role_ranks` | PCA rank override by role |
| `default_rank` | Rank used when no override exists |
| `interaction_scale` | Multiplier on outer-product blocks |
| `alpha_1`, `alpha_2` | BayesianRidge noise-precision hyperpriors |
| `lambda_1`, `lambda_2` | BayesianRidge coefficient-precision hyperpriors |
| `target_transform` | `standard` or `logit` |

Ranks are automatically capped by block width and number of distinct reference rows.

## `trees`

| Key | Meaning |
|---|---|
| `n_estimators` | Number of bootstrapped Extra Trees |
| `max_features` | Per-split feature subset |
| `min_samples_leaf` | Minimum leaf size; use at least 2 for public presets |
| `max_depth` | Optional depth cap |
| `residual_noise_fraction` | Fraction of training residual RMSE added as noise floor |
| `target_transform` | `standard` or `logit` |

Keep `n_jobs=1` inside each model; parallelize independent seeds instead.

## `ensemble`

| Key | Meaning |
|---|---|
| `enabled_models` | Ordered subset of `gp`, `interaction`, `trees`, and opt-in transfer models to fit |
| `initial_weights` | Mapping from model name to prior mixture weight |
| `prequential_learning_rate` | Strength of cumulative predictive log score |
| `minimum_weight` | Nonzero floor for every model |
| `predictive_std_floor` | Minimum yield-scale standard deviation in log scoring |

Constraint:

```text
minimum_weight × number_of_models < 1
```

`residual_transfer` requires a task column, a single-task candidate pool, and both target and auxiliary observations. It fits an auxiliary-only structured GP as a source prior and a target-only structured GP to source-prediction residuals. Their means and independent covariance estimates are added. The residual GP always uses a standard transform because residuals are not bounded yields. The model is inactive unless explicitly listed in `enabled_models`.

`source_expert` also requires a task column and single-task candidate pool. It builds one target-plus-source GP per auxiliary task and gates them using source-only predictive density on currently observed target rows. It is an opt-in research model and is not enabled by production presets.

`low_rank_task_gp` learns a rank-2 positive-semidefinite intrinsic task covariance jointly with the structured condition kernel. A positive task-specific diagonal is added before correlation normalization, and strong loading priors regularize the matrix toward the configured exchangeable `initial_task_rho`. It is opt-in and does not change production presets.

`component_ard_low_rank_task_noise_gp` extends the opt-in component-ARD
low-rank task GP with task-conditioned observation noise. The global noise is
the geometric-mean noise, while an orthonormal zero-sum log-contrast assigns
one relative noise per task. Fixed Gaussian shrinkage toward equal noise and
bounded contrast coordinates stabilize the small-data fit. The model is not
enabled by production presets.

`component_ard_gp` and `component_ard_low_rank_task_gp` retain the structured exact-GP family while learning a separate positive MAP amplitude for each role and configured pair-product kernel. Shrinkage priors are centered on `role_weights` and `initial_pair_scale * interaction_weights`; a shared bias kernel is retained. The low-rank variant multiplies this condition kernel by the learned rank-2 task covariance.

At least one unique model must be enabled; unused entries in `initial_weights` are ignored.

## `policy`

| Key | Meaning |
|---|---|
| `first_query` | `posterior_mean` for the mixture, or an enabled model name such as `gp` |
| `early_end` | Last pure LogEI phase boundary |
| `final_start` | First low-exploration final round |
| `challenge_every` | Middle-phase challenge frequency; 0 disables |
| `challenge_beta` | UCB coefficient for challenger generation |
| `challenge_disagreement_weight` | Standardized disagreement multiplier |
| `challenge_gate_sigma` | LogEI-point lower-bound width for gate |
| `final_beta` | Final-round UCB coefficient |
| `xi_initial`, `xi_final` | Linear LogEI improvement offsets |
| `prior_initial_weight` | Standardized external-prior weight on query 1 |
| `prior_decay` | Exponential prior decay rate |
| `novelty_disagreement_mode` | `off`, pure strain-sensor-style exploration, or a `hybrid` with the phase acquisition |
| `novelty_disagreement_weight` | Standardized novelty-disagreement weight in `hybrid` mode |

The public presets use rounds 1, 2–20, 21–32, and 33–40 as policy phases.

## Train-only metric-frontier presets

`buchwald_sub4_metric_frontier.yaml` and `suzuki_metric_frontier.yaml` are
standalone, frozen, opt-in deployment candidates selected entirely from train
proxies. Both use a single model with weight 1, that model's posterior mean at
query 1, and stable LogEI through query 40 (`early_end: 40`, `final_start: 41`,
`challenge_every: 0`). Buchwald uses `component_ard_low_rank_task_gp`; Suzuki
uses `component_ard_gp`. The Suzuki promotion required a sealed 200-episode
train-only confirmation to pass strict paired AUC and t95 confidence-interval
gates, noninferior initial/best point estimates, a 2.5x runtime bound, and
finite noncollapsed amplitudes. Neither preset enables Thompson sampling,
novelty hybrids, or public test-selected settings, and neither replaces the
general public presets.

The strain-sensor adaptation scores each valid remaining candidate by its encoded Euclidean distance to the nearest observation times the ensemble's between-model predictive disagreement. `pure` uses that score directly. `hybrid` retains posterior-mean selection at round 1, then adds the standardized signal to the configured LogEI/UCB phase score. It never interpolates chemical identities or proposes a reaction outside the benchmark candidate table.

## `runtime`

| Key | Meaning |
|---|---|
| `backend` | `native` or `bgolearn` |
| `output_dir` | Suggested output directory; CLI `--output-dir` overrides it |
| `save_candidate_scores` | Suggested diagnostic setting |
| `strict_reproducibility` | Manifest declaration; future backends may use it to reject nondeterministic settings |

## External-track overlays

The files ending in `.overlay.yaml` are intentionally partial examples. They are not standalone run configurations. Merge the relevant sections into a copy of the complete public preset only after the structure registry and prior have been audited.

Example conceptual merge:

```yaml
# copy configs/buchwald_sub4.yaml, then replace:
features:
  registry_path: ../registry_templates/buchwald_registry.csv
  missing_registry_policy: error
  # retain all other feature keys

policy:
  prior_initial_weight: 0.45
  prior_decay: 0.30
  # retain all other policy keys
```

## Recommended change-control process

Every configuration change should produce:

1. a new config file rather than an overwritten file;
2. a new config hash;
3. a written rationale tied to proxy validation;
4. an ablation result;
5. no access to candidate labels during the decision.
