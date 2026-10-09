# Initial Experimental Investigations

These investigations exercise the supplied Bgolearn-RxN extension against the
current public reaction tables. Labeled test files are accessible only through
`OfflineTableOracle`; encoders, surrogates, and policies receive the initial
training table, unlabeled candidate pool, and prior queried feedback only.

## Suzuki identity contract

The corrected fixed product is:

```text
6-(5-methyl-1-(tetrahydro-2H-pyran-2-yl)-1H-indazol-4-yl)quinoline
```

It remains in `Suzuki/metadata.json` and is emitted as fixed `ProductContext`.
It is not a selectable candidate variable and therefore has no within-task
variance by itself.
The older long boronate name remains valid as a `Nucleophile` candidate. It must
not be globally replaced because that would corrupt the Suzuki candidate space.
The runner checks both facts before starting a Suzuki campaign.

## Policy ablation

The first experiment holds the frozen feature encoder and three-surrogate
ensemble constant while comparing:

- `metric_aware`: posterior mean at query 1, LogEI in the middle, periodic gated
  model-challenge queries, and low-exploration late UCB;
- `logei_only`: posterior mean at query 1 and stable LogEI thereafter;
- `run_knowledge_gradient_study.py`: deterministic finite-pool one-step
  knowledge gradient using full GP posterior covariance and Gauss-Hermite
  integration, evaluated only on staged train-derived episodes;
- `run_rank_weighted_precision_study.py`: deterministic observed-rank training
  precision inspired by Bgolearn `Dynamic_W`, evaluated through staged
  train-only selection and confirmation gates;
- `run_suzuki_surrogate_switch_study.py`: fixed structured GP at Suzuki query
  1 followed by component-ARD LogEI, compared against both pure schedules on
  isolated pilot and confirmation splits;
- `run_suzuki_homologous_committee.py`: opt-in fixed/component-ARD committee
  with honest prequential log-score weights and a conservative disagreement
  gate, evaluated only on the frozen phase-4 train-derived pilot;
- `run_suzuki_ard_hybrid_acquisition.py`: exact frozen phase-2 alternating
  LogEI/correlated-Thompson schedule crossed with fixed and component-ARD GPs
  on the common phase-4 train-derived pilot;
- `run_suzuki_component_ard_final_confirmation.py`: sealed final comparison of
  the unchanged fixed and component-ARD Suzuki GPs on 200 reserved 810k
  train-only episodes, with a strict predeclared promotion gate and no tuning;
- `run_suzuki_divergence_atlas.py`: fresh 910-series paired policy-divergence,
  component-geometry, pair-context, tail-risk, and calibration diagnostics;
- `run_suzuki_strong_heredity_ard.py`: parent-coupled global-local
  interaction-amplitude pilot against the component-ARD frontier;
- `run_suzuki_pair_context_gp.py`: constituent-plus-exact-pair contextual GP
  with random-holdout and leave-pair-out diagnostics;
- `run_suzuki_ard_mode_integration.py`: likelihood-window integration over
  distinct MAP restart modes;
- `run_suzuki_ard_laplace.py`: bounded local Laplace integration inside the
  stable component-ARD MAP basin;
- `run_suzuki_graph_diffusion_gp.py`: product-kernel diffusion over the
  reaction-component graph on the Phase 6 discovery family;
- `run_suzuki_rank_gaussian.py`: rank-Gaussian target modeling with inverse
  transformation of predictive moments;
- `run_suzuki_bayesian_cp.py`: Bayesian CP tensor factorization over reaction
  roles;
- `run_suzuki_consensus_portfolio.py`: fixed top-three ARD/graph consensus
  arbitration;
- `run_suzuki_delete_fold_ard.py`: five-member, yield-balanced delete-fold
  component-ARD posterior;
- `run_suzuki_stage_split_delete_fold.py`: component ARD for query 1 followed
  by delete-fold ARD and stable LogEI;
- `run_suzuki_stage_split_delete_fold_validation_1020.py`: frozen 80-episode
  validation of the stage-split policy;
- `run_suzuki_stage_split_delete_fold_sealed_1030.py`: untouched 100-episode
  sealed confirmation of the promoted stage-split policy;
- `run_suzuki_component_ard_kg.py` and
  `run_suzuki_component_ard_kg_validation_1020.py`: finite-pool one-step
  knowledge-gradient discovery and frozen validation;
- `greedy_mean`: posterior mean at every query.

Run from the dataset root:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=submission/code/bgolearn_rxn_extension/src \
python submission/code/experiments/run_policy_ablation.py \
  --datasets Suzuki Buchwald_sub4 \
  --seeds 100 \
  --budget 40
```

Outputs are written beneath `submission/results/experimental_investigations/`.
They are public-table diagnostics, not evidence for selecting hyperparameters
for a hidden evaluation. The experiment manifest marks this limitation.

To extend only the packaged policy's seed-robustness screen without rerunning
the policy ablations:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python submission/code/experiments/run_policy_ablation.py \
  --datasets Suzuki Buchwald_sub4 \
  --variants metric_aware \
  --seeds 200 300 \
  --budget 40
```

## Random-search reference

Run uniform random sampling without replacement over all 20 official seeds:

```bash
python submission/code/experiments/run_random_baseline.py
```

Candidate selection uses only IDs and the seed. Yields are returned one at a
time through the same isolated oracle used by the model-based campaigns.

## SOTA search and statistics

`run_sota_search.py` evaluates surrogate, policy, product-context, and
strain-sensor acquisition variants with independent process workers. The final
selected variant is `gp_interaction_gp_warmstart_logei`.

Rebuild the 20-run mean, sample SD, 95% confidence interval, and penalized-t95
tables from the saved trajectories:

```bash
python submission/code/experiments/analyze_sota_results.py
```

The final tables are written to
`submission/results/experimental_investigations/sota_search/final/`.

## Train-only transfer proxies

`run_transfer_proxy.py` constructs sequential development episodes exclusively
from the labeled training CSVs. It writes the input SHA256, episode contract,
selected variants, and `public_test_labels_used: false` to each manifest.

Run the focused Buchwald safe-transfer comparison:

```bash
python submission/code/experiments/run_transfer_proxy.py \
  --datasets Buchwald_sub4 \
  --buchwald-repeats 8 \
  --variants target_only_gp pooled_gp_scalar_task \
    safe_transfer_80_20 safe_transfer_50_50 \
  --output-dir submission/results/experimental_investigations/transfer_proxy_safe_mixture
```

Run the Suzuki crossed-role diagnostic:

```bash
python submission/code/experiments/run_transfer_proxy.py \
  --datasets Suzuki \
  --suzuki-episode-mode role \
  --variants target_only_gp pooled_role_task_gp \
  --output-dir submission/results/experimental_investigations/transfer_proxy_suzuki_roles
```

The default Suzuki episode mode remains `pair`. Role mode is a train-side
diagnostic and does not modify the production Suzuki YAML or candidate policy.
See `submission/results/experimental_investigations/TRANSFER_MULTITASK_REPORT.md`
for paired confidence intervals and the method decision.

## Suzuki 40-round development environment

`run_suzuki_40round_landscape_zoo.py` runs the exact 29-initial,
5,731-candidate, 40-query geometry against balanced train-derived additive,
interaction/low-rank, tree, and rugged pseudo-oracles. Development and test
seed families are frozen and disjoint. Public Suzuki candidate yields are
protected and must not be used by this runner.

Run the development reference:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=submission/code/bgolearn_rxn_extension/src \
python submission/code/experiments/run_suzuki_40round_landscape_zoo.py \
  --split dev --episode-count 20 \
  --variants component_ard stage_split --jobs 8
```

The Phase 7 selection is `component_ard_gp`: posterior mean at query 1 and
stable LogEI through query 40. See
`submission/results/experimental_investigations/BGOLEARN_FRONTIER_PHASE7_REPORT.md`
for the sealed test, rejected challengers, confidence intervals, and audit.

## Phase 8 robustness continuation

The same runner now supports `phase8_dev` and `phase8_test`. Phase 8 development
uses fresh seeds over the balanced four-family zoo. Phase 8 test is a disjoint,
unopened panel of sparse-pair and local-neighborhood teacher mechanisms.

Phase 8 evaluated query-one rank consensus, localized jackknife uncertainty,
surprise-gated expert control, and ARD-supported maximin consensus. None passed
the full initial/best/t95/AUC/CVaR/runtime gate, so production remains
`component_ard_gp` and Phase 8 test remains sealed. See
`submission/results/experimental_investigations/BGOLEARN_FRONTIER_PHASE8_REPORT.md`.
