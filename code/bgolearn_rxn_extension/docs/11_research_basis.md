# 11. Research Basis

This document records the main research ideas reflected in the implementation. It is not an exhaustive literature review.

## Bgolearn

**Cao et al., “Bgolearn: a Unified Bayesian Optimization Framework for Accelerating Materials Discovery,” npj Computational Materials, 2026.**

- Paper: https://doi.org/10.1038/s41524-026-02226-3
- Repository: https://github.com/Bin-Cao/Bgolearn
- Documentation: https://bgolearn.netlify.app/getting_started

Relevant lesson: Bgolearn provides an accessible finite-pool BO workflow, multiple surrogate families, and custom-model integration. The extension retains that workflow while specializing the representation and uncertainty model for reactions.

## Stable Log Expected Improvement

**Ament et al., “Unexpected Improvements to Expected Improvement for Bayesian Optimization,” 2023/2024.**

- https://arxiv.org/abs/2310.20708

Relevant lesson: analytic EI can vanish numerically. Evaluating its logarithm through stable expressions can restore meaningful candidate distinctions. `bgolearn_rxn.acquisition.stable_log_expected_improvement` implements the single-objective Gaussian form.

## Designed HTE structure

**Statistical analysis of designed high-throughput Buchwald–Hartwig data, Communications Chemistry, 2026.**

- https://www.nature.com/articles/s42004-025-01866-8

Relevant lesson: factorial design, main effects, sparse interactions, and heredity matter. This motivates role-aware ANOVA kernels and strongly shrunk interaction products.

## Deep-kernel reaction models

**Singh et al., “Deep Kernel learning for reaction outcome prediction and optimization,” Communications Chemistry, 2024.**

- https://www.nature.com/articles/s42004-024-01219-x

Relevant lesson: learned molecular representations can be coupled to GP uncertainty. In very small-data campaigns, frozen or lightly adapted representations are safer than unrestricted end-to-end learning.

## Gaussian processes in chemistry

**Griffiths et al., “GAUCHE: A Library for Gaussian Processes in Chemistry,” NeurIPS, 2023.**

- https://proceedings.neurips.cc/paper_files/paper/2023/hash/f2b1b2e974fa5ea622dd87f22815f423-Abstract-Conference.html

Relevant lesson: chemistry-specific kernels, including fingerprint/Tanimoto kernels, can improve regression, uncertainty quantification, and BO. A dedicated Tanimoto role kernel is a high-priority extension beyond the included frozen feature blocks.

## Multitask chemical reaction optimization

**Multitask Bayesian optimization of chemical reactions, ACS Central Science, 2023.**

- https://doi.org/10.1021/acscentsci.3c00050

Relevant lesson: related reaction histories can accelerate a new campaign. This motivates the Buchwald task kernel and target-specific incumbent separation.

## Robust HEBO modernization

**Zhukov et al., “Modernizing HEBO: a robust Bayesian optimization baseline for practical heteroskedastic and non-stationary problems,” preprint, July 2026.**

- https://arxiv.org/abs/2607.10669

Relevant lesson: surrogate training, output warping, heteroskedasticity, and acquisition implementation can matter as much as headline model class. The default offline tables are deterministic, but robust transforms and training practices remain useful.

## LLM-optimized deep kernels

**Ranković and Schwaller, “GOLLuM: Gaussian Process Optimized LLMs,” preprint, 2025.**

- https://arxiv.org/abs/2504.06265

Relevant lesson: LLM embeddings can be adapted by GP marginal likelihood and improve scientific BO. Because the public Buchwald dataset is a named evaluation in this work, hidden-task validation and contamination controls are essential.

## Bayesian optimization with in-context learning

**Caldas Ramos et al., “Bayesian Optimization of Catalysis with In-Context Learning,” ACS Central Science, 2026.**

- https://doi.org/10.1021/acscentsci.5c02418

Relevant lesson: frozen LLMs can provide regression and uncertainty directly in language space and guide large catalyst searches. In Bgolearn-RxN, such a model is best treated as an auditable ensemble component or weak prior until independently validated on hidden reactions.

## Automated and batch reaction optimization

**Sin et al., “Highly parallel optimisation of chemical reactions through automation and machine intelligence,” Nature Communications, 2025.**

- https://www.nature.com/articles/s41467-025-61803-0

Relevant lesson: practical laboratory optimization includes batch constraints, chemical noise, selectivity, and operational considerations. The current public benchmark is a useful finite-pool `q=1` abstraction, not a complete autonomous-lab simulation.

## Cost-aware reaction optimization

**Schoepfer et al., “Cost-informed Bayesian reaction optimization,” 2024.**

- https://pmc.ncbi.nlm.nih.gov/articles/PMC11465108/

Relevant lesson: acquisition should be extensible to reagent and experiment cost. The default code keeps costs out because the benchmark objective is yield only.

## What the implementation deliberately does not assume

The code does not assume that:

- one surrogate is universally best;
- LLM priors are calibrated;
- public-dataset gains generalize to hidden chemistry;
- deterministic lookup tables capture laboratory noise;
- a more complicated acquisition always beats a better representation;
- source-table row order is a legitimate feature.

## Research priorities for the next version

1. Role-specific Tanimoto and Matérn kernel mixtures.
2. Fully Bayesian low-rank task covariance.
3. Probabilistic tensor factorization with correlated posterior draws.
4. Raw/logit likelihood model averaging.
5. Frozen chemistry-language embeddings with small GP-trained adapters.
6. Proper qLogEI and plate-aware batch constraints.
7. Replicate-aware heteroskedastic likelihoods.
8. Hidden-substrate and hidden-reagent benchmark validation.
