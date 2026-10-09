# Changelog

## Unreleased - 2026-07-28

- Added derived/fixed task-role target masking for train-only proxy tasks.
- Added optional task-descriptor covariance attenuation and shrunk task means
  to the structured multitask GP; both preserve legacy behavior when disabled.
- Added the opt-in `ResidualTransferGP` source-prior/residual-adaptation model.
- Added focused multitask, residual-transfer, and proxy-role tests.
- Added train-only product, substrate-pair, and crossed-role proxy evaluation
  with paired confidence intervals and input-hash manifests.
- Kept all production model presets unchanged after the new transfer variants
  failed to improve discovery AUC.

## 0.1.0 — 2026-07-23

- Added strict participant-side dataset loaders and isolated offline oracle.
- Added frozen role-aware exact/name-rule/reviewed-molecular encoder.
- Added structured ANOVA multitask GP.
- Added Bayesian low-rank interaction surrogate.
- Added bootstrapped Extra Trees hedge.
- Added prequential ensemble weighting.
- Added stable analytic LogEI and metric-aware staged policy.
- Added native and stock-Bgolearn compatibility backends.
- Added stateful ask/tell campaign session.
- Added Buchwald and Suzuki presets, trajectory serialization, validation, tests, and documentation.
