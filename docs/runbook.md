# Operational Runbook

## Train & promote
- `make train` runs `offline_training`: trains, calibrates, registers a new
  version, promotes it to `production`, writes the reference snapshot + model card.
- Every model version is immutable; `production` is a pointer in `registry.json`.

## Before deployment
- CI must be green: ruff, ruff-format, mypy, pytest (with coverage), docker build.
- Review the generated model card (`docs/model_card_churn.md`): ROC-AUC/PR-AUC,
  calibration curve, `f1` vs `f1_at_0.5`, and per-segment AUC for fairness gaps.

## After deployment
- Probe `/health/ready`; confirm `/version` reports the expected model versions.
- Watch `/metrics`: request latency, error rate, `pulse360_churn_probability`
  distribution and `pulse360_feature_drift_psi`.
- Run `make score` (daily batch) and inspect `scored_output/drift_report.json`.

## Rollback
- Re-promote the previous good version: `ModelRegistry.promote("churn", "<version>")`,
  then restart (or call `reset_inference_service`). No redeploy required.

## Rollback conditions
- latency exceeds SLO for 15 minutes
- churn probability distribution collapses or spikes abnormally
- drift report `overall_severity == "significant"` with prediction-PSI breach
- recommendation class mix becomes unstable after a launch
