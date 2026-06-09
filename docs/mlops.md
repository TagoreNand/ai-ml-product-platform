# MLOps: model lifecycle

## Training contract
`build_feature_frame` is the single source of truth for features and is called
by **both** the offline training pipeline and the online inference service. This
eliminates training/serving skew - the most common silent failure in production
ML.

## Modelling decisions
- **Calibration.** Business actions (CSM outreach, save-plays) key off
  *probabilities*, not just rankings, so the churn model is wrapped in
  `CalibratedClassifierCV` (isotonic by default). We report Brier score and a
  reliability curve in the model card.
- **Threshold optimisation.** Instead of a naive 0.5 cut, the operating
  threshold is chosen to maximise F1 on **out-of-fold** predictions (no test
  leakage). The model card shows `f1` vs `f1_at_0.5` so the lift is auditable.
- **Risk bands** (low/medium/high) come from score quantiles, not magic numbers.
- **Model factory.** `CHURN_MODEL_TYPE` switches between LogisticRegression,
  HistGradientBoosting (default) and XGBoost without touching pipeline code.
  XGBoost is optional; if unavailable the factory falls back to HGB.

## Model registry
`ModelRegistry` is a lightweight, file-based, content-addressed registry:
- immutable versions: `models/<name>/<version>/{model.joblib,metadata.json,background.csv}`
- a JSON index (`registry.json`) tracking the `production` pointer per model
- `register()` writes a new version and (optionally) promotes it
- serving resolves `production` / `latest` / an explicit version at load time

Retraining is therefore a **register + promote** operation, not a file
overwrite, and every served prediction can be traced to an exact model version
(returned as `model_version` on every response).

## Retraining policy
`pipelines/retraining.decide_retrain` turns a drift report + live performance
into an auditable decision (`should_retrain`, `reasons`, `severity`). Triggers:
significant data drift, prediction-PSI breach, or live ROC-AUC below a floor.
The decision is surfaced, not auto-applied, keeping a human/PM in the loop.

## Pipelines
- `offline_training` - data -> features -> train+calibrate -> register -> model card.
- `daily_batch_scoring` - vectorised scoring of a snapshot + drift report + retrain recommendation.
