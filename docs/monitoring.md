# Monitoring & drift

## What we watch
- **Data drift** per feature: Population Stability Index (PSI) and a two-sample
  Kolmogorov-Smirnov statistic for numerics; an L1 distribution distance for
  categoricals.
- **Prediction drift**: PSI of the live churn-score distribution vs the training
  reference.
- **Service health**: request latency, error rate, throughput (Prometheus).

## Severity
PSI convention: `< 0.10` stable, `0.10-0.25` moderate, `>= 0.25` significant.
`build_drift_report` rolls per-feature results into an `overall_severity` and a
list of `flagged_features`.

## Reference snapshot
Training persists `artifacts/reference_sample.csv` (representative features +
production churn scores). The batch job and the `/v1/monitoring/drift` endpoint
compare live data against it.

## From signal to action
A drift report feeds `decide_retrain`, which recommends retraining on
significant drift, a prediction-PSI breach, or a live performance drop. The
recommendation is surfaced to operators rather than auto-applied.

## Metrics catalogue
| Metric | Type | Labels |
|---|---|---|
| `pulse360_http_requests_total` | counter | method, path, status |
| `pulse360_http_request_duration_seconds` | histogram | method, path |
| `pulse360_predictions_total` | counter | risk_band, model_version |
| `pulse360_churn_probability` | histogram | - |
| `pulse360_inference_duration_seconds` | histogram | - |
| `pulse360_feature_drift_psi` | gauge | feature |
| `pulse360_model_info` | gauge | model, version |
