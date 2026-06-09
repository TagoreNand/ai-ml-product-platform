# Architecture Notes

## Design goals
- Decouple offline training from online serving.
- Keep product-facing contracts stable while internals evolve.
- Let PM and DS consume outputs (scores + actions) without model internals.
- Provide the production scaffolding (registry, monitoring, observability) that
  a real deployment needs, while staying runnable on a laptop.

## Offline layer
- Synthetic generator stands in for the warehouse / event lake (`data/synthetic.py`).
- `build_feature_frame` is the single feature contract shared by training and
  inference — the key defence against training/serving skew.
- `models/train.py` trains, **calibrates** (isotonic), and **threshold-tunes** the
  churn model on out-of-fold predictions, and evaluates per segment for fairness.
- `models/registry.py` writes immutable, content-addressed versions and promotes
  one to `production`. Retraining is register + promote, never overwrite.

## Online layer
- `api/main.py` is a FastAPI service: lifespan model warm-up, request-id +
  metrics middleware, API-key auth, rate limiting, typed responses, error
  envelopes, Prometheus `/metrics`, liveness/readiness probes.
- `models/inference.py` returns decisions, not just probabilities: calibrated
  score, data-driven risk band, drivers, recommendations and prescriptive actions.
- Scoring is fast by default (heuristic drivers); SHAP explanations are opt-in to
  keep the hot path cheap. Batch scoring is vectorised (one predict call).

## Ops layer
- `pipelines/daily_batch_scoring.py` scores a snapshot and builds a drift report.
- `monitoring/drift.py` computes PSI / KS / categorical distances with severity.
- `pipelines/retraining.py` turns drift + live metrics into an auditable decision.

## Future production upgrades
- Replace synthetic data with warehouse extracts and a feature store.
- Externalise the registry/metrics to MLflow + Grafana/Alertmanager.
- Add async batch queue, shadow/canary rollout, and online feedback logging.
