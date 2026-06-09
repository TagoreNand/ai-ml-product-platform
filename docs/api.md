# API reference

Base URL (local): `http://127.0.0.1:8000`. Interactive docs at `/docs`.

## Operational
| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness probe |
| GET | `/health/ready` | Readiness probe (models loaded) |
| GET | `/version` | Service + model versions |
| GET | `/metrics` | Prometheus exposition |
| GET | `/v1/models` | Registry versions (protected) |

## Inference
| Method | Path | Description |
|---|---|---|
| POST | `/v1/accounts/score?explain=false` | Score one account (fast heuristic drivers) |
| POST | `/v1/accounts/explain` | Score + SHAP attributions (slower, opt-in) |
| POST | `/v1/accounts/score:batch` | Vectorised batch scoring |
| POST | `/v1/accounts/copilot-summary` | PM/CS-ready narrative |
| POST | `/v1/roadmap/prioritize` | Rank initiatives (RICE + strategy + uplift) |
| POST | `/v1/monitoring/drift` | Drift report vs training reference |
| POST | `/v1/accounts/uplift` | Incremental retention effect (uplift model) |
| GET | `/v1/experiments/{experiment}/assignment?unit_id=` | Deterministic A/B variant |
| POST | `/v1/feedback` | Log a served prediction / realised outcome |
| GET | `/v1/feedback/metrics` | Realised accuracy / ROC-AUC from feedback |
| POST | `/v1/serving/shadow` | Production-vs-candidate divergence report |

## Security
- **API key** (optional): set `API_KEY` to require the `X-API-Key` header on
  protected routes. Empty key = open (local dev).
- **Rate limiting**: sliding-window per client (key or IP), `RATE_LIMIT_*`.
- **CORS**: configured via `CORS_ALLOW_ORIGINS`.

## Observability
- Every request gets a correlation id (honours inbound `X-Request-ID`), bound to
  the structured JSON logs and echoed on the response.
- Prometheus metrics: request count/latency, prediction count by risk band,
  churn-probability histogram, inference latency, per-feature drift PSI gauge.

## Response shape (score)
```json
{
  "account_id": "acct_101",
  "model_version": "20260101T120000-ab12cd34ef",
  "churn_probability": 0.78,
  "risk_band": "high",
  "predicted_churn": true,
  "decision_threshold": 0.24,
  "top_risk_drivers": ["..."],
  "protective_factors": ["..."],
  "recommended_features": [{"feature": "copilot_assist", "score": 0.41}],
  "recommended_actions": ["..."],
  "explainer": "heuristic",
  "latency_ms": 7.3
}
```
