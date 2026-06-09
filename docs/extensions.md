# Advanced extensions

These build on the core platform and are all self-contained, tested, and runnable
offline (real integrations are wired with graceful fallbacks when no external
infra/keys are present).

## 1. Warehouse extracts (pluggable data sources)
`data/sources.py` defines a `DataSource` interface with `synthetic`, `file`
(CSV/Parquet/JSON) and `sql` (SQLite/warehouse) implementations, selected by
`DATA_SOURCE`. The training pipeline loads through `get_data_source(...)`, so
moving from demo data to warehouse extracts is a config change, not a code change.

## 2. Feature store
`features/store.py` is a file-backed feature store with **online** retrieval
(latest vector per entity) and **point-in-time-correct historical** retrieval (an
as-of join returning values *as known* at each label time — the key defence
against training-set leakage). The interface mirrors Feast closely enough that a
real store is a drop-in.

## 3. Experiment assignment + uplift modelling
- `experimentation/assignment.py`: deterministic, stateless hash bucketing
  (stable per unit, independent across experiments, weight-respecting splits).
- `models/uplift.py`: a T-learner estimating the **incremental** effect of a CS
  save-play, evaluated with a Qini coefficient. Targets *persuadable* accounts,
  not just high-risk ones. Trained and registered as the `uplift` model.

## 4. Real LLM copilot
`services/llm.py` is a provider-agnostic client (OpenAI / Anthropic via stdlib
HTTP) behind the copilot seam. Configure `LLM_PROVIDER` + `LLM_API_KEY` to use a
real model; otherwise the deterministic template runs (and is the automatic
fallback if a live call fails), so the service never hard-errors.

## 5. Online feedback logging + retraining triggers
`feedback/store.py` appends served predictions and realised outcomes to a JSONL
log and computes **realised** accuracy/ROC-AUC. That realised performance feeds
`pipelines/retraining.decide_retrain`, alongside drift, to gate retraining.

## 6. Shadow + canary rollout
`serving/rollout.py` runs a candidate model in **shadow** (scored alongside
production, divergence logged) and supports **canary** routing of a configurable
traffic fraction (`CANARY_PERCENT`) via deterministic assignment. Rollback is a
registry re-point of `production` — no redeploy.

## New endpoints
| Method | Path | Extension |
|---|---|---|
| POST | `/v1/accounts/uplift` | Uplift (incremental retention effect) |
| GET | `/v1/experiments/{experiment}/assignment?unit_id=` | Experiment variant |
| POST | `/v1/feedback` | Log a served prediction / outcome |
| GET | `/v1/feedback/metrics` | Realised performance from feedback |
| POST | `/v1/serving/shadow` | Production-vs-candidate divergence report |

## Relevant configuration
`DATA_SOURCE`, `DATA_PATH`, `DATA_SQL_PATH`, `FEATURE_STORE_DIR`,
`EXPERIMENT_SALT`, `FEEDBACK_LOG_PATH`, `LLM_PROVIDER`, `LLM_API_KEY`,
`LLM_MODEL`, `SHADOW_ENABLED`, `CANARY_PERCENT`. See `.env.example`.
