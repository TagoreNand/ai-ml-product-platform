"""FastAPI inference service for Pulse360.

Production-shaped: lifespan model warm-up, request-id + metrics middleware,
config-gated API-key auth and rate limiting, typed responses, structured error
envelopes, Prometheus ``/metrics`` and liveness/readiness probes.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import Depends, FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from product_intelligence.api.deps import rate_limit, require_api_key
from product_intelligence.api.middleware import ObservabilityMiddleware
from product_intelligence.api.schemas import (
    AccountFeatures,
    AccountScore,
    AssignmentResponse,
    BatchScoreRequest,
    BatchScoreResponse,
    CopilotResponse,
    FeedbackIn,
    FeedbackResponse,
    HealthResponse,
    PrioritizationRequest,
    ReadinessResponse,
    UpliftResponse,
)
from product_intelligence.core.config import settings
from product_intelligence.core.logging import configure_logging, get_logger, request_id_var
from product_intelligence.core.metrics import CONTENT_TYPE_LATEST, render_latest
from product_intelligence.experimentation.assignment import assign_variant
from product_intelligence.features.builders import (
    CATEGORICAL_COLUMNS,
    MODEL_COLUMNS,
    NUMERIC_COLUMNS,
    build_feature_frame,
)
from product_intelligence.feedback.store import FeedbackStore
from product_intelligence.models.inference import get_inference_service
from product_intelligence.monitoring.drift import build_drift_report
from product_intelligence.pipelines.retraining import decide_retrain
from product_intelligence.services.copilot import build_account_summary
from product_intelligence.services.prioritization import prioritize_initiatives
from product_intelligence.serving.rollout import get_rollout_controller

logger = get_logger(__name__)

PROTECTED = [Depends(rate_limit), Depends(require_api_key)]
feedback_store = FeedbackStore(settings.feedback_log_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(settings.log_level, settings.log_json)
    try:
        get_inference_service()
        logger.info("models warmed up at startup")
    except Exception as exc:  # pragma: no cover - depends on artifacts present
        logger.warning("model warm-up skipped (%s) - train artifacts first", exc)
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Churn risk, next-best-feature recommendations, copilot summaries and roadmap prioritization.",
    lifespan=lifespan,
)
app.add_middleware(ObservabilityMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "detail": exc.errors(),
            "request_id": request_id_var.get(),
        },
    )


@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception):  # pragma: no cover
    logger.exception("unhandled error")
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "detail": str(exc), "request_id": request_id_var.get()},
    )


# ---------------- operational endpoints ----------------
@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    return HealthResponse(status="ok", service=settings.app_name, version=settings.app_version)


@app.get("/health/ready", response_model=ReadinessResponse, tags=["ops"])
def readiness() -> ReadinessResponse:
    try:
        svc = get_inference_service()
        return ReadinessResponse(
            status="ready",
            models_loaded=True,
            churn_version=svc.churn_record.version,
            recommendation_version=svc.rec_record.version,
        )
    except Exception:
        return ReadinessResponse(status="not_ready", models_loaded=False)


@app.get("/version", tags=["ops"])
def version() -> dict:
    svc_models = {}
    try:
        svc = get_inference_service()
        svc_models = {
            "churn": svc.churn_record.version,
            "recommendation": svc.rec_record.version,
        }
    except Exception:
        pass
    return {
        "service_version": settings.app_version,
        "models": svc_models,
        "environment": settings.environment,
    }


@app.get("/metrics", tags=["ops"])
def metrics() -> Response:
    if not settings.enable_metrics:
        return Response(status_code=404)
    return Response(render_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/v1/models", tags=["ops"], dependencies=PROTECTED)
def list_models() -> dict:
    svc = get_inference_service()
    return {
        "churn": svc.registry.list_versions("churn"),
        "recommendation": svc.registry.list_versions("recommendation"),
    }


# ---------------- inference endpoints ----------------
@app.post(
    "/v1/accounts/score", response_model=AccountScore, tags=["inference"], dependencies=PROTECTED
)
def score_account(
    payload: AccountFeatures,
    explain: bool = Query(False, description="Compute SHAP attributions (slower)."),
) -> AccountScore:
    result = get_inference_service().score_account(payload.model_dump(), explain=explain)
    return AccountScore(**result)


@app.post(
    "/v1/accounts/explain", response_model=AccountScore, tags=["inference"], dependencies=PROTECTED
)
def explain_account(payload: AccountFeatures) -> AccountScore:
    result = get_inference_service().score_account(payload.model_dump(), explain=True)
    return AccountScore(**result)


@app.post(
    "/v1/accounts/score:batch",
    response_model=BatchScoreResponse,
    tags=["inference"],
    dependencies=PROTECTED,
)
def score_batch(payload: BatchScoreRequest) -> BatchScoreResponse:
    accounts = payload.accounts[: settings.max_batch_size]
    svc = get_inference_service()
    results = svc.score_batch([a.model_dump() for a in accounts], explain=payload.explain)
    return BatchScoreResponse(
        count=len(results),
        model_version=svc.model_version,
        results=[AccountScore(**r) for r in results],
    )


@app.post(
    "/v1/accounts/copilot-summary",
    response_model=CopilotResponse,
    tags=["inference"],
    dependencies=PROTECTED,
)
def copilot_summary(payload: AccountFeatures) -> CopilotResponse:
    svc = get_inference_service()
    score = svc.score_account(payload.model_dump())
    summary = build_account_summary(score)
    return CopilotResponse(
        account_id=score["account_id"], summary=summary, score=AccountScore(**score)
    )


@app.post("/v1/roadmap/prioritize", tags=["product"], dependencies=PROTECTED)
def roadmap_prioritize(payload: PrioritizationRequest) -> dict:
    ranked = prioritize_initiatives([item.model_dump() for item in payload.initiatives])
    return {"ranked_initiatives": ranked}


# ---------------- monitoring endpoint ----------------
@app.post("/v1/monitoring/drift", tags=["monitoring"], dependencies=PROTECTED)
def drift_check(payload: BatchScoreRequest) -> dict:
    svc = get_inference_service()
    reference = svc.registry.load_background("churn", "production")
    current = build_feature_frame(pd.DataFrame([a.model_dump() for a in payload.accounts]))[
        MODEL_COLUMNS
    ]
    scored = svc.score_batch([a.model_dump() for a in payload.accounts])
    report = build_drift_report(
        reference=reference if reference is not None else current,
        current=current,
        numeric_features=NUMERIC_COLUMNS,
        categorical_features=CATEGORICAL_COLUMNS,
        current_scores=pd.Series([s["churn_probability"] for s in scored]).to_numpy(),
    )
    decision = decide_retrain(report)
    return {"drift": report.to_dict(), "retraining": decision.to_dict()}


# ---------------- experimentation, uplift, feedback, rollout (extensions) ----------------
@app.post(
    "/v1/accounts/uplift", response_model=UpliftResponse, tags=["inference"], dependencies=PROTECTED
)
def uplift(payload: AccountFeatures) -> UpliftResponse:
    return UpliftResponse(**get_inference_service().score_uplift(payload.model_dump()))


@app.get(
    "/v1/experiments/{experiment}/assignment",
    response_model=AssignmentResponse,
    tags=["product"],
    dependencies=PROTECTED,
)
def experiment_assignment(experiment: str, unit_id: str) -> AssignmentResponse:
    a = assign_variant(unit_id, experiment, salt=settings.experiment_salt)
    return AssignmentResponse(unit_id=a.unit_id, experiment=a.experiment, variant=a.variant)


@app.post(
    "/v1/feedback", response_model=FeedbackResponse, tags=["monitoring"], dependencies=PROTECTED
)
def log_feedback(payload: FeedbackIn) -> FeedbackResponse:
    record = feedback_store.log(
        account_id=payload.account_id,
        churn_probability=payload.churn_probability,
        predicted_churn=payload.predicted_churn,
        model_version=payload.model_version,
        actual_churn=payload.actual_churn,
    )
    return FeedbackResponse(logged=True, record=record)


@app.get("/v1/feedback/metrics", tags=["monitoring"], dependencies=PROTECTED)
def feedback_metrics() -> dict:
    return feedback_store.realized_metrics()


@app.post("/v1/serving/shadow", tags=["monitoring"], dependencies=PROTECTED)
def serving_shadow(payload: BatchScoreRequest) -> dict:
    controller = get_rollout_controller()
    return controller.shadow_report([a.model_dump() for a in payload.accounts])
