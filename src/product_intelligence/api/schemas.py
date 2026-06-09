"""Pydantic request/response contracts for the API.

Typed responses give the service a real OpenAPI schema (clients, docs, contract
tests) instead of opaque dicts.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class AccountFeatures(BaseModel):
    account_id: str = Field(examples=["acct_101"])
    company_size: str
    industry: str
    region: str
    plan_tier: str
    contract_value: float = Field(ge=0)
    tenure_months: int = Field(ge=0)
    nps: float = Field(ge=-100, le=100)
    active_days_30d: int = Field(ge=0, le=31)
    weekly_active_users: int = Field(ge=0)
    monthly_active_users: int = Field(ge=0)
    feature_adoption_rate: float = Field(ge=0, le=1)
    workflow_runs_30d: float = Field(ge=0)
    api_calls_30d: float = Field(ge=0)
    support_tickets_90d: int = Field(ge=0)
    open_bug_count: int = Field(ge=0)
    p1_incidents_90d: int = Field(ge=0)
    onboarding_completion_rate: float = Field(ge=0, le=1)
    usage_growth_90d: float
    renewal_days_remaining: int = Field(ge=0)


class RecommendedFeature(BaseModel):
    feature: str
    score: float


class AccountScore(BaseModel):
    account_id: str
    model_version: str
    churn_probability: float
    risk_band: str
    predicted_churn: bool
    decision_threshold: float
    top_risk_drivers: list[str]
    protective_factors: list[str] = []
    recommended_features: list[RecommendedFeature]
    recommended_actions: list[str]
    explainer: str
    latency_ms: float


class CopilotResponse(BaseModel):
    account_id: str
    summary: str
    score: AccountScore


class BatchScoreRequest(BaseModel):
    accounts: list[AccountFeatures]
    explain: bool = False


class BatchScoreResponse(BaseModel):
    count: int
    model_version: str
    results: list[AccountScore]


class Initiative(BaseModel):
    name: str
    reach: float = Field(ge=0, le=10)
    impact: float = Field(ge=0, le=10)
    confidence: float = Field(ge=0, le=10)
    effort: float = Field(ge=1, le=10)
    strategic_alignment: float = Field(ge=0, le=10)
    estimated_model_uplift: float = Field(ge=0, le=1)
    evidence_strength: float = Field(ge=0, le=10)


class PrioritizationRequest(BaseModel):
    initiatives: list[Initiative]


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class ReadinessResponse(BaseModel):
    status: str
    models_loaded: bool
    churn_version: str | None = None
    recommendation_version: str | None = None


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
    request_id: str | None = None


class UpliftResponse(BaseModel):
    account_id: str
    model_version: str
    estimated_uplift: float
    interpretation: str


class AssignmentResponse(BaseModel):
    unit_id: str
    experiment: str
    variant: str


class FeedbackIn(BaseModel):
    account_id: str
    churn_probability: float = Field(ge=0, le=1)
    predicted_churn: bool
    model_version: str
    actual_churn: int | None = Field(default=None, ge=0, le=1)


class FeedbackResponse(BaseModel):
    logged: bool
    record: dict
