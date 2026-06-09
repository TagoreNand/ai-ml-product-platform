"""Account copilot - turns a model score into a PM/CS-ready narrative.

Uses a real LLM provider when configured (`LLM_PROVIDER` + `LLM_API_KEY`),
otherwise a deterministic template. The template is also the automatic fallback
if a live LLM call fails, so the endpoint never hard-errors on provider issues.
"""

from __future__ import annotations

from product_intelligence.core.logging import get_logger
from product_intelligence.services.llm import get_llm_client

logger = get_logger(__name__)

_SYSTEM = (
    "You are a B2B SaaS customer-success copilot. Given a churn-risk assessment, "
    "write a concise, action-oriented summary (<=4 sentences) for a CSM/PM. "
    "Be specific and do not invent numbers."
)


def build_account_summary(score_payload: dict) -> str:
    client = get_llm_client()
    if client is None:
        return _template_summary(score_payload)
    try:
        return client.generate(prompt=_build_prompt(score_payload), system=_SYSTEM)
    except Exception as exc:  # pragma: no cover - network/provider failure path
        logger.warning("LLM copilot failed (%s); using template fallback", exc)
        return _template_summary(score_payload)


def _build_prompt(score_payload: dict) -> str:
    recs = ", ".join(r["feature"] for r in score_payload.get("recommended_features", []))
    return (
        f"Account: {score_payload['account_id']}\n"
        f"Churn probability: {score_payload['churn_probability']:.2f} "
        f"(band: {score_payload['risk_band']}, model {score_payload.get('model_version', 'n/a')})\n"
        f"Risk drivers: {', '.join(score_payload.get('top_risk_drivers', []))}\n"
        f"Protective factors: {', '.join(score_payload.get('protective_factors', [])) or 'none'}\n"
        f"Recommended features to promote: {recs}\n"
        f"Suggested actions: {', '.join(score_payload.get('recommended_actions', []))}\n"
        "Write the summary."
    )


def _template_summary(score_payload: dict) -> str:
    recs = score_payload.get("recommended_features") or [{"feature": "n/a"}]
    top_feature = recs[0]["feature"]
    drivers = score_payload.get("top_risk_drivers", [])
    top_drivers = ", ".join(drivers[:3]) if drivers else "no dominant signal"
    risk = score_payload["risk_band"].upper()
    protective = score_payload.get("protective_factors") or []
    protective_clause = f" Offsetting strengths: {', '.join(protective)}." if protective else ""
    actions = score_payload.get("recommended_actions", [])[:3]
    return (
        f"Account {score_payload['account_id']} is assessed as {risk} churn risk "
        f"(probability {score_payload['churn_probability']:.2f}, model "
        f"{score_payload.get('model_version', 'n/a')}). "
        f"Primary drivers: {top_drivers}.{protective_clause} "
        f"Recommended product intervention: promote {top_feature}. "
        f"Next steps: {'; '.join(actions) if actions else 'monitor'}."
    )
