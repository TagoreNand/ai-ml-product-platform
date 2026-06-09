from product_intelligence.core.config import settings
from product_intelligence.services.copilot import build_account_summary
from product_intelligence.services.llm import get_llm_client

SCORE = {
    "account_id": "acct_x",
    "model_version": "v1",
    "churn_probability": 0.82,
    "risk_band": "high",
    "top_risk_drivers": ["Low active days"],
    "protective_factors": [],
    "recommended_features": [{"feature": "copilot_assist", "score": 0.5}],
    "recommended_actions": ["Trigger CSM outreach within 48 hours"],
}


def test_mock_returns_none_client():
    assert get_llm_client() is None  # mock / no key by default


def test_copilot_falls_back_to_template():
    summary = build_account_summary(SCORE)
    assert "acct_x" in summary and "HIGH" in summary


def test_openai_client_built_when_configured():
    original = (settings.llm_provider, settings.llm_api_key, settings.copilot_use_mock)
    try:
        settings.copilot_use_mock = False
        settings.llm_provider = "openai"
        settings.llm_api_key = "sk-test"
        assert get_llm_client() is not None
    finally:
        settings.copilot_use_mock, settings.llm_provider, settings.llm_api_key = (
            original[2],
            original[0],
            original[1],
        )
