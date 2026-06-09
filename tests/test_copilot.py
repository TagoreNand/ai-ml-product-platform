from product_intelligence.services.copilot import build_account_summary


def test_summary_mentions_account_and_risk():
    score = {
        "account_id": "acct_x",
        "model_version": "v1",
        "churn_probability": 0.82,
        "risk_band": "high",
        "top_risk_drivers": ["Low active days", "Negative usage growth"],
        "protective_factors": ["Strong feature adoption"],
        "recommended_features": [{"feature": "copilot_assist", "score": 0.5}],
        "recommended_actions": ["Trigger CSM outreach within 48 hours"],
    }
    summary = build_account_summary(score)
    assert "acct_x" in summary
    assert "HIGH" in summary
    assert "copilot_assist" in summary
