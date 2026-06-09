from product_intelligence.models.inference import get_inference_service, heuristic_signals


def test_score_account_shape(sample_account):
    result = get_inference_service().score_account(sample_account)
    assert 0.0 <= result["churn_probability"] <= 1.0
    assert result["risk_band"] in {"low", "medium", "high"}
    assert len(result["recommended_features"]) == 3
    assert result["explainer"] == "heuristic"
    assert "model_version" in result and result["model_version"]


def test_explain_path_uses_shap_or_fallback(sample_account):
    result = get_inference_service().score_account(sample_account, explain=True)
    assert result["explainer"] in {"shap", "fallback"}
    assert len(result["top_risk_drivers"]) >= 1


def test_heuristic_signals_high_risk(sample_account):
    drivers, protective = heuristic_signals(sample_account)
    assert any("active days" in d.lower() for d in drivers)
