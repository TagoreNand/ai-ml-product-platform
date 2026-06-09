from product_intelligence.features.builders import MODEL_COLUMNS, build_feature_frame
from product_intelligence.models.explain import ModelExplainer
from product_intelligence.models.train import train_churn_model


def test_fallback_explainer_returns_drivers(labelled_frame, sample_account):
    result = train_churn_model(labelled_frame, model_type="hist_gbdt", calibration="none", seed=7)
    background = labelled_frame[MODEL_COLUMNS].sample(60, random_state=0)
    explainer = ModelExplainer(result.pipeline, background, enable_shap=False)
    assert explainer.mode == "fallback"

    row = build_feature_frame([sample_account])[MODEL_COLUMNS]
    contribs = explainer.contributions(row)
    assert len(contribs) == len(MODEL_COLUMNS)
    drivers = explainer.top_risk_drivers(row, k=5)
    assert 1 <= len(drivers) <= 5
