from product_intelligence.core.config import settings
from product_intelligence.data.synthetic import generate_accounts
from product_intelligence.features.builders import build_feature_frame
from product_intelligence.models.registry import ModelRegistry
from product_intelligence.models.train import train_churn_model
from product_intelligence.serving.rollout import RolloutController


def _register_churn(reg, seed, promote):
    raw = generate_accounts(800, seed=seed)
    df = build_feature_frame(raw)
    df["will_churn"] = raw["will_churn"]
    res = train_churn_model(df, model_type="logreg", calibration="none", seed=seed)
    return reg.register(
        "churn",
        res.pipeline,
        model_type="logreg",
        metrics=res.metrics,
        threshold=res.threshold,
        band_thresholds=res.band_thresholds,
        promote=promote,
    )


def test_shadow_and_canary(tmp_path, monkeypatch):
    reg = ModelRegistry(tmp_path)
    _register_churn(reg, 1, promote=True)  # production
    _register_churn(reg, 2, promote=False)  # candidate (staging, latest)

    controller = RolloutController(reg, candidate_selector="latest")
    assert controller.has_candidate

    payloads = generate_accounts(150, seed=5).to_dict(orient="records")
    report = controller.shadow_report(payloads)
    assert report["has_candidate"] is True
    assert report["n"] == 150
    assert report["mean_abs_divergence"] >= 0

    monkeypatch.setattr(settings, "canary_percent", 0.4)
    served = {"production": 0, "candidate": 0}
    for p in payloads:
        served[controller.score(p).served_by] += 1
    assert served["candidate"] > 0  # canary actually routes some traffic
