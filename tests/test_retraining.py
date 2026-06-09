from product_intelligence.monitoring.drift import DriftReport
from product_intelligence.pipelines.retraining import decide_retrain


def test_no_retrain_when_stable():
    report = DriftReport(n_reference=100, n_current=100, overall_severity="stable")
    decision = decide_retrain(report, current_roc_auc=0.8)
    assert decision.should_retrain is False


def test_retrain_on_significant_drift_or_low_auc():
    report = DriftReport(
        n_reference=100,
        n_current=100,
        overall_severity="significant",
        flagged_features=["nps"],
        prediction_psi=0.4,
    )
    decision = decide_retrain(report, current_roc_auc=0.6)
    assert decision.should_retrain is True
    assert len(decision.reasons) >= 1
