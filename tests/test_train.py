from product_intelligence.models.train import train_churn_model, train_recommendation_model


def test_churn_training_metrics(labelled_frame):
    result = train_churn_model(
        labelled_frame, model_type="hist_gbdt", calibration="isotonic", seed=7
    )
    m = result.metrics
    assert 0.5 <= m["roc_auc"] <= 1.0
    assert 0.0 < result.threshold < 1.0
    # Threshold tuning should not be worse than the naive 0.5 cut.
    assert m["f1"] >= m["f1_at_0.5"] - 1e-9
    assert result.calibration_curve is not None
    assert "prob_true" in result.calibration_curve


def test_churn_logreg_variant(labelled_frame):
    result = train_churn_model(labelled_frame, model_type="logreg", calibration="none", seed=7)
    assert result.metrics["model_type"] == "logreg"
    assert 0.5 <= result.metrics["roc_auc"] <= 1.0


def test_recommendation_training(labelled_frame):
    result = train_recommendation_model(labelled_frame, seed=7)
    assert result.metrics["macro_f1"] > 0.3
    assert result.metrics["n_classes"] == 5
