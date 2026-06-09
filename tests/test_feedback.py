from product_intelligence.feedback.store import FeedbackStore


def test_feedback_log_and_metrics(tmp_path):
    store = FeedbackStore(tmp_path / "fb.jsonl")
    for i in range(40):
        store.log(f"a{i}", 0.7 if i % 2 else 0.2, i % 2 == 1, "v1", actual_churn=i % 2)
    metrics = store.realized_metrics()
    assert metrics["n_logged"] == 40
    assert metrics["n_labelled"] == 40
    assert metrics["accuracy"] == 1.0


def test_feedback_handles_missing_labels(tmp_path):
    store = FeedbackStore(tmp_path / "fb.jsonl")
    store.log("a1", 0.5, True, "v1")  # no actual outcome yet
    m = store.realized_metrics()
    assert m["n_logged"] == 1 and m["n_labelled"] == 0
