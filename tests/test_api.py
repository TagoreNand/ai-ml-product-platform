from fastapi.testclient import TestClient

from product_intelligence.api.main import app

client = TestClient(app)


def test_health_and_ready():
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/health/ready").json()["models_loaded"] is True


def test_version_endpoint():
    body = client.get("/version").json()
    assert "service_version" in body and "models" in body


def test_score_endpoint(sample_account):
    res = client.post("/v1/accounts/score", json=sample_account)
    assert res.status_code == 200
    body = res.json()
    assert body["risk_band"] in {"low", "medium", "high"}
    assert res.headers.get("X-Request-ID")


def test_batch_endpoint(sample_account):
    res = client.post(
        "/v1/accounts/score:batch", json={"accounts": [sample_account, sample_account]}
    )
    assert res.status_code == 200
    assert res.json()["count"] == 2


def test_copilot_endpoint(sample_account):
    res = client.post("/v1/accounts/copilot-summary", json=sample_account)
    assert res.status_code == 200
    assert sample_account["account_id"] in res.json()["summary"]


def test_prioritize_endpoint():
    res = client.post(
        "/v1/roadmap/prioritize",
        json={
            "initiatives": [
                {
                    "name": "A",
                    "reach": 8,
                    "impact": 9,
                    "confidence": 7,
                    "effort": 5,
                    "strategic_alignment": 9,
                    "estimated_model_uplift": 0.11,
                    "evidence_strength": 8,
                }
            ]
        },
    )
    assert res.status_code == 200
    assert res.json()["ranked_initiatives"][0]["name"] == "A"


def test_validation_error_envelope():
    res = client.post("/v1/accounts/score", json={"account_id": "x"})
    assert res.status_code == 422
    assert res.json()["error"] == "validation_error"


def test_metrics_endpoint(sample_account):
    client.post("/v1/accounts/score", json=sample_account)
    res = client.get("/metrics")
    assert res.status_code == 200
    assert b"pulse360_" in res.content


def test_uplift_endpoint(sample_account):
    res = client.post("/v1/accounts/uplift", json=sample_account)
    assert res.status_code == 200
    assert "estimated_uplift" in res.json()


def test_experiment_assignment_endpoint():
    res = client.get("/v1/experiments/save_play/assignment", params={"unit_id": "acct_1"})
    assert res.status_code == 200
    assert res.json()["variant"] in {"control", "treatment"}


def test_feedback_endpoint():
    res = client.post(
        "/v1/feedback",
        json={
            "account_id": "acct_1",
            "churn_probability": 0.6,
            "predicted_churn": True,
            "model_version": "v1",
            "actual_churn": 1,
        },
    )
    assert res.status_code == 200
    assert res.json()["logged"] is True


def test_shadow_endpoint(sample_account):
    res = client.post("/v1/serving/shadow", json={"accounts": [sample_account, sample_account]})
    assert res.status_code == 200
    assert "has_candidate" in res.json()
