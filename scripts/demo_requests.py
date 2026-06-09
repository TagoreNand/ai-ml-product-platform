"""Exercise EVERY endpoint against an in-process TestClient (no running server).

Run after `python scripts/train_models.py`:

    python scripts/demo_requests.py
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from product_intelligence.api.main import app

client = TestClient(app)

ACCOUNT = {
    "account_id": "acct_demo_001",
    "company_size": "mid_market",
    "industry": "fintech",
    "region": "na",
    "plan_tier": "starter",
    "contract_value": 36000,
    "tenure_months": 10,
    "nps": -8,
    "active_days_30d": 5,
    "weekly_active_users": 6,
    "monthly_active_users": 21,
    "feature_adoption_rate": 0.22,
    "workflow_runs_30d": 38,
    "api_calls_30d": 900,
    "support_tickets_90d": 7,
    "open_bug_count": 3,
    "p1_incidents_90d": 1,
    "onboarding_completion_rate": 0.5,
    "usage_growth_90d": -0.25,
    "renewal_days_remaining": 30,
}

INITIATIVES = {
    "initiatives": [
        {
            "name": "AI onboarding copilot",
            "reach": 8,
            "impact": 9,
            "confidence": 7,
            "effort": 5,
            "strategic_alignment": 9,
            "estimated_model_uplift": 0.11,
            "evidence_strength": 8,
        },
        {
            "name": "Advanced export filters",
            "reach": 6,
            "impact": 5,
            "confidence": 8,
            "effort": 3,
            "strategic_alignment": 4,
            "estimated_model_uplift": 0.02,
            "evidence_strength": 5,
        },
    ]
}


def show(title: str, payload, truncate: int | None = None) -> None:
    print(f"\n=== {title} ===")
    text = json.dumps(payload, indent=2) if not isinstance(payload, str) else payload
    print(text[:truncate] + " ..." if truncate and len(text) > truncate else text)


def main() -> None:
    # --- operational ---
    show("GET /health", client.get("/health").json())
    show("GET /health/ready", client.get("/health/ready").json())
    show("GET /version", client.get("/version").json())
    show("GET /v1/models", client.get("/v1/models").json(), truncate=400)
    show("GET /metrics (first lines)", "\n".join(client.get("/metrics").text.splitlines()[:6]))

    # --- inference ---
    show("POST /v1/accounts/score", client.post("/v1/accounts/score", json=ACCOUNT).json())
    show(
        "POST /v1/accounts/score?explain=true",
        client.post("/v1/accounts/score", params={"explain": "true"}, json=ACCOUNT).json(),
        truncate=600,
    )
    show(
        "POST /v1/accounts/explain (SHAP)",
        client.post("/v1/accounts/explain", json=ACCOUNT).json(),
        truncate=600,
    )
    show(
        "POST /v1/accounts/score:batch",
        client.post("/v1/accounts/score:batch", json={"accounts": [ACCOUNT, ACCOUNT]}).json(),
        truncate=400,
    )
    show(
        "POST /v1/accounts/copilot-summary",
        client.post("/v1/accounts/copilot-summary", json=ACCOUNT).json()["summary"],
    )

    # --- product / monitoring ---
    show(
        "POST /v1/roadmap/prioritize",
        client.post("/v1/roadmap/prioritize", json=INITIATIVES).json(),
    )
    show(
        "POST /v1/monitoring/drift",
        client.post("/v1/monitoring/drift", json={"accounts": [ACCOUNT] * 12}).json()["drift"][
            "overall_severity"
        ],
    )

    # --- extensions ---
    show("POST /v1/accounts/uplift", client.post("/v1/accounts/uplift", json=ACCOUNT).json())
    show(
        "GET /v1/experiments/save_play/assignment",
        client.get(
            "/v1/experiments/save_play/assignment", params={"unit_id": "acct_demo_001"}
        ).json(),
    )
    score = client.post("/v1/accounts/score", json=ACCOUNT).json()
    show(
        "POST /v1/feedback",
        client.post(
            "/v1/feedback",
            json={
                "account_id": ACCOUNT["account_id"],
                "churn_probability": score["churn_probability"],
                "predicted_churn": score["predicted_churn"],
                "model_version": score["model_version"],
                "actual_churn": 1,
            },
        ).json(),
    )
    show("GET /v1/feedback/metrics", client.get("/v1/feedback/metrics").json())
    show(
        "POST /v1/serving/shadow",
        client.post("/v1/serving/shadow", json={"accounts": [ACCOUNT, ACCOUNT]}).json(),
    )

    print("\nAll endpoints exercised.")


if __name__ == "__main__":
    main()
