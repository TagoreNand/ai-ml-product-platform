"""Shared fixtures.

Trains a small production registry into a temporary directory once per session
so inference/API tests are hermetic and never depend on committed artifacts.
"""

from __future__ import annotations

import pytest

from product_intelligence.core.config import settings
from product_intelligence.data.synthetic import generate_accounts
from product_intelligence.features.builders import build_feature_frame
from product_intelligence.models.inference import reset_inference_service
from product_intelligence.pipelines.offline_training import run_training_pipeline


@pytest.fixture(scope="session", autouse=True)
def trained_artifacts(tmp_path_factory):
    artifact_dir = tmp_path_factory.mktemp("artifacts")
    settings.model_artifact_dir = artifact_dir
    run_training_pipeline(output_dir=artifact_dir, n_samples=1500, seed=7)
    reset_inference_service()
    yield artifact_dir
    reset_inference_service()


@pytest.fixture
def labelled_frame():
    raw = generate_accounts(n_samples=1200, seed=11)
    df = build_feature_frame(raw)
    df["will_churn"] = raw["will_churn"]
    df["best_next_feature"] = raw["best_next_feature"]
    return df


@pytest.fixture
def sample_account() -> dict:
    return {
        "account_id": "acct_test",
        "company_size": "mid_market",
        "industry": "fintech",
        "region": "emea",
        "plan_tier": "starter",
        "contract_value": 48000,
        "tenure_months": 14,
        "nps": -5,
        "active_days_30d": 4,
        "weekly_active_users": 5,
        "monthly_active_users": 26,
        "feature_adoption_rate": 0.18,
        "workflow_runs_30d": 22,
        "api_calls_30d": 600,
        "support_tickets_90d": 8,
        "open_bug_count": 4,
        "p1_incidents_90d": 2,
        "onboarding_completion_rate": 0.4,
        "usage_growth_90d": -0.3,
        "renewal_days_remaining": 35,
    }
