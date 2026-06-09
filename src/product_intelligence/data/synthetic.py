"""Synthetic but realistic B2B SaaS account telemetry.

The generator encodes a plausible churn data-generating process (engagement,
adoption, support burden, reliability and renewal pressure) plus a
multi-class "best next feature" target. It is deterministic given ``seed`` so
training runs are reproducible and CI can assert metric floors.

Swap this module for warehouse extracts to move from demo to production; the
downstream feature/training/inference contract stays identical.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

DATA_VERSION = "synthetic-v2"

FEATURE_OPTIONS = {
    "company_size": ["smb", "mid_market", "enterprise"],
    "industry": ["fintech", "healthtech", "retail", "saas", "manufacturing"],
    "region": ["na", "emea", "apac", "latam"],
    "plan_tier": ["starter", "growth", "enterprise"],
}

RECOMMENDATION_LABELS = [
    "copilot_assist",
    "workflow_automation",
    "self_serve_reporting",
    "anomaly_alerts",
    "team_dashboard",
]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_accounts(
    n_samples: int = 4000,
    seed: int = 42,
    snapshot: date | None = None,
) -> pd.DataFrame:
    """Generate a labelled account snapshot.

    Returns a ``DataFrame`` containing the raw input columns, the binary
    ``will_churn`` target and the multi-class ``best_next_feature`` target.
    """
    rng = np.random.default_rng(seed)
    snapshot = snapshot or date.today()
    n = n_samples

    df = pd.DataFrame(
        {
            "account_id": [f"acct_{i:05d}" for i in range(n)],
            "snapshot_date": snapshot.isoformat(),
            "company_size": rng.choice(
                FEATURE_OPTIONS["company_size"], size=n, p=[0.45, 0.35, 0.20]
            ),
            "industry": rng.choice(FEATURE_OPTIONS["industry"], size=n),
            "region": rng.choice(FEATURE_OPTIONS["region"], size=n),
            "plan_tier": rng.choice(FEATURE_OPTIONS["plan_tier"], size=n, p=[0.35, 0.45, 0.20]),
            "contract_value": rng.normal(42000, 20000, size=n).clip(4000, 180000),
            "tenure_months": rng.integers(1, 60, size=n),
            "nps": rng.normal(28, 22, size=n).clip(-50, 100),
            "active_days_30d": rng.integers(1, 31, size=n),
            "weekly_active_users": rng.integers(1, 80, size=n),
            "monthly_active_users": rng.integers(2, 200, size=n),
            "feature_adoption_rate": rng.beta(2.2, 2.0, size=n),
            "workflow_runs_30d": rng.gamma(5.0, 25.0, size=n).clip(0, 2500),
            "api_calls_30d": rng.gamma(4.0, 600.0, size=n).clip(0, 60000),
            "support_tickets_90d": rng.poisson(2.2, size=n),
            "open_bug_count": rng.poisson(1.4, size=n),
            "p1_incidents_90d": rng.binomial(3, 0.08, size=n),
            "onboarding_completion_rate": rng.beta(3.0, 1.8, size=n),
            "usage_growth_90d": rng.normal(0.08, 0.20, size=n).clip(-0.8, 1.2),
            "renewal_days_remaining": rng.integers(5, 365, size=n),
        }
    )

    enterprise_boost = (df["company_size"] == "enterprise").astype(int)
    starter_penalty = (df["plan_tier"] == "starter").astype(int)
    low_adoption = (df["feature_adoption_rate"] < 0.35).astype(int)
    low_usage = (df["active_days_30d"] < 8).astype(int)
    high_support = (df["support_tickets_90d"] >= 5).astype(int)
    renewal_pressure = (df["renewal_days_remaining"] < 60).astype(int)

    # Calibrated so the base churn rate is a realistic ~18-20% (annual B2B logo
    # churn) with enough signal that a good model separates risk cleanly.
    churn_logit = (
        0.70
        - 0.020 * df["nps"]
        - 2.5 * df["feature_adoption_rate"]
        - 0.060 * df["active_days_30d"]
        - 1.2 * df["usage_growth_90d"]
        - 1.5 * df["onboarding_completion_rate"]
        + 0.30 * df["support_tickets_90d"]
        + 0.40 * df["open_bug_count"]
        + 0.70 * df["p1_incidents_90d"]
        + 0.60 * starter_penalty
        + 0.50 * renewal_pressure
        + 0.40 * low_adoption
        + rng.normal(0, 0.5, size=n)
    )
    df["will_churn"] = rng.binomial(1, sigmoid(churn_logit))

    rec_scores = pd.DataFrame(
        {
            "copilot_assist": 0.7 * low_usage
            + 0.3 * low_adoption
            + 0.2 * (df["industry"] == "saas").astype(int)
            + 0.1 * enterprise_boost,
            "workflow_automation": 0.45 * (df["workflow_runs_30d"] < 120).astype(int)
            + 0.25 * enterprise_boost
            + 0.15 * (df["industry"] == "manufacturing").astype(int),
            "self_serve_reporting": 0.35 * high_support
            + 0.25 * (df["industry"] == "retail").astype(int)
            + 0.2 * (df["plan_tier"] != "starter").astype(int),
            "anomaly_alerts": 0.5 * (df["api_calls_30d"] > 4000).astype(int)
            + 0.25 * (df["industry"].isin(["fintech", "healthtech"])).astype(int)
            + 0.15 * enterprise_boost,
            "team_dashboard": 0.4 * (df["monthly_active_users"] > 40).astype(int)
            + 0.2 * (df["weekly_active_users"] > 15).astype(int)
            + 0.2 * (df["company_size"] != "smb").astype(int),
        }
    )
    rec_scores = rec_scores + rng.normal(0, 0.05, size=rec_scores.shape)
    df["best_next_feature"] = rec_scores.idxmax(axis=1)

    return df


def generate_intervention_data(n_samples: int = 4000, seed: int = 7) -> pd.DataFrame:
    """Randomised-treatment dataset for uplift modelling.

    Adds a randomly-assigned ``treatment`` (a CS save-play / outreach) and a
    ``retained`` outcome whose treatment effect is **heterogeneous**: the save-play
    helps reachable, mid-risk accounts most and barely moves already-healthy or
    already-lost ones. Uplift models should recover that structure.
    """
    df = generate_accounts(n_samples=n_samples, seed=seed)
    rng = np.random.default_rng(seed + 101)
    n = len(df)

    treatment = rng.binomial(1, 0.5, size=n)

    risk = (
        1.4
        - 2.2 * df["feature_adoption_rate"].to_numpy()
        - 0.05 * df["active_days_30d"].to_numpy()
        - 0.015 * df["nps"].to_numpy()
        + 0.20 * df["support_tickets_90d"].to_numpy()
    )
    # Heterogeneous effect: strongest where there is room to move and goodwill (nps).
    reachable = (df["nps"].to_numpy() > 0).astype(float)
    tau = 1.3 * reachable * (df["feature_adoption_rate"].to_numpy() < 0.5).astype(float)

    churn_logit = risk - tau * treatment + rng.normal(0, 0.5, size=n)
    churned = rng.binomial(1, sigmoid(churn_logit))

    df = df.drop(columns=["will_churn", "best_next_feature"], errors="ignore")
    df["treatment"] = treatment
    df["retained"] = 1 - churned
    return df
