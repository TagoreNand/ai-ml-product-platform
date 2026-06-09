"""Feature engineering contract shared by training and inference.

A single source of truth for column groups guarantees the offline training
pipeline and the online inference service compute *identical* features, which
is the most common source of training/serving skew in production ML systems.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

# --- Raw inputs supplied by upstream systems (product telemetry + CRM) ---
RAW_CATEGORICAL_COLUMNS = [
    "company_size",
    "industry",
    "region",
    "plan_tier",
]

RAW_NUMERIC_COLUMNS = [
    "contract_value",
    "tenure_months",
    "nps",
    "active_days_30d",
    "weekly_active_users",
    "monthly_active_users",
    "feature_adoption_rate",
    "workflow_runs_30d",
    "api_calls_30d",
    "support_tickets_90d",
    "open_bug_count",
    "p1_incidents_90d",
    "onboarding_completion_rate",
    "usage_growth_90d",
    "renewal_days_remaining",
]

REQUIRED_INPUT_COLUMNS = RAW_CATEGORICAL_COLUMNS + RAW_NUMERIC_COLUMNS

# --- Engineered features derived deterministically from the raw inputs ---
DERIVED_NUMERIC_COLUMNS = [
    "adoption_gap",
    "support_burden_ratio",
    "engagement_intensity",
    "api_dependency",
    "wau_mau_ratio",
    "active_day_ratio",
    "reliability_friction",
    "value_per_user",
]

# --- Final model input contract ---
CATEGORICAL_COLUMNS = RAW_CATEGORICAL_COLUMNS
NUMERIC_COLUMNS = RAW_NUMERIC_COLUMNS + DERIVED_NUMERIC_COLUMNS
MODEL_COLUMNS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS

# Human-readable labels used by explainability + copilot narratives.
FEATURE_LABELS: dict[str, str] = {
    "contract_value": "Contract value",
    "tenure_months": "Account tenure",
    "nps": "NPS",
    "active_days_30d": "Active days (30d)",
    "weekly_active_users": "Weekly active users",
    "monthly_active_users": "Monthly active users",
    "feature_adoption_rate": "Feature adoption rate",
    "workflow_runs_30d": "Workflow runs (30d)",
    "api_calls_30d": "API calls (30d)",
    "support_tickets_90d": "Support tickets (90d)",
    "open_bug_count": "Open bug count",
    "p1_incidents_90d": "P1 incidents (90d)",
    "onboarding_completion_rate": "Onboarding completion",
    "usage_growth_90d": "Usage growth (90d)",
    "renewal_days_remaining": "Days to renewal",
    "adoption_gap": "Adoption gap",
    "support_burden_ratio": "Support burden per user",
    "engagement_intensity": "Engagement intensity",
    "api_dependency": "API dependency per user",
    "wau_mau_ratio": "Weekly stickiness (WAU/MAU)",
    "active_day_ratio": "Active-day ratio",
    "reliability_friction": "Reliability friction",
    "value_per_user": "Contract value per user",
    "company_size": "Company size",
    "industry": "Industry",
    "region": "Region",
    "plan_tier": "Plan tier",
}


def build_feature_frame(records: Iterable[dict] | pd.DataFrame) -> pd.DataFrame:
    """Validate raw inputs and append deterministic engineered features.

    Parameters
    ----------
    records:
        An iterable of dict-like account records or a ``DataFrame`` already
        holding the raw input columns.

    Returns
    -------
    pd.DataFrame
        Copy of the input with all ``DERIVED_NUMERIC_COLUMNS`` populated.

    Raises
    ------
    ValueError
        If any required raw input column is missing.
    """
    df = records.copy() if isinstance(records, pd.DataFrame) else pd.DataFrame(records).copy()

    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    mau = df["monthly_active_users"].clip(lower=1)

    df["adoption_gap"] = 1.0 - df["feature_adoption_rate"]
    df["support_burden_ratio"] = df["support_tickets_90d"] / mau
    df["engagement_intensity"] = df["workflow_runs_30d"] / mau
    df["api_dependency"] = df["api_calls_30d"] / mau
    df["wau_mau_ratio"] = (df["weekly_active_users"] / mau).clip(upper=1.0)
    df["active_day_ratio"] = df["active_days_30d"] / 30.0
    df["reliability_friction"] = df["open_bug_count"] + 2.0 * df["p1_incidents_90d"]
    df["value_per_user"] = df["contract_value"] / mau

    # Guard against inf/NaN leaking into the model from edge-case divisions.
    df[DERIVED_NUMERIC_COLUMNS] = (
        df[DERIVED_NUMERIC_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )
    return df


def label_for(column: str) -> str:
    """Return a human-friendly label for a raw or engineered column."""
    return FEATURE_LABELS.get(column, column.replace("_", " ").title())
