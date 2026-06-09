"""Uplift modelling (extension): estimate the *incremental* effect of an action.

Churn probability tells you who is at risk; **uplift** tells you who is
*persuadable* - the accounts where a save-play actually changes the outcome, which
is what should drive spend. Implements a T-learner (separate treatment/control
models) and a Qini coefficient to evaluate targeting quality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from product_intelligence.core.logging import get_logger
from product_intelligence.features.builders import MODEL_COLUMNS, build_feature_frame
from product_intelligence.models.train import _build_preprocessor

logger = get_logger(__name__)


def _trapz(y: np.ndarray) -> float:
    """Trapezoidal area over unit spacing (numpy 1.x/2.x safe)."""
    y = np.asarray(y, dtype=float)
    if y.size < 2:
        return 0.0
    return float(np.sum(y[:-1] + y[1:]) / 2.0)


def _learner(seed: int) -> Pipeline:
    return Pipeline(
        [("preprocessor", _build_preprocessor()), ("model", LogisticRegression(max_iter=1000))]
    )


class UpliftModel:
    """T-learner: P(outcome=1 | treated) - P(outcome=1 | control)."""

    def __init__(self, model_treat: Any, model_control: Any) -> None:
        self.model_treat = model_treat
        self.model_control = model_control

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        frame = X[MODEL_COLUMNS]
        p_t = self.model_treat.predict_proba(frame)[:, 1]
        p_c = self.model_control.predict_proba(frame)[:, 1]
        return p_t - p_c


@dataclass
class UpliftResult:
    model: UpliftModel
    metrics: dict


def qini_score(uplift: np.ndarray, treatment: np.ndarray, outcome: np.ndarray) -> float:
    """Normalised area between the model's Qini curve and random targeting."""
    order = np.argsort(-uplift)
    t, y = treatment[order], outcome[order]
    n = len(uplift)
    nt = max(t.sum(), 1)
    nc = max((1 - t).sum(), 1)
    cum_t = np.cumsum(y * t)
    cum_c = np.cumsum(y * (1 - t))
    qini = cum_t - cum_c * (nt / nc)
    overall = qini[-1]
    rand = overall * (np.arange(1, n + 1) / n)
    auc_model = _trapz(qini)
    auc_rand = _trapz(rand)
    denom = abs(auc_rand) if auc_rand != 0 else 1.0
    return float(round((auc_model - auc_rand) / denom, 4))


def train_uplift_model(
    df: pd.DataFrame,
    treatment_col: str = "treatment",
    outcome_col: str = "retained",
    seed: int = 7,
) -> UpliftResult:
    enriched = build_feature_frame(df)
    enriched[treatment_col] = df[treatment_col].to_numpy()
    enriched[outcome_col] = df[outcome_col].to_numpy()

    train, test = train_test_split(
        enriched, test_size=0.25, random_state=seed, stratify=enriched[treatment_col]
    )

    treated = train[train[treatment_col] == 1]
    control = train[train[treatment_col] == 0]
    m_t = _learner(seed).fit(treated[MODEL_COLUMNS], treated[outcome_col])
    m_c = _learner(seed).fit(control[MODEL_COLUMNS], control[outcome_col])
    model = UpliftModel(m_t, m_c)

    uplift = model.predict_uplift(test)
    qini = qini_score(uplift, test[treatment_col].to_numpy(), test[outcome_col].to_numpy())

    # Targeting lift: mean realised outcome diff in the top-uplift decile vs overall.
    top = np.argsort(-uplift)[: max(len(uplift) // 10, 1)]
    top_t = test.iloc[top]
    metrics = {
        "qini_coefficient": qini,
        "mean_predicted_uplift": round(float(uplift.mean()), 4),
        "top_decile_treated_retention": round(
            float(top_t[top_t[treatment_col] == 1][outcome_col].mean()), 4
        ),
        "top_decile_control_retention": round(
            float(top_t[top_t[treatment_col] == 0][outcome_col].mean()), 4
        ),
        "n_test": int(len(test)),
    }
    logger.info("uplift model trained qini=%s", qini)
    return UpliftResult(model=model, metrics=metrics)
