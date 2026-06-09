"""Retraining decision logic.

Encodes the policy that decides whether a fresh training run should be promoted:
drift severity, prediction drift and a performance floor. Returning a structured
decision (instead of silently retraining) keeps a human/PM in the loop and makes
the trigger auditable.
"""

from __future__ import annotations

from dataclasses import dataclass

from product_intelligence.monitoring.drift import DriftReport


@dataclass
class RetrainDecision:
    should_retrain: bool
    reasons: list[str]
    severity: str

    def to_dict(self) -> dict:
        return {
            "should_retrain": self.should_retrain,
            "reasons": self.reasons,
            "severity": self.severity,
        }


def decide_retrain(
    drift: DriftReport,
    current_roc_auc: float | None = None,
    min_roc_auc: float = 0.70,
    prediction_psi_limit: float = 0.25,
) -> RetrainDecision:
    reasons: list[str] = []

    if drift.overall_severity == "significant":
        reasons.append(f"Significant data drift on: {', '.join(drift.flagged_features) or 'n/a'}")
    if drift.prediction_psi is not None and drift.prediction_psi >= prediction_psi_limit:
        reasons.append(f"Prediction drift PSI={drift.prediction_psi} >= {prediction_psi_limit}")
    if current_roc_auc is not None and current_roc_auc < min_roc_auc:
        reasons.append(f"Live ROC-AUC {current_roc_auc} below floor {min_roc_auc}")

    return RetrainDecision(
        should_retrain=bool(reasons), reasons=reasons, severity=drift.overall_severity
    )
