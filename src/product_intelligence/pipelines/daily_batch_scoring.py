"""Daily batch scoring + drift monitoring pipeline.

Scores a snapshot of accounts with the production model, compares the live
feature/score distributions against the training reference, and emits a drift
report plus a retraining recommendation. This is the offline twin of the online
API and what a scheduler (Airflow/cron) would invoke each morning.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from product_intelligence.core.config import settings
from product_intelligence.core.logging import get_logger
from product_intelligence.data.synthetic import generate_accounts
from product_intelligence.features.builders import (
    CATEGORICAL_COLUMNS,
    MODEL_COLUMNS,
    NUMERIC_COLUMNS,
    build_feature_frame,
)
from product_intelligence.models.inference import get_inference_service
from product_intelligence.monitoring.drift import build_drift_report
from product_intelligence.pipelines.retraining import decide_retrain

logger = get_logger(__name__)


def score_accounts_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Score raw account records and return one row of output per account."""
    service = get_inference_service()
    rows = service.score_batch(df.to_dict(orient="records"))
    return pd.DataFrame(rows)


def run_batch_scoring(
    current: pd.DataFrame,
    output_dir: Path | None = None,
) -> dict:
    artifact_dir = settings.model_artifact_dir
    output_dir = output_dir or Path("scored_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    scored = score_accounts_batch(current)

    summary = {
        "n_scored": int(len(scored)),
        "risk_band_counts": scored["risk_band"].value_counts().to_dict(),
        "mean_churn_probability": round(float(scored["churn_probability"].mean()), 4),
        "high_risk_share": round(float((scored["risk_band"] == "high").mean()), 4),
    }

    drift_payload = None
    reference_path = artifact_dir / "reference_sample.csv"
    if reference_path.exists():
        reference = pd.read_csv(reference_path)
        current_features = build_feature_frame(current)[MODEL_COLUMNS]
        report = build_drift_report(
            reference=reference,
            current=current_features,
            numeric_features=NUMERIC_COLUMNS,
            categorical_features=CATEGORICAL_COLUMNS,
            reference_scores=reference["churn_score"].to_numpy()
            if "churn_score" in reference
            else None,
            current_scores=scored["churn_probability"].to_numpy(),
        )
        decision = decide_retrain(report)
        drift_payload = {"drift": report.to_dict(), "retraining": decision.to_dict()}
        (output_dir / "drift_report.json").write_text(json.dumps(drift_payload, indent=2), "utf-8")
        summary["drift_severity"] = report.overall_severity
        summary["should_retrain"] = decision.should_retrain

    scored.to_csv(output_dir / "scored_accounts.csv", index=False)
    (output_dir / "batch_summary.json").write_text(json.dumps(summary, indent=2), "utf-8")
    logger.info("batch scoring complete %s", summary)
    return {"summary": summary, **(drift_payload or {})}


def main() -> None:
    # Simulate "today's" snapshot with a different seed than training.
    snapshot = generate_accounts(n_samples=800, seed=settings.random_seed + 7)
    result = run_batch_scoring(snapshot)
    print(json.dumps(result["summary"], indent=2))
    if "retraining" in result:
        print("Retraining recommendation:", result["retraining"])


if __name__ == "__main__":
    main()
