"""Online feedback logging (extension).

Closes the loop: log each served prediction and, when the true outcome lands,
the realised label. The append-only JSONL log is the substrate for monitoring
*realised* performance (not just drift) and for feedback-driven retraining
triggers.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from product_intelligence.core.logging import get_logger

logger = get_logger(__name__)


class FeedbackStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        account_id: str,
        churn_probability: float,
        predicted_churn: bool,
        model_version: str,
        actual_churn: int | None = None,
    ) -> dict:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "account_id": account_id,
            "churn_probability": float(churn_probability),
            "predicted_churn": bool(predicted_churn),
            "model_version": model_version,
            "actual_churn": actual_churn,
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        return record

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(
                columns=[
                    "ts",
                    "account_id",
                    "churn_probability",
                    "predicted_churn",
                    "model_version",
                    "actual_churn",
                ]
            )
        rows = [
            json.loads(line) for line in self.path.read_text("utf-8").splitlines() if line.strip()
        ]
        return pd.DataFrame(rows)

    def realized_metrics(self) -> dict:
        """Accuracy / ROC-AUC over rows where the true outcome is known."""
        from sklearn.metrics import accuracy_score, roc_auc_score

        df = self.load()
        labelled = df[df["actual_churn"].notna()] if "actual_churn" in df else df.iloc[:0]
        out: dict = {"n_logged": int(len(df)), "n_labelled": int(len(labelled))}
        if len(labelled) >= 20 and labelled["actual_churn"].nunique() == 2:
            y = labelled["actual_churn"].astype(int)
            out["accuracy"] = round(
                float(accuracy_score(y, labelled["predicted_churn"].astype(int))), 4
            )
            out["roc_auc"] = round(float(roc_auc_score(y, labelled["churn_probability"])), 4)
        return out
