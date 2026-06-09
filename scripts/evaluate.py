"""Offline evaluation harness.

Trains a churn model and prints the headline metrics + the value added by
threshold optimisation. Useful for DS review and for asserting metric floors
in CI before a model is promoted.
"""

from __future__ import annotations

import json

from product_intelligence.core.config import settings
from product_intelligence.data.synthetic import generate_accounts
from product_intelligence.features.builders import build_feature_frame
from product_intelligence.models.train import train_churn_model, train_recommendation_model


def main() -> None:
    raw = generate_accounts(n_samples=6000, seed=settings.random_seed)
    df = build_feature_frame(raw)
    df["will_churn"] = raw["will_churn"]
    df["best_next_feature"] = raw["best_next_feature"]

    churn = train_churn_model(
        df, model_type=settings.churn_model_type, calibration=settings.churn_calibration
    )
    rec = train_recommendation_model(df, model_type=settings.recommendation_model_type)

    print("=== Churn model ===")
    print(json.dumps(churn.metrics, indent=2))
    print(f"F1 lift from threshold tuning: {churn.metrics['f1'] - churn.metrics['f1_at_0.5']:+.4f}")
    print(f"Segment AUC: {json.dumps(churn.segment_metrics)}")
    print("\n=== Recommendation model ===")
    print(json.dumps(rec.metrics, indent=2))


if __name__ == "__main__":
    main()
