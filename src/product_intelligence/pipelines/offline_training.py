"""Offline training pipeline (the 'register & promote' entrypoint).

Steps: generate/load data -> engineer features -> train + calibrate churn and
recommendation models -> register immutable versions -> persist explainer
background + a reference snapshot for drift -> render a model card. Re-running
produces a new version and promotes it to ``production`` atomically via the
registry index.
"""

from __future__ import annotations

import json
from pathlib import Path

from product_intelligence.core.config import settings
from product_intelligence.core.logging import get_logger
from product_intelligence.data.sources import get_data_source
from product_intelligence.data.synthetic import DATA_VERSION, generate_intervention_data
from product_intelligence.features.builders import (
    CATEGORICAL_COLUMNS,
    MODEL_COLUMNS,
    NUMERIC_COLUMNS,
    build_feature_frame,
)
from product_intelligence.models.cards import render_churn_model_card
from product_intelligence.models.registry import ModelRegistry
from product_intelligence.models.train import train_churn_model, train_recommendation_model
from product_intelligence.models.uplift import train_uplift_model

logger = get_logger(__name__)


def run_training_pipeline(
    output_dir: Path | None = None,
    n_samples: int = 5000,
    seed: int | None = None,
) -> dict:
    artifact_dir = output_dir or settings.model_artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    seed = settings.random_seed if seed is None else seed

    raw = get_data_source(n_samples=n_samples, seed=seed).load()
    df = build_feature_frame(raw)
    df["will_churn"] = raw["will_churn"]
    df["best_next_feature"] = raw["best_next_feature"]

    churn_result = train_churn_model(
        df, model_type=settings.churn_model_type, calibration=settings.churn_calibration, seed=seed
    )
    rec_result = train_recommendation_model(
        df, model_type=settings.recommendation_model_type, seed=seed
    )

    registry = ModelRegistry(artifact_dir)
    background = (
        df[MODEL_COLUMNS].sample(min(200, len(df)), random_state=seed).reset_index(drop=True)
    )

    churn_record = registry.register(
        "churn",
        churn_result.pipeline,
        model_type=settings.churn_model_type,
        metrics=churn_result.metrics,
        threshold=churn_result.threshold,
        band_thresholds=churn_result.band_thresholds,
        feature_names=churn_result.feature_names,
        data_version=DATA_VERSION,
        extra={
            "calibration_curve": churn_result.calibration_curve,
            "segment_metrics": churn_result.segment_metrics,
        },
        background=background,
    )
    rec_record = registry.register(
        "recommendation",
        rec_result.pipeline,
        model_type=settings.recommendation_model_type,
        metrics=rec_result.metrics,
        data_version=DATA_VERSION,
    )

    # Uplift model (extension): incremental effect of a CS save-play.
    intervention = generate_intervention_data(n_samples=min(n_samples, 4000), seed=seed + 5)
    uplift_result = train_uplift_model(intervention, seed=seed)
    uplift_record = registry.register(
        "uplift",
        uplift_result.model,
        model_type="t_learner_uplift",
        metrics=uplift_result.metrics,
        data_version=DATA_VERSION,
    )

    # Reference snapshot for drift monitoring (features + production scores).
    ref = df[MODEL_COLUMNS].sample(min(1000, len(df)), random_state=seed + 1).reset_index(drop=True)
    ref_scores = churn_result.pipeline.predict_proba(ref[MODEL_COLUMNS])[:, 1]
    ref_out = ref.copy()
    ref_out["churn_score"] = ref_scores
    ref_out.to_csv(artifact_dir / "reference_sample.csv", index=False)

    # Backward-compatible top-level metadata summary.
    metadata = {
        "data_version": DATA_VERSION,
        "dataset_rows": int(len(df)),
        "churn_version": churn_record.version,
        "recommendation_version": rec_record.version,
        "uplift_version": uplift_record.version,
        "churn_metrics": churn_result.metrics,
        "recommendation_metrics": rec_result.metrics,
        "uplift_metrics": uplift_result.metrics,
        "drift_monitored_numeric": NUMERIC_COLUMNS,
        "drift_monitored_categorical": CATEGORICAL_COLUMNS,
    }
    (artifact_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2), "utf-8")
    df.head(200).to_csv(artifact_dir / "training_sample.csv", index=False)

    # Render model card next to the docs.
    card = render_churn_model_card(
        churn_result, version=churn_record.version, data_version=DATA_VERSION
    )
    docs_dir = Path("docs")
    if docs_dir.exists():
        (docs_dir / "model_card_churn.md").write_text(card, encoding="utf-8")

    logger.info(
        "training pipeline complete churn=%s rec=%s", churn_record.version, rec_record.version
    )
    return {
        "churn": churn_result.metrics,
        "recommendation": rec_result.metrics,
        "churn_version": churn_record.version,
        "recommendation_version": rec_record.version,
        "uplift_version": uplift_record.version,
        "uplift": uplift_result.metrics,
    }


def main() -> None:
    summary = run_training_pipeline()
    print("Training complete")
    for name, values in summary.items():
        print(f"{name}: {values}")


if __name__ == "__main__":
    main()
