"""Offline training with calibration, threshold tuning and segment evaluation.

Design choices that make this "senior level" rather than a toy:

* **Config-driven model factory** - swap LogisticRegression / HistGradientBoosting
  / XGBoost without touching pipeline code (``CHURN_MODEL_TYPE``).
* **Probability calibration** - business actions key off probabilities, so we
  calibrate (isotonic/sigmoid) and report Brier score + a reliability curve.
* **Decision-threshold optimisation** - instead of a naive 0.5 cut we pick the
  F1-optimal operating point from *out-of-fold* predictions (no test leakage).
* **Risk bands from score quantiles** - data-driven low/medium/high cut-offs.
* **Segment evaluation** - per company-size / region AUC to surface fairness gaps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from product_intelligence.core.logging import get_logger
from product_intelligence.features.builders import (
    CATEGORICAL_COLUMNS,
    MODEL_COLUMNS,
    NUMERIC_COLUMNS,
)

logger = get_logger(__name__)


@dataclass
class TrainingResult:
    pipeline: Any
    metrics: dict
    threshold: float = 0.5
    band_thresholds: dict[str, float] = field(default_factory=lambda: {"medium": 0.4, "high": 0.7})
    feature_names: list[str] | None = None
    calibration_curve: dict[str, list[float]] | None = None
    segment_metrics: dict[str, dict[str, float]] | None = None


def _make_churn_estimator(model_type: str, seed: int):
    if model_type == "logreg":
        return LogisticRegression(max_iter=1200, class_weight="balanced")
    if model_type == "xgboost":
        try:
            from xgboost import XGBClassifier

            return XGBClassifier(
                n_estimators=400,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                random_state=seed,
                tree_method="hist",
            )
        except Exception:  # pragma: no cover - depends on optional dep
            logger.warning("xgboost unavailable; falling back to hist_gbdt")
    return HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=350,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        early_stopping=True,
        random_state=seed,
    )


def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                CATEGORICAL_COLUMNS,
            ),
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_COLUMNS,
            ),
        ]
    )


def _best_f1_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    grid = np.linspace(0.05, 0.95, 181)
    f1s = [f1_score(y_true, (proba >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f1s))])


def train_churn_model(
    df: pd.DataFrame,
    model_type: str = "hist_gbdt",
    calibration: str = "isotonic",
    seed: int = 42,
) -> TrainingResult:
    """Train, calibrate and evaluate the churn-risk classifier."""
    X = df[MODEL_COLUMNS]
    y = df["will_churn"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    base = Pipeline(
        [
            ("preprocessor", _build_preprocessor()),
            ("model", _make_churn_estimator(model_type, seed)),
        ]
    )

    # Out-of-fold probabilities on TRAIN -> leakage-free threshold selection.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = cross_val_predict(base, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    threshold = _best_f1_threshold(y_train.to_numpy(), oof)
    cv_roc_auc = float(roc_auc_score(y_train, oof))

    # Final fit on all training data, optionally wrapped in calibration.
    if calibration and calibration != "none":
        model: Any = CalibratedClassifierCV(base, method=calibration, cv=5)
    else:
        model = base
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)

    metrics = {
        "model_type": model_type,
        "calibration": calibration,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "positive_rate": round(float(y.mean()), 4),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_test, proba)), 4),
        "cv_roc_auc": round(cv_roc_auc, 4),
        "brier": round(float(brier_score_loss(y_test, proba)), 4),
        "log_loss": round(float(log_loss(y_test, proba)), 4),
        "threshold": round(threshold, 4),
        "f1": round(float(f1_score(y_test, preds, zero_division=0)), 4),
        "precision": round(float(precision_score(y_test, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, preds, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "f1_at_0.5": round(float(f1_score(y_test, (proba >= 0.5).astype(int), zero_division=0)), 4),
    }

    band_thresholds = {
        "medium": round(float(np.quantile(proba, 0.60)), 4),
        "high": round(float(np.quantile(proba, 0.85)), 4),
    }

    frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10, strategy="quantile")
    calib = {
        "prob_true": [round(float(v), 4) for v in frac_pos],
        "prob_pred": [round(float(v), 4) for v in mean_pred],
    }

    segment_metrics = _segment_auc(X_test.assign(_proba=proba, _y=y_test.to_numpy()))

    # The raw model-input contract is the meaningful feature space for the
    # explainer and model card (it perturbs raw columns, not one-hot dummies).
    feature_names = list(MODEL_COLUMNS)

    logger.info(
        "churn model trained type=%s roc_auc=%s pr_auc=%s f1=%s threshold=%s",
        model_type,
        metrics["roc_auc"],
        metrics["pr_auc"],
        metrics["f1"],
        metrics["threshold"],
    )
    return TrainingResult(
        pipeline=model,
        metrics=metrics,
        threshold=threshold,
        band_thresholds=band_thresholds,
        feature_names=feature_names,
        calibration_curve=calib,
        segment_metrics=segment_metrics,
    )


def _segment_auc(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for col in ("company_size", "region"):
        seg: dict[str, float] = {}
        for value, grp in frame.groupby(col):
            if grp["_y"].nunique() == 2 and len(grp) >= 30:
                seg[str(value)] = round(float(roc_auc_score(grp["_y"], grp["_proba"])), 4)
        if seg:
            out[col] = seg
    return out


def train_recommendation_model(
    df: pd.DataFrame, model_type: str = "random_forest", seed: int = 42
) -> TrainingResult:
    """Train the multi-class next-best-feature recommender."""
    X = df[MODEL_COLUMNS]
    y = df["best_next_feature"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    if model_type == "hist_gbdt":
        estimator: Any = HistGradientBoostingClassifier(random_state=seed, max_iter=300)
    else:
        estimator = RandomForestClassifier(
            n_estimators=240, random_state=seed, class_weight="balanced", n_jobs=-1
        )

    pipeline = Pipeline([("preprocessor", _build_preprocessor()), ("model", estimator)])
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    metrics = {
        "model_type": model_type,
        "n_classes": int(y.nunique()),
        "macro_f1": round(float(f1_score(y_test, preds, average="macro")), 4),
        "accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_test, preds)), 4),
    }
    logger.info(
        "recommendation model trained macro_f1=%s acc=%s", metrics["macro_f1"], metrics["accuracy"]
    )
    return TrainingResult(pipeline=pipeline, metrics=metrics)
