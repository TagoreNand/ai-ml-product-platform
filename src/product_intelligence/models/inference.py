"""Online inference service.

Loads the *production* churn + recommendation models from the registry and turns
a raw account payload into a decision-ready response: calibrated churn
probability, data-driven risk band, risk drivers, ranked feature recommendations
and prescriptive actions.

Latency contract: the default scoring path uses fast, model-informed heuristic
drivers (single-digit milliseconds). SHAP attributions are model-faithful but
cost ~1-2s, so they are *opt-in* (``explain=True`` / the ``/explain`` endpoint)
rather than on the hot path - the standard pattern for serving explainable models
without paying SHAP cost on every request.
"""

from __future__ import annotations

import time
from functools import lru_cache

import pandas as pd

from product_intelligence.core.config import settings
from product_intelligence.core.logging import get_logger
from product_intelligence.core.metrics import (
    CHURN_SCORE,
    INFERENCE_LATENCY,
    MODEL_INFO,
    PREDICTIONS_TOTAL,
)
from product_intelligence.features.builders import MODEL_COLUMNS, build_feature_frame
from product_intelligence.models.explain import ModelExplainer
from product_intelligence.models.registry import ModelRegistry

logger = get_logger(__name__)

_PROB_EPS = 5e-4


def heuristic_signals(payload: dict) -> tuple[list[str], list[str]]:
    """Fast, dependency-free risk drivers + protective factors.

    Used on the latency-sensitive scoring path. Mirrors the structure of the
    SHAP output so callers get a consistent contract regardless of mode.
    """
    drivers: list[str] = []
    protective: list[str] = []

    if payload.get("active_days_30d", 30) < 8:
        drivers.append("Low active days in last 30d")
    if payload.get("usage_growth_90d", 0.0) < 0:
        drivers.append("Negative usage growth")
    if payload.get("onboarding_completion_rate", 1.0) < 0.7:
        drivers.append("Low onboarding completion")
    if payload.get("feature_adoption_rate", 1.0) < 0.4:
        drivers.append("Weak feature adoption")
    if payload.get("support_tickets_90d", 0) >= 5:
        drivers.append("Higher recent support burden")
    if payload.get("open_bug_count", 0) >= 2 or payload.get("p1_incidents_90d", 0) >= 1:
        drivers.append("Product reliability friction")
    if payload.get("renewal_days_remaining", 365) < 60:
        drivers.append("Upcoming renewal pressure")
    if payload.get("nps", 100) < 0:
        drivers.append("Detractor-level NPS")

    if payload.get("feature_adoption_rate", 0.0) >= 0.6:
        protective.append("Strong feature adoption")
    if payload.get("usage_growth_90d", 0.0) > 0.1:
        protective.append("Positive usage growth")
    if payload.get("nps", 0) >= 40:
        protective.append("Promoter-level NPS")

    return (drivers[:5] or ["No dominant risk driver detected"]), protective[:3]


class InferenceService:
    def __init__(self) -> None:
        self.registry = ModelRegistry(settings.model_artifact_dir)
        self.churn_model, self.churn_record = self.registry.load("churn", "production")
        self.recommendation_model, self.rec_record = self.registry.load(
            "recommendation", "production"
        )
        self.threshold = self.churn_record.threshold or 0.5
        self.band_thresholds = self.churn_record.band_thresholds or {"medium": 0.4, "high": 0.7}

        self._background = self.registry.load_background("churn", "production")
        if self._background is None:
            self._background = pd.DataFrame(columns=MODEL_COLUMNS)
        self._explainer: ModelExplainer | None = None  # built lazily (SHAP is heavy)
        self._uplift_loaded = False
        self._uplift_model = None
        self._uplift_record = None

        MODEL_INFO.labels(model="churn", version=self.churn_record.version).set(1)
        MODEL_INFO.labels(model="recommendation", version=self.rec_record.version).set(1)
        logger.info(
            "InferenceService ready churn=%s rec=%s",
            self.churn_record.version,
            self.rec_record.version,
        )

    @property
    def model_version(self) -> str:
        return self.churn_record.version

    @property
    def explainer(self) -> ModelExplainer:
        if self._explainer is None:
            self._explainer = ModelExplainer(
                self.churn_model, self._background, enable_shap=settings.enable_shap
            )
        return self._explainer

    def _rank_recommendations(self, proba_row) -> list[dict]:
        labels = self.recommendation_model.classes_
        return sorted(
            (
                {"feature": str(label), "score": round(float(score), 4)}
                for label, score in zip(labels, proba_row, strict=False)
            ),
            key=lambda x: x["score"],
            reverse=True,
        )

    @property
    def uplift_model(self):
        if not self._uplift_loaded:
            try:
                self._uplift_model, self._uplift_record = self.registry.load("uplift", "production")
            except Exception:  # pragma: no cover - uplift optional
                self._uplift_model, self._uplift_record = None, None
            self._uplift_loaded = True
        return self._uplift_model

    def score_uplift(self, payload: dict) -> dict:
        model = self.uplift_model
        if model is None or self._uplift_record is None:
            raise RuntimeError("No uplift model registered (run training to create one).")
        enriched = build_feature_frame(pd.DataFrame([payload]))
        uplift = float(model.predict_uplift(enriched)[0])
        return {
            "account_id": payload.get("account_id", "unknown"),
            "model_version": self._uplift_record.version,
            "estimated_uplift": round(uplift, 4),
            "interpretation": "expected change in retention probability if the save-play is applied",
        }

    def score_account(self, payload: dict, explain: bool = False) -> dict:
        start = time.perf_counter()
        enriched = build_feature_frame(pd.DataFrame([payload]))
        model_frame = enriched[MODEL_COLUMNS]

        raw_prob = float(self.churn_model.predict_proba(model_frame)[0, 1])
        churn_probability = min(max(raw_prob, _PROB_EPS), 1 - _PROB_EPS)
        risk_band = self._band(churn_probability)

        ranked_recs = self._rank_recommendations(
            self.recommendation_model.predict_proba(model_frame)[0]
        )

        if explain:
            drivers = self.explainer.top_risk_drivers(model_frame, k=5)
            protective = self.explainer.protective_factors(model_frame, k=3)
            mode = self.explainer.mode
        else:
            drivers, protective = heuristic_signals(payload)
            mode = "heuristic"

        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        INFERENCE_LATENCY.observe(latency_ms / 1000)
        CHURN_SCORE.observe(churn_probability)
        PREDICTIONS_TOTAL.labels(risk_band=risk_band, model_version=self.model_version).inc()

        return self._assemble(
            payload,
            churn_probability,
            risk_band,
            drivers,
            protective,
            ranked_recs,
            mode,
            latency_ms,
        )

    def score_batch(self, payloads: list[dict], explain: bool = False) -> list[dict]:
        if not payloads:
            return []
        if explain:  # SHAP is per-instance; use the row path.
            return [self.score_account(p, explain=True) for p in payloads]
        return self.score_frame(pd.DataFrame(payloads))

    def score_frame(self, df_raw: pd.DataFrame) -> list[dict]:
        """Vectorised batch scoring: one ``predict_proba`` call for all rows.

        Avoids the O(n) per-row pipeline overhead that makes naive batch loops
        slow, which matters for the daily scoring job over thousands of accounts.
        """
        start = time.perf_counter()
        X = build_feature_frame(df_raw)[MODEL_COLUMNS]
        churn = self.churn_model.predict_proba(X)[:, 1].clip(_PROB_EPS, 1 - _PROB_EPS)
        rec_proba = self.recommendation_model.predict_proba(X)

        records = df_raw.to_dict(orient="records")
        per_row_ms = round((time.perf_counter() - start) * 1000 / max(len(records), 1), 3)
        results: list[dict] = []
        for i, payload in enumerate(records):
            prob = float(churn[i])
            band = self._band(prob)
            ranked = self._rank_recommendations(rec_proba[i])
            drivers, protective = heuristic_signals(payload)
            CHURN_SCORE.observe(prob)
            PREDICTIONS_TOTAL.labels(risk_band=band, model_version=self.model_version).inc()
            results.append(
                self._assemble(
                    payload, prob, band, drivers, protective, ranked, "heuristic", per_row_ms
                )
            )
        INFERENCE_LATENCY.observe(time.perf_counter() - start)
        return results

    def _assemble(self, payload, prob, band, drivers, protective, ranked, mode, latency_ms) -> dict:
        return {
            "account_id": payload.get("account_id", "unknown"),
            "model_version": self.model_version,
            "churn_probability": round(prob, 4),
            "risk_band": band,
            "predicted_churn": bool(prob >= self.threshold),
            "decision_threshold": round(float(self.threshold), 4),
            "top_risk_drivers": drivers,
            "protective_factors": protective,
            "recommended_features": ranked[:3],
            "recommended_actions": self._recommended_actions(prob, ranked, payload),
            "explainer": mode,
            "latency_ms": latency_ms,
        }

    def _band(self, score: float) -> str:
        if score >= self.band_thresholds.get("high", 0.7):
            return "high"
        if score >= self.band_thresholds.get("medium", 0.4):
            return "medium"
        return "low"

    @staticmethod
    def _recommended_actions(
        churn_probability: float, ranked_recs: list[dict], payload: dict
    ) -> list[str]:
        actions: list[str] = []
        if churn_probability >= 0.7:
            actions.append("Trigger CSM outreach within 48 hours")
        elif churn_probability >= 0.4:
            actions.append("Add to CSM watch-list for proactive check-in")
        if payload.get("onboarding_completion_rate", 1.0) < 0.7:
            actions.append("Offer onboarding recovery workflow")
        if ranked_recs:
            actions.append(f"Promote {ranked_recs[0]['feature']} in-app to increase adoption")
        if payload.get("support_tickets_90d", 0) >= 5:
            actions.append("Review recent support themes with PM and support engineering")
        if payload.get("renewal_days_remaining", 365) < 60 and churn_probability >= 0.4:
            actions.append("Prepare renewal save-play ahead of contract end")
        return actions


@lru_cache(maxsize=1)
def get_inference_service() -> InferenceService:
    return InferenceService()


def reset_inference_service() -> None:
    """Clear the cached singleton (used after retraining / in tests)."""
    get_inference_service.cache_clear()
