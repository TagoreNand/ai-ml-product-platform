"""Shadow + canary rollout (extension).

Safe model promotion on top of the registry's stage pointers:
* **Shadow** - score every request with both the production model and a candidate,
  serve production, and log the divergence. Validates a candidate on live traffic
  with zero user risk.
* **Canary** - deterministically route a configurable fraction of accounts to the
  candidate as the *served* model (stable per account via hash assignment), so a
  new version can be ramped 1% -> 100% with a clean rollback (just re-point
  ``production`` in the registry).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

from product_intelligence.core.config import settings
from product_intelligence.core.logging import get_logger
from product_intelligence.experimentation.assignment import assign_variant
from product_intelligence.features.builders import MODEL_COLUMNS, build_feature_frame
from product_intelligence.models.registry import ModelRegistry

logger = get_logger(__name__)


@dataclass
class RolloutResult:
    account_id: str
    served_by: str  # "production" | "candidate"
    served_version: str
    churn_probability: float
    shadow: dict | None = None


class RolloutController:
    def __init__(
        self, registry: ModelRegistry | None = None, candidate_selector: str = "latest"
    ) -> None:
        self.registry = registry or ModelRegistry(settings.model_artifact_dir)
        self.prod_model, self.prod_record = self.registry.load("churn", "production")
        self.candidate_model = None
        self.candidate_record = None
        try:
            cand_version = self.registry.resolve_version("churn", candidate_selector)
            if cand_version != self.prod_record.version:
                self.candidate_model, self.candidate_record = self.registry.load(
                    "churn", cand_version
                )
        except Exception:  # pragma: no cover
            pass

    @property
    def has_candidate(self) -> bool:
        return self.candidate_model is not None

    def _prob(self, model, payload: dict) -> float:
        X = build_feature_frame(pd.DataFrame([payload]))[MODEL_COLUMNS]
        return float(model.predict_proba(X)[0, 1])

    def canary_variant(self, account_id: str) -> str:
        if not self.has_candidate or settings.canary_percent <= 0:
            return "production"
        weights = {
            "production": 1.0 - settings.canary_percent,
            "candidate": settings.canary_percent,
        }
        return assign_variant(
            account_id, "churn_canary", weights, salt=settings.canary_salt
        ).variant

    def score(self, payload: dict) -> RolloutResult:
        account_id = payload.get("account_id", "unknown")
        prod_prob = self._prob(self.prod_model, payload)

        shadow = None
        if self.has_candidate and settings.shadow_enabled:
            assert self.candidate_record is not None
            cand_prob = self._prob(self.candidate_model, payload)
            shadow = {
                "candidate_version": self.candidate_record.version,
                "candidate_probability": round(cand_prob, 4),
                "abs_divergence": round(abs(cand_prob - prod_prob), 4),
            }

        served_by = self.canary_variant(account_id)
        if served_by == "candidate" and self.has_candidate:
            assert self.candidate_record is not None
            served_prob = self._prob(self.candidate_model, payload)
            served_version = self.candidate_record.version
        else:
            served_by, served_prob, served_version = (
                "production",
                prod_prob,
                self.prod_record.version,
            )

        return RolloutResult(
            account_id=account_id,
            served_by=served_by,
            served_version=served_version,
            churn_probability=round(served_prob, 4),
            shadow=shadow,
        )

    def shadow_report(self, payloads: list[dict]) -> dict:
        """Aggregate divergence between production and candidate over a sample."""
        if not self.has_candidate:
            return {"has_candidate": False}
        assert self.candidate_record is not None
        prod = pd.Series([self._prob(self.prod_model, p) for p in payloads])
        cand = pd.Series([self._prob(self.candidate_model, p) for p in payloads])
        band_flips = int(((prod >= 0.5) != (cand >= 0.5)).sum())
        return {
            "has_candidate": True,
            "candidate_version": self.candidate_record.version,
            "n": len(payloads),
            "mean_abs_divergence": round(float((prod - cand).abs().mean()), 4),
            "max_abs_divergence": round(float((prod - cand).abs().max()), 4),
            "decision_flips": band_flips,
        }


@lru_cache(maxsize=1)
def get_rollout_controller() -> RolloutController:
    return RolloutController()


def reset_rollout_controller() -> None:
    get_rollout_controller.cache_clear()
