"""Per-prediction explainability for the churn model.

We expose *why* an account is risky, not just a probability. The explainer uses
SHAP when available (model-agnostic ``PermutationExplainer`` over the raw feature
space, so attributions map to business-readable columns), and degrades
gracefully to a deterministic perturbation-based attribution when SHAP is not
installed. Either way callers get a stable ``[(feature, signed_contribution)]``
contract where positive values push churn risk *up*.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from product_intelligence.core.logging import get_logger
from product_intelligence.features.builders import MODEL_COLUMNS, label_for

logger = get_logger(__name__)


class ModelExplainer:
    """Wrap a fitted churn pipeline with SHAP / fallback attributions."""

    def __init__(self, pipeline: Any, background: pd.DataFrame, enable_shap: bool = True) -> None:
        self.pipeline = pipeline
        self.columns = list(MODEL_COLUMNS)
        self.background = background[self.columns].reset_index(drop=True)
        self._explainer = None
        self._mode = "fallback"
        if enable_shap:
            self._try_build_shap()

    def _predict_pos(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.columns)
        return self.pipeline.predict_proba(frame[self.columns])[:, 1]

    def _try_build_shap(self) -> None:
        try:
            import shap

            bg = self.background
            if len(bg) > 64:
                bg = bg.sample(64, random_state=0).reset_index(drop=True)
            masker = shap.maskers.Independent(bg, max_samples=len(bg))
            self._explainer = shap.PermutationExplainer(self._predict_pos, masker)
            self._mode = "shap"
            logger.info("SHAP PermutationExplainer initialised over %d background rows", len(bg))
        except Exception as exc:  # pragma: no cover - optional dependency path
            logger.warning("SHAP unavailable (%s); using fallback attributions", exc)
            self._explainer = None
            self._mode = "fallback"

    @property
    def mode(self) -> str:
        return self._mode

    def contributions(self, row: pd.DataFrame) -> list[tuple[str, float]]:
        """Return ``[(feature, signed_contribution)]`` sorted by |impact| desc."""
        row = row[self.columns].reset_index(drop=True)
        if self._explainer is not None:
            try:
                explanation = self._explainer(row, max_evals=2 * len(self.columns) + 1)
                values = explanation.values[0]  # noqa: PD011  (shap Explanation, not a DataFrame)
            except Exception as exc:  # pragma: no cover
                logger.warning("SHAP scoring failed (%s); falling back", exc)
                values = self._fallback_contributions(row)
        else:
            values = self._fallback_contributions(row)
        pairs = list(zip(self.columns, np.asarray(values, dtype=float).ravel(), strict=False))
        return sorted(pairs, key=lambda kv: abs(kv[1]), reverse=True)

    def _fallback_contributions(self, row: pd.DataFrame) -> np.ndarray:
        """Occlusion-style attribution: how much each feature moves the score
        relative to the background median/mode baseline."""
        base = self._predict_pos(self.background).mean()
        contribs = np.zeros(len(self.columns))
        for i, col in enumerate(self.columns):
            perturbed = row.copy()
            if pd.api.types.is_numeric_dtype(self.background[col]):
                perturbed[col] = self.background[col].median()
            else:
                perturbed[col] = self.background[col].mode().iloc[0]
            contribs[i] = float(self._predict_pos(row)[0] - self._predict_pos(perturbed)[0])
        # Centre so contributions sum roughly to (score - base).
        total = self._predict_pos(row)[0] - base
        if contribs.sum() != 0:
            contribs = contribs * (total / contribs.sum())
        return contribs

    def top_risk_drivers(self, row: pd.DataFrame, k: int = 5) -> list[str]:
        """Human-readable drivers that *increase* churn risk."""
        risky = [(c, v) for c, v in self.contributions(row) if v > 0][:k]
        if not risky:
            return ["No dominant risk driver detected"]
        return [f"{label_for(c)} ({_direction(c, row, v)})" for c, v in risky]

    def protective_factors(self, row: pd.DataFrame, k: int = 3) -> list[str]:
        """Human-readable factors that *reduce* churn risk."""
        safe = [(c, v) for c, v in self.contributions(row) if v < 0][:k]
        return [label_for(c) for c, _ in safe]


def _direction(column: str, row: pd.DataFrame, value: float) -> str:
    try:
        raw = row.iloc[0][column]
    except Exception:
        return "elevated risk"
    if isinstance(raw, (int, float, np.floating, np.integer)):
        return f"value={round(float(raw), 3)}"
    return f"segment={raw}"
