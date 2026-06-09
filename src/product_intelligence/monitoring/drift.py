"""Data and prediction drift monitoring.

Provides the standard production toolkit: Population Stability Index (PSI) and
two-sample Kolmogorov-Smirnov for numeric features, plus a categorical
distribution distance, rolled up into a severity-graded ``DriftReport``. The
report is what a scheduled job would emit to alerting and what gates the
retraining decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Conventional PSI thresholds: <0.1 stable, 0.1-0.25 moderate, >0.25 significant.
PSI_MODERATE = 0.10
PSI_SIGNIFICANT = 0.25


def population_stability_index(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    breakpoints = np.quantile(expected, q=np.linspace(0, 1, bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 3:
        return 0.0
    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)
    expected_perc = np.clip(expected_counts / max(expected_counts.sum(), 1), 1e-6, None)
    actual_perc = np.clip(actual_counts / max(actual_counts.sum(), 1), 1e-6, None)
    psi = np.sum((actual_perc - expected_perc) * np.log(actual_perc / expected_perc))
    return float(round(psi, 6))


def ks_statistic(expected: np.ndarray, actual: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic (max CDF gap), no SciPy needed."""
    expected = np.sort(np.asarray(expected, dtype=float))
    actual = np.sort(np.asarray(actual, dtype=float))
    if expected.size == 0 or actual.size == 0:
        return 0.0
    grid = np.concatenate([expected, actual])
    cdf_e = np.searchsorted(expected, grid, side="right") / expected.size
    cdf_a = np.searchsorted(actual, grid, side="right") / actual.size
    return float(round(np.max(np.abs(cdf_e - cdf_a)), 6))


def categorical_l1(expected: pd.Series, actual: pd.Series) -> float:
    """Total-variation-style L1 distance between category distributions."""
    e = expected.value_counts(normalize=True)
    a = actual.value_counts(normalize=True)
    cats = e.index.union(a.index)
    return float(
        round(0.5 * np.abs(e.reindex(cats, fill_value=0) - a.reindex(cats, fill_value=0)).sum(), 6)
    )


def _severity(psi: float) -> str:
    if psi >= PSI_SIGNIFICANT:
        return "significant"
    if psi >= PSI_MODERATE:
        return "moderate"
    return "stable"


@dataclass
class FeatureDrift:
    feature: str
    kind: str
    psi: float | None
    ks: float | None
    l1: float | None
    severity: str


@dataclass
class DriftReport:
    n_reference: int
    n_current: int
    features: list[FeatureDrift] = field(default_factory=list)
    prediction_psi: float | None = None
    overall_severity: str = "stable"
    flagged_features: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_reference": self.n_reference,
            "n_current": self.n_current,
            "overall_severity": self.overall_severity,
            "prediction_psi": self.prediction_psi,
            "flagged_features": self.flagged_features,
            "features": [fd.__dict__ for fd in self.features],
        }


def build_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str] | None = None,
    reference_scores: np.ndarray | None = None,
    current_scores: np.ndarray | None = None,
) -> DriftReport:
    categorical_features = categorical_features or []
    feature_drifts: list[FeatureDrift] = []
    flagged: list[str] = []
    worst = "stable"
    order = {"stable": 0, "moderate": 1, "significant": 2}

    for feat in numeric_features:
        if feat not in reference or feat not in current:
            continue
        psi = population_stability_index(reference[feat].to_numpy(), current[feat].to_numpy())
        ks = ks_statistic(reference[feat].to_numpy(), current[feat].to_numpy())
        sev = _severity(psi)
        feature_drifts.append(FeatureDrift(feat, "numeric", psi, ks, None, sev))
        if sev != "stable":
            flagged.append(feat)
        worst = max(worst, sev, key=lambda s: order[s])

    for feat in categorical_features:
        if feat not in reference or feat not in current:
            continue
        l1 = categorical_l1(reference[feat], current[feat])
        sev = "significant" if l1 >= 0.25 else "moderate" if l1 >= 0.1 else "stable"
        feature_drifts.append(FeatureDrift(feat, "categorical", None, None, l1, sev))
        if sev != "stable":
            flagged.append(feat)
        worst = max(worst, sev, key=lambda s: order[s])

    pred_psi = None
    if reference_scores is not None and current_scores is not None:
        pred_psi = population_stability_index(reference_scores, current_scores)
        worst = max(worst, _severity(pred_psi), key=lambda s: order[s])

    return DriftReport(
        n_reference=len(reference),
        n_current=len(current),
        features=feature_drifts,
        prediction_psi=pred_psi,
        overall_severity=worst,
        flagged_features=flagged,
    )
