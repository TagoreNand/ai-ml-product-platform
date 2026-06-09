import numpy as np

from product_intelligence.data.synthetic import generate_accounts
from product_intelligence.features.builders import (
    CATEGORICAL_COLUMNS,
    MODEL_COLUMNS,
    NUMERIC_COLUMNS,
    build_feature_frame,
)
from product_intelligence.monitoring.drift import (
    build_drift_report,
    ks_statistic,
    population_stability_index,
)


def test_psi_zero_for_same_distribution():
    rng = np.random.default_rng(0)
    x = rng.normal(size=2000)
    assert population_stability_index(x, x) < 1e-6


def test_psi_and_ks_flag_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 2000)
    b = rng.normal(3, 1, 2000)
    assert population_stability_index(a, b) > 0.25
    assert ks_statistic(a, b) > 0.5


def test_drift_report_stable_when_same_source():
    ref = build_feature_frame(generate_accounts(1500, seed=1))[MODEL_COLUMNS]
    cur = build_feature_frame(generate_accounts(1500, seed=2))[MODEL_COLUMNS]
    report = build_drift_report(ref, cur, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS)
    assert report.overall_severity == "stable"
    assert report.flagged_features == []
