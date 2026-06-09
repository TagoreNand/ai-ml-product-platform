import pytest

from product_intelligence.features.builders import (
    DERIVED_NUMERIC_COLUMNS,
    MODEL_COLUMNS,
    build_feature_frame,
    label_for,
)


def test_derived_columns_present(sample_account):
    result = build_feature_frame([sample_account])
    for col in DERIVED_NUMERIC_COLUMNS:
        assert col in result.columns
    assert set(MODEL_COLUMNS).issubset(result.columns)


def test_missing_columns_raise():
    with pytest.raises(ValueError, match="Missing required"):
        build_feature_frame([{"account_id": "x"}])


def test_derived_values_are_finite(sample_account):
    sample_account["monthly_active_users"] = 0  # force divide-by-zero guard
    result = build_feature_frame([sample_account])
    assert result[DERIVED_NUMERIC_COLUMNS].notna().all().all()


def test_label_for_known_and_unknown():
    assert label_for("feature_adoption_rate") == "Feature adoption rate"
    assert label_for("some_new_col") == "Some New Col"
