from product_intelligence.data.synthetic import generate_intervention_data
from product_intelligence.models.uplift import train_uplift_model


def test_uplift_recovers_signal():
    df = generate_intervention_data(2500, seed=7)
    result = train_uplift_model(df, seed=7)
    # Synthetic treatment genuinely helps, so Qini should be clearly positive.
    assert result.metrics["qini_coefficient"] > 0.1
    # Top-uplift decile: treated retention should exceed control retention.
    assert (
        result.metrics["top_decile_treated_retention"]
        >= result.metrics["top_decile_control_retention"]
    )
