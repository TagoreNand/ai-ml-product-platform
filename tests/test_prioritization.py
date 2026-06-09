from product_intelligence.services.prioritization import prioritize_initiatives


def test_ordering_and_rationale():
    items = [
        {
            "name": "low",
            "reach": 2,
            "impact": 2,
            "confidence": 3,
            "effort": 8,
            "strategic_alignment": 2,
            "estimated_model_uplift": 0.0,
            "evidence_strength": 2,
        },
        {
            "name": "high",
            "reach": 9,
            "impact": 9,
            "confidence": 8,
            "effort": 2,
            "strategic_alignment": 9,
            "estimated_model_uplift": 0.2,
            "evidence_strength": 9,
        },
    ]
    ranked = prioritize_initiatives(items)
    assert ranked[0]["name"] == "high"
    assert ranked[0]["priority_score"] > ranked[1]["priority_score"]
    assert "rationale" in ranked[0]
