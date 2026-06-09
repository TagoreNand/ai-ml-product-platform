from __future__ import annotations

from collections.abc import Iterable


def prioritize_initiatives(initiatives: Iterable[dict]) -> list[dict]:
    ranked = []
    for item in initiatives:
        rice = (item["reach"] * item["impact"] * max(item["confidence"], 1)) / max(
            item["effort"], 1
        )
        strategic_bonus = 1.5 * item["strategic_alignment"]
        evidence_bonus = 1.2 * item["evidence_strength"]
        uplift_bonus = 100 * item["estimated_model_uplift"]
        total = round(rice + strategic_bonus + evidence_bonus + uplift_bonus, 2)
        ranked.append(
            {
                **item,
                "priority_score": total,
                "rationale": _build_rationale(item, total),
            }
        )
    return sorted(ranked, key=lambda x: x["priority_score"], reverse=True)


def _build_rationale(item: dict, score: float) -> str:
    strongest = []
    if item["strategic_alignment"] >= 8:
        strongest.append("strong strategic alignment")
    if item["estimated_model_uplift"] >= 0.08:
        strongest.append("meaningful modeled uplift")
    if item["confidence"] >= 7 and item["evidence_strength"] >= 7:
        strongest.append("good decision confidence")
    if item["effort"] <= 4:
        strongest.append("relatively low delivery effort")
    joined = ", ".join(strongest) if strongest else "balanced but less differentiated trade-offs"
    return f"Priority score {score} driven by {joined}."
