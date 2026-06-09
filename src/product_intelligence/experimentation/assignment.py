"""Deterministic experiment assignment (extension).

Stable, stateless bucketing: the same unit always lands in the same variant for a
given experiment (hash of ``salt:experiment:unit``), assignments are independent
across experiments, and traffic splits honour configured weights. No store, no
race conditions - the standard approach used by Optimizely/GrowthBook-style A/B
frameworks.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class Assignment:
    unit_id: str
    experiment: str
    variant: str


def _bucket(unit_id: str, experiment: str, salt: str) -> float:
    digest = hashlib.sha256(f"{salt}:{experiment}:{unit_id}".encode()).hexdigest()
    return int(digest[:12], 16) / float(16**12)  # uniform in [0, 1)


def assign_variant(
    unit_id: str,
    experiment: str,
    variants: dict[str, float] | None = None,
    salt: str = "pulse360",
) -> Assignment:
    variants = variants or {"control": 0.5, "treatment": 0.5}
    total = sum(variants.values())
    if total <= 0:
        raise ValueError("variant weights must sum to a positive number")
    point = _bucket(unit_id, experiment, salt)
    cumulative = 0.0
    chosen = next(iter(variants))
    for name, weight in variants.items():
        cumulative += weight / total
        if point < cumulative:
            chosen = name
            break
    return Assignment(unit_id=unit_id, experiment=experiment, variant=chosen)
