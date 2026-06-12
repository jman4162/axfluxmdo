"""Constraint evaluation records.

Constraint names deliberately match the flat keys of
``AnalyticalResult.to_dict()`` (e.g. ``winding_temp_c``) so that Phase 3's
``optimize_pareto`` can parse SPEC-style constraint strings like
``"winding_temp_c < 140"`` against result dictionaries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ConstraintRecord:
    name: str
    value: float
    limit: float
    satisfied: bool
    margin: float  # normalized headroom: (limit - value) / |limit|; > 0 is feasible
    sense: str = "<="

    def __str__(self) -> str:
        status = "OK" if self.satisfied else "VIOLATED"
        return (
            f"{self.name}: {self.value:.4g} {self.sense} {self.limit:.4g} "
            f"[{status}, margin {self.margin:+.1%}]"
        )


def make_upper_bound(name: str, value: float, limit: float) -> ConstraintRecord:
    """Build a '<=' constraint record with normalized margin."""
    if math.isinf(value):
        return ConstraintRecord(
            name=name, value=value, limit=limit, satisfied=False, margin=-math.inf
        )
    margin = (limit - value) / abs(limit) if limit != 0.0 else -value
    return ConstraintRecord(
        name=name, value=value, limit=limit, satisfied=value <= limit, margin=margin
    )
