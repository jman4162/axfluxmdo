"""Air-gap imperfection models: assembly error, rotor coning, and runout.

These are manufacturing design variables (see SPEC.md): Layer 1
(:class:`~axfluxmdo.models.analytical.AnalyticalModel`) models the perfect
axisymmetric machine and ignores them; only the annular 2.5D model consumes
them. The local gap law is

    g(r, theta) = g0 + gap_offset + coning * (r - r_m)/(r_o - r_i) + runout * cos(theta)

where the coning term is zero-mean at the mean radius and runout is a
once-per-revolution axial oscillation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class GapImperfections:
    """Deviations of the running air gap from its nominal design value (SI meters)."""

    gap_offset_m: float = 0.0  # uniform mean-gap error (positive opens the gap)
    coning_m: float = 0.0  # gap change from r_i to r_o (positive opens outward)
    runout_m: float = 0.0  # peak 1/rev axial runout amplitude, >= 0

    def __post_init__(self) -> None:
        if self.runout_m < 0.0:
            raise ValueError("runout_m must be non-negative")

    @property
    def is_perfect(self) -> bool:
        return self.gap_offset_m == 0.0 and self.coning_m == 0.0 and self.runout_m == 0.0

    def axisymmetric_gap(self, nominal_gap: float, r: float, r_i: float, r_o: float) -> float:
        """Mean-over-theta local gap at radius r (offset and coning, no runout)."""
        r_m = 0.5 * (r_i + r_o)
        return nominal_gap + self.gap_offset_m + self.coning_m * (r - r_m) / (r_o - r_i)

    def local_gap(
        self, nominal_gap: float, r: float, r_i: float, r_o: float, theta: float = 0.0
    ) -> float:
        """Local gap at radius r and circumferential angle theta."""
        return self.axisymmetric_gap(nominal_gap, r, r_i, r_o) + self.runout_m * math.cos(theta)


PERFECT_GAP = GapImperfections()
