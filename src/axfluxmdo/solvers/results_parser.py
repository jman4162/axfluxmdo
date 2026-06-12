"""Parse GetDP output tables into gap-field solutions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def parse_table(path: str | Path) -> np.ndarray:
    """Parse a GetDP ``Format Table`` line print of a 3-component vector field.

    GetDP Table rows carry a variable number of leading metadata columns
    (element type / index), but for a vector field printed OnLine the
    TRAILING SIX columns are always ``x y z vx vy vz``. Returns an (n, 6)
    array of those columns, sorted by x. Sanity-checks that y and z are
    constant (it was a line sample).
    """
    data = np.loadtxt(path, comments="#", ndmin=2)
    if data.shape[1] < 6:
        raise ValueError(
            f"{path}: expected >= 6 columns (x y z vx vy vz at the end), got {data.shape[1]}"
        )
    cols = data[:, -6:]
    if np.ptp(cols[:, 1]) > 1e-9 or np.ptp(cols[:, 2]) > 1e-9:
        raise ValueError(f"{path}: y/z vary along the sample line; not an OnLine table")
    return cols[np.argsort(cols[:, 0])]


@dataclass(frozen=True)
class GapFieldSolution:
    """Axial flux density sampled along the air-gap midline of the unrolled model."""

    x_m: np.ndarray  # ascending positions over one pole pair; may duplicate 0 and L
    by_t: np.ndarray  # axial component of B along the midline
    pole_pitch_m: float
    magnet_arc_ratio: float
    magnet_temp_c: float
    slotted: bool

    @property
    def x_span_m(self) -> float:
        return 2.0 * self.pole_pitch_m

    def _magnet_mask(self) -> np.ndarray:
        """True where x lies under a magnet (either polarity)."""
        tau = self.pole_pitch_m
        half_arc = 0.5 * self.magnet_arc_ratio * tau
        x = np.mod(self.x_m, tau)  # fold both poles onto one pitch
        return np.abs(x - 0.5 * tau) <= half_arc

    @property
    def mean_b_t(self) -> float:
        """Mean |By| over the magnet-covered intervals only.

        This is the load-line ``B_g`` semantics — flux density UNDER the
        magnet — not the full-pitch mean (see ``mean_b_full_pitch_t``).
        """
        mask = self._magnet_mask()
        return float(np.mean(np.abs(self.by_t[mask])))

    @property
    def mean_b_full_pitch_t(self) -> float:
        return float(np.mean(np.abs(self.by_t)))

    @property
    def fundamental_b1_t(self) -> float:
        """Amplitude of the spatial fundamental (period = one pole pair).

        Trapezoid Fourier projection over the periodic span — robust to the
        duplicated x=0 / x=L endpoint that an OnLine sample produces (a naive
        FFT bin would double-count it).
        """
        length = self.x_span_m
        phase = np.exp(-1j * 2.0 * math.pi * self.x_m / length)
        integral = np.trapezoid(self.by_t * phase, self.x_m)
        return float(2.0 / length * np.abs(integral))

    def to_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return self.x_m.copy(), self.by_t.copy()

    @classmethod
    def from_table(
        cls,
        path: str | Path,
        motor,
        *,
        magnet_temp_c: float,
        slotted: bool = False,
    ) -> GapFieldSolution:
        """Build a solution from a (possibly committed golden) GetDP table."""
        cols = parse_table(path)
        return cls(
            x_m=cols[:, 0],
            by_t=cols[:, 4],
            pole_pitch_m=motor.pole_pitch,
            magnet_arc_ratio=motor.magnet_arc_ratio,
            magnet_temp_c=magnet_temp_c,
            slotted=slotted,
        )
