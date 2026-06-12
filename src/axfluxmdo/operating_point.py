"""Operating point definition."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class OperatingPoint:
    """A single electrical/mechanical operating condition."""

    speed_rpm: float
    current_rms: float  # phase RMS current, A
    dc_bus_voltage: float = 48.0
    ambient_temp_c: float = 25.0

    def __post_init__(self) -> None:
        if self.speed_rpm < 0.0:
            raise ValueError("speed_rpm must be non-negative")
        if self.current_rms < 0.0:
            raise ValueError("current_rms must be non-negative")
        if self.dc_bus_voltage <= 0.0:
            raise ValueError("dc_bus_voltage must be positive")

    @property
    def speed_rad_s(self) -> float:
        return self.speed_rpm * 2.0 * math.pi / 60.0
