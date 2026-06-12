"""Design limit values used to evaluate constraints."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Limits:
    """Constraint limit values.

    ``None`` fields are resolved at evaluation time from the motor and
    operating point: max line voltage from the DC bus (SVPWM, V_dc/sqrt(2)),
    max core flux from the steel's saturation knee, and max magnet temperature
    from the magnet grade rating.
    """

    max_winding_temp_c: float = 140.0  # insulation class F continuous proxy
    max_electrical_freq_hz: float = 1000.0
    max_current_density_a_mm2: float = 10.0
    max_line_voltage_v: float | None = None
    max_core_flux_density_t: float | None = None
    max_magnet_temp_c: float | None = None
