"""Electrical conductor properties."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Conductor:
    """Winding conductor material."""

    name: str
    resistivity_20c_ohm_m: float
    temp_coeff_per_c: float  # linear temperature coefficient of resistivity, 1/C
    density_kg_m3: float


COPPER = Conductor(
    "copper", resistivity_20c_ohm_m=1.724e-8, temp_coeff_per_c=0.00393, density_kg_m3=8960.0
)


def resistivity(conductor: Conductor, temp_c: float) -> float:
    """Resistivity at temperature T: rho(T) = rho_20 * (1 + alpha * (T - 20))."""
    return conductor.resistivity_20c_ohm_m * (1.0 + conductor.temp_coeff_per_c * (temp_c - 20.0))
