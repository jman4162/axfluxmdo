"""Simulation-to-analytical residual analysis.

Compares a GetDP open-circuit gap-field solution against the Phase-1 load
line and fundamental, and extracts an effective Carter factor from a
slotless/slotted solution pair.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

from axfluxmdo.geometry.axial_flux import AxialFluxMotor
from axfluxmdo.materials.magnetic import airgap_flux_density
from axfluxmdo.solvers.results_parser import GapFieldSolution


@dataclass(frozen=True)
class OpenCircuitComparison:
    magnet_temp_c: float
    analytical_b_g_t: float
    analytical_b1_t: float
    fea_mean_b_t: float
    fea_b1_t: float
    slotted: bool

    @property
    def residual_b_g_t(self) -> float:
        return self.fea_mean_b_t - self.analytical_b_g_t

    @property
    def residual_b_g_rel(self) -> float:
        return self.residual_b_g_t / self.analytical_b_g_t

    @property
    def residual_b1_t(self) -> float:
        return self.fea_b1_t - self.analytical_b1_t

    @property
    def residual_b1_rel(self) -> float:
        return self.residual_b1_t / self.analytical_b1_t

    def __str__(self) -> str:
        kind = "slotted" if self.slotted else "slotless"
        return "\n".join(
            [
                f"Open-circuit comparison ({kind}, magnets at {self.magnet_temp_c:.0f} °C)",
                f"  B_g: {self.analytical_b_g_t:.4f} T (analytical) vs "
                f"{self.fea_mean_b_t:.4f} T (FEA)  ->  {self.residual_b_g_rel:+.2%}",
                f"  B_1: {self.analytical_b1_t:.4f} T (analytical) vs "
                f"{self.fea_b1_t:.4f} T (FEA)  ->  {self.residual_b1_rel:+.2%}",
            ]
        )


def compare_open_circuit(
    motor: AxialFluxMotor,
    solution: GapFieldSolution,
    *,
    magnet_temp_c: float,
) -> OpenCircuitComparison:
    """Residuals between the analytical layer and an FEA gap-field solution.

    ``magnet_temp_c`` is a required keyword: the caller must state the magnet
    temperature the analytical side is evaluated at (65 °C corresponds to the
    default operating point).
    """
    if abs(magnet_temp_c - solution.magnet_temp_c) > 1e-9:
        warnings.warn(
            f"comparison at {magnet_temp_c} °C but the FEA solution was computed at "
            f"{solution.magnet_temp_c} °C — residuals will mix temperature effects",
            stacklevel=2,
        )
    b_g = airgap_flux_density(motor.magnet, motor.magnet_thickness, motor.air_gap, magnet_temp_c)
    b1 = (4.0 / math.pi) * b_g * math.sin(motor.magnet_arc_ratio * math.pi / 2.0)
    return OpenCircuitComparison(
        magnet_temp_c=magnet_temp_c,
        analytical_b_g_t=b_g,
        analytical_b1_t=b1,
        fea_mean_b_t=solution.mean_b_t,
        fea_b1_t=solution.fundamental_b1_t,
        slotted=solution.slotted,
    )


def measured_carter_factor(
    slotless: GapFieldSolution,
    slotted: GapFieldSolution,
    motor: AxialFluxMotor,
) -> float:
    """Effective Carter factor extracted from a slotless/slotted FEA pair.

    Inverting the load line with BOTH solutions cancels the common fringing
    bias: with B_sl/B_st the under-magnet means,

        k_C = ((B_sl/B_st) * (t_m + mu_r*g) - t_m) / (mu_r * g)

    Closure property: ``airgap_flux_density(..., carter_factor=k_C)`` then
    reproduces the slotted FEA mean (up to the shared fringing bias).
    """
    if slotless.slotted or not slotted.slotted:
        raise ValueError("pass (slotless_solution, slotted_solution) in that order")
    ratio = slotless.mean_b_t / slotted.mean_b_t
    t_m = motor.magnet_thickness
    mu_r_g = motor.magnet.mu_r * motor.air_gap
    return (ratio * (t_m + mu_r_g) - t_m) / mu_r_g
