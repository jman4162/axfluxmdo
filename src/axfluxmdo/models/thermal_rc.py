"""Steady-state lumped thermal network (single winding-to-ambient resistance).

Copper loss rises linearly with winding temperature through rho(T), so the
steady-state fixed point

    T_w = T_amb + R_theta * (P_cu(T_w) + P_other)
    P_cu(T) = P_cu_ref * (1 + alpha * (T - T_ref))

has the closed-form solution

    T_w = [T_amb + R_theta * (P_cu_ref * (1 - alpha * T_ref) + P_other)]
          / [1 - R_theta * P_cu_ref * alpha]

Thermal runaway occurs when the denominator is non-positive: each kelvin of
temperature rise adds more than a kelvin's worth of extra copper loss.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ThermalSolution:
    winding_temp_c: float
    copper_loss_w: float  # copper loss re-evaluated at the solved temperature
    runaway: bool


def solve_winding_temperature(
    p_cu_ref_w: float,
    ref_temp_c: float,
    alpha_per_c: float,
    p_other_w: float,
    r_theta_k_per_w: float,
    ambient_c: float,
) -> ThermalSolution:
    """Closed-form steady-state winding temperature with R(T) coupling.

    p_cu_ref_w: copper loss evaluated at ref_temp_c.
    p_other_w: other losses deposited in the winding node (e.g. a fraction of
        core loss), assumed temperature-independent.
    """
    denominator = 1.0 - r_theta_k_per_w * p_cu_ref_w * alpha_per_c
    if denominator <= 0.0:
        return ThermalSolution(winding_temp_c=math.inf, copper_loss_w=math.inf, runaway=True)
    numerator = ambient_c + r_theta_k_per_w * (
        p_cu_ref_w * (1.0 - alpha_per_c * ref_temp_c) + p_other_w
    )
    t_w = numerator / denominator
    p_cu = p_cu_ref_w * (1.0 + alpha_per_c * (t_w - ref_temp_c))
    return ThermalSolution(winding_temp_c=t_w, copper_loss_w=p_cu, runaway=False)


def solve_winding_temperature_iterative(
    p_cu_ref_w: float,
    ref_temp_c: float,
    alpha_per_c: float,
    p_other_w: float,
    r_theta_k_per_w: float,
    ambient_c: float,
    iterations: int = 50,
) -> float:
    """Fixed-point iteration reference implementation (used to verify the closed form)."""
    t_w = ambient_c
    for _ in range(iterations):
        p_cu = p_cu_ref_w * (1.0 + alpha_per_c * (t_w - ref_temp_c))
        t_w = ambient_c + r_theta_k_per_w * (p_cu + p_other_w)
    return t_w
