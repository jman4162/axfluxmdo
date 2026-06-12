"""Efficiency map over the speed-torque plane.

The Layer-1/2 models have no saturation, so torque is exactly linear in
current at fixed geometry (flux linkage is independent of current and speed;
the magnet temperature is the fixed Phase-1/2 assumption). One probe
evaluation yields torque-per-amp, then each grid cell is a single
``model.evaluate`` at the inverted current. Cells whose result violates any
constraint are masked NaN, with the first violated constraint recorded.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np

from axfluxmdo.geometry.axial_flux import AxialFluxMotor
from axfluxmdo.models.base import Model
from axfluxmdo.operating_point import OperatingPoint


@dataclass
class EfficiencyMap:
    """Gridded results over (torque, speed); arrays are (n_torque, n_speed)."""

    speeds_rpm: np.ndarray
    torques_nm: np.ndarray
    efficiency: np.ndarray  # NaN where infeasible
    current_rms_a: np.ndarray
    copper_loss_w: np.ndarray
    core_loss_w: np.ndarray
    winding_temp_c: np.ndarray
    feasible: np.ndarray  # bool
    binding_constraint: np.ndarray  # str, "" where feasible

    def plot(self, show: bool = False):
        from axfluxmdo.viz.fields import plot_efficiency_map

        return plot_efficiency_map(self, show=show)


def compute_efficiency_map(
    motor: AxialFluxMotor,
    base_op: OperatingPoint,
    *,
    max_speed_rpm: float,
    max_torque_nm: float,
    n_speed: int = 40,
    n_torque: int = 40,
    model: Model | None = None,
) -> EfficiencyMap:
    """Evaluate the motor over a speed-torque grid (bus voltage and ambient from base_op)."""
    if model is None:
        from axfluxmdo.models.annular_2p5d import AnnularModel

        model = AnnularModel()

    # Linear torque-current inversion from one probe evaluation
    probe = model.evaluate(motor, dataclasses.replace(base_op, current_rms=1.0))
    torque_per_amp = probe.torque_nm
    if torque_per_amp <= 0.0:
        raise ValueError("motor produces no torque per amp; check the design")

    # Start one step above zero on both axes (efficiency is degenerate at 0)
    speeds = np.linspace(max_speed_rpm / n_speed, max_speed_rpm, n_speed)
    torques = np.linspace(max_torque_nm / n_torque, max_torque_nm, n_torque)

    shape = (n_torque, n_speed)
    eff = np.full(shape, np.nan)
    current = np.zeros(shape)
    p_cu = np.full(shape, np.nan)
    p_core = np.full(shape, np.nan)
    temp = np.full(shape, np.nan)
    feasible = np.zeros(shape, dtype=bool)
    binding = np.full(shape, "", dtype=object)

    for i, torque in enumerate(torques):
        i_rms = torque / torque_per_amp
        for j, speed in enumerate(speeds):
            r = model.evaluate(
                motor, dataclasses.replace(base_op, speed_rpm=speed, current_rms=i_rms)
            )
            current[i, j] = i_rms
            feasible[i, j] = r.feasible
            if r.feasible:
                eff[i, j] = r.efficiency
                p_cu[i, j] = r.copper_loss_w
                p_core[i, j] = r.core_loss_w
                temp[i, j] = r.winding_temp_c if math.isfinite(r.winding_temp_c) else np.nan
            else:
                binding[i, j] = next(c.name for c in r.constraints if not c.satisfied)

    return EfficiencyMap(
        speeds_rpm=speeds,
        torques_nm=torques,
        efficiency=eff,
        current_rms_a=current,
        copper_loss_w=p_cu,
        core_loss_w=p_core,
        winding_temp_c=temp,
        feasible=feasible,
        binding_constraint=binding,
    )
