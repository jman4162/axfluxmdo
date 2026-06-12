"""Fast analytical sizing model (Phase 1, Layer 1).

Electromagnetic formulation
---------------------------
The air-gap field comes from the magnet load line (``materials.magnetic``),
its fundamental being B1 = (4/pi) * B_g * sin(alpha_m * pi / 2). The
fundamental flux per pole over the annulus is

    Phi_p = (2/pi) * B1 * A_g / (2p) = B1 * A_g / (pi * p)

and the peak phase flux linkage lambda = k_w * N * Phi_p. Torque and
back-EMF are BOTH derived from this same flux linkage (sinusoidal machine,
currents aligned with EMF / MTPA for a surface-PM machine):

    E_rms = omega_e * lambda / sqrt(2)        (equivalently 4.44 f N k_w Phi_p)
    T     = m * p * lambda * I_rms / sqrt(2)

so the power identity m * E_rms * I_rms == T * omega_m holds to machine
precision by construction. The SPEC's shear-stress form
T = (2 pi sigma_t / 3)(r_o^3 - r_i^3) is the equivalent integral formulation;
it differs from the flux-linkage form only by the geometry factor
(2/3)(r_o^3 - r_i^3) / (r_m (r_o^2 - r_i^2)) (~1.09 for the reference motor).
The result reports ``shear_stress_pa`` as the average shear implied by the
computed torque, T / ((2 pi / 3)(r_o^3 - r_i^3)).

Phase-1 simplifications (documented, lifted in later phases): slotless field
(Carter factor 1 unless supplied), single air gap, magnets evaluated at
ambient + 40 C (NOT coupled to the solved winding temperature — estimate the
magnet thermal path separately for temperature-sensitive designs), inductive
voltage drop neglected, zero mechanical loss by default.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from axfluxmdo.geometry.axial_flux import AxialFluxMotor
from axfluxmdo.limits import Limits
from axfluxmdo.materials.magnetic import airgap_flux_density
from axfluxmdo.models.constraints import ConstraintRecord, make_upper_bound
from axfluxmdo.models.losses import (
    copper_loss,
    core_flux_density_proxy,
    mass_rollup,
    mechanical_loss,
    phase_resistance,
    steinmetz_core_loss,
)
from axfluxmdo.models.thermal_rc import solve_winding_temperature
from axfluxmdo.operating_point import OperatingPoint

MAGNET_TEMP_RISE_C = 40.0  # Phase-1 assumption: magnets run this far above ambient
CORE_LOSS_TO_WINDING_FRACTION = 0.5  # fraction of core loss heating the winding node


def build_constraints(
    motor: AxialFluxMotor,
    op: OperatingPoint,
    limits: Limits,
    *,
    winding_temp_c: float,
    f_e_hz: float,
    current_density_a_mm2: float,
    back_emf_v_rms: float,
    phase_resistance_ohm: float,
    b_yoke_t: float,
    magnet_temp_c: float,
) -> list[ConstraintRecord]:
    """The Phase-1 constraint set, shared by all fidelity layers.

    ``None`` limits resolve from the motor/operating point (see ``Limits``).
    Voltage: V_required = sqrt(3)*(E + I*R) vs V_dc/sqrt(2) (SVPWM line-line
    fundamental). The INDUCTIVE drop I*X_L is neglected — fine at low
    electrical frequency, but materially optimistic when f_e is high AND the
    voltage margin is tight (e.g. ~100 uH at several kRPM adds tens of
    volts). Measure or estimate winding inductance before trusting a
    near-binding voltage constraint at high speed.
    """
    v_limit = (
        limits.max_line_voltage_v
        if limits.max_line_voltage_v is not None
        else op.dc_bus_voltage / math.sqrt(2.0)
    )
    v_required = (
        math.sqrt(3.0) * (back_emf_v_rms + op.current_rms * phase_resistance_ohm)
        if math.isfinite(phase_resistance_ohm)
        else math.inf
    )
    b_limit = (
        limits.max_core_flux_density_t
        if limits.max_core_flux_density_t is not None
        else motor.steel.b_sat_t
    )
    magnet_temp_limit = (
        limits.max_magnet_temp_c
        if limits.max_magnet_temp_c is not None
        else motor.magnet.max_operating_temp_c
    )
    return [
        make_upper_bound("winding_temp_c", winding_temp_c, limits.max_winding_temp_c),
        make_upper_bound("electrical_frequency_hz", f_e_hz, limits.max_electrical_freq_hz),
        make_upper_bound(
            "current_density_a_mm2", current_density_a_mm2, limits.max_current_density_a_mm2
        ),
        make_upper_bound("line_voltage_v", v_required, v_limit),
        make_upper_bound("core_flux_density_t", b_yoke_t, b_limit),
        make_upper_bound("magnet_temp_c", magnet_temp_c, magnet_temp_limit),
    ]


@dataclass(frozen=True)
class AnalyticalResult:
    """Evaluated performance of one motor at one operating point (SI + named units)."""

    torque_nm: float
    back_emf_v_rms: float  # per-phase line-neutral EMF at the operating speed
    electrical_frequency_hz: float
    airgap_flux_density_t: float
    shear_stress_pa: float
    phase_resistance_ohm: float  # at the solved winding temperature
    current_density_a_mm2: float
    copper_loss_w: float
    core_loss_w: float
    mechanical_loss_w: float
    output_power_w: float
    input_power_w: float
    efficiency: float
    winding_temp_c: float
    mass_kg: float
    mass_breakdown: dict[str, float] = field(repr=False)
    constraints: list[ConstraintRecord] = field(repr=False)

    @property
    def torque_density_nm_kg(self) -> float:
        return self.torque_nm / self.mass_kg

    @property
    def feasible(self) -> bool:
        return all(c.satisfied for c in self.constraints)

    def to_dict(self) -> dict[str, float]:
        """Flat scalar dict for sweeps/DataFrames; keys match constraint names."""
        return {
            "torque_nm": self.torque_nm,
            "back_emf_v_rms": self.back_emf_v_rms,
            "electrical_frequency_hz": self.electrical_frequency_hz,
            "airgap_flux_density_t": self.airgap_flux_density_t,
            "shear_stress_pa": self.shear_stress_pa,
            "phase_resistance_ohm": self.phase_resistance_ohm,
            "current_density_a_mm2": self.current_density_a_mm2,
            "copper_loss_w": self.copper_loss_w,
            "core_loss_w": self.core_loss_w,
            "mechanical_loss_w": self.mechanical_loss_w,
            "output_power_w": self.output_power_w,
            "input_power_w": self.input_power_w,
            "efficiency": self.efficiency,
            "winding_temp_c": self.winding_temp_c,
            "mass_kg": self.mass_kg,
            "torque_density_nm_kg": self.torque_density_nm_kg,
            "feasible": float(self.feasible),
        }

    def __str__(self) -> str:
        lines = [
            "AnalyticalResult",
            f"  torque:            {self.torque_nm:.3f} N·m",
            f"  torque density:    {self.torque_density_nm_kg:.3f} N·m/kg",
            f"  back-EMF (rms):    {self.back_emf_v_rms:.2f} V/phase",
            f"  elec frequency:    {self.electrical_frequency_hz:.1f} Hz",
            f"  air-gap B:         {self.airgap_flux_density_t:.3f} T",
            f"  shear stress:      {self.shear_stress_pa / 1e3:.2f} kPa",
            f"  current density:   {self.current_density_a_mm2:.2f} A/mm²",
            f"  copper loss:       {self.copper_loss_w:.1f} W",
            f"  core loss:         {self.core_loss_w:.2f} W",
            f"  output power:      {self.output_power_w:.1f} W",
            f"  efficiency:        {self.efficiency:.4f}",
            f"  winding temp:      {self.winding_temp_c:.1f} °C",
            f"  mass:              {self.mass_kg:.3f} kg",
            "  constraints:",
        ]
        lines += [f"    {c}" for c in self.constraints]
        return "\n".join(lines)


class AnalyticalModel:
    """Layer-1 analytical sizing model.

    carter_factor multiplies the magnetic air gap in the load line (default
    1.0 = slotless). The Phase-4 FEA validation found the uncorrected load
    line OVERESTIMATES the gap field (about -11% on the under-magnet mean and
    -7% on the fundamental for the reference motor, from inter-magnet leakage
    and fringing), and measured an effective k_C = 1.44 for the slotted
    variant — measure your own with
    :func:`axfluxmdo.validation.measured_carter_factor` and pass it here for
    corrected predictions.
    """

    def __init__(self, limits: Limits | None = None, carter_factor: float = 1.0):
        self.limits = limits or Limits()
        self.carter_factor = carter_factor

    def evaluate(self, motor: AxialFluxMotor, op: OperatingPoint) -> AnalyticalResult:
        # 1. Frequencies
        omega_m = op.speed_rad_s
        f_e = motor.pole_pairs * op.speed_rpm / 60.0
        magnet_temp_c = op.ambient_temp_c + MAGNET_TEMP_RISE_C

        # 2. Air-gap field and fundamental flux linkage
        b_gap = airgap_flux_density(
            motor.magnet,
            motor.magnet_thickness,
            motor.air_gap,
            magnet_temp_c,
            self.carter_factor,
        )
        b1 = (4.0 / math.pi) * b_gap * math.sin(motor.magnet_arc_ratio * math.pi / 2.0)
        flux_per_pole = b1 * motor.airgap_area / (math.pi * motor.pole_pairs)
        flux_linkage = motor.winding_factor * motor.turns_per_phase * flux_per_pole

        # 3. Torque and back-EMF from the same flux linkage (see module docstring)
        torque = motor.phases * motor.pole_pairs * flux_linkage * op.current_rms / math.sqrt(2.0)
        back_emf_rms = motor.pole_pairs * omega_m * flux_linkage / math.sqrt(2.0)
        shear_stress = torque / (
            (2.0 * math.pi / 3.0) * (motor.outer_radius**3 - motor.inner_radius**3)
        )

        # 4. Core loss (temperature-independent here; computed before the thermal solve)
        b_yoke = core_flux_density_proxy(motor, b_gap)
        p_core = steinmetz_core_loss(motor, f_e, b_yoke)

        # 5. Copper loss at reference temperature, then thermal steady state with R(T)
        ref_temp_c = 20.0
        r_ref = phase_resistance(motor, ref_temp_c)
        p_cu_ref = copper_loss(motor.phases, op.current_rms, r_ref)
        thermal = solve_winding_temperature(
            p_cu_ref_w=p_cu_ref,
            ref_temp_c=ref_temp_c,
            alpha_per_c=motor.conductor.temp_coeff_per_c,
            p_other_w=CORE_LOSS_TO_WINDING_FRACTION * p_core,
            r_theta_k_per_w=motor.thermal_resistance_k_per_w,
            ambient_c=op.ambient_temp_c,
        )
        if thermal.runaway:
            winding_temp = math.inf
            r_phase = math.inf
            p_cu = math.inf
        else:
            winding_temp = thermal.winding_temp_c
            r_phase = phase_resistance(motor, winding_temp)
            p_cu = copper_loss(motor.phases, op.current_rms, r_phase)

        # 6. Mechanical loss placeholder, powers, efficiency
        p_mech = mechanical_loss(omega_m)
        p_em = torque * omega_m
        p_out = p_em - p_mech
        p_in = p_em + p_cu + p_core
        efficiency = p_out / p_in if p_in > 0.0 and math.isfinite(p_in) else 0.0

        # 7. Mass rollup
        masses = mass_rollup(motor)

        # 8. Constraints
        current_density = op.current_rms / (motor.conductor_area * 1e6)  # A/mm^2
        constraints = build_constraints(
            motor,
            op,
            self.limits,
            winding_temp_c=winding_temp,
            f_e_hz=f_e,
            current_density_a_mm2=current_density,
            back_emf_v_rms=back_emf_rms,
            phase_resistance_ohm=r_phase,
            b_yoke_t=b_yoke,
            magnet_temp_c=magnet_temp_c,
        )

        return AnalyticalResult(
            torque_nm=torque,
            back_emf_v_rms=back_emf_rms,
            electrical_frequency_hz=f_e,
            airgap_flux_density_t=b_gap,
            shear_stress_pa=shear_stress,
            phase_resistance_ohm=r_phase,
            current_density_a_mm2=current_density,
            copper_loss_w=p_cu,
            core_loss_w=p_core,
            mechanical_loss_w=p_mech,
            output_power_w=p_out,
            input_power_w=p_in,
            efficiency=efficiency,
            winding_temp_c=winding_temp,
            mass_kg=masses["total"],
            mass_breakdown=masses,
            constraints=constraints,
        )
