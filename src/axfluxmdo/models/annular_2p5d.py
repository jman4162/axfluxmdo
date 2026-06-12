"""2.5D annular slice model (Phase 2, Layer 2).

The disk machine is split into ``n_slices`` radial annuli; the Phase-1
flux-linkage chain is evaluated per slice and summed:

    dA_k     = pi * (r_{k+1}^2 - r_k^2)                 (exact annulus areas)
    dlambda_k = k_w * N * B1(r_k) * dA_k / (pi * p)
    lambda    = fsum(dlambda_k)
    T = m*p*lambda*I/sqrt(2),   E_rms = p*omega_m*lambda/sqrt(2)

Because torque and EMF come from the same summed flux linkage, the power
identity m*E*I == T*omega and the energy balance hold at any slice count, and
with radius-uniform parameters the model reproduces ``AnalyticalModel`` to
machine precision (``n_slices=1`` matches on every ``to_dict()`` key — the
parity tests pin this).

Radius dependence enters through: the local axisymmetric gap (offset +
coning from ``motor.tolerances``), the local magnet arc (``magnet_shape ==
"rectangular"`` gives alpha(r) = min(1, alpha_m * r_m / r)), the local pole
pitch in the yoke flux proxy (low pole counts saturate first at the outer
radius), optional edge fringing, and the 1/r current loading. Runout enters
as analytic circumferential averages of the load line
(:mod:`axfluxmdo.materials.magnetic`); note the convexity consequence — mean
torque slightly *increases* with runout, the penalty being the 1/rev
modulation reported as ``torque_ripple_proxy`` and the axial pull
``axial_force_n``. Rotor tilting moment is not modeled in Phase 2.

Copper resistance and the thermal network stay lumped (single RC), per SPEC.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from axfluxmdo.geometry.axial_flux import AxialFluxMotor
from axfluxmdo.limits import Limits
from axfluxmdo.materials.magnetic import (
    airgap_b_squared_runout_mean,
    airgap_flux_density_runout_extremes,
    airgap_flux_density_runout_mean,
)
from axfluxmdo.models.analytical import (
    CORE_LOSS_TO_WINDING_FRACTION,
    MAGNET_TEMP_RISE_C,
    AnalyticalResult,
    build_constraints,
)
from axfluxmdo.models.losses import (
    copper_loss,
    mass_rollup,
    mechanical_loss,
    phase_resistance,
)
from axfluxmdo.models.thermal_rc import solve_winding_temperature
from axfluxmdo.operating_point import OperatingPoint

MU_0 = 4.0e-7 * math.pi


@dataclass(frozen=True)
class AnnularResult(AnalyticalResult):
    """AnalyticalResult plus imperfection metrics and per-slice field/torque profiles."""

    torque_ripple_proxy: float  # 1/rev flux-linkage modulation depth; 0 for a perfect gap
    axial_force_n: float  # mean magnetic axial pull on the rotor
    n_slices: int
    slice_radii_m: np.ndarray = field(repr=False, compare=False)
    slice_airgap_b_t: np.ndarray = field(repr=False, compare=False)
    slice_b1_t: np.ndarray = field(repr=False, compare=False)
    slice_torque_nm: np.ndarray = field(repr=False, compare=False)
    slice_shear_pa: np.ndarray = field(repr=False, compare=False)
    slice_yoke_b_t: np.ndarray = field(repr=False, compare=False)
    slice_current_loading_a_m: np.ndarray = field(repr=False, compare=False)

    def to_dict(self) -> dict[str, float]:
        """Phase-1 keys (stable interface) plus the Phase-2 scalars."""
        return super().to_dict() | {
            "torque_ripple_proxy": self.torque_ripple_proxy,
            "axial_force_n": self.axial_force_n,
        }

    def __str__(self) -> str:
        extra = [
            f"  torque ripple:     {self.torque_ripple_proxy:.4f} (1/rev proxy)",
            f"  axial force:       {self.axial_force_n:.1f} N "
            "(one-sided pull, single-gap topology — bearings must carry it)",
            f"  slices:            {self.n_slices}",
        ]
        return super().__str__() + "\n" + "\n".join(extra)


class AnnularModel:
    """Layer-2 annular slice model.

    Parameters
    ----------
    n_slices : torque/EMF are exact at any count (the flux-linkage sum is
        additive); only core loss and the saturation constraint are
        discretized, and both are smooth in radius — 32 slices is well within
        0.1 % of converged.
    edge_fringe_length_m : optional flux derating length at the annulus edges,
        B1 *= (1-exp(-(r-r_i)/L))*(1-exp(-(r_o-r)/L)). Zero (default) disables
        it and preserves exact Phase-1 parity.
    k_bearing, k_windage : mechanical loss coefficients (see
        :func:`~axfluxmdo.models.losses.mechanical_loss`), default zero.
    carter_factor : multiplies the magnetic gap in the load line (default 1.0
        = slotless). Phase-4 FEA measured k_C = 1.44 for the slotted reference
        motor; see :func:`axfluxmdo.validation.measured_carter_factor`.
    """

    def __init__(
        self,
        limits: Limits | None = None,
        n_slices: int = 32,
        edge_fringe_length_m: float = 0.0,
        k_bearing: float = 0.0,
        k_windage: float = 0.0,
        carter_factor: float = 1.0,
    ):
        if n_slices < 1:
            raise ValueError("n_slices must be at least 1")
        self.limits = limits or Limits()
        self.n_slices = n_slices
        self.edge_fringe_length_m = edge_fringe_length_m
        self.k_bearing = k_bearing
        self.k_windage = k_windage
        self.carter_factor = carter_factor

    def evaluate(self, motor: AxialFluxMotor, op: OperatingPoint) -> AnnularResult:
        m, p, n_turns = motor.phases, motor.pole_pairs, motor.turns_per_phase
        k_w = motor.winding_factor
        r_i, r_o = motor.inner_radius, motor.outer_radius
        tol = motor.tolerances

        omega_m = op.speed_rad_s
        f_e = p * op.speed_rpm / 60.0
        magnet_temp_c = op.ambient_temp_c + MAGNET_TEMP_RISE_C

        # 1. Radial slices: midpoints and exact annulus areas
        edges = np.linspace(r_i, r_o, self.n_slices + 1)
        radii = 0.5 * (edges[:-1] + edges[1:])
        areas = math.pi * (edges[1:] ** 2 - edges[:-1] ** 2)

        # 2. Per-slice field: local axisymmetric gap, runout-averaged load line,
        #    local magnet arc, optional edge fringing
        gaps = np.array([tol.axisymmetric_gap(motor.air_gap, r, r_i, r_o) for r in radii])
        b_gap = np.array(
            [
                airgap_flux_density_runout_mean(
                    motor.magnet,
                    motor.magnet_thickness,
                    g,
                    tol.runout_m,
                    magnet_temp_c,
                    self.carter_factor,
                )
                for g in gaps
            ]
        )
        alpha = self._magnet_arc(motor, radii)
        fringe = self._fringe_factor(radii, r_i, r_o)
        b1 = (4.0 / math.pi) * b_gap * np.sin(alpha * math.pi / 2.0) * fringe

        # 3. Flux linkage sum -> torque, EMF (energy-consistent by construction)
        dlambda = k_w * n_turns * b1 * areas / (math.pi * p)
        flux_linkage = math.fsum(dlambda)
        torque = m * p * flux_linkage * op.current_rms / math.sqrt(2.0)
        back_emf_rms = p * omega_m * flux_linkage / math.sqrt(2.0)
        slice_torque = m * p * dlambda * op.current_rms / math.sqrt(2.0)
        shear_stress = torque / ((2.0 * math.pi / 3.0) * (r_o**3 - r_i**3))
        slice_shear = slice_torque / (radii * areas)

        # 4. Per-slice yoke flux and core loss (local pole pitch tau_p(r) = pi*r/p)
        t_core = motor.stator_core_thickness
        stacking = motor.steel.stacking_factor
        b_yoke = b_gap * alpha * math.pi * radii / (2.0 * p * t_core * stacking)
        core_mass = areas * t_core * stacking * motor.steel.density_kg_m3
        p_core = math.fsum(
            motor.steel.core_loss_w_per_kg(f_e, b) * mass_
            for b, mass_ in zip(b_yoke, core_mass, strict=True)
        )

        # 5. Lumped copper + thermal (identical to Phase 1)
        ref_temp_c = 20.0
        r_ref = phase_resistance(motor, ref_temp_c)
        p_cu_ref = copper_loss(m, op.current_rms, r_ref)
        thermal = solve_winding_temperature(
            p_cu_ref_w=p_cu_ref,
            ref_temp_c=ref_temp_c,
            alpha_per_c=motor.conductor.temp_coeff_per_c,
            p_other_w=CORE_LOSS_TO_WINDING_FRACTION * p_core,
            r_theta_k_per_w=motor.thermal_resistance_k_per_w,
            ambient_c=op.ambient_temp_c,
        )
        if thermal.runaway:
            winding_temp, r_phase, p_cu = math.inf, math.inf, math.inf
        else:
            winding_temp = thermal.winding_temp_c
            r_phase = phase_resistance(motor, winding_temp)
            p_cu = copper_loss(m, op.current_rms, r_phase)

        # 6. Mechanical loss, powers, efficiency
        p_mech = mechanical_loss(omega_m, self.k_bearing, self.k_windage)
        p_em = torque * omega_m
        p_out = p_em - p_mech
        p_in = p_em + p_cu + p_core
        efficiency = p_out / p_in if p_in > 0.0 and math.isfinite(p_in) else 0.0

        # 7. Runout modulation proxy: flux linkage at the gap extremes
        ripple = self._ripple_proxy(motor, gaps, alpha, fringe, areas, magnet_temp_c)

        # 8. Mean axial magnetic pull under the magnet coverage
        b_sq = np.array(
            [
                airgap_b_squared_runout_mean(
                    motor.magnet,
                    motor.magnet_thickness,
                    g,
                    tol.runout_m,
                    magnet_temp_c,
                    self.carter_factor,
                )
                for g in gaps
            ]
        )
        axial_force = float(np.sum(b_sq / (2.0 * MU_0) * alpha * areas))

        # 9. Mass, current loading, constraints
        masses = mass_rollup(motor)
        current_loading = m * n_turns * op.current_rms / (math.pi * radii)
        current_density = op.current_rms / (motor.conductor_area * 1e6)
        constraints = build_constraints(
            motor,
            op,
            self.limits,
            winding_temp_c=winding_temp,
            f_e_hz=f_e,
            current_density_a_mm2=current_density,
            back_emf_v_rms=back_emf_rms,
            phase_resistance_ohm=r_phase,
            b_yoke_t=float(np.max(b_yoke)),
            magnet_temp_c=magnet_temp_c,
        )

        return AnnularResult(
            torque_nm=torque,
            back_emf_v_rms=back_emf_rms,
            electrical_frequency_hz=f_e,
            airgap_flux_density_t=float(np.dot(b_gap, areas) / np.sum(areas)),
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
            torque_ripple_proxy=ripple,
            axial_force_n=axial_force,
            n_slices=self.n_slices,
            slice_radii_m=radii,
            slice_airgap_b_t=b_gap,
            slice_b1_t=b1,
            slice_torque_nm=slice_torque,
            slice_shear_pa=slice_shear,
            slice_yoke_b_t=b_yoke,
            slice_current_loading_a_m=current_loading,
        )

    @staticmethod
    def _magnet_arc(motor: AxialFluxMotor, radii: np.ndarray) -> np.ndarray:
        if motor.magnet_shape == "rectangular":
            return np.minimum(1.0, motor.magnet_arc_ratio * motor.mean_radius / radii)
        return np.full_like(radii, motor.magnet_arc_ratio)

    def _fringe_factor(self, radii: np.ndarray, r_i: float, r_o: float) -> np.ndarray:
        if self.edge_fringe_length_m <= 0.0:
            return np.ones_like(radii)
        length = self.edge_fringe_length_m
        return (1.0 - np.exp(-(radii - r_i) / length)) * (1.0 - np.exp(-(r_o - radii) / length))

    def _ripple_proxy(
        self,
        motor: AxialFluxMotor,
        gaps: np.ndarray,
        alpha: np.ndarray,
        fringe: np.ndarray,
        areas: np.ndarray,
        magnet_temp_c: float,
    ) -> float:
        """1/rev flux-linkage modulation depth (lam_max - lam_min)/(lam_max + lam_min)."""
        runout = motor.tolerances.runout_m
        if runout == 0.0:
            return 0.0
        lam = {"tight": 0.0, "wide": 0.0}
        for g, a, fr, area in zip(gaps, alpha, fringe, areas, strict=True):
            b_tight, b_wide = airgap_flux_density_runout_extremes(
                motor.magnet,
                motor.magnet_thickness,
                g,
                runout,
                magnet_temp_c,
                self.carter_factor,
            )
            weight = (4.0 / math.pi) * math.sin(a * math.pi / 2.0) * fr * area
            lam["tight"] += b_tight * weight
            lam["wide"] += b_wide * weight
        return (lam["tight"] - lam["wide"]) / (lam["tight"] + lam["wide"])
