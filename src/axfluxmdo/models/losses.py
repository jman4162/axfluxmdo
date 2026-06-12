"""Loss components and mass rollup — small pure functions reused by all models."""

from __future__ import annotations

from axfluxmdo.geometry.axial_flux import AxialFluxMotor
from axfluxmdo.materials.electrical import resistivity


def phase_resistance(motor: AxialFluxMotor, temp_c: float) -> float:
    """DC phase resistance at winding temperature T: rho(T) * N * L_turn / A_cond."""
    rho = resistivity(motor.conductor, temp_c)
    return rho * motor.turns_per_phase * motor.mean_turn_length / motor.conductor_area


def copper_loss(phases: int, current_rms: float, resistance_ohm: float) -> float:
    """Total winding copper loss P_cu = m * I_rms^2 * R_phase."""
    return phases * current_rms**2 * resistance_ohm


def steinmetz_core_loss(motor: AxialFluxMotor, f_hz: float, b_peak_t: float) -> float:
    """Stator core loss: specific Steinmetz loss times stator core mass."""
    return motor.steel.core_loss_w_per_kg(f_hz, b_peak_t) * stator_core_mass(motor)


def mechanical_loss(
    omega_m_rad_s: float, k_bearing: float = 0.0, k_windage: float = 0.0
) -> float:
    """Bearing + windage placeholder: k_b * omega + k_w * omega^3 (defaults zero).

    Fidelity item for Phase 2; the field exists so the efficiency rollup and
    result schema do not change later.
    """
    return k_bearing * omega_m_rad_s + k_windage * omega_m_rad_s**3


def stator_core_mass(motor: AxialFluxMotor) -> float:
    """Stator yoke lamination mass including stacking factor."""
    return motor.stator_core_volume * motor.steel.stacking_factor * motor.steel.density_kg_m3


def core_flux_density_proxy(motor: AxialFluxMotor, b_gap_t: float) -> float:
    """Peak stator-yoke flux density proxy for core loss and saturation checks.

    Half of one pole's air-gap flux returns through the yoke cross-section
    (active_length * stator_core_thickness * stacking), giving

        B_yoke = B_g * alpha_m * tau_p / (2 * t_core * stacking)
    """
    return (
        b_gap_t
        * motor.magnet_arc_ratio
        * motor.pole_pitch
        / (2.0 * motor.stator_core_thickness * motor.steel.stacking_factor)
    )


def mass_rollup(motor: AxialFluxMotor) -> dict[str, float]:
    """Component masses in kg; 'total' includes the structure factor."""
    magnets = motor.magnet_volume * motor.magnet.density_kg_m3
    back_iron = motor.back_iron_volume * motor.steel.density_kg_m3
    stator_core = stator_core_mass(motor)
    copper = motor.copper_volume * motor.conductor.density_kg_m3
    active = magnets + back_iron + stator_core + copper
    structure = motor.structure_mass_factor * active
    return {
        "magnets": magnets,
        "back_iron": back_iron,
        "stator_core": stator_core,
        "copper": copper,
        "structure": structure,
        "total": active + structure,
    }
