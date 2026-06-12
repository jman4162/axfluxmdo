"""Golden regression values for the SPEC reference motor at the reference operating point.

Captured from the first hand-verified run and locked to guard refactors.

Hand verification of the torque chain (matches to 3+ significant figures):
  Br(65 C)   = 1.30 * (1 - 0.0012 * 45)                = 1.2298 T
  B_g        = 1.2298 * 0.004 / (0.004 + 1.05*0.0008)  = 1.0163 T
  B_1        = (4/pi) * B_g * sin(0.85*pi/2)           = 1.2583 T
  Phi_p      = B_1 * pi*(0.08^2-0.025^2) / (pi*14)     = 1.4965e-3 Wb  (fundamental, per pole)
  lambda     = 0.933 * 24 * Phi_p                      = 3.3510e-2 Wb
  T          = 3 * 14 * lambda * 25 / sqrt(2)          = 8.63 N*m
  E_rms      = 14 * 52.36 * lambda / sqrt(2)           = 6.02 V  ->  3*E*I = T*omega exactly
Thermal: P_cu(20 C) = 17.94 W, closed form -> T_w = 49.6 C, P_cu(T_w) = 20.0 W.
"""

import pytest

from axfluxmdo.models import AnalyticalModel

GOLDEN = {
    "torque_nm": 8.62943,
    "back_emf_v_rms": 6.02448,
    "electrical_frequency_hz": 116.667,
    "airgap_flux_density_t": 1.01636,
    "shear_stress_pa": 8300.67,
    "phase_resistance_ohm": 0.0106771,
    "current_density_a_mm2": 4.04203,
    "copper_loss_w": 20.0195,
    "core_loss_w": 0.913686,
    "efficiency": 0.955722,
    "winding_temp_c": 49.5716,
    "mass_kg": 3.65092,
    "torque_density_nm_kg": 2.36363,
}


def test_reference_motor_golden_values(reference_motor, reference_op):
    result = AnalyticalModel().evaluate(reference_motor, reference_op)
    d = result.to_dict()
    for key, expected in GOLDEN.items():
        assert d[key] == pytest.approx(expected, rel=1e-5), key


def test_reference_motor_feasible(reference_motor, reference_op):
    assert AnalyticalModel().evaluate(reference_motor, reference_op).feasible


# --- Phase 2: annular model with every imperfection active -------------------
#
# Hand verification of slice 0 (innermost of 32, midpoint r = 25.859 mm):
#   gap(r)  = 0.0008 + 1e-4 + 2e-4*(r - 0.0525)/0.055      = 0.80313 mm
#   a       = t_m + mu_r*g = 0.004 + 1.05*0.00080313       = 4.84328e-3
#   d       = mu_r*runout  = 1.05*3e-4                     = 3.15e-4
#   Br(65C) = 1.30*(1 - 0.0012*45)                         = 1.2298 T
#   <B>     = Br*t_m/sqrt(a^2 - d^2)                       = 1.01783 T   (matches model)
# Axial pull ~ B^2/(2*mu0)*alpha*A_g ~ 0.4 MPa * 0.014 m^2 ~ 6 kN scale: OK.

ANNULAR_GOLDEN = {
    "torque_nm": 7.92548,
    "torque_ripple_proxy": 0.0635438,
    "axial_force_n": 5617.34,
    "core_loss_w": 0.808411,
    "efficiency": 0.952219,
    "winding_temp_c": 49.5026,
    "airgap_flux_density_t": 0.99325,
}


def test_annular_imperfect_golden_values(reference_motor, reference_op):
    import dataclasses

    from axfluxmdo import GapImperfections
    from axfluxmdo.models import AnnularModel

    motor = dataclasses.replace(
        reference_motor,
        tolerances=GapImperfections(gap_offset_m=1e-4, coning_m=2e-4, runout_m=3e-4),
        magnet_shape="rectangular",
    )
    d = AnnularModel(n_slices=32).evaluate(motor, reference_op).to_dict()
    for key, expected in ANNULAR_GOLDEN.items():
        assert d[key] == pytest.approx(expected, rel=1e-5), key
