"""Synthetic-data tests for the sim-to-analytical comparison layer (no binaries)."""

import math

import numpy as np
import pytest

from axfluxmdo.materials import N42, airgap_flux_density
from axfluxmdo.solvers import GapFieldSolution
from axfluxmdo.validation import compare_open_circuit, measured_carter_factor

TAU = 0.011781  # reference motor pole pitch
ALPHA = 0.85


def make_solution(by_func, n=721, slotted=False, include_duplicate_endpoint=True):
    span = 2 * TAU
    x = np.linspace(0.0, span, n if include_duplicate_endpoint else n - 1, endpoint=include_duplicate_endpoint)
    return GapFieldSolution(
        x_m=x,
        by_t=by_func(x),
        pole_pitch_m=TAU,
        magnet_arc_ratio=ALPHA,
        magnet_temp_c=65.0,
        slotted=slotted,
    )


class TestFundamental:
    @pytest.mark.parametrize("dup", [True, False])
    def test_pure_sinusoid_recovers_amplitude(self, dup):
        b1 = 1.234
        sol = make_solution(
            lambda x: b1 * np.sin(math.pi * x / TAU), include_duplicate_endpoint=dup
        )
        assert sol.fundamental_b1_t == pytest.approx(b1, rel=1e-6)

    def test_square_wave_fundamental(self):
        """Square wave of amplitude B_g under the magnet arcs -> (4/pi)*B_g*sin(alpha*pi/2)."""
        b_g = 1.0

        def square(x):
            xm = np.mod(x, TAU)
            under = np.abs(xm - TAU / 2) <= ALPHA * TAU / 2
            sign = np.where(np.mod(x, 2 * TAU) < TAU, 1.0, -1.0)
            return np.where(under, sign * b_g, 0.0)

        sol = make_solution(square, n=20001)
        expected = (4 / math.pi) * b_g * math.sin(ALPHA * math.pi / 2)
        assert sol.fundamental_b1_t == pytest.approx(expected, rel=1e-3)


class TestMeans:
    def test_square_wave_under_magnet_mean_is_exact(self):
        b_g = 0.987

        def square(x):
            xm = np.mod(x, TAU)
            under = np.abs(xm - TAU / 2) <= ALPHA * TAU / 2
            sign = np.where(np.mod(x, 2 * TAU) < TAU, 1.0, -1.0)
            return np.where(under, sign * b_g, 0.0)

        sol = make_solution(square, n=20001)
        assert sol.mean_b_t == pytest.approx(b_g, rel=1e-6)
        # full-pitch mean is diluted by the uncovered span
        assert sol.mean_b_full_pitch_t == pytest.approx(b_g * ALPHA, rel=1e-2)


class TestComparison:
    @pytest.fixture
    def motor(self, reference_motor):
        return reference_motor

    def test_zero_residual_for_loadline_square_wave(self, motor):
        b_g = airgap_flux_density(N42, motor.magnet_thickness, motor.air_gap, 65.0)

        def square(x):
            xm = np.mod(x, motor.pole_pitch)
            under = np.abs(xm - motor.pole_pitch / 2) <= ALPHA * motor.pole_pitch / 2
            sign = np.where(np.mod(x, 2 * motor.pole_pitch) < motor.pole_pitch, 1.0, -1.0)
            return np.where(under, sign * b_g, 0.0)

        span = 2 * motor.pole_pitch
        x = np.linspace(0, span, 20001)
        sol = GapFieldSolution(
            x_m=x,
            by_t=square(x),
            pole_pitch_m=motor.pole_pitch,
            magnet_arc_ratio=motor.magnet_arc_ratio,
            magnet_temp_c=65.0,
            slotted=False,
        )
        cmp_ = compare_open_circuit(motor, sol, magnet_temp_c=65.0)
        assert cmp_.residual_b_g_rel == pytest.approx(0.0, abs=1e-5)
        assert cmp_.residual_b1_rel == pytest.approx(0.0, abs=1e-3)
        assert "B_g" in str(cmp_)

    def test_temperature_mismatch_warns(self, motor):
        sol = make_solution(lambda x: np.sin(math.pi * x / TAU))
        with pytest.warns(UserWarning, match="temperature"):
            compare_open_circuit(motor, sol, magnet_temp_c=20.0)


class TestCarterFactor:
    def test_round_trip(self, reference_motor):
        """Synthesize slotless/slotted means from the load line with known k_C."""
        motor = reference_motor
        k_c_true = 1.15
        b_slotless = airgap_flux_density(
            motor.magnet, motor.magnet_thickness, motor.air_gap, 65.0, carter_factor=1.0
        )
        b_slotted = airgap_flux_density(
            motor.magnet, motor.magnet_thickness, motor.air_gap, 65.0, carter_factor=k_c_true
        )
        flat = lambda b: (lambda x: np.full_like(x, b))  # noqa: E731
        # mean under magnet of a constant field is the constant itself
        sl = make_solution(flat(b_slotless), slotted=False)
        st = make_solution(flat(b_slotted), slotted=True)
        k_c = measured_carter_factor(sl, st, motor)
        assert k_c == pytest.approx(k_c_true, rel=1e-12)

    def test_argument_order_enforced(self, reference_motor):
        sl = make_solution(lambda x: np.ones_like(x), slotted=False)
        st = make_solution(lambda x: np.ones_like(x), slotted=True)
        with pytest.raises(ValueError, match="order"):
            measured_carter_factor(st, sl, reference_motor)
