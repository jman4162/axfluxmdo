import math

import pytest

from axfluxmdo.models.thermal_rc import (
    solve_winding_temperature,
    solve_winding_temperature_iterative,
)

ALPHA_CU = 0.00393


class TestClosedForm:
    def test_matches_fixed_point_iteration(self):
        sol = solve_winding_temperature(
            p_cu_ref_w=50.0,
            ref_temp_c=20.0,
            alpha_per_c=ALPHA_CU,
            p_other_w=10.0,
            r_theta_k_per_w=1.2,
            ambient_c=25.0,
        )
        t_iter = solve_winding_temperature_iterative(
            p_cu_ref_w=50.0,
            ref_temp_c=20.0,
            alpha_per_c=ALPHA_CU,
            p_other_w=10.0,
            r_theta_k_per_w=1.2,
            ambient_c=25.0,
        )
        assert sol.winding_temp_c == pytest.approx(t_iter, rel=1e-9)
        assert not sol.runaway

    def test_satisfies_fixed_point_equation(self):
        sol = solve_winding_temperature(30.0, 20.0, ALPHA_CU, 5.0, 0.8, 25.0)
        rhs = 25.0 + 0.8 * (sol.copper_loss_w + 5.0)
        assert sol.winding_temp_c == pytest.approx(rhs, rel=1e-12)

    def test_no_loss_means_ambient(self):
        sol = solve_winding_temperature(0.0, 20.0, ALPHA_CU, 0.0, 1.2, 25.0)
        assert sol.winding_temp_c == pytest.approx(25.0)
        assert sol.copper_loss_w == pytest.approx(0.0)

    def test_temperature_rises_with_thermal_resistance(self):
        lo = solve_winding_temperature(50.0, 20.0, ALPHA_CU, 0.0, 0.5, 25.0)
        hi = solve_winding_temperature(50.0, 20.0, ALPHA_CU, 0.0, 1.5, 25.0)
        assert hi.winding_temp_c > lo.winding_temp_c

    def test_copper_loss_grows_above_reference(self):
        sol = solve_winding_temperature(50.0, 20.0, ALPHA_CU, 0.0, 1.2, 25.0)
        assert sol.copper_loss_w > 50.0  # winding ends up hotter than the 20 C reference


class TestRunaway:
    def test_runaway_flag(self):
        # alpha * R_theta * P_cu_ref >= 1 -> divergence
        p_critical = 1.0 / (ALPHA_CU * 1.2)
        sol = solve_winding_temperature(p_critical * 1.01, 20.0, ALPHA_CU, 0.0, 1.2, 25.0)
        assert sol.runaway
        assert math.isinf(sol.winding_temp_c)

    def test_just_below_critical_is_finite(self):
        p_critical = 1.0 / (ALPHA_CU * 1.2)
        sol = solve_winding_temperature(p_critical * 0.95, 20.0, ALPHA_CU, 0.0, 1.2, 25.0)
        assert not sol.runaway
        assert math.isfinite(sol.winding_temp_c)
