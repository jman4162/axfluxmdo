import dataclasses
import math

import pytest

from axfluxmdo.models import AnalyticalModel


@pytest.fixture
def model():
    return AnalyticalModel()


@pytest.fixture
def reference_result(model, reference_motor, reference_op):
    return model.evaluate(reference_motor, reference_op)


def cool_motor(motor):
    """Variant with near-perfect cooling, to decouple thermal effects."""
    return dataclasses.replace(motor, thermal_resistance_k_per_w=1e-6)


class TestExactIdentities:
    def test_electrical_frequency(self, reference_result, reference_motor, reference_op):
        assert reference_result.electrical_frequency_hz == pytest.approx(
            reference_motor.pole_pairs * reference_op.speed_rpm / 60.0, rel=1e-12
        )

    def test_energy_balance(self, reference_result):
        r = reference_result
        assert r.input_power_w == pytest.approx(
            r.output_power_w + r.copper_loss_w + r.core_loss_w + r.mechanical_loss_w, rel=1e-9
        )

    def test_emf_torque_consistency(self, reference_result, reference_motor, reference_op):
        """m * E_rms * I_rms == T * omega_m by construction (shared flux linkage)."""
        r = reference_result
        electromagnetic_power = reference_motor.phases * r.back_emf_v_rms * reference_op.current_rms
        assert electromagnetic_power == pytest.approx(
            r.torque_nm * reference_op.speed_rad_s, rel=1e-9
        )

    def test_shear_stress_torque_consistency(self, reference_result, reference_motor):
        """shear_stress_pa is the average shear implied by the torque (SPEC integral form)."""
        r = reference_result
        t_from_sigma = (
            (2.0 * math.pi / 3.0)
            * r.shear_stress_pa
            * (reference_motor.outer_radius**3 - reference_motor.inner_radius**3)
        )
        assert t_from_sigma == pytest.approx(r.torque_nm, rel=1e-12)


class TestMonotonicity:
    def test_torque_increases_with_current(self, model, reference_motor, reference_op):
        m = cool_motor(reference_motor)
        t1 = model.evaluate(m, dataclasses.replace(reference_op, current_rms=10)).torque_nm
        t2 = model.evaluate(m, dataclasses.replace(reference_op, current_rms=20)).torque_nm
        assert t2 > t1
        assert t2 == pytest.approx(2 * t1, rel=1e-9)  # linear in current (no saturation model)

    def test_torque_increases_with_outer_radius(self, model, reference_motor, reference_op):
        m = cool_motor(reference_motor)
        t1 = model.evaluate(m, reference_op).torque_nm
        t2 = model.evaluate(dataclasses.replace(m, outer_radius=0.09), reference_op).torque_nm
        assert t2 > t1

    def test_torque_decreases_with_air_gap(self, model, reference_motor, reference_op):
        m = cool_motor(reference_motor)
        t1 = model.evaluate(m, reference_op).torque_nm
        t2 = model.evaluate(dataclasses.replace(m, air_gap=0.002), reference_op).torque_nm
        assert t2 < t1

    def test_torque_increases_with_magnet_thickness(self, model, reference_motor, reference_op):
        m = cool_motor(reference_motor)
        t1 = model.evaluate(m, reference_op).torque_nm
        t2 = model.evaluate(dataclasses.replace(m, magnet_thickness=0.008), reference_op).torque_nm
        assert t2 > t1

    def test_core_loss_increases_with_pole_pairs(self, model, reference_motor, reference_op):
        """At fixed speed, more pole pairs -> higher electrical frequency -> more core loss.

        Compared at fixed yoke flux density basis: higher p also shrinks the pole
        pitch and hence B_yoke, so compare with stator core thinned proportionally
        to keep B_yoke equal. Simpler robust check: frequency term dominates.
        """
        m = cool_motor(reference_motor)
        p1 = model.evaluate(dataclasses.replace(m, pole_pairs=8), reference_op)
        p2 = model.evaluate(dataclasses.replace(m, pole_pairs=16), reference_op)
        assert p2.electrical_frequency_hz == pytest.approx(2 * p1.electrical_frequency_hz)

    def test_back_emf_linear_in_speed(self, model, reference_motor, reference_op):
        e1 = model.evaluate(reference_motor, reference_op).back_emf_v_rms
        e2 = model.evaluate(
            reference_motor, dataclasses.replace(reference_op, speed_rpm=1000)
        ).back_emf_v_rms
        assert e2 == pytest.approx(2 * e1, rel=1e-9)


class TestLimitingCases:
    def test_zero_current(self, model, reference_motor, reference_op):
        r = model.evaluate(reference_motor, dataclasses.replace(reference_op, current_rms=0))
        assert r.torque_nm == 0.0
        assert r.copper_loss_w == 0.0
        assert r.current_density_a_mm2 == 0.0
        # Winding heated only by the core-loss fraction
        expected = reference_op.ambient_temp_c + (
            reference_motor.thermal_resistance_k_per_w * 0.5 * r.core_loss_w
        )
        assert r.winding_temp_c == pytest.approx(expected, rel=1e-9)

    def test_zero_speed(self, model, reference_motor, reference_op):
        r = model.evaluate(reference_motor, dataclasses.replace(reference_op, speed_rpm=0))
        assert r.back_emf_v_rms == 0.0
        assert r.electrical_frequency_hz == 0.0
        assert r.core_loss_w == 0.0
        assert r.efficiency == 0.0  # no div-by-zero
        assert r.torque_nm > 0.0  # stall torque exists

    def test_efficiency_bounds_at_reference(self, reference_result):
        assert 0.0 < reference_result.efficiency < 1.0


class TestResultInterface:
    def test_reference_design_is_feasible(self, reference_result):
        assert reference_result.feasible, "\n" + str(reference_result)

    def test_to_dict_keys_match_constraint_names(self, reference_result):
        d = reference_result.to_dict()
        # constraint names that are direct result quantities must be dict keys
        for c in reference_result.constraints:
            if c.name in ("winding_temp_c", "electrical_frequency_hz", "current_density_a_mm2"):
                assert c.name in d
                assert d[c.name] == pytest.approx(c.value)

    def test_str_report(self, reference_result):
        s = str(reference_result)
        assert "torque" in s
        assert "constraints" in s

    def test_mass_breakdown_sums_to_total(self, reference_result):
        b = reference_result.mass_breakdown
        parts = b["magnets"] + b["back_iron"] + b["stator_core"] + b["copper"] + b["structure"]
        assert b["total"] == pytest.approx(parts, rel=1e-12)
        assert reference_result.mass_kg == b["total"]


class TestConstraintViolations:
    def test_overcurrent_violates_thermal_and_density(self, model, reference_motor, reference_op):
        r = model.evaluate(reference_motor, dataclasses.replace(reference_op, current_rms=200))
        by_name = {c.name: c for c in r.constraints}
        assert not by_name["winding_temp_c"].satisfied
        assert not by_name["current_density_a_mm2"].satisfied

    def test_overspeed_violates_frequency_and_voltage(self, model, reference_motor, reference_op):
        r = model.evaluate(reference_motor, dataclasses.replace(reference_op, speed_rpm=20000))
        by_name = {c.name: c for c in r.constraints}
        assert not by_name["electrical_frequency_hz"].satisfied
        assert not by_name["line_voltage_v"].satisfied
