import pytest

from axfluxmdo.optimize import compute_sensitivities


class TestComputeSensitivities:
    def test_physics_signs(self, reference_motor, reference_op):
        sens = compute_sensitivities(
            reference_motor,
            reference_op,
            ["air_gap", "outer_radius", "magnet_thickness"],
            output="torque_nm",
        )
        by_var = {e.variable: e for e in sens.entries}
        assert by_var["air_gap"].swing < 0  # opening the gap loses torque
        assert by_var["outer_radius"].swing > 0
        assert by_var["magnet_thickness"].swing > 0

    def test_sorted_by_swing_magnitude(self, reference_motor, reference_op):
        sens = compute_sensitivities(
            reference_motor,
            reference_op,
            ["air_gap", "outer_radius", "fill_factor", "magnet_thickness"],
            output="torque_density",  # alias accepted
        )
        magnitudes = [abs(e.swing) for e in sens.entries]
        assert magnitudes == sorted(magnitudes, reverse=True)
        assert sens.output == "torque_density_nm_kg"

    def test_dict_spec_clamps_to_bounds(self, reference_motor, reference_op):
        sens = compute_sensitivities(
            reference_motor,
            reference_op,
            {"outer_radius": (0.079, 0.081)},  # tighter than +/-5%
            output="torque_nm",
        )
        entry = sens.entries[0]
        assert entry.low_input == pytest.approx(0.079)
        assert entry.high_input == pytest.approx(0.081)

    def test_choice_spec_steps_one_option(self, reference_motor, reference_op):
        sens = compute_sensitivities(
            reference_motor,
            reference_op,
            {"pole_pairs": [10, 12, 14, 16, 18]},
            output="electrical_frequency_hz",
        )
        entry = sens.entries[0]
        assert entry.low_input == 12  # one below baseline 14
        assert entry.high_input == 16  # one above
        assert entry.swing > 0  # f_e grows with p

    def test_integer_field_perturbation(self, reference_motor, reference_op):
        sens = compute_sensitivities(
            reference_motor, reference_op, ["turns_per_phase"], output="torque_nm"
        )
        entry = sens.entries[0]
        assert entry.low_input == 23 or entry.low_input == pytest.approx(22.8, abs=1)
        assert entry.high_input > entry.low_input
        assert entry.swing > 0  # more turns, more torque
