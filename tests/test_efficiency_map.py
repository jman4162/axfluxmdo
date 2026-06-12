import dataclasses

import numpy as np
import pytest

from axfluxmdo.models import AnnularModel, compute_efficiency_map


@pytest.fixture
def small_map(reference_motor, reference_op):
    return compute_efficiency_map(
        reference_motor,
        reference_op,
        max_speed_rpm=3000,
        max_torque_nm=12,
        n_speed=12,
        n_torque=10,
    )


class TestEfficiencyMap:
    def test_shapes_and_grids(self, small_map):
        assert small_map.speeds_rpm.shape == (12,)
        assert small_map.torques_nm.shape == (10,)
        for arr in (
            small_map.efficiency,
            small_map.current_rms_a,
            small_map.copper_loss_w,
            small_map.core_loss_w,
            small_map.winding_temp_c,
            small_map.feasible,
            small_map.binding_constraint,
        ):
            assert arr.shape == (10, 12)
        assert small_map.speeds_rpm[-1] == 3000
        assert small_map.torques_nm[-1] == 12

    def test_torque_round_trip(self, small_map, reference_motor, reference_op):
        """Linear inversion: a feasible cell's evaluated torque equals the grid torque."""
        model = AnnularModel()
        i, j = 4, 6
        r = model.evaluate(
            reference_motor,
            dataclasses.replace(
                reference_op,
                speed_rpm=small_map.speeds_rpm[j],
                current_rms=small_map.current_rms_a[i, j],
            ),
        )
        assert r.torque_nm == pytest.approx(small_map.torques_nm[i], rel=1e-9)

    def test_efficiency_bounds_and_masking(self, small_map):
        feasible_eff = small_map.efficiency[small_map.feasible]
        assert feasible_eff.size > 0
        assert np.all((feasible_eff > 0) & (feasible_eff < 1))
        assert np.all(np.isnan(small_map.efficiency[~small_map.feasible]))

    def test_copper_loss_monotone_in_torque(self, small_map):
        j = 5  # fixed speed column
        col = small_map.copper_loss_w[:, j]
        valid = ~np.isnan(col)
        assert np.all(np.diff(col[valid]) > 0)

    def test_binding_constraints_recorded(self, reference_motor, reference_op):
        emap = compute_efficiency_map(
            reference_motor,
            reference_op,
            max_speed_rpm=12000,  # drives f_e and voltage violations at high speed
            max_torque_nm=60,  # drives thermal/current violations at high torque
            n_speed=8,
            n_torque=8,
        )
        assert (~emap.feasible).any()
        names = set(emap.binding_constraint[~emap.feasible])
        assert names <= {
            "winding_temp_c",
            "electrical_frequency_hz",
            "current_density_a_mm2",
            "line_voltage_v",
            "core_flux_density_t",
            "magnet_temp_c",
        }
        assert len(names) >= 2

    def test_spot_check_against_direct_evaluate(self, small_map, reference_motor, reference_op):
        model = AnnularModel()
        i, j = 2, 3
        if small_map.feasible[i, j]:
            r = model.evaluate(
                reference_motor,
                dataclasses.replace(
                    reference_op,
                    speed_rpm=small_map.speeds_rpm[j],
                    current_rms=small_map.current_rms_a[i, j],
                ),
            )
            assert small_map.efficiency[i, j] == pytest.approx(r.efficiency, rel=1e-12)
