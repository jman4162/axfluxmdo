import numpy as np
import pytest

from axfluxmdo.sweeps import sweep_parameter, sweep_pole_pairs


class TestSweepParameter:
    def test_length_and_order(self, reference_motor, reference_op):
        gaps = [0.0005, 0.001, 0.0015]
        sweep = sweep_parameter(reference_motor, reference_op, "air_gap", gaps)
        assert sweep.values == gaps
        assert len(sweep.results) == 3
        torques = [r.torque_nm for r in sweep.results]
        assert torques == sorted(torques, reverse=True)  # torque falls as gap grows

    def test_original_motor_unmodified(self, reference_motor, reference_op):
        sweep_parameter(reference_motor, reference_op, "air_gap", [0.002])
        assert reference_motor.air_gap == 0.0008

    def test_to_arrays(self, reference_motor, reference_op):
        sweep = sweep_parameter(reference_motor, reference_op, "air_gap", [0.0005, 0.001])
        data = sweep.to_arrays("torque_nm", "efficiency")
        assert set(data) == {"air_gap", "torque_nm", "efficiency"}
        assert all(isinstance(v, np.ndarray) and v.shape == (2,) for v in data.values())


class TestSweepPolePairs:
    def test_frequency_exactly_linear_in_p(self, reference_motor, reference_op):
        sweep = sweep_pole_pairs(reference_motor, reference_op, range(4, 21, 4))
        data = sweep.to_arrays("electrical_frequency_hz")
        expected = np.array(sweep.values) * reference_op.speed_rpm / 60.0
        np.testing.assert_allclose(data["electrical_frequency_hz"], expected, rtol=1e-12)

    def test_plot_returns_figure(self, reference_motor, reference_op):
        import matplotlib.pyplot as plt

        sweep = sweep_pole_pairs(reference_motor, reference_op, [8, 12, 16])
        fig = sweep.plot()
        assert fig is not None
        plt.close(fig)

    def test_invalid_field_raises(self, reference_motor, reference_op):
        sweep = sweep_pole_pairs(reference_motor, reference_op, [8, 12])
        with pytest.raises(KeyError):
            sweep.to_arrays("not_a_field")
