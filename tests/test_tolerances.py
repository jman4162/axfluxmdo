import math

import pytest

from axfluxmdo import AxialFluxMotor
from axfluxmdo.geometry.tolerances import PERFECT_GAP, GapImperfections

R_I, R_O = 0.025, 0.08
R_M = 0.5 * (R_I + R_O)
G0 = 0.001


class TestLocalGapLaw:
    def test_perfect_gap_is_nominal_everywhere(self):
        for r in (R_I, R_M, R_O):
            for theta in (0.0, math.pi / 2, math.pi):
                assert PERFECT_GAP.local_gap(G0, r, R_I, R_O, theta) == pytest.approx(G0)

    def test_offset_shifts_uniformly(self):
        tol = GapImperfections(gap_offset_m=2e-4)
        for r in (R_I, R_M, R_O):
            assert tol.local_gap(G0, r, R_I, R_O) == pytest.approx(G0 + 2e-4)

    def test_coning_zero_mean_at_mean_radius(self):
        tol = GapImperfections(coning_m=4e-4)
        assert tol.axisymmetric_gap(G0, R_M, R_I, R_O) == pytest.approx(G0)
        assert tol.axisymmetric_gap(G0, R_O, R_I, R_O) == pytest.approx(G0 + 2e-4)
        assert tol.axisymmetric_gap(G0, R_I, R_I, R_O) == pytest.approx(G0 - 2e-4)

    def test_runout_cosine_in_theta(self):
        tol = GapImperfections(runout_m=3e-4)
        assert tol.local_gap(G0, R_M, R_I, R_O, theta=0.0) == pytest.approx(G0 + 3e-4)
        assert tol.local_gap(G0, R_M, R_I, R_O, theta=math.pi) == pytest.approx(G0 - 3e-4)
        assert tol.local_gap(G0, R_M, R_I, R_O, theta=math.pi / 2) == pytest.approx(G0)

    def test_is_perfect(self):
        assert PERFECT_GAP.is_perfect
        assert not GapImperfections(runout_m=1e-5).is_perfect


class TestValidation:
    def test_negative_runout_rejected(self):
        with pytest.raises(ValueError, match="runout"):
            GapImperfections(runout_m=-1e-4)

    def test_motor_rejects_gap_closing_tolerances(self):
        with pytest.raises(ValueError, match="close the air gap"):
            AxialFluxMotor(
                outer_radius=R_O,
                inner_radius=R_I,
                air_gap=0.0008,
                pole_pairs=14,
                tolerances=GapImperfections(runout_m=0.0008),
            )

    def test_motor_accepts_safe_tolerances(self):
        motor = AxialFluxMotor(
            outer_radius=R_O,
            inner_radius=R_I,
            air_gap=0.0008,
            pole_pairs=14,
            tolerances=GapImperfections(gap_offset_m=1e-4, coning_m=2e-4, runout_m=3e-4),
        )
        assert motor.tolerances.runout_m == 3e-4

    def test_magnet_shape_validated(self):
        with pytest.raises(ValueError, match="magnet_shape"):
            AxialFluxMotor(
                outer_radius=R_O,
                inner_radius=R_I,
                air_gap=0.0008,
                pole_pairs=14,
                magnet_shape="trapezoidal",
            )

    def test_default_motor_is_perfect(self, reference_motor):
        assert reference_motor.tolerances.is_perfect
        assert reference_motor.magnet_shape == "wedge"
