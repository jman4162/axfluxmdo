import dataclasses
import math

import pytest

from axfluxmdo import AxialFluxMotor, OperatingPoint


class TestDerivedGeometry:
    def test_mean_radius(self, reference_motor):
        assert reference_motor.mean_radius == pytest.approx(0.5 * (0.08 + 0.025))

    def test_active_length(self, reference_motor):
        assert reference_motor.active_length == pytest.approx(0.08 - 0.025)

    def test_airgap_area(self, reference_motor):
        assert reference_motor.airgap_area == pytest.approx(math.pi * (0.08**2 - 0.025**2))

    def test_pole_pitch(self, reference_motor):
        assert reference_motor.pole_pitch == pytest.approx(
            math.pi * reference_motor.mean_radius / 14
        )

    def test_magnet_volume(self, reference_motor):
        expected = reference_motor.airgap_area * 0.85 * 0.004
        assert reference_motor.magnet_volume == pytest.approx(expected)

    def test_conductor_area_positive_and_sensible(self, reference_motor):
        # Sub-cm^2 conductor for a 16 cm motor
        assert 0.0 < reference_motor.conductor_area < 1e-4

    def test_mean_turn_length(self, reference_motor):
        expected = 2 * reference_motor.active_length + 2 * 1.4 * reference_motor.pole_pitch
        assert reference_motor.mean_turn_length == pytest.approx(expected)


class TestValidation:
    def test_inner_radius_must_be_less_than_outer(self):
        with pytest.raises(ValueError, match="outer_radius"):
            AxialFluxMotor(outer_radius=0.02, inner_radius=0.025, air_gap=0.001, pole_pairs=10)

    def test_zero_air_gap_rejected(self):
        with pytest.raises(ValueError, match="air_gap"):
            AxialFluxMotor(outer_radius=0.08, inner_radius=0.025, air_gap=0.0, pole_pairs=10)

    def test_fill_factor_bounds(self):
        with pytest.raises(ValueError, match="fill_factor"):
            AxialFluxMotor(
                outer_radius=0.08,
                inner_radius=0.025,
                air_gap=0.001,
                pole_pairs=10,
                fill_factor=0.9,
            )

    def test_pole_pairs_minimum(self):
        with pytest.raises(ValueError, match="pole_pairs"):
            AxialFluxMotor(outer_radius=0.08, inner_radius=0.025, air_gap=0.001, pole_pairs=0)


class TestImmutability:
    def test_frozen(self, reference_motor):
        with pytest.raises(dataclasses.FrozenInstanceError):
            reference_motor.outer_radius = 0.1

    def test_replace_round_trip(self, reference_motor):
        variant = dataclasses.replace(reference_motor, pole_pairs=10)
        assert variant.pole_pairs == 10
        assert reference_motor.pole_pairs == 14
        back = dataclasses.replace(variant, pole_pairs=14)
        assert back == reference_motor


class TestOperatingPoint:
    def test_speed_rad_s(self, reference_op):
        assert reference_op.speed_rad_s == pytest.approx(500 * 2 * math.pi / 60)

    def test_negative_speed_rejected(self):
        with pytest.raises(ValueError):
            OperatingPoint(speed_rpm=-100, current_rms=10)

    def test_frozen(self, reference_op):
        with pytest.raises(dataclasses.FrozenInstanceError):
            reference_op.speed_rpm = 1000
