import pytest

from axfluxmdo.materials import (
    COPPER,
    M19_29GA,
    N42,
    airgap_flux_density,
    resistivity,
)


class TestConductor:
    def test_copper_resistivity_at_20c(self):
        assert resistivity(COPPER, 20.0) == pytest.approx(1.724e-8, rel=1e-12)

    def test_copper_resistivity_ratio_100c(self):
        ratio = resistivity(COPPER, 100.0) / resistivity(COPPER, 20.0)
        assert ratio == pytest.approx(1.0 + 0.00393 * 80.0, rel=1e-12)
        assert ratio == pytest.approx(1.314, rel=1e-3)


class TestMagnet:
    def test_remanence_derates_with_temperature(self):
        assert N42.remanence_at(100.0) < N42.remanence_at(20.0)
        assert N42.remanence_at(20.0) == pytest.approx(N42.remanence_t)

    def test_remanence_derating_magnitude(self):
        # -0.12 %/C over 80 C -> -9.6 %
        assert N42.remanence_at(100.0) == pytest.approx(1.30 * (1 - 0.0012 * 80), rel=1e-12)


class TestSteinmetz:
    def test_m19_datasheet_band_60hz_1p5t(self):
        """M-19 29ga typical loss at 60 Hz / 1.5 T is roughly 1.4-1.6 W/kg."""
        p = M19_29GA.core_loss_w_per_kg(60.0, 1.5)
        assert 1.2 < p < 1.9

    def test_zero_frequency_or_flux_gives_zero(self):
        assert M19_29GA.core_loss_w_per_kg(0.0, 1.5) == 0.0
        assert M19_29GA.core_loss_w_per_kg(60.0, 0.0) == 0.0

    def test_loss_increases_with_frequency_and_flux(self):
        assert M19_29GA.core_loss_w_per_kg(120.0, 1.5) > M19_29GA.core_loss_w_per_kg(60.0, 1.5)
        assert M19_29GA.core_loss_w_per_kg(60.0, 1.6) > M19_29GA.core_loss_w_per_kg(60.0, 1.5)


class TestAirgapFluxDensity:
    def test_increases_with_magnet_thickness(self):
        b_thin = airgap_flux_density(N42, 0.002, 0.001)
        b_thick = airgap_flux_density(N42, 0.006, 0.001)
        assert b_thick > b_thin

    def test_decreases_with_air_gap(self):
        b_small = airgap_flux_density(N42, 0.004, 0.0005)
        b_large = airgap_flux_density(N42, 0.004, 0.002)
        assert b_small > b_large

    def test_approaches_remanence_as_gap_vanishes(self):
        assert airgap_flux_density(N42, 0.004, 0.0) == pytest.approx(N42.remanence_t)

    def test_hot_magnet_gives_lower_flux(self):
        cold = airgap_flux_density(N42, 0.004, 0.0008, magnet_temp_c=20.0)
        hot = airgap_flux_density(N42, 0.004, 0.0008, magnet_temp_c=100.0)
        assert hot < cold

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError):
            airgap_flux_density(N42, 0.0, 0.001)
        with pytest.raises(ValueError):
            airgap_flux_density(N42, 0.004, -0.001)


class TestRunoutClosedForms:
    """Closed-form theta-averages verified against trapezoidal quadrature.

    The trapezoid rule is spectrally accurate on smooth periodic integrands,
    so 64 points already reach near machine precision.
    """

    T_M, G, DELTA = 0.004, 0.0008, 0.0003

    def _quad_mean(self, func, n):
        import math

        from axfluxmdo.materials import airgap_flux_density

        total = 0.0
        for k in range(n):
            theta = 2 * math.pi * k / n
            g = self.G + self.DELTA * math.cos(theta)
            total += func(airgap_flux_density(N42, self.T_M, g))
        return total / n

    def test_mean_b_matches_quadrature(self):
        from axfluxmdo.materials.magnetic import airgap_flux_density_runout_mean

        closed = airgap_flux_density_runout_mean(N42, self.T_M, self.G, self.DELTA)
        for n in (64, 256):
            assert closed == pytest.approx(self._quad_mean(lambda b: b, n), rel=1e-10)

    def test_mean_b_squared_matches_quadrature(self):
        from axfluxmdo.materials.magnetic import airgap_b_squared_runout_mean

        closed = airgap_b_squared_runout_mean(N42, self.T_M, self.G, self.DELTA)
        for n in (64, 256):
            assert closed == pytest.approx(self._quad_mean(lambda b: b * b, n), rel=1e-10)

    def test_zero_runout_reduces_to_load_line(self):
        from axfluxmdo.materials.magnetic import (
            airgap_b_squared_runout_mean,
            airgap_flux_density_runout_mean,
        )

        b0 = airgap_flux_density(N42, self.T_M, self.G)
        assert airgap_flux_density_runout_mean(N42, self.T_M, self.G, 0.0) == pytest.approx(
            b0, rel=1e-15
        )
        assert airgap_b_squared_runout_mean(N42, self.T_M, self.G, 0.0) == pytest.approx(
            b0**2, rel=1e-15
        )

    def test_jensen_mean_exceeds_b_at_mean_gap(self):
        from axfluxmdo.materials.magnetic import airgap_flux_density_runout_mean

        mean_b = airgap_flux_density_runout_mean(N42, self.T_M, self.G, self.DELTA)
        assert mean_b > airgap_flux_density(N42, self.T_M, self.G)

    def test_extremes_bracket_mean(self):
        from axfluxmdo.materials.magnetic import (
            airgap_flux_density_runout_extremes,
            airgap_flux_density_runout_mean,
        )

        b_tight, b_wide = airgap_flux_density_runout_extremes(N42, self.T_M, self.G, self.DELTA)
        mean_b = airgap_flux_density_runout_mean(N42, self.T_M, self.G, self.DELTA)
        assert b_wide < mean_b < b_tight

    def test_gap_closing_runout_raises(self):
        from axfluxmdo.materials.magnetic import airgap_flux_density_runout_mean

        with pytest.raises(ValueError, match="runout"):
            airgap_flux_density_runout_mean(N42, self.T_M, 0.0008, 0.01)
