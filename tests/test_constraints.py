import math

from axfluxmdo.limits import Limits
from axfluxmdo.models.constraints import make_upper_bound


class TestConstraintRecord:
    def test_satisfied_with_positive_margin(self):
        c = make_upper_bound("winding_temp_c", 100.0, 140.0)
        assert c.satisfied
        assert c.margin > 0.0
        assert c.margin == (140.0 - 100.0) / 140.0

    def test_violated_with_negative_margin(self):
        c = make_upper_bound("winding_temp_c", 180.0, 140.0)
        assert not c.satisfied
        assert c.margin < 0.0

    def test_boundary_is_satisfied(self):
        c = make_upper_bound("x", 140.0, 140.0)
        assert c.satisfied
        assert c.margin == 0.0

    def test_infinite_value_is_violated(self):
        c = make_upper_bound("winding_temp_c", math.inf, 140.0)
        assert not c.satisfied
        assert c.margin == -math.inf

    def test_str_rendering(self):
        c = make_upper_bound("electrical_frequency_hz", 116.7, 1000.0)
        s = str(c)
        assert "electrical_frequency_hz" in s
        assert "OK" in s


class TestLimits:
    def test_defaults(self):
        lim = Limits()
        assert lim.max_winding_temp_c == 140.0
        assert lim.max_electrical_freq_hz == 1000.0
        assert lim.max_current_density_a_mm2 == 10.0
        assert lim.max_line_voltage_v is None
        assert lim.max_core_flux_density_t is None
