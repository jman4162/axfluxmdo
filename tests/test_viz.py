import matplotlib.pyplot as plt
import pytest
from matplotlib.patches import Wedge

from axfluxmdo.viz import plot_geometry


class TestPlotGeometry:
    def test_both_views_return_figure(self, reference_motor):
        fig = plot_geometry(reference_motor)
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_front_view_has_2p_magnet_wedges(self, reference_motor):
        fig = plot_geometry(reference_motor, view="front")
        wedges = [p for p in fig.axes[0].patches if isinstance(p, Wedge)]
        assert len(wedges) == 2 * reference_motor.pole_pairs
        plt.close(fig)

    def test_section_view(self, reference_motor):
        fig = plot_geometry(reference_motor, view="section")
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_invalid_view_raises(self, reference_motor):
        with pytest.raises(ValueError, match="view"):
            plot_geometry(reference_motor, view="side")


class TestFieldPlots:
    def test_radial_profiles_returns_figure(self, reference_motor, reference_op):
        from axfluxmdo.models import AnnularModel
        from axfluxmdo.viz import plot_radial_profiles

        result = AnnularModel().evaluate(reference_motor, reference_op)
        fig = plot_radial_profiles(result)
        assert len(fig.axes) == 4
        plt.close(fig)

    def test_efficiency_map_returns_figure(self, reference_motor, reference_op):
        from axfluxmdo.models import compute_efficiency_map
        from axfluxmdo.viz import plot_efficiency_map

        emap = compute_efficiency_map(
            reference_motor,
            reference_op,
            max_speed_rpm=2000,
            max_torque_nm=10,
            n_speed=6,
            n_torque=5,
        )
        fig = plot_efficiency_map(emap)
        assert fig is not None
        plt.close(fig)
