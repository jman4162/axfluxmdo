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
