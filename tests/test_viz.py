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


class TestOptimizationPlots:
    def test_tornado_returns_figure(self, reference_motor, reference_op):
        from axfluxmdo.optimize import compute_sensitivities
        from axfluxmdo.viz import plot_tornado

        sens = compute_sensitivities(
            reference_motor, reference_op, ["air_gap", "outer_radius"], output="torque_nm"
        )
        fig = plot_tornado(sens)
        assert fig is not None
        plt.close(fig)

    def test_plot_pareto_with_colorbar(self, reference_motor, reference_op):
        pytest.importorskip("pymoo")
        from axfluxmdo.optimize import optimize_pareto
        from axfluxmdo.viz import plot_pareto

        study = optimize_pareto(
            reference_motor,
            reference_op,
            variables={"outer_radius": (0.06, 0.10), "fill_factor": (0.35, 0.55)},
            objectives=["maximize_torque_density", "maximize_efficiency"],
            pop_size=8,
            n_gen=3,
            seed=3,
        )
        fig = plot_pareto(study, x="torque_density", y="efficiency", color="winding_temp_c")
        assert len(fig.axes) == 2  # scatter + colorbar
        plt.close(fig)

        fig2 = plot_pareto(study, x="outer_radius", y="torque_nm", annotate_best=True)
        assert fig2 is not None
        plt.close(fig2)
