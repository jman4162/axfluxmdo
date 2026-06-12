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


class TestGapFieldPlot:
    def test_returns_figure_with_overlays(self, reference_motor):
        import math

        import numpy as np

        from axfluxmdo.solvers import GapFieldSolution
        from axfluxmdo.validation import compare_open_circuit
        from axfluxmdo.viz import plot_gap_field

        tau = reference_motor.pole_pitch
        x = np.linspace(0, 2 * tau, 401)
        sol = GapFieldSolution(
            x_m=x,
            by_t=1.0 * np.sin(math.pi * x / tau),
            pole_pitch_m=tau,
            magnet_arc_ratio=reference_motor.magnet_arc_ratio,
            magnet_temp_c=65.0,
            slotted=False,
        )
        cmp_ = compare_open_circuit(reference_motor, sol, magnet_temp_c=65.0)
        fig = plot_gap_field(sol, cmp_)
        assert fig is not None
        plt.close(fig)


class TestBayesoptPlots:
    @pytest.fixture(scope="class")
    def bo_study(self):
        pytest.importorskip("sklearn")
        from axfluxmdo import AxialFluxMotor, OperatingPoint
        from axfluxmdo.optimize import bayesian_optimize

        motor = AxialFluxMotor(outer_radius=0.08, inner_radius=0.025, air_gap=0.0008, pole_pairs=14)
        op = OperatingPoint(speed_rpm=500, current_rms=25, dc_bus_voltage=48)
        return bayesian_optimize(
            motor,
            op,
            variables={"outer_radius": (0.06, 0.10), "pole_pairs": [10, 12, 14]},
            objective="maximize_torque_density",
            n_initial=5,
            n_iterations=4,
            seed=9,
        )

    def test_convergence_plot(self, bo_study):
        from axfluxmdo.viz import plot_convergence

        fig = plot_convergence(bo_study)
        assert fig is not None
        plt.close(fig)

    def test_surrogate_slice_continuous_and_choice(self, bo_study):
        from axfluxmdo.viz import plot_surrogate_slice

        for var in ("outer_radius", "pole_pairs"):
            fig = plot_surrogate_slice(bo_study, var)
            assert fig is not None
            plt.close(fig)

    def test_unknown_variable_raises(self, bo_study):
        from axfluxmdo.viz import plot_surrogate_slice

        with pytest.raises(ValueError, match="unknown design variable"):
            plot_surrogate_slice(bo_study, "bogus")


class TestReviewRegressions:
    def test_surrogate_slice_string_choices(self, reference_motor, reference_op):
        """plot_surrogate_slice must handle non-numeric choice variables."""
        pytest.importorskip("sklearn")
        from axfluxmdo.optimize import bayesian_optimize
        from axfluxmdo.viz import plot_surrogate_slice

        study = bayesian_optimize(
            reference_motor,
            reference_op,
            variables={
                "outer_radius": (0.06, 0.10),
                "magnet_shape": ["wedge", "rectangular"],
            },
            objective="maximize_torque_density",
            n_initial=5,
            n_iterations=3,
            seed=4,
        )
        fig = plot_surrogate_slice(study, "magnet_shape")
        assert fig is not None
        plt.close(fig)

    def test_plot_pareto_invalid_color_key(self, reference_motor, reference_op):
        pytest.importorskip("pymoo")
        from axfluxmdo.optimize import optimize_pareto
        from axfluxmdo.optimize.problem import UnknownKeyError
        from axfluxmdo.viz import plot_pareto

        study = optimize_pareto(
            reference_motor,
            reference_op,
            variables={"outer_radius": (0.06, 0.10)},
            objectives=["maximize_torque_density", "maximize_efficiency"],
            pop_size=8,
            n_gen=2,
            seed=1,
        )
        with pytest.raises(UnknownKeyError):
            plot_pareto(study, color="not_a_key")

    def test_sweep_plot_bogus_field(self, reference_motor, reference_op):
        from axfluxmdo.sweeps import sweep_parameter

        sweep = sweep_parameter(reference_motor, reference_op, "air_gap", [0.0008, 0.001])
        with pytest.raises(KeyError):
            sweep.plot(fields=("bogus_field",))
