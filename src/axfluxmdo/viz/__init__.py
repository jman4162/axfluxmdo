"""Visualization utilities."""

from axfluxmdo.viz.bayesopt import plot_convergence, plot_surrogate_slice
from axfluxmdo.viz.fields import plot_efficiency_map, plot_gap_field, plot_radial_profiles
from axfluxmdo.viz.geometry_plot import plot_geometry
from axfluxmdo.viz.pareto import plot_pareto
from axfluxmdo.viz.sensitivity import plot_tornado

__all__ = [
    "animate_exploded",
    "animate_rotation",
    "build_motor_assembly",
    "can_render",
    "plot_convergence",
    "plot_efficiency_map",
    "plot_gap_field",
    "plot_geometry",
    "plot_motor_3d",
    "plot_pareto",
    "plot_radial_profiles",
    "plot_surrogate_slice",
    "plot_tornado",
]

_PYVISTA_NAMES = {
    "animate_exploded",
    "animate_rotation",
    "build_motor_assembly",
    "can_render",
    "plot_motor_3d",
}


def __getattr__(name: str):
    if name in _PYVISTA_NAMES:
        from axfluxmdo.viz import pyvista_3d

        return getattr(pyvista_3d, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
