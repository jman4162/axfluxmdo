"""Field/profile visualization: radial profiles and efficiency maps."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from axfluxmdo.models.annular_2p5d import AnnularResult
    from axfluxmdo.models.efficiency_map import EfficiencyMap
    from axfluxmdo.solvers.results_parser import GapFieldSolution
    from axfluxmdo.validation.sim2real import OpenCircuitComparison


def plot_radial_profiles(result: AnnularResult, show: bool = False) -> Figure:
    """2x2 grid of per-slice quantities vs radius from an AnnularResult."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4))
    r_mm = result.slice_radii_m * 1e3

    ax = axes[0, 0]
    ax.plot(r_mm, result.slice_airgap_b_t, "o-", label=r"$\langle B_g \rangle$")
    ax.plot(r_mm, result.slice_b1_t, "s-", label="$B_1$ (fundamental)")
    ax.set_ylabel("flux density (T)")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    dr = np.gradient(result.slice_radii_m)
    ax.plot(r_mm, result.slice_torque_nm / dr, "o-")
    ax.set_ylabel("torque density dT/dr (N·m/m)")

    ax = axes[1, 0]
    ax.plot(r_mm, result.slice_yoke_b_t, "o-")
    ax.axhline(
        max(c.limit for c in result.constraints if c.name == "core_flux_density_t"),
        color="r",
        ls="--",
        lw=1,
        label="saturation limit",
    )
    ax.set_ylabel("yoke flux density (T)")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(r_mm, result.slice_current_loading_a_m / 1e3, "o-")
    ax.set_ylabel("current loading (kA/m)")

    for ax in axes.flat:
        ax.set_xlabel("radius (mm)")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_efficiency_map(emap: EfficiencyMap, show: bool = False) -> Figure:
    """Efficiency contours over the speed-torque plane; infeasible region greyed."""
    fig, ax = plt.subplots(figsize=(8, 5.5))

    infeasible = np.where(emap.feasible, np.nan, 1.0)
    ax.contourf(
        emap.speeds_rpm,
        emap.torques_nm,
        infeasible,
        levels=[0.5, 1.5],
        colors=["0.85"],
    )

    levels = np.linspace(np.nanmin(emap.efficiency), np.nanmax(emap.efficiency), 20)
    cf = ax.contourf(
        emap.speeds_rpm, emap.torques_nm, emap.efficiency, levels=levels, cmap="viridis"
    )
    fig.colorbar(cf, ax=ax, label="efficiency")
    line_levels = [lv for lv in (0.80, 0.90, 0.95, 0.97) if lv < np.nanmax(emap.efficiency)]
    if line_levels:
        cs = ax.contour(
            emap.speeds_rpm,
            emap.torques_nm,
            emap.efficiency,
            levels=line_levels,
            colors="w",
            linewidths=0.8,
        )
        ax.clabel(cs, fmt="%.2f", fontsize=8)

    ax.set_xlabel("speed (rpm)")
    ax.set_ylabel("torque (N·m)")
    ax.set_title("Efficiency map (grey = infeasible)")
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_gap_field(
    solution: GapFieldSolution,
    comparison: OpenCircuitComparison | None = None,
    show: bool = False,
) -> Figure:
    """FEA air-gap field By(x) along the midline, with analytical overlays."""
    import math

    fig, ax = plt.subplots(figsize=(8.5, 5))
    tau = solution.pole_pitch_m
    x_norm = solution.x_m / tau
    ax.plot(x_norm, solution.by_t, lw=1.5, label="FEA $B_y$ (gap midline)")

    # Shade the magnet-covered spans
    half_arc = 0.5 * solution.magnet_arc_ratio
    for k in (0, 1):
        ax.axvspan(
            k + 0.5 - half_arc,
            k + 0.5 + half_arc,
            color="0.9",
            zorder=0,
            label="magnet arc" if k == 0 else None,
        )

    if comparison is not None:
        b_g, b1 = comparison.analytical_b_g_t, comparison.analytical_b1_t
        for k, sign in ((0, +1), (1, -1)):
            ax.hlines(
                sign * b_g,
                k + 0.5 - half_arc,
                k + 0.5 + half_arc,
                colors="r",
                linestyles="--",
                lw=1.2,
                label="analytical $\\pm B_g$" if k == 0 else None,
            )
        xs = np.linspace(0, 2, 400)
        ax.plot(
            xs,
            b1 * np.sin(math.pi * xs),
            "g:",
            lw=1.2,
            label="analytical $B_1 \\sin(\\pi x/\\tau_p)$",
        )
        ax.set_title(
            f"Gap field — residuals: $B_g$ {comparison.residual_b_g_rel:+.1%}, "
            f"$B_1$ {comparison.residual_b1_rel:+.1%}"
        )
    else:
        ax.set_title("Gap field along the air-gap midline")

    ax.set_xlabel(r"$x / \tau_p$ (pole pitches)")
    ax.set_ylabel("$B_y$ (T)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    if show:
        plt.show()
    return fig
