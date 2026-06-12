"""2D geometry visualization: front view and axial cross-section."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle, Wedge

from axfluxmdo.geometry.axial_flux import AxialFluxMotor

AIR_GAP_EXAGGERATION = 4.0  # the real gap is invisible at scale; annotated on the plot

N_COLOR = "#d62728"
S_COLOR = "#1f77b4"


def plot_geometry(motor: AxialFluxMotor, view: str = "both", show: bool = False) -> Figure:
    """Plot the motor geometry.

    view: "front" (magnet disk seen along the axis), "section" (r-z axial
    cross-section), or "both".
    """
    if view not in ("front", "section", "both"):
        raise ValueError("view must be 'front', 'section', or 'both'")
    if view == "both":
        fig, (ax_front, ax_section) = plt.subplots(1, 2, figsize=(12, 5.5))
        _draw_front(motor, ax_front)
        _draw_section(motor, ax_section)
    else:
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        if view == "front":
            _draw_front(motor, ax)
        else:
            _draw_section(motor, ax)
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def _draw_front(motor: AxialFluxMotor, ax: plt.Axes) -> None:
    r_o, r_i = motor.outer_radius, motor.inner_radius
    n_poles = 2 * motor.pole_pairs
    pole_angle_deg = 360.0 / n_poles
    magnet_arc_deg = motor.magnet_arc_ratio * pole_angle_deg

    ax.add_patch(Circle((0, 0), r_o, fill=False, color="0.3", lw=1.5))
    ax.add_patch(Circle((0, 0), r_i, fill=False, color="0.3", lw=1.5))
    for k in range(n_poles):
        start = k * pole_angle_deg + 0.5 * (pole_angle_deg - magnet_arc_deg)
        color = N_COLOR if k % 2 == 0 else S_COLOR
        ax.add_patch(
            Wedge(
                (0, 0),
                r_o,
                start,
                start + magnet_arc_deg,
                width=r_o - r_i,
                facecolor=color,
                edgecolor="k",
                lw=0.4,
                alpha=0.85,
            )
        )

    ax.annotate(
        f"$r_o$ = {r_o * 1e3:.1f} mm",
        xy=(r_o * math.cos(0.4), r_o * math.sin(0.4)),
        xytext=(1.25 * r_o, 0.9 * r_o),
        arrowprops={"arrowstyle": "->"},
    )
    ax.annotate(
        f"$r_i$ = {r_i * 1e3:.1f} mm",
        xy=(r_i * math.cos(0.9), r_i * math.sin(0.9)),
        xytext=(1.25 * r_o, 0.7 * r_o),
        arrowprops={"arrowstyle": "->"},
    )
    ax.set_title(f"Rotor front view — {n_poles} poles (p = {motor.pole_pairs})")
    ax.set_xlim(-1.7 * r_o, 1.7 * r_o)
    ax.set_ylim(-1.3 * r_o, 1.3 * r_o)
    ax.set_aspect("equal")
    ax.axis("off")


def _draw_section(motor: AxialFluxMotor, ax: plt.Axes) -> None:
    """r-z cross-section through one circumferential cut (single-gap stack)."""
    r_i, r_o = motor.inner_radius, motor.outer_radius
    width = r_o - r_i
    gap_drawn = AIR_GAP_EXAGGERATION * motor.air_gap

    layers = [  # bottom-up: (axial thickness, label, color)
        (motor.back_iron_thickness, "rotor back iron", "0.55"),
        (motor.magnet_thickness, "magnets", N_COLOR),
        (
            gap_drawn,
            f"air gap ({motor.air_gap * 1e3:.2f} mm, drawn ×{AIR_GAP_EXAGGERATION:.0f})",
            "white",
        ),
        (motor.slot_depth, "winding window", "#b87333"),
        (motor.stator_core_thickness, "stator core", "0.55"),
    ]
    z = 0.0
    for thickness, label, color in layers:
        ax.add_patch(
            Rectangle((r_i, z), width, thickness, facecolor=color, edgecolor="k", lw=0.6, alpha=0.9)
        )
        ax.annotate(
            f"{label}",
            xy=(r_o, z + thickness / 2),
            xytext=(r_o + 0.15 * width, z + thickness / 2),
            va="center",
            fontsize=9,
            arrowprops={"arrowstyle": "-"},
        )
        z += thickness

    ax.set_xlim(r_i - 0.15 * width, r_o + 0.9 * width)
    ax.set_ylim(-0.2 * z, 1.2 * z)
    ax.set_xlabel("radius (m)")
    ax.set_ylabel("axial position (m)")
    ax.set_title("Axial cross-section (r–z)")
    ax.set_aspect("auto")
