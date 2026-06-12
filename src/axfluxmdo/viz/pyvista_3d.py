"""PyVista 3D assembly, static views, and animations (SPEC viz/pyvista_3d.py).

Every component is built from one primitive: a closed hexahedral annular
sector as a ``pv.StructuredGrid`` over a numpy (r, theta, z) vertex grid.
Full rings are 360-degree sectors; magnets, teeth, slots, and the cutaway
are partial sectors. Vertices lie exactly on the bounding circles (bounds
are exact) and the solids have meaningful ``.volume`` (tested against the
motor's analytic volume properties).

Conventions:
- Axial stack matches ``solvers/gmsh_export.py`` / ``Linear2DLayout`` with
  the air-gap midline at z = 0.
- The optional cutaway wedge applies to STATOR-side parts only — cutting the
  rotor would slice a moving part mid-animation.
- Animations are GIF-only (GitHub renders GIFs in READMEs; MP4 would pull in
  the imageio-ffmpeg binary wheel for no documentation benefit).

This module imports pyvista at module level but is only reachable lazily
from ``axfluxmdo.viz`` (PEP 562) — the base package never imports VTK.
"""

from __future__ import annotations

import functools
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

try:
    import pyvista as pv
except ImportError as exc:  # pragma: no cover - exercised only without [viz3d]
    raise ImportError(
        "3D visualization requires pyvista; install with: pip install 'axfluxmdo[viz3d]'"
    ) from exc

from axfluxmdo.geometry.axial_flux import AxialFluxMotor

COLORS = {
    "magnet_n": "#d62728",
    "magnet_s": "#1f77b4",
    "copper": "#b87333",
    "iron": "#808080",
}


def _annular_sector(
    r_inner: float,
    r_outer: float,
    z0: float,
    z1: float,
    theta0: float,
    theta1: float,
    *,
    theta_resolution_deg: float = 3.0,
) -> pv.StructuredGrid:
    """Closed hexahedral annular-sector solid (exact vertices on both circles)."""
    n = max(3, math.ceil(math.degrees(theta1 - theta0) / theta_resolution_deg))
    theta = np.linspace(theta0, theta1, n + 1)
    r = np.array([r_inner, r_outer])
    z = np.array([z0, z1])
    # meshgrid with indexing="ij" gives axes (r, theta, z); StructuredGrid
    # dimensions are (nx, ny, nz) for points ordered x-fastest, so build the
    # point array with r varying fastest by raveling in Fortran-like order
    # over (r, theta, z).
    R = r[:, None, None] * np.ones((2, n + 1, 2))
    T = theta[None, :, None] * np.ones((2, n + 1, 2))
    Z = z[None, None, :] * np.ones((2, n + 1, 2))
    points = np.column_stack(
        [
            (R * np.cos(T)).ravel(order="F"),
            (R * np.sin(T)).ravel(order="F"),
            Z.ravel(order="F"),
        ]
    )
    grid = pv.StructuredGrid()
    grid.points = points
    grid.dimensions = (2, n + 1, 2)
    return grid


def _rectangular_magnet(
    r_inner: float,
    r_outer: float,
    z0: float,
    z1: float,
    center_angle: float,
    width: float,
) -> pv.StructuredGrid:
    """Straight-sided magnet bar along the pole centerline (rectangular shape).

    Constant circumferential width (the mean-radius width convention of the
    annular model); the radial ends are chords, not arcs.
    """
    u = np.array([r_inner, r_outer])  # along the centerline
    v = np.array([-width / 2.0, width / 2.0])  # across
    z = np.array([z0, z1])
    U = u[:, None, None] * np.ones((2, 2, 2))
    V = v[None, :, None] * np.ones((2, 2, 2))
    Z = z[None, None, :] * np.ones((2, 2, 2))
    c, s = math.cos(center_angle), math.sin(center_angle)
    X = U * c - V * s
    Y = U * s + V * c
    grid = pv.StructuredGrid()
    grid.points = np.column_stack([X.ravel(order="F"), Y.ravel(order="F"), Z.ravel(order="F")])
    grid.dimensions = (2, 2, 2)
    return grid


def build_motor_assembly(
    motor: AxialFluxMotor,
    *,
    theta_cutaway_deg: float | None = 90.0,
    rotor_angle_rad: float = 0.0,
    theta_resolution_deg: float = 3.0,
) -> dict:
    """Full-360° motor assembly as PyVista solids.

    Returns a dict with stable keys: ``rotor_iron`` (StructuredGrid),
    ``magnets`` (MultiBlock of 2p sectors), ``stator_teeth``,
    ``stator_coils``, ``stator_yoke``. The cutaway wedge (over
    ``[0, theta_cutaway_deg]``) removes STATOR-side material only.
    """
    g = motor.air_gap
    t_m = motor.magnet_thickness
    z_rotor = (-g / 2.0 - t_m - motor.back_iron_thickness, -g / 2.0 - t_m)
    z_magnet = (-g / 2.0 - t_m, -g / 2.0)
    z_slot = (g / 2.0, g / 2.0 + motor.slot_depth)
    z_yoke = (z_slot[1], z_slot[1] + motor.stator_core_thickness)
    r_i, r_o = motor.inner_radius, motor.outer_radius
    res = theta_resolution_deg

    stator_start = math.radians(theta_cutaway_deg) if theta_cutaway_deg else 0.0

    rotor_iron = _annular_sector(r_i, r_o, *z_rotor, 0.0, 2.0 * math.pi, theta_resolution_deg=res)

    pole_angle = math.pi / motor.pole_pairs
    magnets = pv.MultiBlock()
    for k in range(2 * motor.pole_pairs):
        center = (k + 0.5) * pole_angle
        if motor.magnet_shape == "rectangular":
            width = motor.magnet_arc_ratio * motor.pole_pitch
            block = _rectangular_magnet(r_i, r_o, *z_magnet, center, width)
        else:
            half_arc = 0.5 * motor.magnet_arc_ratio * pole_angle
            block = _annular_sector(
                r_i,
                r_o,
                *z_magnet,
                center - half_arc,
                center + half_arc,
                theta_resolution_deg=res,
            )
        magnets.append(block, name=f"magnet_{k}")

    if rotor_angle_rad != 0.0:
        rotor_iron.rotate_z(math.degrees(rotor_angle_rad), inplace=True)
        for block in magnets:
            block.rotate_z(math.degrees(rotor_angle_rad), inplace=True)

    # Slotted stator band: alternating copper slots and iron teeth.
    n_slots = 2 * motor.phases * motor.pole_pairs
    slot_pitch = 2.0 * math.pi / n_slots
    opening = motor.slot_width_fraction * slot_pitch
    teeth_parts: list[pv.StructuredGrid] = []
    coil_parts: list[pv.StructuredGrid] = []

    def clipped(theta0: float, theta1: float) -> tuple[float, float] | None:
        """Clip a sector to the non-cutaway span [stator_start, 2*pi]."""
        lo, hi = max(theta0, stator_start), min(theta1, 2.0 * math.pi)
        return (lo, hi) if hi - lo > 1e-9 else None

    for k in range(n_slots):
        s0 = k * slot_pitch + 0.5 * (slot_pitch - opening)
        s1 = s0 + opening
        for span, bucket in (
            ((k * slot_pitch, s0), teeth_parts),
            ((s0, s1), coil_parts),
            ((s1, (k + 1) * slot_pitch), teeth_parts),
        ):
            kept = clipped(*span)
            if kept is not None:
                bucket.append(_annular_sector(r_i, r_o, *z_slot, *kept, theta_resolution_deg=res))

    stator_teeth = pv.merge(teeth_parts)
    stator_coils = pv.merge(coil_parts)
    stator_yoke = _annular_sector(
        r_i, r_o, *z_yoke, stator_start, 2.0 * math.pi, theta_resolution_deg=res
    )

    return {
        "rotor_iron": rotor_iron,
        "magnets": magnets,
        "stator_teeth": stator_teeth,
        "stator_coils": stator_coils,
        "stator_yoke": stator_yoke,
    }


@functools.lru_cache(maxsize=1)
def _can_render() -> bool:
    """Probe whether an off-screen VTK render is possible.

    Run in a SUBPROCESS: without a usable GL context VTK can segfault rather
    than raise, and an in-process probe would take the caller down with it.
    On Linux without a display, try pyvista's xvfb helper first.
    """
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        try:
            pv.start_xvfb()
        except OSError:
            return False
    code = (
        "import pyvista as pv; p = pv.Plotter(off_screen=True, window_size=(2, 2)); "
        "p.add_mesh(pv.Sphere()); p.screenshot(None); p.close()"
    )
    try:
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):  # pragma: no cover
        return False
    return proc.returncode == 0


def _add_assembly(plotter: pv.Plotter, assembly: dict) -> None:
    plotter.add_mesh(assembly["rotor_iron"], color=COLORS["iron"], smooth_shading=True)
    for k, block in enumerate(assembly["magnets"]):
        color = COLORS["magnet_n"] if k % 2 == 0 else COLORS["magnet_s"]
        plotter.add_mesh(block, color=color, smooth_shading=True, specular=0.3)
    plotter.add_mesh(assembly["stator_teeth"], color=COLORS["iron"], smooth_shading=True)
    plotter.add_mesh(
        assembly["stator_coils"], color=COLORS["copper"], smooth_shading=True, specular=0.4
    )
    plotter.add_mesh(assembly["stator_yoke"], color=COLORS["iron"], smooth_shading=True)


def _set_camera(plotter: pv.Plotter, motor: AxialFluxMotor) -> None:
    """Fixed isometric-ish view angled into the cutaway opening (first quadrant)."""
    r = motor.outer_radius
    plotter.camera_position = [
        (3.2 * r, 2.4 * r, 2.6 * r),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    ]


def plot_motor_3d(
    motor: AxialFluxMotor,
    *,
    theta_cutaway_deg: float | None = 90.0,
    show: bool = False,
    screenshot: str | Path | None = None,
    window_size: tuple[int, int] = (960, 720),
) -> pv.Plotter:
    """Static 3D view; returns the Plotter (the Figure analogue of this layer)."""
    assembly = build_motor_assembly(motor, theta_cutaway_deg=theta_cutaway_deg)
    plotter = pv.Plotter(off_screen=not show, window_size=list(window_size))
    _add_assembly(plotter, assembly)
    _set_camera(plotter, motor)
    if screenshot is not None:
        plotter.screenshot(str(Path(screenshot).resolve()))
    if show:  # pragma: no cover - interactive path
        plotter.show()
    return plotter


def animate_rotation(
    motor: AxialFluxMotor,
    path: str | Path,
    *,
    n_frames: int = 72,
    fps: int = 15,
    theta_cutaway_deg: float | None = 90.0,
    window_size: tuple[int, int] = (640, 480),
) -> Path:
    """One full mechanical revolution of the rotor over the (cutaway) stator."""
    path = _require_gif(path)
    assembly = build_motor_assembly(motor, theta_cutaway_deg=theta_cutaway_deg)
    plotter = pv.Plotter(off_screen=True, window_size=list(window_size))
    _add_assembly(plotter, assembly)
    _set_camera(plotter, motor)
    plotter.open_gif(str(path), fps=fps)
    step_deg = 360.0 / n_frames
    rotor_parts = [assembly["rotor_iron"], *assembly["magnets"]]
    for _ in range(n_frames):
        for mesh in rotor_parts:
            mesh.rotate_z(step_deg, inplace=True)
        plotter.write_frame()
    plotter.close()
    return path


def animate_exploded(
    motor: AxialFluxMotor,
    path: str | Path,
    *,
    n_frames: int = 60,
    fps: int = 15,
    travel: float | None = None,
    window_size: tuple[int, int] = (640, 480),
) -> Path:
    """Components separate axially, hold, and reassemble (ease-in-out)."""
    path = _require_gif(path)
    assembly = build_motor_assembly(motor, theta_cutaway_deg=None)
    stack_height = (
        motor.back_iron_thickness
        + motor.magnet_thickness
        + motor.air_gap
        + motor.slot_depth
        + motor.stator_core_thickness
    )
    travel = travel if travel is not None else 3.0 * stack_height

    multipliers = {
        "rotor_iron": -1.0,
        "magnets": -0.5,
        "stator_coils": 0.5,
        "stator_teeth": 0.5,
        "stator_yoke": 1.0,
    }
    parts: list[tuple[pv.DataSet, float, np.ndarray]] = []
    for name, mult in multipliers.items():
        meshes = assembly[name] if name == "magnets" else [assembly[name]]
        for mesh in meshes:
            parts.append((mesh, mult, mesh.points.copy()))

    plotter = pv.Plotter(off_screen=True, window_size=list(window_size))
    _add_assembly(plotter, assembly)
    _set_camera(plotter, motor)
    plotter.open_gif(str(path), fps=fps)

    n_out = max(1, int(0.4 * n_frames))
    n_hold = max(1, int(0.2 * n_frames))
    n_in = n_frames - n_out - n_hold
    profile = (
        [0.5 * (1 - math.cos(math.pi * t / n_out)) for t in range(n_out)]
        + [1.0] * n_hold
        + [0.5 * (1 + math.cos(math.pi * t / max(1, n_in - 1))) for t in range(n_in)]
    )
    for s in profile:
        for mesh, mult, base in parts:
            # absolute positioning from cached base points: no incremental drift
            mesh.points = base + np.array([0.0, 0.0, mult * travel * s])
        plotter.write_frame()
    plotter.close()
    return path


def _require_gif(path: str | Path) -> Path:
    path = Path(path).resolve()
    if path.suffix.lower() != ".gif":
        raise ValueError(
            f"animations are GIF-only (got {path.suffix!r}); GitHub renders GIFs "
            "in READMEs and MP4 would require the imageio-ffmpeg binary wheel"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
