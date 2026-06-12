"""Gmsh geometry and mesh export.

Two models:

- **2D unrolled** (``build_linear_2d_model`` / ``export_mesh``): the annulus
  unrolled at the mean radius into a linear-machine cross-section — one pole
  pair, x circumferential in [0, 2*tau_p], y axial with y=0 at the air-gap
  center. Solvable as planar magnetostatics by GetDP; the slotless variant
  places stator iron directly at +g/2 so the magnetic gap is exactly
  ``motor.air_gap`` (the load line's circuit). The slotted variant opens
  winding slots so the Carter factor can be MEASURED.
- **3D sector** (``export_3d_sector``): an OCC annular sector with rotor /
  magnet / gap / winding / stator volumes for visualization and downstream
  meshing credibility; no solver hookup.

gmsh is a process-global singleton: every session goes through
``_gmsh_session`` (try/finally finalize), and all paths are resolved to
absolute before writing (gmsh writes CWD-relative otherwise). Mesh files are
written as MSH 2.2, the format GetDP handles best.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from axfluxmdo.geometry.axial_flux import AxialFluxMotor


def _import_gmsh():
    try:
        import gmsh
    except ImportError as exc:  # pragma: no cover - exercised only without [fea]
        raise ImportError(
            "mesh export requires gmsh; install with: pip install 'axfluxmdo[fea]'"
        ) from exc
    return gmsh


@contextmanager
def _gmsh_session(*, terminal: bool = False):
    gmsh = _import_gmsh()
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1 if terminal else 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        yield gmsh
    finally:
        gmsh.finalize()


@dataclass(frozen=True)
class Linear2DLayout:
    """Everything the GetDP template and the parser need to know about the mesh."""

    x_span_m: float  # L = n_pole_pairs_modeled * 2 * tau_p
    gap_midline_y_m: float  # 0.0
    depth_m: float  # active length r_o - r_i (out-of-plane depth)
    pole_pitch_m: float
    magnet_arc_ratio: float
    slotted: bool
    group_tags: dict[str, int]  # physical group name -> tag (surfaces + boundaries)
    y_interfaces_m: dict[str, float]  # named axial interface coordinates


def build_linear_2d_model(
    motor: AxialFluxMotor,
    *,
    slotted: bool = False,
    n_pole_pairs_modeled: int = 1,
    airgap_layers: int = 4,
    mesh_size_factor: float = 1.0,
    slots_per_pole_pair: int | None = None,
) -> Linear2DLayout:
    """Populate the CURRENT gmsh model with the unrolled cross-section.

    Requires an active gmsh session (see ``export_mesh`` for the session-owning
    wrapper). Returns the layout descriptor; the mesh is not yet generated.
    """
    gmsh = _import_gmsh()
    geo = gmsh.model.geo

    tau = motor.pole_pitch
    span = n_pole_pairs_modeled * 2.0 * tau
    g = motor.air_gap
    t_m = motor.magnet_thickness
    t_bi = motor.back_iron_thickness
    d_slot = motor.slot_depth
    t_sc = motor.stator_core_thickness

    y_rotor_bottom = -g / 2.0 - t_m - t_bi
    y_magnet_bottom = -g / 2.0 - t_m
    y_gap_bottom = -g / 2.0
    y_gap_top = +g / 2.0
    y_slot_top = y_gap_top + d_slot
    y_stator_top = y_slot_top + t_sc

    size_fine = mesh_size_factor * min(g, t_m) / 2.0
    size_coarse = 4.0 * size_fine

    point_cache: dict[tuple[float, float], int] = {}

    def pt(x: float, y: float, size: float) -> int:
        key = (round(x, 12), round(y, 12))
        if key not in point_cache:
            point_cache[key] = geo.addPoint(x, y, 0.0, size)
        return point_cache[key]

    line_cache: dict[tuple[int, int], int] = {}

    def ln(p1: int, p2: int) -> int:
        """Shared line cache so adjacent rectangles are conformal."""
        if (p1, p2) in line_cache:
            return line_cache[(p1, p2)]
        if (p2, p1) in line_cache:
            return -line_cache[(p2, p1)]
        tag = geo.addLine(p1, p2)
        line_cache[(p1, p2)] = tag
        return tag

    def rect(x0: float, x1: float, y0: float, y1: float, size: float) -> int:
        """Conformal rectangle surface; returns the surface tag."""
        p_bl, p_br = pt(x0, y0, size), pt(x1, y0, size)
        p_tr, p_tl = pt(x1, y1, size), pt(x0, y1, size)
        loop = geo.addCurveLoop([ln(p_bl, p_br), ln(p_br, p_tr), ln(p_tr, p_tl), ln(p_tl, p_bl)])
        return geo.addPlaneSurface([loop])

    surfaces: dict[str, list[int]] = {
        "ROTOR_IRON": [],
        "MAGNET_N": [],
        "MAGNET_S": [],
        "AIR": [],
        "AIRGAP": [],
        "WINDING": [],
        "STATOR_IRON": [],
    }

    # The magnet band needs x-breakpoints at magnet edges; gap and iron bands
    # share those breakpoints so every vertical interface is conformal.
    x_breaks: list[float] = [0.0]
    magnet_intervals: list[tuple[float, float, str]] = []
    for k in range(2 * n_pole_pairs_modeled):
        x0 = k * tau + 0.5 * tau * (1.0 - motor.magnet_arc_ratio)
        x1 = k * tau + 0.5 * tau * (1.0 + motor.magnet_arc_ratio)
        name = "MAGNET_N" if k % 2 == 0 else "MAGNET_S"
        magnet_intervals.append((x0, x1, name))
        x_breaks += [x0, x1]
    x_breaks.append(span)
    x_breaks = sorted(set(round(x, 12) for x in x_breaks))

    def band(y0: float, y1: float, size: float, classify) -> None:
        for xa, xb in zip(x_breaks[:-1], x_breaks[1:], strict=False):
            xm = 0.5 * (xa + xb)
            surfaces[classify(xm)].append(rect(xa, xb, y0, y1, size))

    # Rotor back iron (single band, coarse)
    band(y_rotor_bottom, y_magnet_bottom, size_coarse, lambda _x: "ROTOR_IRON")

    # Magnet band: magnets where covered, AIR between
    def classify_magnet(xm: float) -> str:
        for x0, x1, name in magnet_intervals:
            if x0 <= xm <= x1:
                return name
        return "AIR"

    band(y_magnet_bottom, y_gap_bottom, size_fine, classify_magnet)

    # Air gap (fine)
    band(y_gap_bottom, y_gap_top, size_fine, lambda _x: "AIRGAP")

    # Stator
    if not slotted:
        # Slotless: iron face directly at +g/2 — the load-line magnetic circuit.
        band(y_gap_top, y_stator_top, size_coarse, lambda _x: "STATOR_IRON")
    else:
        n_slots = slots_per_pole_pair or 2 * motor.phases
        n_slots *= n_pole_pairs_modeled
        slot_pitch = span / n_slots
        opening = motor.slot_width_fraction * slot_pitch
        slot_intervals = []
        for k in range(n_slots):
            center = (k + 0.5) * slot_pitch
            slot_intervals.append((center - opening / 2.0, center + opening / 2.0))
        slot_breaks = sorted(
            set(
                x_breaks
                + [round(x0, 12) for x0, _ in slot_intervals]
                + [round(x1, 12) for _, x1 in slot_intervals]
            )
        )

        def classify_slot(xm: float) -> str:
            for x0, x1 in slot_intervals:
                if x0 <= xm <= x1:
                    return "WINDING"
            return "STATOR_IRON"

        for xa, xb in zip(slot_breaks[:-1], slot_breaks[1:], strict=False):
            xm = 0.5 * (xa + xb)
            surfaces[classify_slot(xm)].append(rect(xa, xb, y_gap_top, y_slot_top, size_fine))
        # Yoke above the slots (conformal with the slot band via shared breakpoints)
        for xa, xb in zip(slot_breaks[:-1], slot_breaks[1:], strict=False):
            surfaces["STATOR_IRON"].append(rect(xa, xb, y_slot_top, y_stator_top, size_coarse))

    geo.synchronize()

    # Periodic constraint: right edges are translated copies of left edges.
    def edges_at_x(x_target: float) -> list[int]:
        tags = []
        for (p1, p2), tag in line_cache.items():
            x1y = point_cache_inv[p1]
            x2y = point_cache_inv[p2]
            if abs(x1y[0] - x_target) < 1e-12 and abs(x2y[0] - x_target) < 1e-12:
                tags.append(tag)
        return sorted(tags)

    point_cache_inv = {tag: xy for xy, tag in point_cache.items()}
    left_edges = edges_at_x(0.0)
    right_edges = edges_at_x(round(span, 12))
    translation = [1, 0, 0, span, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    gmsh.model.mesh.setPeriodic(1, right_edges, left_edges, translation)

    # Boundary groups
    def horizontal_edges_at_y(y_target: float) -> list[int]:
        tags = []
        for (p1, p2), tag in line_cache.items():
            y1 = point_cache_inv[p1][1]
            y2 = point_cache_inv[p2][1]
            if abs(y1 - y_target) < 1e-12 and abs(y2 - y_target) < 1e-12:
                tags.append(tag)
        return sorted(tags)

    group_tags: dict[str, int] = {}
    for name, surfs in surfaces.items():
        if surfs:
            tag = gmsh.model.addPhysicalGroup(2, surfs)
            gmsh.model.setPhysicalName(2, tag, name)
            group_tags[name] = tag
    for name, edges in (
        ("PERIODIC_LEFT", left_edges),
        ("PERIODIC_RIGHT", right_edges),
        ("OUTER", horizontal_edges_at_y(y_rotor_bottom) + horizontal_edges_at_y(y_stator_top)),
    ):
        tag = gmsh.model.addPhysicalGroup(1, [abs(t) for t in edges])
        gmsh.model.setPhysicalName(1, tag, name)
        group_tags[name] = tag

    return Linear2DLayout(
        x_span_m=span,
        gap_midline_y_m=0.0,
        depth_m=motor.active_length,
        pole_pitch_m=tau,
        magnet_arc_ratio=motor.magnet_arc_ratio,
        slotted=slotted,
        group_tags=group_tags,
        y_interfaces_m={
            "rotor_bottom": y_rotor_bottom,
            "magnet_bottom": y_magnet_bottom,
            "gap_bottom": y_gap_bottom,
            "gap_top": y_gap_top,
            "slot_top": y_slot_top,
            "stator_top": y_stator_top,
        },
    )


def export_mesh(
    motor: AxialFluxMotor,
    path: str | Path,
    *,
    geo_unrolled: bool = False,
    **build_kwargs,
) -> tuple[Path, Linear2DLayout]:
    """Build the unrolled 2D model, mesh it, and write a .msh (MSH 2.2)."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _gmsh_session() as gmsh:
        gmsh.model.add("axfluxmdo_linear2d")
        layout = build_linear_2d_model(motor, **build_kwargs)
        gmsh.model.mesh.generate(2)
        gmsh.write(str(path))
        if geo_unrolled:
            gmsh.write(str(path.with_suffix(".geo_unrolled")))
    return path, layout


def export_3d_sector(
    motor: AxialFluxMotor,
    path: str | Path,
    *,
    sector_poles: int = 1,
    mesh_size_factor: float = 1.0,
) -> Path:
    """Export a 3D annular sector mesh (rotor/magnets/gap/winding/stator volumes)."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    angle = sector_poles * math.pi / motor.pole_pairs  # one pole = pi/p
    g = motor.air_gap
    t_m = motor.magnet_thickness

    with _gmsh_session() as gmsh:
        gmsh.model.add("axfluxmdo_sector3d")
        occ = gmsh.model.occ

        def annular_sector(z0: float, dz: float) -> int:
            outer = occ.addCylinder(0, 0, z0, 0, 0, dz, motor.outer_radius, angle=angle)
            inner = occ.addCylinder(0, 0, z0, 0, 0, dz, motor.inner_radius, angle=angle)
            out, _ = occ.cut([(3, outer)], [(3, inner)])
            return out[0][1]

        z = 0.0
        volumes: list[tuple[str, int]] = []
        layers = [
            ("ROTOR_IRON", motor.back_iron_thickness),
            ("MAGNET", t_m),
            ("AIRGAP", g),
            ("WINDING", motor.slot_depth),
            ("STATOR_IRON", motor.stator_core_thickness),
        ]
        for name, thickness in layers:
            if name == "MAGNET":
                # magnet arc sub-sector centered in the pole(s); rest of the band is air
                full = annular_sector(z, thickness)
                magnet_angle = motor.magnet_arc_ratio * math.pi / motor.pole_pairs
                magnet_tags = []
                for k in range(sector_poles):
                    pole_start = k * math.pi / motor.pole_pairs
                    rotate_by = pole_start + 0.5 * (math.pi / motor.pole_pairs - magnet_angle)
                    outer = occ.addCylinder(
                        0, 0, z, 0, 0, thickness, motor.outer_radius, angle=magnet_angle
                    )
                    inner = occ.addCylinder(
                        0, 0, z, 0, 0, thickness, motor.inner_radius, angle=magnet_angle
                    )
                    cut_out, _ = occ.cut([(3, outer)], [(3, inner)])
                    occ.rotate(cut_out, 0, 0, 0, 0, 0, 1, rotate_by)
                    magnet_tags.append(cut_out[0][1])
                air_parts, _ = occ.cut([(3, full)], [(3, t) for t in magnet_tags], removeTool=False)
                for dim_tag in air_parts:
                    volumes.append(("AIR", dim_tag[1]))
                for k, t in enumerate(magnet_tags):
                    volumes.append((f"MAGNET_{'N' if k % 2 == 0 else 'S'}", t))
            else:
                volumes.append((name, annular_sector(z, thickness)))
            z += thickness

        occ.fragment([(3, t) for _, t in volumes], [])
        occ.synchronize()

        groups: dict[str, list[int]] = {}
        for name, tag in volumes:
            groups.setdefault(name, []).append(tag)
        for name, tags in groups.items():
            ptag = gmsh.model.addPhysicalGroup(3, tags)
            gmsh.model.setPhysicalName(3, ptag, name)

        size = mesh_size_factor * min(g, t_m)
        gmsh.option.setNumber("Mesh.MeshSizeMin", size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 8.0 * size)
        gmsh.model.mesh.generate(3)
        gmsh.write(str(path))
    return path
