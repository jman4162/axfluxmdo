import dataclasses
import math

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from axfluxmdo.viz.pyvista_3d import (  # noqa: E402
    _annular_sector,
    _can_render,
    animate_rotation,
    build_motor_assembly,
    plot_motor_3d,
)

EXPECTED_KEYS = {"rotor_iron", "magnets", "stator_teeth", "stator_coils", "stator_yoke"}


@pytest.fixture
def reference_motor_local():
    from axfluxmdo import AxialFluxMotor

    return AxialFluxMotor(outer_radius=0.08, inner_radius=0.025, air_gap=0.0008, pole_pairs=14)


class TestAnnularSector:
    def test_known_ring_volume(self):
        """The watch-out: StructuredGrid dimensions vs ravel order."""
        ring = _annular_sector(0.5, 1.0, 0.0, 2.0, 0.0, 2 * math.pi)
        expected = math.pi * (1.0**2 - 0.5**2) * 2.0
        assert ring.volume == pytest.approx(expected, rel=2e-3)

    def test_quarter_sector_volume(self):
        sector = _annular_sector(0.5, 1.0, 0.0, 1.0, 0.0, math.pi / 2)
        expected = 0.25 * math.pi * (1.0 - 0.25) * 1.0
        assert sector.volume == pytest.approx(expected, rel=2e-3)


class TestAssemblyGeometry:
    def test_keys(self, reference_motor_local):
        assembly = build_motor_assembly(reference_motor_local)
        assert set(assembly) == EXPECTED_KEYS

    def test_magnet_count(self, reference_motor_local):
        assembly = build_motor_assembly(reference_motor_local)
        assert assembly["magnets"].n_blocks == 2 * reference_motor_local.pole_pairs

    def test_component_volumes_match_motor_properties(self, reference_motor_local):
        m = reference_motor_local
        assembly = build_motor_assembly(m, theta_cutaway_deg=None)
        assert assembly["rotor_iron"].volume == pytest.approx(m.back_iron_volume, rel=0.02)
        magnet_total = sum(b.volume for b in assembly["magnets"])
        assert magnet_total == pytest.approx(m.magnet_volume, rel=0.02)
        assert assembly["stator_yoke"].volume == pytest.approx(m.stator_core_volume, rel=0.02)
        coil_expected = m.airgap_area * m.slot_depth * m.slot_width_fraction
        teeth_expected = m.airgap_area * m.slot_depth * (1 - m.slot_width_fraction)
        assert assembly["stator_coils"].volume == pytest.approx(coil_expected, rel=0.02)
        assert assembly["stator_teeth"].volume == pytest.approx(teeth_expected, rel=0.02)

    def test_cutaway_reduces_stator_only(self, reference_motor_local):
        full = build_motor_assembly(reference_motor_local, theta_cutaway_deg=None)
        cut = build_motor_assembly(reference_motor_local, theta_cutaway_deg=90.0)
        assert cut["stator_yoke"].volume == pytest.approx(
            0.75 * full["stator_yoke"].volume, rel=0.02
        )
        assert cut["rotor_iron"].volume == pytest.approx(full["rotor_iron"].volume, rel=1e-6)

    def test_bounds(self, reference_motor_local):
        m = reference_motor_local
        assembly = build_motor_assembly(m, theta_cutaway_deg=None)
        all_bounds = np.array([assembly[k].bounds for k in ("rotor_iron", "stator_yoke")])
        assert all_bounds[:, 1].max() == pytest.approx(m.outer_radius, abs=1e-12)
        z_bottom = -m.air_gap / 2 - m.magnet_thickness - m.back_iron_thickness
        z_top = m.air_gap / 2 + m.slot_depth + m.stator_core_thickness
        assert assembly["rotor_iron"].bounds[4] == pytest.approx(z_bottom, abs=1e-12)
        assert assembly["stator_yoke"].bounds[5] == pytest.approx(z_top, abs=1e-12)

    def test_rotor_angle_rotates_magnets_not_stator(self, reference_motor_local):
        angle = 0.21
        base = build_motor_assembly(reference_motor_local, rotor_angle_rad=0.0)
        rotated = build_motor_assembly(reference_motor_local, rotor_angle_rad=angle)

        def centroid_angle(mesh):
            c = mesh.points.mean(axis=0)  # point mean (mesh.center is the bounds center)
            return math.atan2(c[1], c[0])

        shift = centroid_angle(rotated["magnets"][0]) - centroid_angle(base["magnets"][0])
        assert shift == pytest.approx(angle, abs=1e-9)
        np.testing.assert_allclose(
            rotated["stator_yoke"].points.mean(axis=0),
            base["stator_yoke"].points.mean(axis=0),
            atol=1e-15,
        )

    def test_rectangular_magnet_shape(self, reference_motor_local):
        m = dataclasses.replace(reference_motor_local, magnet_shape="rectangular")
        assembly = build_motor_assembly(m)
        block = assembly["magnets"][0]
        assert assembly["magnets"].n_blocks == 2 * m.pole_pairs
        # width across the pole centerline equals the mean-radius arc width
        expected_width = m.magnet_arc_ratio * m.pole_pitch
        pts = block.points
        center = (0.5) * math.pi / m.pole_pairs
        across = -pts[:, 0] * math.sin(center) + pts[:, 1] * math.cos(center)
        assert np.ptp(across) == pytest.approx(expected_width, rel=1e-9)

    def test_gif_suffix_required(self, reference_motor_local, tmp_path):
        with pytest.raises(ValueError, match="GIF-only"):
            animate_rotation(reference_motor_local, tmp_path / "out.mp4")


needs_render = pytest.mark.skipif(
    not _can_render(), reason="no usable GL context for VTK rendering"
)


@needs_render
class TestRendering:
    def test_screenshot_written(self, reference_motor_local, tmp_path):
        target = tmp_path / "motor.png"
        plotter = plot_motor_3d(reference_motor_local, screenshot=target)
        plotter.close()
        assert target.is_file() and target.stat().st_size > 0
        assert target.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    def test_rotation_gif(self, reference_motor_local, tmp_path):
        from PIL import Image

        target = animate_rotation(
            reference_motor_local, tmp_path / "spin.gif", n_frames=8, window_size=(160, 120)
        )
        assert target.read_bytes()[:4] == b"GIF8"
        assert Image.open(target).n_frames == 8

    def test_exploded_gif(self, reference_motor_local, tmp_path):
        from axfluxmdo.viz.pyvista_3d import animate_exploded

        target = animate_exploded(
            reference_motor_local, tmp_path / "explode.gif", n_frames=6, window_size=(160, 120)
        )
        assert target.is_file()
        assert target.read_bytes()[:4] == b"GIF8"
