import numpy as np
import pytest

gmsh = pytest.importorskip("gmsh")

from axfluxmdo.solvers.gmsh_export import export_3d_sector, export_mesh  # noqa: E402


@pytest.fixture(autouse=True)
def gmsh_clean():
    """No gmsh state leaks between tests, even when an assertion fires mid-session."""
    yield
    if gmsh.isInitialized():
        gmsh.finalize()


EXPECTED_2D_GROUPS = {
    "ROTOR_IRON",
    "MAGNET_N",
    "MAGNET_S",
    "AIR",
    "AIRGAP",
    "STATOR_IRON",
    "PERIODIC_LEFT",
    "PERIODIC_RIGHT",
    "OUTER",
}


def read_groups(path) -> dict[str, tuple[int, int]]:
    """name -> (dim, tag) of all physical groups in a mesh file."""
    gmsh.initialize()
    gmsh.open(str(path))
    out = {
        gmsh.model.getPhysicalName(dim, tag): (dim, tag)
        for dim, tag in gmsh.model.getPhysicalGroups()
    }
    gmsh.finalize()
    return out


class TestExportMesh2D:
    def test_msh_written_and_version_2p2(self, reference_motor, tmp_path):
        path, layout = export_mesh(reference_motor, tmp_path / "m.msh")
        assert path.is_file() and path.stat().st_size > 1000
        head = path.read_text().splitlines()[:2]
        assert head[0] == "$MeshFormat"
        assert head[1].startswith("2.2")

    def test_physical_groups_slotless(self, reference_motor, tmp_path):
        path, layout = export_mesh(reference_motor, tmp_path / "m.msh")
        groups = read_groups(path)
        assert EXPECTED_2D_GROUPS <= set(groups)
        assert "WINDING" not in groups
        assert set(layout.group_tags) == set(groups)

    def test_physical_groups_slotted(self, reference_motor, tmp_path):
        path, layout = export_mesh(reference_motor, tmp_path / "m.msh", slotted=True)
        groups = read_groups(path)
        assert "WINDING" in groups
        assert layout.slotted

    def test_periodic_edges_have_matching_nodes(self, reference_motor, tmp_path):
        """Every node on x=0 has a partner at x=L with the same y — what GetDP's
        Link constraint requires. Collected from global coordinates (the
        per-entity API has includeBoundary quirks on periodic slave curves)."""
        path, layout = export_mesh(reference_motor, tmp_path / "m.msh")
        gmsh.initialize()
        gmsh.open(str(path))
        _tags, coords, _ = gmsh.model.mesh.getNodes()
        gmsh.finalize()
        xyz = np.asarray(coords).reshape(-1, 3)
        tol = 1e-9
        left_ys = np.unique(np.round(xyz[np.abs(xyz[:, 0]) < tol, 1], 9))
        right_ys = np.unique(np.round(xyz[np.abs(xyz[:, 0] - layout.x_span_m) < tol, 1], 9))
        assert left_ys.size > 10  # the boundary is actually populated
        assert left_ys.shape == right_ys.shape
        np.testing.assert_allclose(left_ys, right_ys, atol=1e-9)

    @pytest.mark.parametrize("slotted", [False, True])
    def test_mesh_is_conformal_no_duplicate_nodes(self, reference_motor, tmp_path, slotted):
        """Coincident duplicate nodes mean a cracked interface — which the A_z
        formulation silently treats as an infinite-permeability boundary and
        decouples the regions (this bug made the slotted model behave slotless)."""
        path, _ = export_mesh(reference_motor, tmp_path / "m.msh", slotted=slotted)
        gmsh.initialize()
        gmsh.open(str(path))
        _, coords, _ = gmsh.model.mesh.getNodes()
        gmsh.finalize()
        xyz = np.round(np.asarray(coords).reshape(-1, 3), 12)
        unique = np.unique(xyz, axis=0)
        assert unique.shape[0] == xyz.shape[0], (
            f"{xyz.shape[0] - unique.shape[0]} coincident duplicate nodes (mesh crack)"
        )

    def test_deterministic_node_count(self, reference_motor, tmp_path):
        def node_count(p):
            gmsh.initialize()
            gmsh.open(str(p))
            n = len(gmsh.model.mesh.getNodes()[0])
            gmsh.finalize()
            return n

        p1, _ = export_mesh(reference_motor, tmp_path / "a.msh")
        p2, _ = export_mesh(reference_motor, tmp_path / "b.msh")
        assert node_count(p1) == node_count(p2)

    def test_layout_numbers(self, reference_motor, tmp_path):
        _, layout = export_mesh(reference_motor, tmp_path / "m.msh")
        assert layout.x_span_m == pytest.approx(2 * reference_motor.pole_pitch)
        assert layout.gap_midline_y_m == 0.0
        assert layout.depth_m == pytest.approx(reference_motor.active_length)
        assert layout.y_interfaces_m["gap_top"] == pytest.approx(reference_motor.air_gap / 2)


class TestExport3DSector:
    def test_sector_mesh_and_volumes(self, reference_motor, tmp_path):
        path = export_3d_sector(reference_motor, tmp_path / "sector.msh", mesh_size_factor=2.0)
        assert path.is_file() and path.stat().st_size > 1000
        gmsh.initialize()
        gmsh.open(str(path))
        names = {
            gmsh.model.getPhysicalName(dim, tag) for dim, tag in gmsh.model.getPhysicalGroups(3)
        }
        gmsh.finalize()
        assert {"ROTOR_IRON", "MAGNET_N", "AIRGAP", "WINDING", "STATOR_IRON", "AIR"} <= names

    def test_magnet_volume_sanity(self, reference_motor, tmp_path):
        """Sum of magnet tet volumes in a one-pole sector ~ analytic magnet volume / 2p.

        Computed from the mesh elements (a reopened .msh has no CAD kernel,
        so occ.getMass is unavailable).
        """
        path = export_3d_sector(reference_motor, tmp_path / "sector.msh", mesh_size_factor=2.0)
        gmsh.initialize()
        gmsh.open(str(path))
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        xyz = {t: coords[3 * i : 3 * i + 3] for i, t in enumerate(node_tags)}
        total = 0.0
        for dim, tag in gmsh.model.getPhysicalGroups(3):
            if gmsh.model.getPhysicalName(dim, tag).startswith("MAGNET"):
                for ent in gmsh.model.getEntitiesForPhysicalGroup(dim, tag):
                    etypes, _etags, enodes = gmsh.model.mesh.getElements(3, ent)
                    for etype, nodes in zip(etypes, enodes, strict=True):
                        if etype != 4:  # linear tetrahedra
                            continue
                        quad = np.array(nodes).reshape(-1, 4)
                        for tet in quad:
                            p = np.array([xyz[t] for t in tet])
                            total += (
                                abs(
                                    np.linalg.det(
                                        np.column_stack([p[1] - p[0], p[2] - p[0], p[3] - p[0]])
                                    )
                                )
                                / 6.0
                            )
        gmsh.finalize()
        expected = reference_motor.magnet_volume / (2 * reference_motor.pole_pairs)
        assert total == pytest.approx(expected, rel=0.05)
