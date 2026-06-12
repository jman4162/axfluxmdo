from pathlib import Path

import numpy as np
import pytest

from axfluxmdo.materials import N42, airgap_flux_density
from axfluxmdo.solvers import find_getdp, parse_table

DATA_DIR = Path(__file__).parent / "data"

HAS_GMSH = True
try:
    import gmsh  # noqa: F401
except ImportError:
    HAS_GMSH = False

needs_solver = pytest.mark.skipif(
    find_getdp() is None or not HAS_GMSH,
    reason="getdp binary and/or gmsh not available",
)


class TestRenderPro:
    @pytest.fixture
    def rendered(self, reference_motor, tmp_path):
        pytest.importorskip("gmsh")
        from axfluxmdo.solvers.getdp_templates import render_open_circuit_pro
        from axfluxmdo.solvers.gmsh_export import export_mesh

        _, layout = export_mesh(reference_motor, tmp_path / "m.msh")
        return render_open_circuit_pro(reference_motor, layout, magnet_temp_c=65.0)

    def test_contains_motor_numbers(self, rendered, reference_motor):
        br_65 = N42.remanence_at(65.0)
        assert repr(br_65) in rendered  # ~1.2298
        assert repr(reference_motor.magnet.mu_r) in rendered
        assert repr(reference_motor.steel.mu_r) in rendered
        assert repr(2 * reference_motor.pole_pitch) in rendered
        assert "720" in rendered

    def test_no_unsubstituted_placeholders(self, rendered):
        import re

        # $ followed by an identifier means a missed substitution
        assert not re.search(r"\$[A-Za-z_]", rendered), "unsubstituted $placeholder"

    def test_slotless_has_no_winding_region(self, rendered):
        assert "WINDING" not in rendered

    def test_slotted_includes_winding(self, reference_motor, tmp_path):
        pytest.importorskip("gmsh")
        from axfluxmdo.solvers.getdp_templates import render_open_circuit_pro
        from axfluxmdo.solvers.gmsh_export import export_mesh

        _, layout = export_mesh(reference_motor, tmp_path / "m.msh", slotted=True)
        pro = render_open_circuit_pro(reference_motor, layout, magnet_temp_c=65.0)
        assert "WINDING     = Region[" in pro
        assert "AIR, WINDING}" in pro  # winding included in the non-magnetic region


class TestParseTable:
    def test_fixture_parses(self):
        cols = parse_table(DATA_DIR / "gap_field_table_sample.dat")
        assert cols.shape == (11, 6)
        assert np.all(np.diff(cols[:, 0]) > 0)  # sorted by x
        assert cols[0, 4] == pytest.approx(1.017)

    def test_rejects_varying_y(self, tmp_path):
        bad = tmp_path / "bad.dat"
        bad.write_text("1 0.0 0.0 0.0 0.0 1.0 0.0\n1 0.1 0.5 0.0 0.0 1.0 0.0\n")
        with pytest.raises(ValueError, match="y/z vary"):
            parse_table(bad)

    def test_rejects_too_few_columns(self, tmp_path):
        bad = tmp_path / "bad.dat"
        bad.write_text("1.0 2.0 3.0\n")
        with pytest.raises(ValueError, match="6 columns"):
            parse_table(bad)


@needs_solver
class TestLiveSolve:
    """End-to-end pipeline; runs only where a getdp binary is available."""

    @pytest.fixture(scope="class")
    def slotless(self, request):
        from axfluxmdo.solvers import solve_open_circuit

        motor = request.getfixturevalue("reference_motor")
        return solve_open_circuit(motor, magnet_temp_c=65.0)

    def test_mean_b_within_fringing_band_of_load_line(self, reference_motor):
        """The 1D load line is an UPPER bound: 2D FEA sees inter-magnet leakage
        and circumferential gap fringing it cannot. For the reference motor
        (alpha_m=0.85, inter-magnet gap ~ magnet thickness/2) the measured
        midline under-magnet mean is ~11% below the load line (GetDP 3.5.0)."""
        from axfluxmdo.solvers import solve_open_circuit

        solution = solve_open_circuit(reference_motor, magnet_temp_c=65.0)
        b_g = airgap_flux_density(
            reference_motor.magnet,
            reference_motor.magnet_thickness,
            reference_motor.air_gap,
            65.0,
        )
        assert 0.80 * b_g < solution.mean_b_t < 1.00 * b_g

    def test_fundamental_within_fringing_band_of_b1(self, reference_motor):
        """B1 is flux-weighted and less midline-sensitive than the mean:
        measured ~7% below the analytical fundamental for the reference motor."""
        import math

        from axfluxmdo.solvers import solve_open_circuit

        solution = solve_open_circuit(reference_motor, magnet_temp_c=65.0)
        b_g = airgap_flux_density(
            reference_motor.magnet,
            reference_motor.magnet_thickness,
            reference_motor.air_gap,
            65.0,
        )
        b1 = (4 / math.pi) * b_g * math.sin(reference_motor.magnet_arc_ratio * math.pi / 2)
        assert 0.85 * b1 < solution.fundamental_b1_t < 1.00 * b1

    def test_garbage_pro_raises_solver_error(self, tmp_path, reference_motor):
        pytest.importorskip("gmsh")
        from axfluxmdo.solvers import SolverError, run_getdp
        from axfluxmdo.solvers.gmsh_export import export_mesh

        msh, _ = export_mesh(reference_motor, tmp_path / "m.msh")
        bad_pro = tmp_path / "bad.pro"
        bad_pro.write_text("This is not a valid GetDP problem file.\n")
        with pytest.raises(SolverError):
            run_getdp(bad_pro, msh, workdir=tmp_path)
