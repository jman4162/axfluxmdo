import subprocess
import sys

import pytest

from axfluxmdo import OperatingPoint
from axfluxmdo.models.analytical import MAGNET_TEMP_RISE_C
from axfluxmdo.solvers import GETDP_ENV_VAR, SolverError, find_getdp
from axfluxmdo.solvers.getdp_runner import _DEFAULT_AMBIENT_C


class TestLazyImport:
    def test_solvers_import_does_not_pull_gmsh(self):
        """Run in a clean interpreter: importing the solvers package must not import gmsh."""
        code = (
            "import sys; import axfluxmdo.solvers; "
            "assert 'gmsh' not in sys.modules, 'gmsh imported eagerly'"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        assert proc.returncode == 0, proc.stderr


class TestFindGetdp:
    def test_env_override_used(self, tmp_path, monkeypatch):
        fake = tmp_path / "getdp"
        fake.write_text("#!/bin/sh\n")
        fake.chmod(0o755)
        monkeypatch.setenv(GETDP_ENV_VAR, str(fake))
        assert find_getdp() == str(fake)

    def test_env_override_missing_is_loud(self, monkeypatch):
        monkeypatch.setenv(GETDP_ENV_VAR, "/nonexistent/getdp")
        with pytest.raises(SolverError, match="does not point"):
            find_getdp()

    def test_no_env_no_path_returns_none(self, monkeypatch, tmp_path):
        monkeypatch.delenv(GETDP_ENV_VAR, raising=False)
        monkeypatch.setenv("PATH", str(tmp_path))  # empty dir: no getdp
        assert find_getdp() is None


class TestTemperatureConsistency:
    def test_default_magnet_temp_matches_analytical_model(self):
        """The default solve must evaluate magnets at the same temperature the
        analytical model uses for the default operating point."""
        assert _DEFAULT_AMBIENT_C == OperatingPoint(speed_rpm=1, current_rms=0).ambient_temp_c
        assert _DEFAULT_AMBIENT_C + MAGNET_TEMP_RISE_C == 65.0
