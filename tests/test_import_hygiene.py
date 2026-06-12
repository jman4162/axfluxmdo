"""Optional heavy dependencies must never load at package-import time."""

import subprocess
import sys

import pytest


def _assert_clean_import(module: str, forbidden: list[str]):
    checks = "; ".join(
        f"assert not any(m == {name!r} or m.startswith({name + '.'!r}) "
        f"for m in sys.modules), {name!r} + ' imported eagerly'"
        for name in forbidden
    )
    code = f"import sys; import {module}; {checks}"
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


@pytest.mark.parametrize(
    "module,forbidden",
    [
        ("axfluxmdo", ["pymoo", "openmdao", "sklearn", "gmsh", "pyvista", "vtk"]),
        ("axfluxmdo.optimize", ["pymoo", "openmdao", "sklearn"]),
        ("axfluxmdo.solvers", ["gmsh"]),
        ("axfluxmdo.viz", ["pyvista", "vtk"]),
    ],
)
def test_no_eager_optional_imports(module, forbidden):
    _assert_clean_import(module, forbidden)
