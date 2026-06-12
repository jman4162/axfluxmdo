"""External solver hooks (Layer 3).

gmsh is optional (``pip install "axfluxmdo[fea]"``) and is never imported at
package-import time — the gmsh-backed names resolve through PEP 562 module
``__getattr__``. GetDP is an external binary found via the AXFLUXMDO_GETDP
environment variable or PATH; solver tests skip cleanly without it.
"""

from axfluxmdo.solvers.getdp_runner import (
    GETDP_ENV_VAR,
    SolverError,
    find_getdp,
    run_getdp,
    solve_open_circuit,
)
from axfluxmdo.solvers.results_parser import GapFieldSolution, parse_table

__all__ = [
    "GETDP_ENV_VAR",
    "GapFieldSolution",
    "Linear2DLayout",
    "SolverError",
    "build_linear_2d_model",
    "export_3d_sector",
    "export_mesh",
    "find_getdp",
    "parse_table",
    "run_getdp",
    "solve_open_circuit",
]

_GMSH_NAMES = {"Linear2DLayout", "build_linear_2d_model", "export_mesh", "export_3d_sector"}


def __getattr__(name: str):
    if name in _GMSH_NAMES:
        from axfluxmdo.solvers import gmsh_export

        return getattr(gmsh_export, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
