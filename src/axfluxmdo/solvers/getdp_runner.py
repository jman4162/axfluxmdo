"""GetDP solver orchestration.

GetDP (https://getdp.info) is an external binary, never a Python dependency.
Discovery order: the ``AXFLUXMDO_GETDP`` environment variable (must point at
an existing file — a set-but-wrong override fails loudly), then ``getdp`` on
PATH. All solver tests skip cleanly when the binary is absent.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from axfluxmdo.geometry.axial_flux import AxialFluxMotor
from axfluxmdo.models.analytical import MAGNET_TEMP_RISE_C
from axfluxmdo.solvers.results_parser import GapFieldSolution, parse_table

# Must match OperatingPoint's default ambient (pinned by a test) so that the
# default solve compares apples-to-apples with AnalyticalModel.evaluate at the
# default operating point: magnet temp = 25 + 40 = 65 C.
_DEFAULT_AMBIENT_C = 25.0

GETDP_ENV_VAR = "AXFLUXMDO_GETDP"


class SolverError(RuntimeError):
    """GetDP invocation failed; the message carries the output tail."""


def find_getdp() -> str | None:
    """Locate the getdp binary (env var override first, then PATH)."""
    override = os.environ.get(GETDP_ENV_VAR)
    if override:
        path = Path(override).expanduser()
        if not path.is_file():
            raise SolverError(f"{GETDP_ENV_VAR}={override!r} does not point to an existing file")
        return str(path)
    return shutil.which("getdp")


def run_getdp(
    pro_path: str | Path,
    msh_path: str | Path,
    *,
    resolution: str = "Magnetostatics2D",
    post_operation: str = "gap_field",
    table_filename: str = "gap_field.dat",
    workdir: str | Path | None = None,
    timeout: float = 120.0,
) -> Path:
    """Run getdp on a rendered .pro + .msh; return the output table path."""
    getdp = find_getdp()
    if getdp is None:
        raise SolverError(
            "getdp binary not found; install GetDP (e.g. the ONELAB bundle from "
            f"https://onelab.info) and/or set {GETDP_ENV_VAR}=/path/to/getdp"
        )
    pro_path = Path(pro_path).resolve()
    msh_path = Path(msh_path).resolve()
    cwd = Path(workdir).resolve() if workdir is not None else pro_path.parent
    cmd = [
        getdp,
        str(pro_path),
        "-msh",
        str(msh_path),
        "-solve",
        resolution,
        "-pos",
        post_operation,
        "-v",
        "2",
    ]
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise SolverError(f"getdp timed out after {timeout}s: {' '.join(cmd)}") from exc
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout + "\n" + proc.stderr).splitlines()[-30:])
        raise SolverError(f"getdp failed (exit {proc.returncode}):\n{tail}")
    table = cwd / table_filename
    if not table.is_file():
        tail = "\n".join((proc.stdout + "\n" + proc.stderr).splitlines()[-30:])
        raise SolverError(f"getdp produced no output table at {table}:\n{tail}")
    return table


def solve_open_circuit(
    motor: AxialFluxMotor,
    workdir: str | Path | None = None,
    *,
    slotted: bool = False,
    n_samples: int = 720,
    magnet_temp_c: float | None = None,
    mesh_size_factor: float = 1.0,
) -> GapFieldSolution:
    """One-call pipeline: mesh export -> .pro render -> getdp -> parsed gap field.

    magnet_temp_c defaults to ambient (25 C) + MAGNET_TEMP_RISE_C = 65 C —
    exactly the magnet temperature AnalyticalModel uses at the default
    operating point, so default residual comparisons are apples-to-apples.
    """
    from axfluxmdo.solvers.getdp_templates import render_open_circuit_pro
    from axfluxmdo.solvers.gmsh_export import export_mesh

    if magnet_temp_c is None:
        magnet_temp_c = _DEFAULT_AMBIENT_C + MAGNET_TEMP_RISE_C

    if workdir is None:
        with tempfile.TemporaryDirectory(prefix="axfluxmdo_getdp_") as tmp:
            return solve_open_circuit(
                motor,
                tmp,
                slotted=slotted,
                n_samples=n_samples,
                magnet_temp_c=magnet_temp_c,
                mesh_size_factor=mesh_size_factor,
            )

    wd = Path(workdir).resolve()
    wd.mkdir(parents=True, exist_ok=True)
    msh_path, layout = export_mesh(
        motor, wd / "model.msh", slotted=slotted, mesh_size_factor=mesh_size_factor
    )
    pro_text = render_open_circuit_pro(
        motor, layout, magnet_temp_c=magnet_temp_c, n_samples=n_samples
    )
    pro_path = wd / "model.pro"
    pro_path.write_text(pro_text)
    table = run_getdp(pro_path, msh_path, workdir=wd)
    cols = parse_table(table)
    return GapFieldSolution(
        x_m=cols[:, 0],
        by_t=cols[:, 4],
        pole_pitch_m=motor.pole_pitch,
        magnet_arc_ratio=motor.magnet_arc_ratio,
        magnet_temp_c=magnet_temp_c,
        slotted=slotted,
    )
