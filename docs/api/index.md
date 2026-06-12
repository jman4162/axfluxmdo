# API reference

The package keeps heavy optional dependencies out of import paths: `import
axfluxmdo` (and `axfluxmdo.optimize` / `axfluxmdo.solvers` / `axfluxmdo.viz`)
never imports pymoo, OpenMDAO, scikit-learn, gmsh, or PyVista. Names that
need an optional backend resolve lazily (PEP 562) and raise a helpful
`ImportError` naming the extra to install:

| Extra | Enables | Backends |
| ----- | ------- | -------- |
| *(core)* | analytical + annular models, sweeps, 2D plots | numpy, matplotlib |
| `[opt]` | `optimize_pareto`, OpenMDAO component, surrogates, Bayesian optimization | pymoo, OpenMDAO, scikit-learn, scipy |
| `[fea]` | mesh export, GetDP pipeline | gmsh (+ external `getdp` binary) |
| `[viz3d]` | 3D assembly rendering and animations | PyVista |

!!! note "Stable result keys"
    `AnalyticalResult.to_dict()` keys (and their aliases, e.g.
    `torque_density` → `torque_density_nm_kg`) are a stable interface. The
    optimization grammar (`"maximize_torque_density"`,
    `"winding_temp_c < 140"`) parses against them.

