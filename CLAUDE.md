# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

Phases 1–2 are implemented: the analytical workbench (`AnalyticalModel`) and the 2.5D annular slice model (`AnnularModel` with `GapImperfections`, ripple/axial-force metrics, `compute_efficiency_map`). Phases 3–5 (MDO, external solvers, surrogates) are not started. `SPEC.md` remains the source of truth for scope, architecture, equations, and roadmap.

Physics invariants enforced by tests (do not break):
- Torque and back-EMF derive from the same flux linkage so `m·E_rms·I_rms == T·ω_m` holds to ~1e-9 relative, and the energy balance `P_in == P_out + P_cu + P_core + P_mech` likewise — in BOTH models, for any imperfections.
- `AnnularModel(n_slices=1)` reproduces `AnalyticalModel` on every `to_dict()` key at 1e-12 relative (perfect gap, wedge magnets), and torque/EMF match at ANY slice count for uniform parameters (the flux-linkage sum is additive; keep `math.fsum`).
- `AnnularResult.to_dict()` is additive-only over `AnalyticalResult` keys (the flat keys are a stable interface for Phase 3's `optimize_pareto`).
- Runout sign: the load line is convex in the gap, so mean torque slightly *increases* with runout (Jensen); the penalties are `torque_ripple_proxy` and axial force. Never "fix" tests toward the intuitive wrong sign.
- `tests/test_regression.py` pins golden values for the reference motor (both models) — if a deliberate physics change shifts them, re-verify by hand (hand calculations are in that file) before updating.

`AnalyticalResult.to_dict()` keys and constraint names (`winding_temp_c`, `electrical_frequency_hz`, ...) are a stable interface — Phase 3's `optimize_pareto` will parse SPEC-style constraint strings against them.

Examples in `examples/` are jupytext-paired: author the `.py` percent script, then regenerate the executed `.ipynb` (`jupytext --to ipynb` + `jupyter nbconvert --execute --inplace`, without MPLBACKEND=Agg so figures render inline). Keep both in sync.

## What this package is (and isn't)

It sits **above** existing tools (a "reusable science layer"), not a custom FEA solver. Three fidelity layers:

1. **Layer 1 — analytical sizing model** (the MVP): closed-form torque, back EMF, copper/core losses, lumped thermal RC, mass, efficiency. Equations are in `SPEC.md`.
2. **Layer 2 — 2.5D annular slice model**: split the disk motor into radial annuli, integrate torque/loss across radius. The "sweet spot" fidelity for MDO.
3. **Layer 3 — external solver hooks**: export geometry/meshes via Gmsh and call GetDP/ONELAB or Elmer. Never write a custom FEM solver.

Explicit non-goals (from SPEC.md "What not to build first"):
- No full transient 3D EM FEA / CFD / structural / inverter-switching simulation in v1.
- ML/Bayesian optimization is **not** the core of v1 — physics first, BO later (Phase 5).
- No dependencies on proprietary tools (Motor-CAD, Ansys Maxwell, COMSOL). Core must work with analytical models and open tooling.
- Never optimize torque density alone — thermal headroom, ripple, controllability, and manufacturability are co-equal objectives.

## Intended architecture

Top-level subpackages planned in `SPEC.md` (see the full tree there):
`geometry/` (parametric rotor/stator/magnet/winding, tolerances), `materials/`, `models/` (analytical, annular 2.5D, losses, thermal RC, inverter, manufacturing proxies), `solvers/` (Gmsh export, GetDP/Elmer runners), `optimize/` (OpenMDAO components, pymoo Pareto runner, BoTorch/skopt, surrogates), `viz/` (geometry, fields, Pareto, sensitivity, PyVista 3D), `validation/` (sim-to-real calibration, metrics), `examples/` (notebooks).

Key user-facing API shape (defined in SPEC.md): `AxialFluxMotor` (parametric design object), `OperatingPoint`, model classes with `.evaluate(motor, op)` returning results with torque/efficiency/temps/constraints, and `optimize_pareto(motor, variables=..., objectives=..., constraints=...)`.

The design-variable table in SPEC.md (geometry, magnetics, windings, thermal, inverter, manufacturing) defines what must be first-class optimization variables — keep new physics parameters compatible with that optimization interface.

## Tooling stack decisions (already made in SPEC.md)

- **OpenMDAO** for system integration / coupled MDO
- **pymoo** for multi-objective Pareto optimization
- **scikit-optimize** first, **BoTorch** later for Bayesian optimization
- **Gmsh** (Python API) for geometry/meshing; **GetDP/ONELAB** or **Elmer** as optional open-source solvers
- **PyVista** for 3D visualization

## Roadmap order

Build Phase 1 (analytical workbench) then Phase 2 (2.5D annular model) as the polished prototype before any MDO (Phase 3), external solvers (Phase 4), or surrogates/BO (Phase 5). Don't pull later-phase features forward.

## Commands

```bash
pip install -e ".[dev]"                  # editable install (venv at .venv/)
pytest                                   # full suite
pytest tests/test_analytical.py          # one file
pytest tests/test_analytical.py::TestExactIdentities::test_energy_balance   # one test
ruff check . && ruff format --check .    # lint + format check (CI enforces both)
MPLBACKEND=Agg python examples/01_basic_axial_flux_motor.py   # run an example headless
```
