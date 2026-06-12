# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

Greenfield. The repository currently contains only `SPEC.md` — the design document for `axfluxmdo`, an open-source Python toolkit for parametric modeling, simulation, visualization, and multidisciplinary design optimization (MDO) of axial-flux permanent-magnet motors. There is no code, package scaffolding, or test infrastructure yet. Read `SPEC.md` in full before doing any implementation work; it is the source of truth for scope, architecture, equations, and roadmap.

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

None yet — no build/test/lint tooling exists. When scaffolding the package, set up commands and document them here (build, test, single-test invocation, lint).
