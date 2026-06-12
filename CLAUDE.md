# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

ALL FIVE SPEC phases are implemented, plus PyVista 3D visualization (v0.6.0, the SPEC's viz/pyvista_3d.py slot): the analytical workbench (`AnalyticalModel`), the 2.5D annular slice model (`AnnularModel` with `GapImperfections`, ripple/axial-force metrics, `compute_efficiency_map`), the MDO layer (`optimize_pareto` via pymoo, `DesignProblem`, `compute_sensitivities`, OpenMDAO `MotorComponent`), external solver hooks (Gmsh 2D-unrolled/3D-sector export, GetDP open-circuit magnetostatics pipeline, `validation/sim2real.py` residuals; Elmer deferred by user decision), and surrogates + Bayesian optimization (`DesignDataset`, `GPSurrogate`/`RandomForestSurrogate`, `bayesian_optimize`/`BOStudy`). `SPEC.md` remains the historical design doc; where decisions superseded it (Elmer deferral; **surrogate backend = scikit-learn GP + hand-rolled EI**, not the SPEC's "scikit-optimize first, BoTorch later" — both remain future alternative backends), this file wins.

Surrogate/BO invariants:
- sklearn/scipy live in `[opt]` and are NEVER imported by `import axfluxmdo` or `import axfluxmdo.optimize` (PEP 562, test-enforced in `tests/test_import_hygiene.py`). `optimize/dataset.py` is numpy-only and eagerly exported.
- BO determinism-by-seed is test-pinned (same seed → identical trajectory).
- `DesignDataset` JSONL header `axfluxmdo-dataset-v1` is a stable on-disk format.
- `BOStudy` reports human-sign objective values (extends the `ParetoStudy.F` invariant); minimize-space lives only inside `bayesopt.py`.
- Surrogate training uses SOFT penalties for infeasible points (worst feasible + 10% of range) — never feed `PENALTY_OBJECTIVE=1e9` into the GP.
- GP kernel must keep per-dimension (ARD) length scales — the ordinal feature encoding mixes scales.

Release/PyPI invariants (v0.7.0):
- README image links must stay ABSOLUTE (raw.githubusercontent.com) — relative paths break on the PyPI page.
- `src/axfluxmdo/py.typed` ships in the wheel (PEP 561); keep it.
- Publishing: tag `v*` triggers `.github/workflows/release.yml` (trusted publishing, no tokens). One-time setup John must do on pypi.org (and test.pypi.org for dry runs): add trusted publisher for project axfluxmdo / repo jman4162/axfluxmdo / workflow release.yml. `workflow_dispatch` with target=testpypi does a dry run.
- sdist deliberately includes examples/docs (~8.5 MB, portfolio value); it excludes `.venv*` and `.github` — iCloud can drop `".venv 2"`-style duplicate dirs into the tree, which the gitignore (`.venv*/`) and sdist excludes both guard against.
- `carter_factor` defaults to 1.0 on both models — all golden values unchanged; the knob exists so measured FEA corrections can be fed back.
- CHANGELOG.md is keep-a-changelog; add an entry per release.
- At each release, sync the version in THREE places: pyproject.toml, the README Citation BibTeX block, and CITATION.cff (plus its date-released).

Future-work decisions (2026-06-12, don't re-litigate without new information):
- Elmer integration: assessed and declined as the next step — it would duplicate the GetDP magnetostatics validation with a second runner/format/parser for zero new science, and its distribution is heavier (no simple macOS standalone). Revisit ONLY when the package wants coupled multiphysics FEA: thermal conduction replacing the lumped RC, or rotor-disk structural deflection under the ~5.6 kN single-gap axial pull (that analysis would be genuinely new capability and is the natural Elmer demo).
- Higher-impact next candidates, in rough order: double-gap topology support (top documented limitation; YASA/TORUS), saturation/nonlinear BH, winding-factor calculator, winding-inductance estimate (fixes the voltage constraint's neglected I·X_L).

Docs site (MkDocs Material → GitHub Pages at https://jman4162.github.io/axfluxmdo/):
- `mkdocs.yml` at root, `docs_dir: docs` — `docs/images/` doubles as README assets (README image URLs stay ABSOLUTE for PyPI; site pages use relative `images/...`).
- `docs/examples/*.ipynb` are BUILD-TIME COPIES of `examples/*.ipynb` (gitignored, never edit) — regenerate notebooks via the jupytext flow and the site picks them up. Local build: `mkdir -p docs/examples && cp examples/*.ipynb docs/examples/ && mkdocs build --strict`.
- `mkdocs build --strict` must pass; deploy = `.github/workflows/docs.yml` on push to main (paths-filtered).
- mkdocstrings uses griffe STATIC analysis — `[opt]/[fea]/[viz3d]` are NOT needed for docs builds; API pages target leaf modules (PEP 562 packages don't expand statically).
- `docs/limitations.md` is the canonical limitations doc (README keeps the summary); `docs/changelog.md` is a snippets include of CHANGELOG.md.
- Guide pages carry the LaTeX derivations; docstrings keep plain-text math (terminal help() readability) — don't move equations from guides into docstrings or vice versa.

3D-viz invariants:
- pyvista/imageio live in the `[viz3d]` extra; `import axfluxmdo.viz` never imports pyvista/vtk (PEP 562, test-enforced).
- Animations are GIF-only (no imageio-ffmpeg/MP4).
- The cutaway wedge applies to STATOR-side parts only (never slice the rotating parts).
- Committed renders (docs/images/08_*.png/.gif) are regenerated locally on macOS via `python examples/08_3d_animation.py`; CI only smoke-runs the example under xvfb. Rendering tests skip without a GL context (`_can_render()` subprocess probe — VTK segfaults rather than raises without GL).

Solver-layer invariants:
- `import axfluxmdo.solvers` never imports gmsh (test-enforced; PEP 562 pattern). Every gmsh session goes through `_gmsh_session` (try/finally finalize); paths absolute; .msh files written as MSH 2.2 (GetDP compatibility).
- GetDP tests SKIP (never fail) without the binary; `AXFLUXMDO_GETDP` overrides PATH and fails loudly if set wrong. The advisory `getdp-pipeline` CI job runs the live solver with a pinned GetDP and uploads gap-field tables as artifacts.
- **Measured fringing facts (GetDP 3.5.0, reference motor): the 1D load line is an UPPER bound — FEA midline under-magnet mean is ~11% below B_g, fundamental ~7% below B1** (inter-magnet leakage + gap fringing). The live tests assert these bands; do not "tighten" them back toward zero residual.
- `GapFieldSolution.mean_b_t` is the UNDER-MAGNET mean (load-line semantics); the full-pitch mean is a separate property. B1 uses a trapezoid Fourier projection (never a naive FFT — the OnLine sample duplicates the x=0/L endpoint).
- Golden tables in `examples/data/` carry provenance headers; regenerate only from a documented GetDP run (the CI artifact path).
- Magnet-temperature chain: solver default = 25 °C ambient + `MAGNET_TEMP_RISE_C` = 65 °C, pinned to `OperatingPoint` defaults by test; `compare_open_circuit` requires `magnet_temp_c` explicitly.

Local environment note: the dev venv lives at `~/.venvs/axfluxmdo` (NOT inside the repo — `~/Documents` is iCloud-synced, and iCloud sets the macOS hidden flag on files, which makes Python ≥3.12 silently skip editable-install `.pth` files).

Optimization-layer invariants:
- pymoo and OpenMDAO live in the `[opt]` extra and are NEVER imported at `axfluxmdo.optimize` import time (pymoo lazily inside `optimize_pareto`; OpenMDAO names via PEP 562 `__getattr__`). A test/check: `import axfluxmdo.optimize` must not put `pymoo` or `openmdao` in `sys.modules`.
- The alias map (`ALIASES`/`resolve_key` in `optimize/problem.py`) is the single source for short-name resolution — viz and sensitivity import it; never re-declare aliases.
- Every `ParetoStudy` point is feasible (user constraints AND `result.feasible`); `enforce_model_constraints=False` is the only sanctioned escape.
- `ParetoStudy.F` stores human-readable (un-negated) objective values; the minimize-space sign handling lives only in `EvalRecord.f_min`.
- Invalid design vectors (geometry `ValueError`) are penalized with large finite values in `DesignProblem.evaluate`, never raised to the optimizer.

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
pip install -e ".[dev,opt,fea]"          # venv at ~/.venvs/axfluxmdo (see note above); fea = gmsh
pytest                                   # full suite
pytest tests/test_analytical.py          # one file
pytest tests/test_analytical.py::TestExactIdentities::test_energy_balance   # one test
ruff check . && ruff format --check .    # lint + format check (CI enforces both)
MPLBACKEND=Agg python examples/01_basic_axial_flux_motor.py   # run an example headless
```
