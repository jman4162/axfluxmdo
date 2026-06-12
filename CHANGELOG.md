# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
semantic versioning (0.x: minor bumps may change APIs).

## [0.7.0] — 2026-06-12

First PyPI release. Hardening pass driven by a three-track review (science,
packaging, code quality).

### Added
- `carter_factor` parameter on `AnalyticalModel` and `AnnularModel` — feed a
  measured Carter factor (e.g. from `validation.measured_carter_factor`) back
  into the load line.
- Public `axfluxmdo.viz.can_render()` (was private), `py.typed` marker
  (PEP 561), `UnknownKeyError` for alias-resolution failures.
- "Model fidelity & known limitations" README section consolidating the
  documented simplifications (single-gap axial pull, no saturation, inductive
  drop neglected, magnet-temperature decoupling, omitted loss mechanisms).
- Tag-triggered PyPI release workflow (trusted publishing).
- Python 3.11 added to the CI matrix.

### Fixed
- `plot_surrogate_slice` crashed on non-numeric choice variables; now sweeps
  option indices with labeled ticks.
- `replace_field` rejects multi-level dotted paths with a clear error.
- README image links absolute (render correctly on PyPI).

## [0.6.0] — 2026-06-12

### Added
- PyVista 3D visualization (`viz/pyvista_3d.py`, `[viz3d]` extra): full-360°
  parametric assembly (exact-volume hexahedral sectors), cutaway static view,
  spinning-rotor and exploded-assembly GIF animations; committed renders in
  the README.

## [0.5.0] — 2026-06-12

### Added
- Phase 5 (final SPEC phase): `DesignDataset` (JSONL persistence),
  `GPSurrogate` (ARD Matérn + CV diagnostics) and `RandomForestSurrogate`,
  `bayesian_optimize`/`BOStudy` with expected-improvement acquisition,
  soft-penalty constraint handling, `expensive_fn` hook for FEA-in-the-loop,
  uncertainty-aware `recommend()`; convergence and surrogate-slice plots.
  Backend decision: scikit-learn GP + hand-rolled EI (BoTorch/skopt remain
  future alternatives).

## [0.4.0] — 2026-06-12

### Added
- Phase 4: Gmsh 2D-unrolled and 3D-sector mesh export, GetDP open-circuit
  magnetostatics pipeline, `validation/sim2real.py` residual analysis and
  measured Carter factor, committed golden FEA tables. Headline finding: the
  1D load line overestimates the gap field (−11% mean / −7% fundamental,
  measured k_C = 1.44). Elmer integration deferred.

### Fixed
- Conformal slotted-stator mesh (a cracked gap interface had made the slotted
  model behave slotless).

## [0.3.0] — 2026-06-12

### Added
- Phase 3 MDO layer: `DesignProblem` (SPEC-style objective/constraint string
  parsing against stable result keys), `optimize_pareto` via pymoo
  mixed-variable GA, `ParetoStudy`, OpenMDAO `MotorComponent` + SLSQP demo,
  one-at-a-time sensitivities with tornado charts, Pareto-front plotting.
  `[opt]` extra.

## [0.2.0] — 2026-06-11

### Added
- Phase 2: 2.5D annular slice model (`AnnularModel`) with machine-precision
  parity to the analytical layer, `GapImperfections` (offset/coning/runout
  with analytic circumferential averages), torque-ripple proxy, axial force,
  constraint-aware efficiency maps, radial-profile plots.

## [0.1.0] — 2026-06-11

### Added
- Phase 1 analytical workbench: parametric `AxialFluxMotor`,
  energy-consistent `AnalyticalModel` (torque/EMF from a shared flux linkage),
  Steinmetz core loss, closed-form thermal RC with runaway detection, named
  constraint records, parameter/pole-pair sweeps, 2D geometry visualization,
  materials library, hand-verified golden regression values.

[0.7.0]: https://github.com/jman4162/axfluxmdo/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/jman4162/axfluxmdo/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/jman4162/axfluxmdo/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/jman4162/axfluxmdo/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/jman4162/axfluxmdo/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/jman4162/axfluxmdo/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/jman4162/axfluxmdo/releases/tag/v0.1.0
