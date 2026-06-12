I’d recommend building this as an **axial-flux motor MDO workbench**, not as a full custom FEA solver. The highest-ROI package would connect **parametric axial-flux geometry, analytical/2.5D physics models, visualization, external solver hooks, and optimization loops**. That lets you demonstrate exactly the IRG-relevant skillset: applied EM, simulation-to-real thinking, and system design optimization.

## Recommended package concept

**Package name idea:** `axfluxmdo` or `afmdo`

**One-line mission:**

> Open-source Python toolkit for parametric modeling, simulation, visualization, and multidisciplinary design optimization of axial-flux permanent-magnet motors.

The package should sit above existing tools rather than replacing them. Pyleecan already provides an open-source multiphysics design/optimization framework for electrical machines and drives, so your package should learn from that pattern but specialize in axial-flux geometry, air-gap sensitivity, pole-pair tradeoffs, torque-density optimization, and robotics-relevant actuator constraints. ([PYLEECAN][1])

## Core recommendation

Build it in **three fidelity layers**:

### Layer 1 — Fast analytical sizing model

This should be the MVP. It gives instant estimates for torque, back EMF, losses, temperature rise, pole-pair tradeoffs, mass, and efficiency.

Core equations:

[
T \approx \frac{2\pi\sigma_t}{3}(r_o^3-r_i^3)
]

[
\omega_e = p\omega_m
]

[
f_e = \frac{p n_{rpm}}{60}
]

[
P_{cu}=3I_{rms}^2R_s
]

[
P_{core}=k_h f B^\alpha+k_e f^2B^2
]

[
T_{ss}=T_{amb}+P_{loss}R_\theta
]

This lets users quickly explore questions like: “What happens if I increase pole pairs, reduce air gap, increase outer radius, improve fill factor, or raise current density?”

### Layer 2 — 2.5D annular slice model

Axial-flux machines are naturally disk-shaped, so a useful approximation is to split the motor into radial annuli and integrate performance across radius:

[
T \approx \sum_i 2\pi r_i^2 \sigma_t(r_i)\Delta r_i
]

This is probably the sweet spot for an open-source package: much faster than full 3D FEA, more informative than one-shot analytical sizing, and very useful for MDO.

### Layer 3 — External FEA / multiphysics hooks

Do **not** write your own FEM solver initially. Generate geometry, meshes, and solver input files, then call external tools. Gmsh is a strong fit for parametric geometry and meshing because it is an open-source 3D finite-element mesh generator with CAD and post-processing capabilities, and it exposes APIs including Python. ([Gmsh][2])

For open-source solvers, consider optional integrations with GetDP/ONELAB or Elmer. ONELAB bundles Gmsh, GetDP, and optimization tooling in an open interface, while GetDP is an open-source finite-element solver commonly paired with Gmsh. ([ONELAB][3]) Elmer is also relevant because it supports multiphysics PDE solving including heat transfer, structural mechanics, fluid dynamics, and electromagnetics. ([GitHub][4])

## Proposed architecture

```text
axfluxmdo/
  geometry/
    axial_flux.py          # rotor/stator/magnet/winding geometry
    slots.py               # slot and winding layouts
    tolerances.py          # air-gap, runout, bearing imperfection models

  materials/
    magnetic.py            # B-H curves, magnet properties, steel loss coeffs
    thermal.py             # conductivity, heat capacity, cooling params
    electrical.py          # copper, insulation, inverter params

  models/
    analytical.py          # fast sizing equations
    annular_2p5d.py         # radial-slice quasi-3D model
    losses.py              # copper, core, eddy, mechanical losses
    thermal_rc.py           # lumped thermal network
    inverter.py             # voltage/current/frequency constraints
    manufacturing.py        # fill factor, tolerance, cost proxies

  solvers/
    gmsh_export.py          # geometry/mesh export
    getdp_runner.py         # optional GetDP integration
    elmer_runner.py         # optional Elmer integration
    results_parser.py       # parse field/force/loss outputs

  optimize/
    problem.py              # design variables, objectives, constraints
    openmdao_components.py  # coupled MDO components
    pymoo_runner.py         # Pareto optimization
    bayesopt.py             # expensive black-box optimization
    surrogate.py            # GP / ensemble surrogate models

  viz/
    geometry_plot.py        # 2D cross-sections
    fields.py               # B-field / flux / loss maps
    pareto.py               # Pareto fronts
    sensitivity.py          # tornado charts / constraint plots
    pyvista_3d.py           # 3D visualization

  validation/
    test_data.py            # bench/dyno data schemas
    sim2real.py             # calibration and residual analysis
    metrics.py              # RMSE, bias, uncertainty, confidence intervals

  examples/
    01_basic_axial_flux_motor.ipynb
    02_pole_pair_tradeoff.ipynb
    03_torque_density_optimization.ipynb
    04_air_gap_sensitivity.ipynb
    05_gmsh_export.ipynb
```

## Key design variables

Your package should make these first-class optimization variables:

| Category      | Variables                                                                          |
| ------------- | ---------------------------------------------------------------------------------- |
| Geometry      | outer radius, inner radius, air gap, rotor/stator thickness, magnet thickness      |
| Magnetics     | pole pairs, magnet arc ratio, magnet grade, remanence, back-iron thickness         |
| Windings      | turns, phases, fill factor, conductor area, current density, slot layout           |
| Thermal       | cooling coefficient, thermal resistance, winding temperature limit                 |
| Inverter      | DC bus voltage, current limit, switching frequency, max electrical frequency       |
| Manufacturing | air-gap tolerance, runout, magnet placement error, fill-factor limit               |
| Objectives    | torque density, efficiency, continuous torque, ripple proxy, mass, cost            |
| Constraints   | saturation, thermal limit, voltage limit, current limit, stress, manufacturability |

## MDO stack

Use **OpenMDAO** as the main system-integration layer because it is designed for multidisciplinary optimization in Python, supports decomposed coupled models, and emphasizes efficient derivatives and high-fidelity analysis integration. ([OpenMDAO][5])

Use **pymoo** for Pareto-front exploration because it is an open-source Python framework for single- and multi-objective optimization with visualization and decision-making support. ([pymoo][6])

Use **BoTorch** for Bayesian optimization once simulations become expensive because it is modular, built on PyTorch, supports scalable Gaussian processes through GPyTorch, and is intended for efficient Monte Carlo Bayesian optimization. ([BoTorch][7]) For a simpler first pass, `scikit-optimize` is easier; it explicitly supports sequential model-based optimization and Gaussian-process-based `gp_minimize`. ([Scikit-Optimize][8])

For visualization, use **PyVista** for 3D meshes and field plots because it provides Python-native 3D visualization and mesh analysis built around the scientific Python stack. ([PyVista][9])

## Minimum viable product

I would start with a package that can answer five questions well:

1. **Pole-pair tradeoff:** how do torque, electrical frequency, core loss, inverter burden, and ripple proxy change with pole pairs?
2. **Geometry tradeoff:** how do (r_o), (r_i), air gap, and magnet thickness affect torque density?
3. **Thermal limit:** what is peak torque vs continuous torque under winding/core/inverter loss?
4. **Manufacturing sensitivity:** how much does air-gap error, runout, or magnet placement error hurt performance?
5. **Pareto optimization:** what is the frontier across torque density, efficiency, thermal headroom, mass, and manufacturability?

A simple user API could look like:

```python
from axfluxmdo import AxialFluxMotor, OperatingPoint
from axfluxmdo.models import AnalyticalModel, AnnularModel
from axfluxmdo.optimize import optimize_pareto
from axfluxmdo.viz import plot_geometry, plot_pareto

motor = AxialFluxMotor(
    outer_radius=0.08,
    inner_radius=0.025,
    air_gap=0.0008,
    pole_pairs=14,
    phases=3,
    turns_per_phase=24,
    fill_factor=0.45,
    magnet_thickness=0.004,
    back_iron_thickness=0.006,
)

op = OperatingPoint(speed_rpm=500, current_rms=25, dc_bus_voltage=48)

result = AnalyticalModel().evaluate(motor, op)

print(result.torque_nm)
print(result.efficiency)
print(result.winding_temp_c)
print(result.constraints)

study = optimize_pareto(
    motor,
    variables={
        "outer_radius": (0.05, 0.12),
        "pole_pairs": [8, 10, 12, 14, 16, 18, 20],
        "air_gap": (0.0005, 0.0015),
        "fill_factor": (0.30, 0.60),
    },
    objectives=["maximize_torque_density", "maximize_efficiency", "minimize_mass"],
    constraints=["winding_temp_c < 140", "electrical_frequency_hz < 1000"],
)

plot_pareto(study, x="torque_density", y="efficiency", color="winding_temp_c")
```

## What not to build first

Do **not** start with full transient 3D EM FEA, CFD cooling, structural deformation, and inverter switching simulation in one package. That is too ambitious and will likely stall.

Do **not** make ML the core of v1. For this domain, the credible package starts with physics and uses ML/BO later for expensive design loops.

Do **not** make it depend on proprietary tools like Motor-CAD, Ansys Maxwell, or COMSOL. You can add optional adapters later, but the core open-source story should work with analytical models and open tooling.

Do **not** optimize only torque density. A robotics motor that wins on torque density but loses on thermal headroom, torque ripple, controllability, or manufacturability is not actually a better actuator.

## Recommended roadmap

**Phase 1: Analytical workbench**

* Parametric axial-flux motor object.
* Torque, back EMF, electrical frequency, copper loss, core loss, mass, efficiency.
* Thermal RC model.
* Pole-pair sweep.
* Geometry visualization.
* Basic constraints.

**Phase 2: 2.5D annular model**

* Radial segmentation.
* Radius-dependent flux, current loading, shear stress, losses.
* Air-gap/runout sensitivity.
* Magnet arc and pole-pitch effects.
* Efficiency map over speed/torque.

**Phase 3: MDO**

* OpenMDAO components.
* pymoo Pareto optimization.
* Constraint handling.
* Sensitivity analysis.
* Pareto-front dashboard.

**Phase 4: External solver integration**

* Gmsh geometry export.
* Optional GetDP/Elmer solver pipeline.
* Field map import.
* Simulation-to-analytical residual analysis.

**Phase 5: Surrogate and Bayesian optimization**

* Dataset of evaluated designs.
* Gaussian-process / ensemble surrogate.
* Bayesian optimization for expensive FEA calls.
* Uncertainty-aware design recommendations.

## Why this would be useful for IRG-style work

This package would demonstrate the exact mechanism Yuri seemed to care about: **start from physics, model the actuator as a coupled EM/thermal/mechanical/control system, visualize the constraints, and optimize design tradeoffs instead of doing brute-force intuition-only iteration.**

The strongest positioning would be:

> “I would not try to replace expert motor designers or high-fidelity FEA. I would build the reusable science layer around them: parametric design, fast physics models, solver automation, validation against hardware, Pareto-front analysis, and uncertainty-aware optimization. For axial-flux robotic actuators, that could help identify which constraints are actually binding: torque density, efficiency, thermal headroom, pole-pair electrical frequency, torque ripple, air-gap tolerance, bearing/runout imperfections, or manufacturability.”

My recommendation: build **Phase 1 + Phase 2 first** as a polished open-source prototype. That is enough to be impressive, technically grounded, and directly relevant without getting trapped trying to build an industrial-grade motor solver.

[1]: https://www.pyleecan.org/?utm_source=chatgpt.com "PYLEECAN — PYthon Library for Electrical Engineering ..."
[2]: https://gmsh.info/?utm_source=chatgpt.com "Gmsh: a three-dimensional finite element mesh generator with ..."
[3]: https://onelab.info/?utm_source=chatgpt.com "ONELAB: Open Numerical Engineering LABoratory"
[4]: https://github.com/elmercsc/elmerfem?utm_source=chatgpt.com "ElmerCSC/elmerfem: Official git repository of Elmer FEM ..."
[5]: https://openmdao.org/?utm_source=chatgpt.com "OpenMDAO.org | An open-source framework for efficient ..."
[6]: https://pymoo.org/?utm_source=chatgpt.com "pymoo: Multi-objective Optimization in Python — pymoo: Multi ..."
[7]: https://botorch.org/?utm_source=chatgpt.com "BoTorch"
[8]: https://scikit-optimize.github.io/?utm_source=chatgpt.com "scikit-optimize: sequential model-based optimization in Python ..."
[9]: https://pyvista.org/?utm_source=chatgpt.com "PyVista | 3D plotting & analysis made easy"
