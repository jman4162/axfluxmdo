# axfluxmdo

> Open-source Python toolkit for parametric modeling, simulation, visualization, and
> multidisciplinary design optimization of axial-flux permanent-magnet motors.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

`axfluxmdo` is a **reusable science layer** that sits above expert motor designers and
high-fidelity FEA — not a replacement for either. It provides parametric axial-flux
geometry, fast analytical physics models, constraint visualization, and (in later phases)
solver automation and Pareto-front optimization, so that design tradeoffs can be explored
systematically instead of by intuition-only iteration.

## Install

```bash
git clone https://github.com/jman4162/axfluxmdo.git
cd axfluxmdo
pip install -e ".[dev]"
```

## Quickstart

```python
from axfluxmdo import AxialFluxMotor, OperatingPoint
from axfluxmdo.models import AnalyticalModel
from axfluxmdo.viz import plot_geometry

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

plot_geometry(motor)
```

## Roadmap

| Phase | Scope | Status |
| ----- | ----- | ------ |
| 1 | Analytical workbench: parametric motor, torque/EMF/losses/thermal RC, constraints, pole-pair sweep, geometry viz | ✅ |
| 2 | 2.5D annular slice model: radius-dependent flux/loading/losses, air-gap & runout sensitivity, efficiency maps | — |
| 3 | MDO: OpenMDAO components, pymoo Pareto optimization, sensitivity analysis | — |
| 4 | External solver integration: Gmsh export, GetDP/Elmer pipelines, sim-to-analytical residuals | — |
| 5 | Surrogates & Bayesian optimization for expensive design loops | — |

## Non-goals

- No custom FEM solver — geometry/mesh export targets open tools (Gmsh, GetDP/ONELAB, Elmer).
- No full transient 3D EM FEA / CFD / structural / inverter-switching simulation in v1.
- No dependencies on proprietary tools (Motor-CAD, Ansys Maxwell, COMSOL).
- Torque density is never optimized alone — thermal headroom, ripple, controllability, and
  manufacturability are co-equal objectives.

## License

[MIT](LICENSE)
