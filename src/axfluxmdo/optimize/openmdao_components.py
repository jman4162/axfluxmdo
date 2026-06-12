"""OpenMDAO integration: the motor model as an ExplicitComponent.

pymoo handles multi-objective Pareto exploration; this layer exposes the
motor model to OpenMDAO's gradient-based drivers and to larger coupled MDO
groups (the SPEC's system-integration story). Discrete design variables
(integer fields, choice lists) are frozen at their baseline values here —
gradient drivers cannot move them; use ``optimize_pareto`` for those.

This module imports ``openmdao.api`` at import time (the component must
subclass ``om.ExplicitComponent``), so it is exposed from
``axfluxmdo.optimize`` only through lazy PEP 562 ``__getattr__``.
"""

from __future__ import annotations

import numpy as np

try:
    import openmdao.api as om
except ImportError as exc:  # pragma: no cover - exercised only without [opt]
    raise ImportError(
        "OpenMDAO integration requires openmdao; install with: pip install 'axfluxmdo[opt]'"
    ) from exc

from axfluxmdo.geometry.axial_flux import AxialFluxMotor
from axfluxmdo.models.base import Model
from axfluxmdo.operating_point import OperatingPoint
from axfluxmdo.optimize.problem import DesignProblem

CLAMP = 1e6  # keep SLSQP line searches numeric through thermal runaway etc.


class MotorComponent(om.ExplicitComponent):
    """Motor model wrapped for OpenMDAO with finite-difference partials.

    Inputs: the DesignProblem's continuous variables (baseline values as
    defaults). Outputs: the requested result keys plus one ``g_<name>``
    violation output per constraint (feasible when <= 0).
    """

    def initialize(self):
        self.options.declare("problem", types=DesignProblem)
        self.options.declare(
            "outputs", default=None, desc="to_dict keys to expose (default: objective keys)"
        )

    def setup(self):
        problem: DesignProblem = self.options["problem"]
        for name, (lo, hi) in problem.continuous.items():
            baseline = problem._baseline_value(name)
            self.add_input(_om_name(name), val=float(baseline))
            del lo, hi  # bounds are applied at the driver level
        output_keys = self.options["outputs"] or [obj.name for obj in problem.objectives]
        self._output_keys = list(dict.fromkeys(output_keys))
        for key in self._output_keys:
            self.add_output(key, val=1.0)
        self._constraint_outputs = [f"g_{i}" for i in range(problem.n_constr)]
        for g_name in self._constraint_outputs:
            self.add_output(g_name, val=0.0)

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd", step_calc="rel_avg", step=1e-4)

    def compute(self, inputs, outputs):
        problem: DesignProblem = self.options["problem"]
        x = {name: float(inputs[_om_name(name)][0]) for name in problem.continuous}
        record = problem.evaluate(x)
        if record.result is None:
            for key in self._output_keys:
                outputs[key] = CLAMP
            for g_name in self._constraint_outputs:
                outputs[g_name] = CLAMP
            return
        d = record.result.to_dict()
        for key in self._output_keys:
            outputs[key] = float(np.clip(d[key], -CLAMP, CLAMP))
        for g_name, g_val in zip(self._constraint_outputs, record.g, strict=True):
            outputs[g_name] = float(np.clip(g_val, -CLAMP, CLAMP))


def _om_name(name: str) -> str:
    """OpenMDAO variable names cannot contain dots."""
    return name.replace(".", "__")


def build_motor_group(problem: DesignProblem, *, objective_index: int = 0) -> om.Problem:
    """An om.Problem with the motor component, SLSQP driver, and constraints wired.

    The objective is the DesignProblem objective at ``objective_index``
    (negated internally when it is a maximize objective — OpenMDAO minimizes).
    """
    obj = problem.objectives[objective_index]

    om_prob = om.Problem(reports=False)
    om_prob.model.add_subsystem("motor", MotorComponent(problem=problem), promotes=["*"])

    for name, (lo, hi) in problem.continuous.items():
        om_prob.model.add_design_var(_om_name(name), lower=lo, upper=hi)
    om_prob.model.add_objective(obj.name, scaler=-1.0 if obj.sense > 0 else 1.0)
    for i in range(problem.n_constr):
        om_prob.model.add_constraint(f"g_{i}", upper=0.0)

    om_prob.driver = om.ScipyOptimizeDriver(optimizer="SLSQP", tol=1e-6, maxiter=60)
    return om_prob


def run_openmdao_demo(
    motor: AxialFluxMotor,
    op: OperatingPoint,
    *,
    variables: dict,
    objective: str,
    constraints: tuple = (),
    model: Model | None = None,
) -> dict:
    """Single-objective gradient refinement over the continuous variables.

    Returns {'x': optimal design dict, 'result': final to_dict(), 'success': bool}.
    """
    problem = DesignProblem(
        motor, op, variables=variables, objectives=[objective], constraints=constraints, model=model
    )
    if not problem.continuous:
        raise ValueError("run_openmdao_demo needs at least one continuous variable")
    om_prob = build_motor_group(problem)
    om_prob.setup()
    driver_out = om_prob.run_driver()
    # OpenMDAO >= 3.31 returns a DriverResult; older versions return `failed`
    driver_ok = driver_out.success if hasattr(driver_out, "success") else not driver_out
    x = {name: float(om_prob.get_val(_om_name(name))[0]) for name in problem.continuous}
    record = problem.evaluate(x)
    return {
        "x": x,
        "result": record.result.to_dict() if record.result else None,
        "success": bool(driver_ok) and record.result is not None,
    }
