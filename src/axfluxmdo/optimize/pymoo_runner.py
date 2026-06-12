"""Pareto-front optimization via pymoo's mixed-variable GA.

pymoo is imported lazily so the base package works without the ``[opt]``
extra. Mixed continuous/integer/choice design variables use
``MixedVariableGA`` with rank-and-crowding survival (the multi-objective
configuration recommended by pymoo 0.6.x for mixed search spaces) — never
NSGA2-with-rounding, which breaks Choice semantics and duplicate
elimination.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from axfluxmdo.geometry.axial_flux import AxialFluxMotor
from axfluxmdo.models.analytical import AnalyticalResult
from axfluxmdo.models.base import Model
from axfluxmdo.operating_point import OperatingPoint
from axfluxmdo.optimize.problem import DesignProblem, Objective, resolve_key


def _require_pymoo():
    try:
        import pymoo  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without [opt]
        raise ImportError(
            "optimize_pareto requires pymoo; install with: pip install 'axfluxmdo[opt]'"
        ) from exc


@dataclass
class ParetoStudy:
    """Feasible non-dominated designs from one optimize_pareto run.

    ``F`` holds human-readable objective values (maximize objectives are NOT
    negated here; the sign convention lives only inside the optimizer).
    Every point satisfies the user constraints and ``result.feasible``.
    """

    variables: list[str]
    objectives: list[Objective]
    X: list[dict]
    F: np.ndarray  # (n_points, n_obj), human sign
    results: list[AnalyticalResult]
    motors: list[AxialFluxMotor]
    seed: int
    pop_size: int
    n_gen: int
    problem: DesignProblem = field(repr=False)

    def __len__(self) -> int:
        return len(self.X)

    def to_records(self) -> list[dict]:
        """One flat dict per point: design variables merged with result.to_dict()."""
        return [x | r.to_dict() for x, r in zip(self.X, self.results, strict=True)]

    def to_arrays(self, *fields_: str) -> dict[str, np.ndarray]:
        """Extract named fields (aliases, to_dict keys, or design variables) as arrays."""
        records = self.to_records()
        available = records[0].keys()
        out = {}
        for name in fields_:
            key = name if name in available else resolve_key(name, available)
            out[name] = np.array([rec[key] for rec in records])
        return out

    def best(self, objective: str) -> int:
        """Index of the best point for one objective (alias-aware).

        'Best' respects the study's sense for that key if it is an objective
        (max for maximize_*), otherwise defaults to maximum.
        """
        records = self.to_records()
        key = objective if objective in records[0] else resolve_key(objective, records[0].keys())
        values = np.array([rec[key] for rec in records])
        sense = next((obj.sense for obj in self.objectives if obj.name == key), +1)
        return int(np.argmax(sense * values))

    def summary(self) -> str:
        lines = [
            f"ParetoStudy: {len(self)} feasible non-dominated designs "
            f"(pop={self.pop_size}, gen={self.n_gen}, seed={self.seed})"
        ]
        for i, obj in enumerate(self.objectives):
            col = self.F[:, i]
            lines.append(f"  {obj.label}: {col.min():.4g} … {col.max():.4g}")
        return "\n".join(lines)


def optimize_pareto(
    motor: AxialFluxMotor,
    op: OperatingPoint,
    *,
    variables: Mapping[str, object],
    objectives: Sequence[str],
    constraints: Sequence[str] = (),
    model: Model | None = None,
    algorithm=None,
    pop_size: int = 60,
    n_gen: int = 40,
    seed: int = 1,
    verbose: bool = False,
    enforce_model_constraints: bool = True,
) -> ParetoStudy:
    """Multi-objective Pareto optimization of the motor design (SPEC flagship API).

    Note: unlike the SPEC sketch, the operating point is an explicit argument —
    every objective (torque, efficiency, winding temperature) depends on it.
    """
    _require_pymoo()
    from pymoo.core.mixed import MixedVariableGA
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.core.variable import Choice, Integer, Real
    from pymoo.optimize import minimize

    try:
        from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
    except ImportError:  # pragma: no cover - older pymoo layout
        from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival as RankAndCrowding

    design = DesignProblem(
        motor,
        op,
        variables=variables,
        objectives=objectives,
        constraints=constraints,
        model=model,
        enforce_model_constraints=enforce_model_constraints,
    )

    pymoo_vars: dict[str, object] = {}
    for name, (lo, hi) in design.continuous.items():
        pymoo_vars[name] = Real(bounds=(lo, hi))
    for name, (lo, hi) in design.integer.items():
        pymoo_vars[name] = Integer(bounds=(lo, hi))
    for name, options in design.choices.items():
        pymoo_vars[name] = Choice(options=options)

    class _MotorProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(vars=pymoo_vars, n_obj=design.n_obj, n_ieq_constr=design.n_constr)

        def _evaluate(self, x, out, *args, **kwargs):
            record = design.evaluate(x)
            out["F"] = record.f_min
            out["G"] = record.g

    if algorithm is None:
        algorithm = MixedVariableGA(pop_size=pop_size, survival=RankAndCrowding())

    res = minimize(_MotorProblem(), algorithm, ("n_gen", n_gen), seed=seed, verbose=verbose)

    if res.X is None:
        raise RuntimeError(
            "optimize_pareto found no feasible designs; relax the constraints, widen "
            "the variable bounds, or increase pop_size/n_gen"
        )
    xs = [dict(x) for x in np.atleast_1d(res.X)]

    # Re-evaluate the front for full results; keep strictly feasible points only.
    points = []
    for x in xs:
        record = design.evaluate(x)
        if record.result is not None and record.result.feasible and np.all(record.g <= 0):
            points.append((x, record))
    if not points:
        raise RuntimeError(
            "optimize_pareto found no feasible designs after re-evaluation; relax the "
            "constraints or increase the budget"
        )

    return ParetoStudy(
        variables=design.variable_names,
        objectives=design.objectives,
        X=[x for x, _ in points],
        F=np.array([design.objective_values(rec.result) for _, rec in points]),
        results=[rec.result for _, rec in points],
        motors=[rec.motor for _, rec in points],
        seed=seed,
        pop_size=pop_size,
        n_gen=n_gen,
        problem=design,
    )
