"""Optimization layer: design problems, Pareto runs, sensitivities.

pymoo and OpenMDAO are optional (``pip install "axfluxmdo[opt]"``) and are
never imported at package-import time: ``optimize_pareto`` imports pymoo
lazily inside the call, and the OpenMDAO names resolve through PEP 562
module ``__getattr__``.
"""

from axfluxmdo.optimize.dataset import DesignDataset
from axfluxmdo.optimize.problem import (
    ALIASES,
    DesignProblem,
    Objective,
    UserConstraint,
    parse_constraint,
    parse_objective,
    resolve_key,
)
from axfluxmdo.optimize.pymoo_runner import ParetoStudy, optimize_pareto
from axfluxmdo.optimize.sensitivity import (
    SensitivityEntry,
    SensitivityResult,
    compute_sensitivities,
)

__all__ = [
    "ALIASES",
    "BOStudy",
    "DesignDataset",
    "DesignProblem",
    "GPSurrogate",
    "MotorComponent",
    "RandomForestSurrogate",
    "Objective",
    "ParetoStudy",
    "SensitivityEntry",
    "SensitivityResult",
    "UserConstraint",
    "bayesian_optimize",
    "build_motor_group",
    "compute_sensitivities",
    "optimize_pareto",
    "parse_constraint",
    "parse_objective",
    "resolve_key",
    "run_openmdao_demo",
]

_OPENMDAO_NAMES = {"MotorComponent", "build_motor_group", "run_openmdao_demo"}
_SURROGATE_NAMES = {"GPSurrogate", "RandomForestSurrogate", "Surrogate"}
_BAYESOPT_NAMES = {"BOStudy", "bayesian_optimize"}


def __getattr__(name: str):
    if name in _OPENMDAO_NAMES:
        from axfluxmdo.optimize import openmdao_components

        return getattr(openmdao_components, name)
    if name in _SURROGATE_NAMES:
        from axfluxmdo.optimize import surrogate

        return getattr(surrogate, name)
    if name in _BAYESOPT_NAMES:
        from axfluxmdo.optimize import bayesopt

        return getattr(bayesopt, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
