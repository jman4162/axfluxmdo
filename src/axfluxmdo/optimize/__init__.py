"""Optimization layer: design problems, Pareto runs, sensitivities.

pymoo and OpenMDAO are optional (``pip install "axfluxmdo[opt]"``) and are
never imported at package-import time: ``optimize_pareto`` imports pymoo
lazily inside the call, and the OpenMDAO names resolve through PEP 562
module ``__getattr__``.
"""

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
    "DesignProblem",
    "MotorComponent",
    "Objective",
    "ParetoStudy",
    "SensitivityEntry",
    "SensitivityResult",
    "UserConstraint",
    "build_motor_group",
    "compute_sensitivities",
    "optimize_pareto",
    "parse_constraint",
    "parse_objective",
    "resolve_key",
    "run_openmdao_demo",
]

_OPENMDAO_NAMES = {"MotorComponent", "build_motor_group", "run_openmdao_demo"}


def __getattr__(name: str):
    if name in _OPENMDAO_NAMES:
        from axfluxmdo.optimize import openmdao_components

        return getattr(openmdao_components, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
