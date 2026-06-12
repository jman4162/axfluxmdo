"""Backend-agnostic optimization problem specification.

This module is the single source of truth for parsing the SPEC-style
optimization grammar against the stable ``AnalyticalResult.to_dict()`` keys:

- variables: ``{"outer_radius": (0.05, 0.12), "pole_pairs": [8, 10, 12]}`` —
  2-tuples are continuous bounds (or integer bounds when the motor field is an
  int), lists are discrete choices; dotted paths reach nested fields
  (``"tolerances.runout_m"``).
- objectives: ``"maximize_torque_density"`` / ``"minimize_mass"`` — the
  suffix resolves through the alias map to a ``to_dict()`` key.
- constraints: ``"winding_temp_c < 140"`` — parsed to the pymoo/OpenMDAO
  ``g <= 0`` convention. The model's built-in ConstraintRecords are enforced
  in addition by default, so optimizer output is always ``result.feasible``.

Design vectors that violate geometric validation (frozen-dataclass
``__post_init__`` ValueError, e.g. inner_radius >= outer_radius) are
penalized with large finite objective/constraint values rather than raising —
GA constraint-domination buries them and gradient drivers stay numeric.

Only numpy is required here; pymoo/OpenMDAO backends import this module, not
the other way around.
"""

from __future__ import annotations

import math
import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from axfluxmdo.geometry.axial_flux import AxialFluxMotor
from axfluxmdo.models.analytical import AnalyticalModel, AnalyticalResult
from axfluxmdo.models.base import Model
from axfluxmdo.operating_point import OperatingPoint
from axfluxmdo.sweeps import replace_field

# SPEC short names -> canonical to_dict() keys. Every canonical key also
# resolves as itself. This map is the single source for alias resolution —
# viz and sensitivity import resolve_key; never re-declare aliases elsewhere.
ALIASES: dict[str, str] = {
    "torque": "torque_nm",
    "torque_density": "torque_density_nm_kg",
    "mass": "mass_kg",
    "winding_temp": "winding_temp_c",
    "electrical_frequency": "electrical_frequency_hz",
    "current_density": "current_density_a_mm2",
    "back_emf": "back_emf_v_rms",
    "copper_loss": "copper_loss_w",
    "core_loss": "core_loss_w",
    "ripple": "torque_ripple_proxy",
    "axial_force": "axial_force_n",
}

PENALTY_OBJECTIVE = 1e9
PENALTY_CONSTRAINT = 1e3
MARGIN_CLAMP = 1e6  # |-inf| margins (thermal runaway) clamp to this violation

_CONSTRAINT_RE = re.compile(r"^\s*([\w.]+)\s*(<=|>=|<|>)\s*([-+0-9.eE]+|inf)\s*$")


def resolve_key(name: str, available: Collection[str]) -> str:
    """Resolve a user-facing name (alias or canonical) to a to_dict() key."""
    candidate = ALIASES.get(name, name)
    if candidate in available:
        return candidate
    options = sorted(set(available) | set(ALIASES))
    raise ValueError(f"unknown result key {name!r}; available keys and aliases: {options}")


@dataclass(frozen=True)
class Objective:
    name: str  # canonical to_dict() key
    sense: int  # +1 maximize, -1 minimize
    label: str  # original user string


@dataclass(frozen=True)
class UserConstraint:
    key: str  # canonical to_dict() key
    op: str  # "<", "<=", ">", ">="
    bound: float
    label: str

    def violation(self, value: float) -> float:
        """g <= 0 convention: positive means violated."""
        if not math.isfinite(value):
            return MARGIN_CLAMP
        if self.op in ("<", "<="):
            return value - self.bound
        return self.bound - value


def parse_objective(spec: str, available: Collection[str]) -> Objective:
    """Parse 'maximize_<key>' / 'minimize_<key>' (key may be an alias)."""
    if spec.startswith("maximize_"):
        sense, raw = +1, spec[len("maximize_") :]
    elif spec.startswith("minimize_"):
        sense, raw = -1, spec[len("minimize_") :]
    else:
        raise ValueError(
            f"objective {spec!r} must start with 'maximize_' or 'minimize_' "
            f"(e.g. 'maximize_torque_density')"
        )
    key = resolve_key(raw, available)
    if key == "feasible":
        raise ValueError("'feasible' is not a valid objective; constraints handle feasibility")
    return Objective(name=key, sense=sense, label=spec)


def parse_constraint(spec: str, available: Collection[str]) -> UserConstraint:
    """Parse '<key> < bound' style constraint strings (<=, >=, <, > supported)."""
    match = _CONSTRAINT_RE.match(spec)
    if match is None:
        raise ValueError(
            f"cannot parse constraint {spec!r}; expected '<key> < bound' with one of <, <=, >, >="
        )
    raw_key, op, bound = match.groups()
    return UserConstraint(
        key=resolve_key(raw_key, available), op=op, bound=float(bound), label=spec
    )


@dataclass
class EvalRecord:
    """One design evaluation in optimizer space."""

    f_min: np.ndarray  # objectives, MINIMIZE convention (maximize negated)
    g: np.ndarray  # inequality violations, <= 0 feasible
    result: AnalyticalResult | None  # None when the geometry was invalid
    motor: AxialFluxMotor | None


class DesignProblem:
    """Validated design variables / objectives / constraints around a seed motor."""

    def __init__(
        self,
        motor: AxialFluxMotor,
        op: OperatingPoint,
        *,
        variables: Mapping[str, object],
        objectives: Sequence[str],
        constraints: Sequence[str] = (),
        model: Model | None = None,
        enforce_model_constraints: bool = True,
    ):
        self.motor = motor
        self.op = op
        self.model = model or AnalyticalModel()
        self.enforce_model_constraints = enforce_model_constraints

        # Baseline evaluation defines the available result keys and sanity-checks
        # that the seed design is evaluable.
        self.baseline_result = self.model.evaluate(motor, op)
        available = self.baseline_result.to_dict().keys()

        self.continuous: dict[str, tuple[float, float]] = {}
        self.integer: dict[str, tuple[int, int]] = {}
        self.choices: dict[str, list] = {}
        for name, spec in variables.items():
            baseline_value = self._baseline_value(name)  # validates the field path
            if isinstance(spec, tuple) and len(spec) == 2:
                lo, hi = spec
                if lo >= hi:
                    raise ValueError(f"variable {name!r}: lower bound must be < upper bound")
                if isinstance(baseline_value, int) and not isinstance(baseline_value, bool):
                    self.integer[name] = (int(lo), int(hi))
                else:
                    self.continuous[name] = (float(lo), float(hi))
            elif isinstance(spec, (list, range, np.ndarray)):
                options = list(spec)
                if not options:
                    raise ValueError(f"variable {name!r}: empty choice list")
                self.choices[name] = options
            else:
                raise ValueError(
                    f"variable {name!r}: spec must be a (low, high) tuple or a list of "
                    f"choices, got {spec!r}"
                )

        self.objectives = [parse_objective(s, available) for s in objectives]
        if not self.objectives:
            raise ValueError("at least one objective is required")
        self.user_constraints = [parse_constraint(s, available) for s in constraints]

        self.model_constraint_names = (
            [c.name for c in self.baseline_result.constraints] if enforce_model_constraints else []
        )

    # -- introspection --------------------------------------------------------

    @property
    def variable_names(self) -> list[str]:
        return [*self.continuous, *self.integer, *self.choices]

    @property
    def n_obj(self) -> int:
        return len(self.objectives)

    @property
    def n_constr(self) -> int:
        return len(self.user_constraints) + len(self.model_constraint_names)

    @property
    def constraint_names(self) -> list[str]:
        return [c.label for c in self.user_constraints] + [
            f"model:{n}" for n in self.model_constraint_names
        ]

    def _baseline_value(self, name: str):
        obj = self.motor
        try:
            for part in name.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            fields = [f for f in vars(self.motor)] + [
                f"tolerances.{f}" for f in vars(self.motor.tolerances)
            ]
            raise ValueError(
                f"unknown design variable {name!r}; motor fields: {sorted(fields)}"
            ) from None
        return obj

    # -- evaluation ------------------------------------------------------------

    def apply(self, x: Mapping[str, object]) -> AxialFluxMotor:
        """Build the motor variant for a design vector; may raise ValueError."""
        motor = self.motor
        for name, value in x.items():
            if name in self.integer or name in self.choices:
                baseline = self._baseline_value(name)
                if isinstance(baseline, int) and not isinstance(value, int):
                    value = int(round(float(value)))  # noqa: PLW2901
            motor = replace_field(motor, name, value)
        return motor

    def objective_values(self, result: AnalyticalResult) -> np.ndarray:
        """Human-readable (un-negated) objective values."""
        d = result.to_dict()
        return np.array([d[obj.name] for obj in self.objectives])

    def evaluate(self, x: Mapping[str, object]) -> EvalRecord:
        try:
            motor = self.apply(x)
        except ValueError:
            return EvalRecord(
                f_min=np.full(self.n_obj, PENALTY_OBJECTIVE),
                g=np.full(self.n_constr, PENALTY_CONSTRAINT),
                result=None,
                motor=None,
            )
        result = self.model.evaluate(motor, self.op)
        d = result.to_dict()

        f_min = np.array(
            [-d[obj.name] if obj.sense > 0 else d[obj.name] for obj in self.objectives]
        )
        f_min = np.where(np.isfinite(f_min), f_min, PENALTY_OBJECTIVE)

        g_user = [c.violation(d[c.key]) for c in self.user_constraints]
        g_model = []
        if self.enforce_model_constraints:
            for record in result.constraints:
                margin = record.margin
                g_model.append(-margin if math.isfinite(margin) else MARGIN_CLAMP)
        g = np.clip(np.array(g_user + g_model, dtype=float), -MARGIN_CLAMP, MARGIN_CLAMP)
        return EvalRecord(f_min=f_min, g=g, result=result, motor=motor)
