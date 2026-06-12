"""One-at-a-time design sensitivities for tornado charts."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from axfluxmdo.geometry.axial_flux import AxialFluxMotor
from axfluxmdo.models.analytical import AnalyticalModel
from axfluxmdo.models.base import Model
from axfluxmdo.operating_point import OperatingPoint
from axfluxmdo.optimize.problem import resolve_key
from axfluxmdo.sweeps import replace_field


@dataclass(frozen=True)
class SensitivityEntry:
    variable: str
    low_input: float
    high_input: float
    low_output: float
    high_output: float

    @property
    def swing(self) -> float:
        return self.high_output - self.low_output


@dataclass
class SensitivityResult:
    output: str  # canonical to_dict key
    baseline: float
    entries: list[SensitivityEntry]  # sorted by |swing| descending


def compute_sensitivities(
    motor: AxialFluxMotor,
    op: OperatingPoint,
    variables: Sequence[str] | Mapping[str, object],
    *,
    output: str = "torque_density_nm_kg",
    rel_step: float = 0.05,
    model: Model | None = None,
) -> SensitivityResult:
    """One-at-a-time low/high perturbations of each variable (tornado-chart input).

    ``variables`` may be a sequence of field names (perturbed by ±rel_step of
    the baseline value) or an optimize-style dict: tuple bounds clamp the
    perturbations, choice lists step one option below/above the baseline's
    nearest choice. Integer fields are rounded and de-duplicated. A
    perturbation that produces invalid geometry is skipped with a warning
    (the baseline output stands in for that side).
    """
    model = model or AnalyticalModel()
    baseline_result = model.evaluate(motor, op)
    key = resolve_key(output, baseline_result.to_dict().keys())
    baseline = baseline_result.to_dict()[key]

    specs: dict[str, object | None]
    if isinstance(variables, Mapping):
        specs = dict(variables)
    else:
        specs = {name: None for name in variables}

    entries = []
    for name, spec in specs.items():
        base_value = _get_field(motor, name)
        low_in, high_in = _perturbations(base_value, spec, rel_step)
        if low_in == high_in:
            warnings.warn(f"variable {name!r}: no usable perturbation; skipped", stacklevel=2)
            continue
        low_out = _evaluate_side(model, motor, op, name, low_in, key, baseline)
        high_out = _evaluate_side(model, motor, op, name, high_in, key, baseline)
        entries.append(
            SensitivityEntry(
                variable=name,
                low_input=float(low_in),
                high_input=float(high_in),
                low_output=low_out,
                high_output=high_out,
            )
        )

    entries.sort(key=lambda e: abs(e.swing), reverse=True)
    return SensitivityResult(output=key, baseline=baseline, entries=entries)


def _get_field(motor: AxialFluxMotor, name: str):
    obj = motor
    for part in name.split("."):
        obj = getattr(obj, part)
    return obj


def _perturbations(base_value, spec, rel_step: float) -> tuple:
    is_int = isinstance(base_value, int) and not isinstance(base_value, bool)
    if isinstance(spec, (list, range)):
        options = sorted(spec)
        nearest = min(range(len(options)), key=lambda i: abs(options[i] - base_value))
        low = options[max(0, nearest - 1)]
        high = options[min(len(options) - 1, nearest + 1)]
        return low, high
    low = base_value * (1.0 - rel_step)
    high = base_value * (1.0 + rel_step)
    if isinstance(spec, tuple) and len(spec) == 2:
        low = max(low, spec[0])
        high = min(high, spec[1])
    if is_int:
        low, high = int(round(low)), int(round(high))
        if low == high == base_value:
            low, high = base_value - 1, base_value + 1
    return low, high


def _evaluate_side(model, motor, op, name, value, key, baseline) -> float:
    try:
        variant = replace_field(motor, name, value)
    except ValueError:
        warnings.warn(
            f"variable {name!r}={value!r}: invalid geometry; using baseline for this side",
            stacklevel=3,
        )
        return baseline
    return model.evaluate(variant, op).to_dict()[key]
