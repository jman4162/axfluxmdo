"""Parameter sweeps over motor design variables.

Sweeps never mutate the input motor: each point is evaluated on a
``dataclasses.replace`` variant, the same mechanism later optimization
drivers use.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from axfluxmdo.geometry.axial_flux import AxialFluxMotor
from axfluxmdo.models.analytical import AnalyticalModel, AnalyticalResult
from axfluxmdo.operating_point import OperatingPoint


@dataclass
class SweepResult:
    parameter: str
    values: list
    results: list[AnalyticalResult]

    def to_arrays(self, *fields: str) -> dict[str, np.ndarray]:
        """Extract named result fields (keys of ``AnalyticalResult.to_dict()``) as arrays."""
        dicts = [r.to_dict() for r in self.results]
        out = {self.parameter: np.asarray(self.values)}
        for name in fields:
            out[name] = np.array([d[name] for d in dicts])
        return out

    def plot(
        self,
        fields: Sequence[str] = (
            "torque_nm",
            "efficiency",
            "core_loss_w",
            "winding_temp_c",
        ),
        show: bool = False,
    ):
        """Plot each field against the swept parameter on a grid of axes."""
        import matplotlib.pyplot as plt

        n = len(fields)
        ncols = 2
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.2 * nrows), squeeze=False)
        data = self.to_arrays(*fields)
        x = data[self.parameter]
        for ax, name in zip(axes.flat, fields, strict=False):
            ax.plot(x, data[name], "o-")
            ax.set_xlabel(self.parameter)
            ax.set_ylabel(name)
            ax.grid(True, alpha=0.3)
        for ax in axes.flat[n:]:
            ax.set_visible(False)
        fig.tight_layout()
        if show:
            plt.show()
        return fig


def sweep_parameter(
    motor: AxialFluxMotor,
    op: OperatingPoint,
    name: str,
    values: Sequence,
    model: AnalyticalModel | None = None,
) -> SweepResult:
    """Evaluate the motor across values of one design field (e.g. ``air_gap``)."""
    model = model or AnalyticalModel()
    results = [model.evaluate(dataclasses.replace(motor, **{name: v}), op) for v in values]
    return SweepResult(parameter=name, values=list(values), results=results)


def sweep_pole_pairs(
    motor: AxialFluxMotor,
    op: OperatingPoint,
    pole_pairs: Sequence[int] = tuple(range(4, 24, 2)),
    model: AnalyticalModel | None = None,
) -> SweepResult:
    """The SPEC MVP question #1: performance tradeoffs across pole-pair count."""
    return sweep_parameter(motor, op, "pole_pairs", list(pole_pairs), model)
