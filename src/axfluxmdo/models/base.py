"""Shared model interface."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from axfluxmdo.geometry.axial_flux import AxialFluxMotor
    from axfluxmdo.models.analytical import AnalyticalResult
    from axfluxmdo.operating_point import OperatingPoint


class Model(Protocol):
    """Anything with an ``evaluate(motor, op) -> AnalyticalResult``-compatible method."""

    def evaluate(self, motor: AxialFluxMotor, op: OperatingPoint) -> AnalyticalResult: ...
