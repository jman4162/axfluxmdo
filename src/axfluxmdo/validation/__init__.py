"""Validation against external simulations (and, later, hardware data)."""

from axfluxmdo.validation.sim2real import (
    OpenCircuitComparison,
    compare_open_circuit,
    measured_carter_factor,
)

__all__ = ["OpenCircuitComparison", "compare_open_circuit", "measured_carter_factor"]
