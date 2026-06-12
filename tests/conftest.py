import matplotlib
import pytest

matplotlib.use("Agg")

from axfluxmdo import AxialFluxMotor, OperatingPoint  # noqa: E402


@pytest.fixture
def reference_motor() -> AxialFluxMotor:
    """The SPEC.md quickstart motor."""
    return AxialFluxMotor(
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


@pytest.fixture
def reference_op() -> OperatingPoint:
    """The SPEC.md quickstart operating point."""
    return OperatingPoint(speed_rpm=500, current_rms=25, dc_bus_voltage=48)
