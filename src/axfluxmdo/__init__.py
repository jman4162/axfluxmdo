"""axfluxmdo: parametric modeling, analysis, and MDO of axial-flux PM motors."""

from importlib.metadata import PackageNotFoundError, version

from axfluxmdo.geometry.axial_flux import AxialFluxMotor
from axfluxmdo.geometry.tolerances import GapImperfections
from axfluxmdo.operating_point import OperatingPoint

try:
    __version__ = version("axfluxmdo")
except PackageNotFoundError:  # pragma: no cover - package not installed
    __version__ = "0.0.0+unknown"

__all__ = ["AxialFluxMotor", "GapImperfections", "OperatingPoint", "__version__"]
