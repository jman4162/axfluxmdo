"""axfluxmdo: parametric modeling, analysis, and MDO of axial-flux PM motors."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("axfluxmdo")
except PackageNotFoundError:  # pragma: no cover - package not installed
    __version__ = "0.0.0+unknown"
