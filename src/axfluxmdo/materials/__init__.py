"""Material property definitions and defaults."""

from axfluxmdo.materials.electrical import COPPER, Conductor, resistivity
from axfluxmdo.materials.magnetic import (
    M19_29GA,
    N35,
    N42,
    N42SH,
    N48,
    ElectricalSteel,
    MagnetMaterial,
    airgap_flux_density,
)

__all__ = [
    "COPPER",
    "M19_29GA",
    "N35",
    "N42",
    "N42SH",
    "N48",
    "Conductor",
    "ElectricalSteel",
    "MagnetMaterial",
    "airgap_flux_density",
    "resistivity",
]
