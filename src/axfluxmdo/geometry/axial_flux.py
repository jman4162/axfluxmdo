"""Parametric axial-flux permanent-magnet motor geometry.

Single-gap (one rotor disk, one stator disk) topology in Phase 1. All
dimensions are SI (meters); the dataclass is frozen so design variants are
produced with ``dataclasses.replace(motor, ...)`` — the mechanism used by
sweeps and, later, optimization drivers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from axfluxmdo.geometry.tolerances import PERFECT_GAP, GapImperfections
from axfluxmdo.materials.electrical import COPPER, Conductor
from axfluxmdo.materials.magnetic import M19_29GA, N42, ElectricalSteel, MagnetMaterial


@dataclass(frozen=True)
class AxialFluxMotor:
    """Parametric axial-flux PM motor design.

    The nine leading fields form the primary design vector (see SPEC.md);
    the remaining fields are Phase-1 modeling parameters with sensible
    defaults so the quickstart snippet runs unchanged.
    """

    # --- primary design variables (SPEC API) ---
    outer_radius: float  # m
    inner_radius: float  # m
    air_gap: float  # m
    pole_pairs: int
    phases: int = 3
    turns_per_phase: int = 24
    fill_factor: float = 0.45
    magnet_thickness: float = 0.004  # m
    back_iron_thickness: float = 0.006  # m, rotor back iron

    # --- winding / magnet layout ---
    magnet_arc_ratio: float = 0.85  # pole coverage alpha_m (fraction of pole pitch)
    magnet_shape: str = "wedge"  # "wedge": alpha(r)=alpha_m; "rectangular": constant width
    slot_depth: float = 0.012  # m, axial depth of the stator winding window
    slot_width_fraction: float = 0.5  # circumferential fraction of annulus open to copper
    # Fundamental winding factor (distribution x pitch) for the assumed
    # integral-slot 3-phase layout the 2D/3D geometry uses (2*phases slots per
    # pole pair, full-pitch). Changing phases/pole/slot combinations does NOT
    # update this automatically — supply the winding factor for your layout.
    winding_factor: float = 0.933
    end_turn_factor: float = 1.4  # end-turn length as multiple of pole pitch at mean radius
    stator_core_thickness: float = 0.008  # m, stator yoke behind the winding window

    # --- materials ---
    magnet: MagnetMaterial = field(default=N42)
    steel: ElectricalSteel = field(default=M19_29GA)
    conductor: Conductor = field(default=COPPER)

    # --- thermal / structure lumped parameters ---
    thermal_resistance_k_per_w: float = 1.2  # winding -> ambient
    structure_mass_factor: float = 0.25  # housing/shaft mass as fraction of active mass

    # --- manufacturing imperfections (consumed by AnnularModel only) ---
    tolerances: GapImperfections = field(default=PERFECT_GAP)

    def __post_init__(self) -> None:
        if self.inner_radius <= 0.0:
            raise ValueError("inner_radius must be positive")
        if self.outer_radius <= self.inner_radius:
            raise ValueError("outer_radius must exceed inner_radius")
        if self.air_gap <= 0.0:
            raise ValueError("air_gap must be positive")
        if self.pole_pairs < 1:
            raise ValueError("pole_pairs must be at least 1")
        if self.phases < 1:
            raise ValueError("phases must be at least 1")
        if self.turns_per_phase < 1:
            raise ValueError("turns_per_phase must be at least 1")
        if not 0.0 < self.fill_factor < 0.8:
            raise ValueError("fill_factor must be in (0, 0.8)")
        if not 0.0 < self.magnet_arc_ratio <= 1.0:
            raise ValueError("magnet_arc_ratio must be in (0, 1]")
        if self.magnet_thickness <= 0.0:
            raise ValueError("magnet_thickness must be positive")
        if not 0.0 < self.slot_width_fraction <= 1.0:
            raise ValueError("slot_width_fraction must be in (0, 1]")
        if self.magnet_shape not in ("wedge", "rectangular"):
            raise ValueError("magnet_shape must be 'wedge' or 'rectangular'")
        min_gap = (
            self.air_gap
            + self.tolerances.gap_offset_m
            - abs(self.tolerances.coning_m) / 2.0
            - self.tolerances.runout_m
        )
        if min_gap <= 0.0:
            raise ValueError("tolerances close the air gap (minimum local gap <= 0)")

    # --- derived geometry ---

    @property
    def mean_radius(self) -> float:
        return 0.5 * (self.outer_radius + self.inner_radius)

    @property
    def active_length(self) -> float:
        """Radial length of the active annulus (the 'stack length' analogue)."""
        return self.outer_radius - self.inner_radius

    @property
    def airgap_area(self) -> float:
        """Annular air-gap area, pi * (r_o^2 - r_i^2)."""
        return math.pi * (self.outer_radius**2 - self.inner_radius**2)

    @property
    def pole_pitch(self) -> float:
        """Circumferential pole pitch at the mean radius."""
        return math.pi * self.mean_radius / self.pole_pairs

    @property
    def magnet_volume(self) -> float:
        """Total magnet volume across all 2p poles."""
        return self.airgap_area * self.magnet_arc_ratio * self.magnet_thickness

    @property
    def back_iron_volume(self) -> float:
        """Rotor back-iron disk volume."""
        return self.airgap_area * self.back_iron_thickness

    @property
    def stator_core_volume(self) -> float:
        """Stator yoke volume (solid, before stacking factor)."""
        return self.airgap_area * self.stator_core_thickness

    @property
    def copper_window_area(self) -> float:
        """Circumferential copper window cross-section at the mean radius (m^2).

        The winding window is slot_depth deep (axially) and occupies
        slot_width_fraction of the circumference at the mean radius; this is
        the total area available to all phase conductors crossing the annulus.
        """
        return self.slot_depth * 2.0 * math.pi * self.mean_radius * self.slot_width_fraction

    @property
    def conductor_area(self) -> float:
        """Cross-section of one conductor (one turn of one phase), m^2.

        Each phase places 2*N conductors through the window (go and return
        paths of N turns); copper fills fill_factor of the window.
        """
        total_conductors = 2.0 * self.phases * self.turns_per_phase
        return self.fill_factor * self.copper_window_area / total_conductors

    @property
    def mean_turn_length(self) -> float:
        """Mean length of one winding turn: two radial runs plus two end turns."""
        return 2.0 * self.active_length + 2.0 * self.end_turn_factor * self.pole_pitch

    @property
    def copper_volume(self) -> float:
        """Total copper volume across all phases."""
        return self.phases * self.turns_per_phase * self.mean_turn_length * self.conductor_area
