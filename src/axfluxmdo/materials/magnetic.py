"""Magnetic material properties: NdFeB magnet grades and electrical steel.

Steinmetz coefficients for M-19 29-gauge are tuned to reproduce typical
manufacturer data (AK Steel / Proto Laminations): roughly 1.4-1.6 W/kg at
60 Hz, 1.5 T. The classical (two-term) Steinmetz model is used:

    P_v = k_h * f * B**alpha + k_e * f**2 * B**2    [W/kg]

which assumes sinusoidal flux. Coefficients vary noticeably between sources
and gauges; a unit test pins the 60 Hz / 1.5 T point to the datasheet band.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MagnetMaterial:
    """Sintered permanent-magnet material (NdFeB unless noted)."""

    name: str
    remanence_t: float  # Br at 20 C, tesla
    mu_r: float = 1.05  # recoil relative permeability
    temp_coeff_br_per_c: float = -0.0012  # reversible Br coefficient, 1/C (-0.12 %/C)
    density_kg_m3: float = 7500.0
    max_operating_temp_c: float = 80.0

    def remanence_at(self, temp_c: float) -> float:
        """Temperature-derated remanence Br(T) = Br20 * (1 + alpha_Br * (T - 20))."""
        return self.remanence_t * (1.0 + self.temp_coeff_br_per_c * (temp_c - 20.0))


N35 = MagnetMaterial("N35", remanence_t=1.19)
N42 = MagnetMaterial("N42", remanence_t=1.30)
N48 = MagnetMaterial("N48", remanence_t=1.39)
N42SH = MagnetMaterial(
    "N42SH", remanence_t=1.29, temp_coeff_br_per_c=-0.0011, max_operating_temp_c=150.0
)


@dataclass(frozen=True)
class ElectricalSteel:
    """Non-oriented electrical steel lamination with Steinmetz loss coefficients.

    Loss model (per unit mass): P_v = k_h * f * B**alpha + k_e * f**2 * B**2.
    """

    name: str
    density_kg_m3: float
    k_h: float  # hysteresis coefficient, W/kg per (Hz * T**alpha)
    alpha: float  # Steinmetz flux-density exponent
    k_e: float  # eddy-current coefficient, W/kg per (Hz**2 * T**2)
    b_sat_t: float = 1.6  # saturation knee used as a design-limit proxy, tesla
    stacking_factor: float = 0.95

    def core_loss_w_per_kg(self, f_hz: float, b_peak_t: float) -> float:
        """Specific core loss at electrical frequency f and peak flux density B."""
        if f_hz <= 0.0 or b_peak_t <= 0.0:
            return 0.0
        return self.k_h * f_hz * b_peak_t**self.alpha + self.k_e * f_hz**2 * b_peak_t**2


# Tuned to ~1.50 W/kg at 60 Hz / 1.5 T (~70% hysteresis / 30% eddy at 60 Hz):
#   0.00886*60*1.5^1.68 + 5.56e-5*60^2*1.5^2 = 1.05 + 0.45 = 1.50 W/kg
M19_29GA = ElectricalSteel("M-19 29ga", density_kg_m3=7650.0, k_h=0.00886, alpha=1.68, k_e=5.56e-5)


def airgap_flux_density(
    magnet: MagnetMaterial,
    magnet_thickness: float,
    air_gap: float,
    magnet_temp_c: float = 20.0,
    carter_factor: float = 1.0,
) -> float:
    """Open-circuit air-gap flux density from the magnet load line.

    Series magnetic circuit of a surface magnet over an air gap (slotless;
    Carter factor defaults to 1.0 in Phase 1, parameterized for later):

        B_g = Br(T) * t_m / (t_m + mu_r * k_C * g)

    Approaches Br(T) as g -> 0 (with mu_r -> 1), and falls off as the gap grows.
    """
    if magnet_thickness <= 0.0:
        raise ValueError("magnet_thickness must be positive")
    if air_gap < 0.0:
        raise ValueError("air_gap must be non-negative")
    br = magnet.remanence_at(magnet_temp_c)
    return br * magnet_thickness / (magnet_thickness + magnet.mu_r * carter_factor * air_gap)


def airgap_flux_density_runout_mean(
    magnet: MagnetMaterial,
    magnet_thickness: float,
    mean_gap: float,
    runout_amp: float,
    magnet_temp_c: float = 20.0,
    carter_factor: float = 1.0,
) -> float:
    """Circumferential mean of B over a 1/rev runout g(theta) = g_mean + d*cos(theta).

    The load line B(g) = Br*t_m/(t_m + mu_r*k_C*g) is a Moebius function of g,
    so the theta-average is analytic:

        <B> = Br*t_m / sqrt((t_m + mu_r*k_C*g_mean)^2 - (mu_r*k_C*d)^2)

    B(g) is convex in g, so by Jensen's inequality <B> >= B(g_mean): runout
    slightly *increases* the mean flux density (and torque); its real penalties
    are the 1/rev torque/force modulation. Reduces exactly to
    :func:`airgap_flux_density` at runout_amp = 0.
    """
    a = magnet_thickness + magnet.mu_r * carter_factor * mean_gap
    d = magnet.mu_r * carter_factor * runout_amp
    if d >= a:
        raise ValueError("runout closes the air gap (runout_amp too large)")
    br = magnet.remanence_at(magnet_temp_c)
    return br * magnet_thickness / math.sqrt(a**2 - d**2)


def airgap_flux_density_runout_extremes(
    magnet: MagnetMaterial,
    magnet_thickness: float,
    mean_gap: float,
    runout_amp: float,
    magnet_temp_c: float = 20.0,
    carter_factor: float = 1.0,
) -> tuple[float, float]:
    """(B at the tightest gap g_mean - d, B at the widest gap g_mean + d)."""
    b_tight = airgap_flux_density(
        magnet, magnet_thickness, mean_gap - runout_amp, magnet_temp_c, carter_factor
    )
    b_wide = airgap_flux_density(
        magnet, magnet_thickness, mean_gap + runout_amp, magnet_temp_c, carter_factor
    )
    return b_tight, b_wide


def airgap_b_squared_runout_mean(
    magnet: MagnetMaterial,
    magnet_thickness: float,
    mean_gap: float,
    runout_amp: float,
    magnet_temp_c: float = 20.0,
    carter_factor: float = 1.0,
) -> float:
    """Circumferential mean of B^2 over a 1/rev runout (for axial magnetic pull).

    <B^2> = Br^2*t_m^2*(t_m + mu_r*k_C*g_mean)
            / ((t_m + mu_r*k_C*g_mean)^2 - (mu_r*k_C*d)^2)^(3/2)
    """
    a = magnet_thickness + magnet.mu_r * carter_factor * mean_gap
    d = magnet.mu_r * carter_factor * runout_amp
    if d >= a:
        raise ValueError("runout closes the air gap (runout_amp too large)")
    br = magnet.remanence_at(magnet_temp_c)
    return br**2 * magnet_thickness**2 * a / (a**2 - d**2) ** 1.5
