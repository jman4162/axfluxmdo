# Model fidelity & known limitations

The fast layers are deliberately simple. This page is the canonical statement
of what they leave out — know it before trusting absolute numbers. Each item
is also documented at the relevant docstring.

## Topology

**Single-gap machine** (one rotor disk, one stator disk). Real axial-flux
machines are often double-gap (TORUS, YASA, AFIR); a single-sided rotor
carries a large **unbalanced axial pull** — `AnnularResult.axial_force_n`
reports it (≈5–6 kN for the reference motor) and the bearing stack must
absorb it. Torque, mass, and inertia for double-gap variants are not modeled.

## Magnetics

**The 1D load line is an upper bound on the gap field.** FEA validation
([guide](guide/fea-validation.md)) measured, for the reference motor:

- under-magnet mean flux density: **−11.2%** vs the load line,
- fundamental $B_1$: **−6.8%**,

from inter-magnet leakage and circumferential fringing the 1D magnetic
circuit cannot see, plus a measured Carter factor $k_C = 1.44$ for the
slotted stator. Both models accept `carter_factor=` to fold a measured
correction back in; the residual fringing bias remains otherwise.

**No magnetic saturation** — torque is exactly linear in current. The
yoke-flux-density constraint (vs the steel's saturation knee) and the
current-density limit are the guards; near or beyond them the linear
prediction is optimistic.

**Fixed winding factor** (default 0.933 for the assumed integral-slot
3-phase layout). Changing phase/pole/slot combinations does not update it.

## Electrical

**The voltage constraint neglects inductive drop** ($I\,X_L$): it compares
$\sqrt{3}\,(E + I R)$ against $V_{dc}/\sqrt{2}$. Fine at low electrical
frequency; optimistic when $f_e$ is high *and* the bus margin is tight
(≈100 µH at several kRPM adds tens of volts).

## Thermal

**Single lumped RC** from winding to ambient with a constant thermal
resistance — no speed-dependent cooling, no radial/axial temperature
distribution. Half of core loss is assigned to the winding node.

**Magnet temperature is fixed at ambient + 40 °C**, not coupled to the solved
winding temperature. Remanence derating uses that assumption everywhere.

## Losses omitted

- AC copper loss (skin and proximity effects) — relevant above ~1 kHz
  electrical frequency.
- Magnet eddy-current loss.
- PWM harmonic losses (sinusoidal current assumed).
- Mechanical loss defaults to **zero** (bearing/windage coefficients are
  parameters of `AnnularModel`); efficiency is therefore optimistic unless
  you set them.

## Numerical conventions

- The runout average uses the exact circumferential mean of the load line;
  because the load line is convex in the gap, mean torque *rises* slightly
  with runout — the real penalties are the 1/rev ripple proxy and the axial
  force modulation. This sign is test-pinned; it is not a bug.
- The 2D FEA validation linearizes the annulus at the mean radius (one pole
  pair, periodic) — radial end effects are outside its scope.
