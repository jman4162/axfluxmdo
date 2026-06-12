# The analytical model (Layer 1)

The foundation of the package: a closed-form sizing model that evaluates a
complete motor design — torque, back-EMF, losses, steady-state temperature,
mass, and constraint margins — in microseconds. Every higher-fidelity layer
reproduces it in a limit (the [annular model](annular-model.md) matches it to
1×10⁻¹² with one slice; tests pin this), so the physics derived here is the
backbone of everything else.

Code: [`axfluxmdo.models.analytical`](../api/models.md),
[`axfluxmdo.materials.magnetic`](../api/materials.md),
[`axfluxmdo.models.losses`](../api/models.md),
[`axfluxmdo.models.thermal_rc`](../api/models.md).

---

## 1. The air-gap field from the magnet load line

**Assumptions:** a surface permanent magnet of thickness $h_m$ faces an iron
stator across an air gap $g$; iron permeability is effectively infinite; the
flux path is one-dimensional (no fringing — see
[Limitations](../limitations.md) for the FEA-measured consequence).

Apply Ampère's law around the magnet–gap loop. With infinitely permeable
iron, the only MMF drops are in the magnet and the gap:

$$
H_m h_m + H_g\, k_C\, g = 0 ,
$$

where $k_C \ge 1$ is the **Carter factor**, the classical correction that
inflates the effective gap to account for slot openings ($k_C = 1$ for a
slotless face). Flux continuity through the series circuit gives
$B_m = B_g$. The magnet operates on its **recoil line**,

$$
B_m = \mu_0 \mu_r H_m + B_r(T) ,
$$

with recoil permeability $\mu_r \approx 1.05$ and temperature-derated
remanence $B_r(T) = B_{r,20}\,[1 + \alpha_{B_r}(T - 20)]$ (NdFeB:
$\alpha_{B_r} \approx -0.12\,\%/°\mathrm{C}$). In the gap,
$B_g = \mu_0 H_g$. Eliminating the field intensities:

$$
\boxed{\; B_g \;=\; \frac{B_r(T)\, h_m}{h_m + \mu_r\, k_C\, g} \;}
$$

This is `airgap_flux_density()` in
[`materials/magnetic.py`](../api/materials.md). Two sanity limits: as
$g \to 0$ (with $\mu_r \to 1$), $B_g \to B_r$; thickening the magnet pushes
$B_g$ toward $B_r$ asymptotically — diminishing returns that the
[sensitivity tornado](optimization.md) makes visible.

**Reference motor numbers** (N42 at 65 °C, $h_m$ = 4 mm, $g$ = 0.8 mm):
$B_r(65) = 1.30 \times (1 - 0.0012 \cdot 45) = 1.2298$ T, so
$B_g = 1.2298 \cdot 4/(4 + 1.05 \cdot 0.8) = 1.016$ T.

!!! warning "The load line is an upper bound"
    2D FEA measures the real under-magnet mean ≈ 11% lower (inter-magnet
    leakage + fringing) — see [FEA validation](fea-validation.md). Pass a
    measured `carter_factor` to either model to fold corrections back in.

## 2. The fundamental of the gap field

The magnets create a square-wave field pattern: $+B_g$ under N poles, $-B_g$
under S poles, zero in the uncovered fraction $(1-\alpha_m)$ of each pole
pitch. Sinusoidal machine theory works with the **fundamental** of that
wave. For a square wave of amplitude $B_g$ spanning the central fraction
$\alpha_m$ of each half-period, the Fourier fundamental is

$$
B_1 \;=\; \frac{4}{\pi}\, B_g \sin\!\Big(\frac{\alpha_m \pi}{2}\Big).
$$

*Derivation sketch:* $b_1 = \frac{2}{\tau}\int_{-\tau/2}^{\tau/2} B(x)\sin(\pi x/\tau)\,dx$
with $B(x) = \pm B_g$ over the magnet arc; the integral over the covered span
$[\tau(1-\alpha_m)/2,\ \tau(1+\alpha_m)/2]$ evaluates to the $\sin(\alpha_m\pi/2)$
factor. Note the diminishing payoff: $\alpha_m: 0.85 \to 1.0$ raises
$\sin(\alpha_m\pi/2)$ only from 0.972 to 1.0 — visible in example 04's magnet-arc sweep.

## 3. Flux linkage, torque, and back-EMF — from one quantity

The fundamental flux per pole over the annular gap area
$A_g = \pi(r_o^2 - r_i^2)$ is the average of the half-sine over a pole:

$$
\Phi_p = \frac{2}{\pi}\, B_1\, \frac{A_g}{2p} = \frac{B_1 A_g}{\pi p},
\qquad
\lambda = k_w N\, \Phi_p ,
$$

with $k_w$ the winding factor and $N$ turns per phase. The package then
derives **both** torque and back-EMF from this single $\lambda$:

$$
E_\mathrm{rms} = \frac{\omega_e \lambda}{\sqrt{2}},
\qquad
T = \frac{m\, p\, \lambda\, I_\mathrm{rms}}{\sqrt{2}} .
$$

**Why this matters — the energy-balance proof.** Multiply:

$$
m\, E_\mathrm{rms} I_\mathrm{rms}
= m\,\frac{p\,\omega_m \lambda}{\sqrt 2}\, I_\mathrm{rms}
= \omega_m \cdot \frac{m\, p\, \lambda\, I_\mathrm{rms}}{\sqrt 2}
= T\,\omega_m .
$$

Electrical power into the EMF equals mechanical power out, *identically* —
not approximately. The test suite enforces this to 10⁻⁹ relative
(`tests/test_analytical.py::TestExactIdentities`), which pins down every
$\sqrt2$ and $\pi$ in the chain. Models that compute torque and EMF from
separate formulas routinely leak a few percent here.

??? info "Equivalence to the classical sizing equation"
    The SPEC's shear-stress form $T = \tfrac{2\pi\sigma_t}{3}(r_o^3 - r_i^3)$
    integrates a uniform air-gap shear over the annulus. It differs from the
    flux-linkage form only by the geometry factor
    $\tfrac{2}{3}(r_o^3-r_i^3)\,/\,[r_m(r_o^2-r_i^2)] \approx 1.09$ for the
    reference motor — two valid Phase-1 conventions; the package uses the
    flux-linkage form internally and reports the implied average shear.

## 4. Losses

**Copper.** DC resistance per phase from geometry,
$R_\mathrm{ph}(T) = \rho(T)\, N L_\mathrm{turn} / A_\mathrm{cond}$, with
$\rho(T) = \rho_{20}[1 + 0.00393\,(T-20)]$ and
$L_\mathrm{turn} = 2(r_o - r_i) + 2 k_\mathrm{end}\tau_p$. Then
$P_\mathrm{cu} = m I^2 R_\mathrm{ph}$ — note copper loss is **linear in
temperature**, which is what makes the thermal solution closed-form below.

**Core (Steinmetz).** The classical two-term model per unit mass:

$$
P_v = k_h\, f\, B^{\alpha} \;+\; k_e\, f^2 B^2 ,
$$

hysteresis (area of the B–H loop traversed $f$ times per second, with the
empirical Steinmetz exponent $\alpha \approx 1.6\!-\!1.8$) plus classical
eddy currents (induced EMF ∝ $fB$, loss ∝ EMF², hence $f^2B^2$). The M-19
coefficients are pinned to the manufacturer's 60 Hz / 1.5 T datasheet point
by a unit test. The flux density used is the **stator-yoke** value
$B_y = B_g \alpha_m \tau_p / (2 t_\mathrm{core} k_\mathrm{stack})$ — half of
each pole's flux returns through the yoke cross-section.

## 5. The thermal RC and its closed form

A single thermal resistance $R_\theta$ couples the winding node to ambient.
At steady state,

$$
T_w = T_\mathrm{amb} + R_\theta \big( P_\mathrm{cu}(T_w) + \gamma P_\mathrm{core} \big),
$$

a fixed-point equation because copper loss rises with its own temperature.
Since $P_\mathrm{cu}(T) = P_\mathrm{cu,ref}\,[1 + \alpha(T - T_\mathrm{ref})]$
is **linear** in $T$, the fixed point solves in closed form:

$$
T_w \;=\;
\frac{ T_\mathrm{amb} + R_\theta\big( P_\mathrm{cu,ref}(1-\alpha T_\mathrm{ref}) + \gamma P_\mathrm{core} \big) }
     { 1 - \alpha R_\theta P_\mathrm{cu,ref} } .
$$

The denominator tells a physical story: each kelvin of rise adds
$\alpha R_\theta P_\mathrm{cu,ref}$ kelvin of *additional* rise through the
resistivity feedback. When

$$
\alpha\, R_\theta\, P_\mathrm{cu,ref} \;\ge\; 1
$$

the geometric series diverges — **thermal runaway**. The model flags it
(`ThermalSolution.runaway`) and forces the thermal constraint to violated
rather than reporting a meaningless negative temperature. A 50-iteration
fixed-point reference implementation verifies the closed form in tests.

## 6. Constraints

Every evaluation reports six named constraints with normalized margins
$(\mathrm{limit} - \mathrm{value})/|\mathrm{limit}|$: winding temperature,
electrical frequency, current density, line voltage
($\sqrt3\,(E + IR) \le V_{dc}/\sqrt2$, inductive drop neglected — see
[Limitations](../limitations.md)), yoke flux density vs the steel's
saturation knee, and magnet temperature vs its grade rating. The constraint
names double as the optimization grammar's vocabulary
(`"winding_temp_c < 140"`).

---

**Try it:** [example 01](../examples/01_basic_axial_flux_motor.ipynb) walks
the full chain on the reference motor;
[example 02](../examples/02_pole_pair_tradeoff.ipynb) shows why torque is
independent of pole count at fixed loading while yoke saturation and
electrical frequency form the real tradeoff.
