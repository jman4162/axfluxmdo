# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Pole-pair tradeoff study
#
# How do torque, back-EMF, electrical frequency, core loss, efficiency, and
# winding temperature change with pole-pair count?
#
# In the Layer-1 model, torque at fixed air-gap field and electrical loading
# is independent of pole-pair count: flux per pole falls as 1/p while the
# pole count rises as p, and the two cancel in the torque expression. The
# tradeoff lives elsewhere:
#
# - Low `p` needs a thick stator yoke. Each pole's flux return scales with
#   the pole pitch, so a fixed yoke saturates at low pole counts.
# - High `p` raises electrical frequency linearly, which raises inverter
#   switching burden and ripple, while shorter end turns reduce copper loss.
#
# The second sweep at the end of this notebook shows where the familiar
# "more poles is better" intuition actually comes from: resizing the iron
# with p raises torque density several-fold while torque stays constant.

# %%
from pathlib import Path

from axfluxmdo import AxialFluxMotor, OperatingPoint
from axfluxmdo.sweeps import sweep_pole_pairs

OUTPUT_DIR = Path(__file__).parent / "output" if "__file__" in globals() else Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# %%
motor = AxialFluxMotor(
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
op = OperatingPoint(speed_rpm=500, current_rms=25, dc_bus_voltage=48)

# %% [markdown]
# ## Sweep p = 4 … 20

# %%
sweep = sweep_pole_pairs(motor, op, pole_pairs=range(4, 21, 2))

data = sweep.to_arrays(
    "torque_nm", "electrical_frequency_hz", "core_loss_w", "efficiency", "winding_temp_c"
)
for p, t, f, pc in zip(
    data["pole_pairs"],
    data["torque_nm"],
    data["electrical_frequency_hz"],
    data["core_loss_w"],
    strict=True,
):
    print(f"p={p:3.0f}  torque={t:6.2f} N·m  f_e={f:6.1f} Hz  core loss={pc:5.2f} W")

# %% [markdown]
# ## Tradeoff plots

# %%
fig = sweep.plot(
    fields=(
        "torque_nm",
        "electrical_frequency_hz",
        "core_loss_w",
        "efficiency",
        "winding_temp_c",
        "back_emf_v_rms",
    )
)
fig.savefig(OUTPUT_DIR / "02_pole_pair_tradeoff.png", dpi=150, bbox_inches="tight")
fig

# %% [markdown]
# ## Where do constraints bind?
#
# At the low end, p = 4 is infeasible: with the stator core thickness held
# fixed, the wide pole pitch pushes the yoke flux density past saturation. At
# the high end the electrical-frequency margin shrinks linearly with `p`; at
# 500 rpm it never binds, but at higher shaft speeds it becomes the limiting
# constraint well before thermal limits.

# %%
for p, r in zip(sweep.values, sweep.results, strict=True):
    by_name = {c.name: c for c in r.constraints}
    f_margin = by_name["electrical_frequency_hz"].margin
    b_yoke = by_name["core_flux_density_t"].value
    print(f"p={p:3d}  f_e margin {f_margin:+.1%}  B_yoke={b_yoke:.2f} T  feasible={r.feasible}")

# %% [markdown]
# ## Where the "more poles = more torque" intuition comes from
#
# Torque is flat above because the sweep holds the geometry fixed. The
# benefit of high pole counts is torque *density*: each pole's flux return
# scales with the pole pitch, so the required yoke thickness scales as 1/p.
# Resizing the rotor back iron and stator core accordingly (holding yoke
# flux density constant) leaves torque unchanged but removes iron mass.

# %%
import dataclasses

from axfluxmdo.models import AnalyticalModel

model = AnalyticalModel()
P_REF = motor.pole_pairs  # yokes sized for p=14 in the reference design

print("fixed geometry            vs   yokes resized as 1/p (constant B_yoke)")
for p in range(4, 21, 2):
    fixed = model.evaluate(dataclasses.replace(motor, pole_pairs=p), op)
    scale = P_REF / p
    scaled_motor = dataclasses.replace(
        motor,
        pole_pairs=p,
        stator_core_thickness=motor.stator_core_thickness * scale,
        back_iron_thickness=motor.back_iron_thickness * scale,
    )
    scaled = model.evaluate(scaled_motor, op)
    print(
        f"p={p:3d}  T={fixed.torque_nm:5.2f} N·m  {fixed.torque_density_nm_kg:5.2f} N·m/kg"
        f"   |   T={scaled.torque_nm:5.2f} N·m  {scaled.torque_density_nm_kg:5.2f} N·m/kg"
        f"  (mass {scaled.mass_kg:5.2f} kg)"
    )

# %% [markdown]
# Torque is identical in both columns at every pole count. In the fixed
# sweep, torque density rises only ~13% (shorter end turns shrink the copper
# mass). In the design-scaled sweep it rises ~3.4x, because the iron shrinks
# with p. Thinner yokes at constant air-gap shear, not extra torque, are why
# optimized high-pole machines win in robotics joints and direct-drive
# applications. The
# [pole-pair explainer](https://jman4162.github.io/axfluxmdo/guide/analytical-model/#4-pole-pairs-and-torque-a-common-misconception)
# in the docs works through the algebra.
