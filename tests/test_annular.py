import dataclasses
import math

import numpy as np
import pytest

from axfluxmdo.geometry.tolerances import GapImperfections
from axfluxmdo.models import AnalyticalModel, AnnularModel


@pytest.fixture
def imperfect_motor(reference_motor):
    """Reference motor with every Phase-2 imperfection switched on."""
    return dataclasses.replace(
        reference_motor,
        tolerances=GapImperfections(gap_offset_m=1e-4, coning_m=2e-4, runout_m=3e-4),
        magnet_shape="rectangular",
    )


class TestPhase1Parity:
    def test_single_slice_matches_analytical_on_every_key(self, reference_motor, reference_op):
        """n_slices=1 with a perfect gap reproduces AnalyticalModel exactly."""
        analytical = AnalyticalModel().evaluate(reference_motor, reference_op).to_dict()
        annular = AnnularModel(n_slices=1).evaluate(reference_motor, reference_op).to_dict()
        for key, expected in analytical.items():
            assert annular[key] == pytest.approx(expected, rel=1e-12), key

    @pytest.mark.parametrize("n_slices", [2, 7, 32, 200])
    def test_torque_emf_exact_at_any_slice_count(self, reference_motor, reference_op, n_slices):
        """The flux-linkage sum is additive: uniform case is exact, not just convergent."""
        analytical = AnalyticalModel().evaluate(reference_motor, reference_op)
        annular = AnnularModel(n_slices=n_slices).evaluate(reference_motor, reference_op)
        assert annular.torque_nm == pytest.approx(analytical.torque_nm, rel=1e-12)
        assert annular.back_emf_v_rms == pytest.approx(analytical.back_emf_v_rms, rel=1e-12)
        assert annular.airgap_flux_density_t == pytest.approx(
            analytical.airgap_flux_density_t, rel=1e-12
        )

    def test_perfect_gap_has_zero_ripple(self, reference_motor, reference_op):
        r = AnnularModel().evaluate(reference_motor, reference_op)
        assert r.torque_ripple_proxy == 0.0


class TestIdentitiesImperfect:
    def test_emf_torque_consistency(self, imperfect_motor, reference_op):
        r = AnnularModel().evaluate(imperfect_motor, reference_op)
        em_power = imperfect_motor.phases * r.back_emf_v_rms * reference_op.current_rms
        assert em_power == pytest.approx(r.torque_nm * reference_op.speed_rad_s, rel=1e-9)

    def test_energy_balance(self, imperfect_motor, reference_op):
        r = AnnularModel().evaluate(imperfect_motor, reference_op)
        assert r.input_power_w == pytest.approx(
            r.output_power_w + r.copper_loss_w + r.core_loss_w + r.mechanical_loss_w, rel=1e-9
        )

    def test_slice_torques_sum_to_total(self, imperfect_motor, reference_op):
        r = AnnularModel().evaluate(imperfect_motor, reference_op)
        assert math.fsum(r.slice_torque_nm) == pytest.approx(r.torque_nm, rel=1e-12)


class TestConvergence:
    def test_imperfect_case_converges(self, imperfect_motor, reference_op):
        r64 = AnnularModel(n_slices=64).evaluate(imperfect_motor, reference_op)
        r256 = AnnularModel(n_slices=256).evaluate(imperfect_motor, reference_op)
        for key in ("torque_nm", "core_loss_w", "axial_force_n", "torque_ripple_proxy"):
            assert r64.to_dict()[key] == pytest.approx(r256.to_dict()[key], rel=1e-3), key

    def test_error_shrinks_with_slice_count(self, imperfect_motor, reference_op):
        ref = AnnularModel(n_slices=512).evaluate(imperfect_motor, reference_op).core_loss_w
        errors = [
            abs(AnnularModel(n_slices=n).evaluate(imperfect_motor, reference_op).core_loss_w - ref)
            for n in (4, 16, 64)
        ]
        assert errors[0] > errors[1] > errors[2]


class TestImperfectionSigns:
    def test_runout_increases_mean_torque_and_creates_ripple(self, reference_motor, reference_op):
        """Convex load line -> Jensen: mean B (and torque) rises slightly with runout."""
        model = AnnularModel()
        perfect = model.evaluate(reference_motor, reference_op)
        with_runout = model.evaluate(
            dataclasses.replace(reference_motor, tolerances=GapImperfections(runout_m=3e-4)),
            reference_op,
        )
        assert with_runout.torque_nm > perfect.torque_nm
        assert with_runout.torque_nm == pytest.approx(perfect.torque_nm, rel=0.05)  # tiny effect
        assert with_runout.torque_ripple_proxy > 0.0

    def test_positive_gap_offset_reduces_torque(self, reference_motor, reference_op):
        model = AnnularModel()
        nominal = model.evaluate(reference_motor, reference_op)
        opened = model.evaluate(
            dataclasses.replace(reference_motor, tolerances=GapImperfections(gap_offset_m=3e-4)),
            reference_op,
        )
        assert opened.torque_nm < nominal.torque_nm

    def test_coning_sign(self, reference_motor, reference_op):
        """Opening the gap outward (+coning) hurts more than closing it (-coning) helps the
        same slices: large radii dominate the area weighting, so +coning reduces torque."""
        model = AnnularModel(n_slices=64)
        nominal = model.evaluate(reference_motor, reference_op)
        opened_out = model.evaluate(
            dataclasses.replace(reference_motor, tolerances=GapImperfections(coning_m=4e-4)),
            reference_op,
        )
        closed_out = model.evaluate(
            dataclasses.replace(reference_motor, tolerances=GapImperfections(coning_m=-4e-4)),
            reference_op,
        )
        assert opened_out.torque_nm < nominal.torque_nm < closed_out.torque_nm

    def test_axial_force_positive_and_grows_as_gap_shrinks(self, reference_motor, reference_op):
        model = AnnularModel()
        nominal = model.evaluate(reference_motor, reference_op)
        tighter = model.evaluate(dataclasses.replace(reference_motor, air_gap=0.0005), reference_op)
        assert nominal.axial_force_n > 0.0
        assert tighter.axial_force_n > nominal.axial_force_n

    def test_rectangular_magnet_arc_clipped(self, reference_motor, reference_op):
        motor = dataclasses.replace(
            reference_motor, magnet_shape="rectangular", magnet_arc_ratio=0.95
        )
        r = AnnularModel(n_slices=64).evaluate(motor, reference_op)
        # alpha(r) = min(1, alpha_m * r_m / r): exceeds 1 unclipped at small radii
        inner_b1_ratio = r.slice_b1_t[0] / r.slice_airgap_b_t[0]
        assert inner_b1_ratio <= 4.0 / math.pi + 1e-12  # sin capped at 1


class TestConstraints:
    def test_saturation_constraint_uses_max_slice_yoke(self, reference_motor, reference_op):
        r = AnnularModel(n_slices=64).evaluate(reference_motor, reference_op)
        by_name = {c.name: c for c in r.constraints}
        assert by_name["core_flux_density_t"].value == pytest.approx(
            float(np.max(r.slice_yoke_b_t)), rel=1e-12
        )
        # Max over radius is at r_o and exceeds the Phase-1 mean-radius proxy
        analytical = AnalyticalModel().evaluate(reference_motor, reference_op)
        by_name_1 = {c.name: c for c in analytical.constraints}
        assert by_name["core_flux_density_t"].value > by_name_1["core_flux_density_t"].value

    def test_imperfect_reference_still_feasible(self, imperfect_motor, reference_op):
        assert AnnularModel().evaluate(imperfect_motor, reference_op).feasible


class TestResultInterface:
    def test_to_dict_is_additive_over_phase1(self, imperfect_motor, reference_op):
        annular_keys = set(AnnularModel().evaluate(imperfect_motor, reference_op).to_dict())
        analytical_keys = set(AnalyticalModel().evaluate(imperfect_motor, reference_op).to_dict())
        assert analytical_keys <= annular_keys
        assert annular_keys - analytical_keys == {"torque_ripple_proxy", "axial_force_n"}

    def test_str_includes_phase2_lines(self, imperfect_motor, reference_op):
        s = str(AnnularModel().evaluate(imperfect_motor, reference_op))
        assert "ripple" in s
        assert "axial force" in s

    def test_slice_arrays_shapes(self, imperfect_motor, reference_op):
        r = AnnularModel(n_slices=32).evaluate(imperfect_motor, reference_op)
        for name in (
            "slice_radii_m",
            "slice_airgap_b_t",
            "slice_b1_t",
            "slice_torque_nm",
            "slice_shear_pa",
            "slice_yoke_b_t",
            "slice_current_loading_a_m",
        ):
            assert getattr(r, name).shape == (32,), name
