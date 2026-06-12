import numpy as np
import pytest

from axfluxmdo.models import AnnularModel
from axfluxmdo.optimize import (
    DesignProblem,
    parse_constraint,
    parse_objective,
    resolve_key,
)

AVAILABLE = {"torque_nm", "torque_density_nm_kg", "efficiency", "winding_temp_c", "mass_kg"}


class TestResolveKey:
    def test_alias_and_canonical_both_resolve(self):
        assert resolve_key("torque_density", AVAILABLE) == "torque_density_nm_kg"
        assert resolve_key("torque_density_nm_kg", AVAILABLE) == "torque_density_nm_kg"

    def test_unknown_key_lists_options(self):
        with pytest.raises(ValueError, match="torque_density_nm_kg"):
            resolve_key("not_a_key", AVAILABLE)


class TestParseObjective:
    def test_maximize_and_minimize(self):
        obj = parse_objective("maximize_torque_density", AVAILABLE)
        assert obj.name == "torque_density_nm_kg"
        assert obj.sense == +1
        obj = parse_objective("minimize_mass", AVAILABLE)
        assert obj.name == "mass_kg"
        assert obj.sense == -1

    def test_bare_key_rejected(self):
        with pytest.raises(ValueError, match="maximize_"):
            parse_objective("torque_density", AVAILABLE)

    def test_feasible_rejected(self):
        with pytest.raises(ValueError, match="feasible"):
            parse_objective("maximize_feasible", AVAILABLE | {"feasible"})


class TestParseConstraint:
    @pytest.mark.parametrize(
        "spec,op,bound",
        [
            ("winding_temp_c < 140", "<", 140.0),
            ("winding_temp_c<=140", "<=", 140.0),
            (" efficiency > 0.9 ", ">", 0.9),
            ("mass_kg >= 1.5e-1", ">=", 0.15),
        ],
    )
    def test_valid_forms(self, spec, op, bound):
        c = parse_constraint(spec, AVAILABLE)
        assert c.op == op
        assert c.bound == bound

    def test_alias_in_constraint(self):
        c = parse_constraint("winding_temp < 140", AVAILABLE)
        assert c.key == "winding_temp_c"

    @pytest.mark.parametrize("spec", ["winding_temp_c = 140", "x ~ 1", "nonsense"])
    def test_garbage_rejected(self, spec):
        with pytest.raises(ValueError):
            parse_constraint(spec, AVAILABLE)

    def test_violation_signs(self):
        lt = parse_constraint("winding_temp_c < 140", AVAILABLE)
        assert lt.violation(120.0) < 0  # satisfied
        assert lt.violation(160.0) > 0  # violated
        gt = parse_constraint("efficiency > 0.9", AVAILABLE)
        assert gt.violation(0.95) < 0
        assert gt.violation(0.85) > 0


class TestDesignProblem:
    def make(self, reference_motor, reference_op, **kw):
        defaults = dict(
            variables={
                "outer_radius": (0.05, 0.12),
                "pole_pairs": [8, 10, 12, 14],
                "turns_per_phase": (10, 40),
                "tolerances.runout_m": (0.0, 3e-4),
            },
            objectives=["maximize_torque_density", "minimize_mass"],
            constraints=["winding_temp_c < 140"],
        )
        defaults.update(kw)
        return DesignProblem(reference_motor, reference_op, **defaults)

    def test_variable_classification(self, reference_motor, reference_op):
        p = self.make(reference_motor, reference_op)
        assert "outer_radius" in p.continuous
        assert "tolerances.runout_m" in p.continuous  # float field, tuple spec
        assert p.integer["turns_per_phase"] == (10, 40)  # int field, tuple spec
        assert p.choices["pole_pairs"] == [8, 10, 12, 14]

    def test_unknown_variable_fails_fast(self, reference_motor, reference_op):
        with pytest.raises(ValueError, match="unknown design variable"):
            self.make(reference_motor, reference_op, variables={"bogus_field": (0, 1)})

    def test_evaluate_baseline_point(self, reference_motor, reference_op):
        p = self.make(reference_motor, reference_op)
        record = p.evaluate(
            {
                "outer_radius": 0.08,
                "pole_pairs": 14,
                "turns_per_phase": 24,
                "tolerances.runout_m": 0.0,
            }
        )
        assert record.result is not None
        # maximize objectives negated in minimize space
        d = record.result.to_dict()
        assert record.f_min[0] == pytest.approx(-d["torque_density_nm_kg"])
        assert record.f_min[1] == pytest.approx(d["mass_kg"])
        assert np.all(record.g <= 0)  # baseline is feasible

    def test_invalid_geometry_penalized_not_raised(self, reference_motor, reference_op):
        p = self.make(reference_motor, reference_op)
        # outer_radius below inner_radius (0.025) -> __post_init__ ValueError path
        record = p.evaluate(
            {
                "outer_radius": 0.05,
                "pole_pairs": 14,
                "turns_per_phase": 24,
                "tolerances.runout_m": 2.9e-4,
            }
        )
        assert record.result is not None  # 0.05 > 0.025 is actually valid; use a real invalid
        bad = p.evaluate(
            {
                "outer_radius": 0.01,  # below inner radius -> invalid
                "pole_pairs": 14,
                "turns_per_phase": 24,
                "tolerances.runout_m": 0.0,
            }
        )
        assert bad.result is None
        assert bad.motor is None
        assert np.all(np.isfinite(bad.f_min))
        assert np.all(bad.f_min >= 1e8)
        assert np.all(bad.g > 0)

    def test_model_constraints_wired_as_negative_margin(self, reference_motor, reference_op):
        p = self.make(reference_motor, reference_op)
        x = {
            "outer_radius": 0.08,
            "pole_pairs": 14,
            "turns_per_phase": 24,
            "tolerances.runout_m": 0.0,
        }
        record = p.evaluate(x)
        margins = [c.margin for c in record.result.constraints]
        n_user = 1
        for g_val, margin in zip(record.g[n_user:], margins, strict=True):
            assert g_val == pytest.approx(-margin)

    def test_annular_only_keys_available_with_annular_model(self, reference_motor, reference_op):
        p = self.make(
            reference_motor,
            reference_op,
            model=AnnularModel(n_slices=8),
            objectives=["maximize_torque_density", "minimize_ripple"],
        )
        assert p.objectives[1].name == "torque_ripple_proxy"

    def test_enforce_model_constraints_off(self, reference_motor, reference_op):
        p = self.make(reference_motor, reference_op, enforce_model_constraints=False)
        assert p.n_constr == 1  # only the user constraint
