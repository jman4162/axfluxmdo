import numpy as np
import pytest

pytest.importorskip("pymoo")

from axfluxmdo.optimize import optimize_pareto  # noqa: E402

VARIABLES = {
    "outer_radius": (0.06, 0.10),
    "pole_pairs": [10, 12, 14, 16],
    "fill_factor": (0.35, 0.55),
}
OBJECTIVES = ["maximize_torque_density", "maximize_efficiency"]
CONSTRAINTS = ["winding_temp_c < 140", "electrical_frequency_hz < 1000"]
BUDGET = dict(pop_size=12, n_gen=5, seed=7)


# module-scoped fixtures can't use the function-scoped conftest fixtures; rebuild them
@pytest.fixture(scope="module")
def reference_motor_module():
    from axfluxmdo import AxialFluxMotor

    return AxialFluxMotor(
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


@pytest.fixture(scope="module")
def reference_op_module():
    from axfluxmdo import OperatingPoint

    return OperatingPoint(speed_rpm=500, current_rms=25, dc_bus_voltage=48)


@pytest.fixture(scope="module")
def pareto(reference_motor_module, reference_op_module):
    return optimize_pareto(
        reference_motor_module,
        reference_op_module,
        variables=VARIABLES,
        objectives=OBJECTIVES,
        constraints=CONSTRAINTS,
        **BUDGET,
    )


class TestOptimizePareto:
    def test_all_points_feasible(self, pareto):
        assert len(pareto) > 0
        for result in pareto.results:
            assert result.feasible
            d = result.to_dict()
            assert d["winding_temp_c"] < 140
            assert d["electrical_frequency_hz"] < 1000

    def test_variables_respect_bounds_and_choices(self, pareto):
        for x in pareto.X:
            assert 0.06 <= x["outer_radius"] <= 0.10
            assert x["pole_pairs"] in [10, 12, 14, 16]
            assert 0.35 <= x["fill_factor"] <= 0.55

    def test_front_is_non_dominated(self, pareto):
        # Convert to minimize space and check no point dominates another
        senses = np.array([obj.sense for obj in pareto.objectives])
        f_min = -senses * pareto.F
        n = len(pareto)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                dominates = np.all(f_min[i] <= f_min[j]) and np.any(f_min[i] < f_min[j])
                assert not dominates, f"point {i} dominates point {j}"

    def test_f_is_human_readable(self, pareto):
        # maximize objectives are NOT negated: torque density and efficiency positive
        assert np.all(pareto.F[:, 0] > 0)  # N·m/kg, ~1-10
        assert np.all((pareto.F[:, 1] > 0) & (pareto.F[:, 1] < 1))  # efficiency

    def test_reproducible_with_same_seed(self, reference_motor_module, reference_op_module, pareto):
        again = optimize_pareto(
            reference_motor_module,
            reference_op_module,
            variables=VARIABLES,
            objectives=OBJECTIVES,
            constraints=CONSTRAINTS,
            **BUDGET,
        )
        np.testing.assert_allclose(again.F, pareto.F)

    def test_records_arrays_best(self, pareto):
        records = pareto.to_records()
        assert len(records) == len(pareto)
        assert "outer_radius" in records[0] and "efficiency" in records[0]
        arrays = pareto.to_arrays("torque_density", "efficiency", "pole_pairs")
        assert all(v.shape == (len(pareto),) for v in arrays.values())
        idx = pareto.best("torque_density")
        assert arrays["torque_density"][idx] == pytest.approx(arrays["torque_density"].max())

    def test_summary(self, pareto):
        s = pareto.summary()
        assert "feasible non-dominated" in s
        assert "maximize_torque_density" in s

    def test_impossible_constraint_raises(self, reference_motor_module, reference_op_module):
        with pytest.raises(RuntimeError, match="no feasible"):
            optimize_pareto(
                reference_motor_module,
                reference_op_module,
                variables={"outer_radius": (0.06, 0.10)},
                objectives=["maximize_torque_density"],
                constraints=["efficiency > 1.5"],
                pop_size=8,
                n_gen=3,
                seed=1,
            )
