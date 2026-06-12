import numpy as np
import pytest

pytest.importorskip("sklearn")

from axfluxmdo.optimize import bayesian_optimize  # noqa: E402

VARIABLES = {
    "outer_radius": (0.06, 0.10),
    "fill_factor": (0.35, 0.55),
    "pole_pairs": [10, 12, 14, 16],
}
CONSTRAINTS = ["winding_temp_c < 140"]
BUDGET = dict(n_initial=6, n_iterations=8, seed=5)


@pytest.fixture(scope="module")
def study():
    from axfluxmdo import AxialFluxMotor, OperatingPoint

    motor = AxialFluxMotor(outer_radius=0.08, inner_radius=0.025, air_gap=0.0008, pole_pairs=14)
    op = OperatingPoint(speed_rpm=500, current_rms=25, dc_bus_voltage=48)
    return bayesian_optimize(
        motor,
        op,
        variables=VARIABLES,
        objective="maximize_torque_density",
        constraints=CONSTRAINTS,
        **BUDGET,
    )


class TestBayesianOptimize:
    def test_improves_over_initial_design(self, study):
        initial_best = np.nanmax(
            np.where(study.feasible[: study.n_initial], study.y[: study.n_initial], np.nan)
        )
        assert study.best_value >= initial_best

    def test_bounds_and_choices_respected(self, study):
        for x in study.X:
            assert 0.06 <= x["outer_radius"] <= 0.10
            assert 0.35 <= x["fill_factor"] <= 0.55
            assert x["pole_pairs"] in [10, 12, 14, 16]
            assert isinstance(x["pole_pairs"], int)

    def test_best_is_feasible(self, study):
        assert study.best_result is not None
        assert study.best_result.feasible
        assert study.best_result.to_dict()["winding_temp_c"] < 140

    def test_history_monotone_nondecreasing(self, study):
        h = study.history[~np.isnan(study.history)]
        assert np.all(np.diff(h) >= -1e-12)

    def test_dataset_records_every_evaluation(self, study):
        assert len(study.dataset) == study.n_initial + study.n_iterations
        assert len(study.X) == len(study.dataset)

    def test_recommend_returns_feasible_with_uncertainty(self, study):
        recs = study.recommend(k=3)
        assert 1 <= len(recs) <= 3
        for rec in recs:
            assert rec["predicted_std"] >= 0
            assert "observed" in rec and "score" in rec

    def test_summary(self, study):
        s = study.summary()
        assert "maximize_torque_density" in s
        assert "best" in s


class TestDeterminism:
    def test_same_seed_same_trajectory(self, study):
        from axfluxmdo import AxialFluxMotor, OperatingPoint

        motor = AxialFluxMotor(outer_radius=0.08, inner_radius=0.025, air_gap=0.0008, pole_pairs=14)
        op = OperatingPoint(speed_rpm=500, current_rms=25, dc_bus_voltage=48)
        again = bayesian_optimize(
            motor,
            op,
            variables=VARIABLES,
            objective="maximize_torque_density",
            constraints=CONSTRAINTS,
            **BUDGET,
        )
        np.testing.assert_allclose(again.y, study.y)
        assert again.best_x == study.best_x


class TestExpensiveFn:
    @pytest.mark.parametrize("prefix,sense", [("maximize", +1), ("minimize", -1)])
    def test_expensive_fn_drives_objective(self, reference_motor, reference_op, prefix, sense):
        calls = []

        def expensive(motor, op):
            value = motor.outer_radius * 100.0  # novel objective: bigger r_o wins/loses
            calls.append(value)
            return {"my_metric": value}

        study = bayesian_optimize(
            reference_motor,
            reference_op,
            variables={"outer_radius": (0.06, 0.10)},
            objective=f"{prefix}_my_metric",
            expensive_fn=expensive,
            n_initial=5,
            n_iterations=5,
            seed=2,
        )
        assert len(calls) == 10  # called once per evaluation
        # BO should push toward the corresponding bound
        if sense > 0:
            assert study.best_x["outer_radius"] > 0.09
        else:
            assert study.best_x["outer_radius"] < 0.07
        assert study.best_value == pytest.approx(study.best_x["outer_radius"] * 100, rel=1e-9)
