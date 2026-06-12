import numpy as np
import pytest

pytest.importorskip("openmdao.api")

from axfluxmdo.optimize import DesignProblem, run_openmdao_demo  # noqa: E402
from axfluxmdo.optimize.openmdao_components import MotorComponent, build_motor_group  # noqa: E402


@pytest.fixture
def problem(reference_motor, reference_op):
    return DesignProblem(
        reference_motor,
        reference_op,
        variables={"outer_radius": (0.06, 0.10), "fill_factor": (0.35, 0.55)},
        objectives=["maximize_torque_density"],
        constraints=["winding_temp_c < 140"],
    )


class TestMotorComponent:
    def test_compute_matches_direct_evaluate(self, problem):
        import openmdao.api as om

        om_prob = om.Problem(reports=False)
        om_prob.model.add_subsystem("motor", MotorComponent(problem=problem), promotes=["*"])
        om_prob.setup()
        om_prob.set_val("outer_radius", 0.08)
        om_prob.set_val("fill_factor", 0.45)
        om_prob.run_model()

        record = problem.evaluate({"outer_radius": 0.08, "fill_factor": 0.45})
        d = record.result.to_dict()
        assert om_prob.get_val("torque_density_nm_kg")[0] == pytest.approx(
            d["torque_density_nm_kg"], rel=1e-12
        )
        assert om_prob.get_val("g_0")[0] == pytest.approx(record.g[0], rel=1e-12)

    def test_partials_finite_and_sane(self, problem):
        import openmdao.api as om

        om_prob = om.Problem(reports=False)
        om_prob.model.add_subsystem("motor", MotorComponent(problem=problem), promotes=["*"])
        om_prob.setup(force_alloc_complex=False)
        om_prob.run_model()
        data = om_prob.check_partials(method="fd", step=1e-5, out_stream=None)
        for (_of, _wrt), info in data["motor"].items():
            assert np.all(np.isfinite(info["J_fwd"]))


class TestSLSQPDemo:
    def test_refinement_improves_objective(self, reference_motor, reference_op, problem):
        baseline = problem.baseline_result.to_dict()["torque_density_nm_kg"]
        out = run_openmdao_demo(
            reference_motor,
            reference_op,
            variables={"outer_radius": (0.06, 0.10), "fill_factor": (0.35, 0.55)},
            objective="maximize_torque_density",
            constraints=("winding_temp_c < 140",),
        )
        assert out["success"]
        assert out["result"]["torque_density_nm_kg"] > baseline
        assert out["result"]["winding_temp_c"] < 140
        assert out["result"]["feasible"] == 1.0

    def test_group_builder_wires_constraints(self, problem):
        om_prob = build_motor_group(problem)
        om_prob.setup()
        om_prob.run_model()
        # one user constraint + six model constraints
        assert problem.n_constr == 7
