import numpy as np
import pytest

pytest.importorskip("sklearn")

from scipy.stats import qmc  # noqa: E402

from axfluxmdo.optimize import DesignProblem  # noqa: E402
from axfluxmdo.optimize.dataset import DesignDataset  # noqa: E402
from axfluxmdo.optimize.surrogate import (  # noqa: E402
    GPSurrogate,
    RandomForestSurrogate,
    Surrogate,
)

BOUNDS_LO = [0.05, 0.0005]
BOUNDS_HI = [0.12, 0.0015]


@pytest.fixture(scope="module")
def training_data():
    """40 LHS evaluations of torque density over (outer_radius, air_gap)."""
    from axfluxmdo import AxialFluxMotor, OperatingPoint

    motor = AxialFluxMotor(outer_radius=0.08, inner_radius=0.025, air_gap=0.0008, pole_pairs=14)
    op = OperatingPoint(speed_rpm=500, current_rms=25, dc_bus_voltage=48)
    problem = DesignProblem(
        motor,
        op,
        variables={"outer_radius": (0.05, 0.12), "air_gap": (0.0005, 0.0015)},
        objectives=["maximize_torque_density"],
    )
    sampler = qmc.LatinHypercube(d=2, seed=11)
    points = qmc.scale(sampler.random(50), BOUNDS_LO, BOUNDS_HI)
    ds = DesignDataset.from_evaluations(
        problem, [{"outer_radius": p[0], "air_gap": p[1]} for p in points]
    )
    X, y = ds.to_arrays("torque_density")
    return ds, X, y


class TestGPSurrogate:
    def test_held_out_accuracy(self, training_data):
        _, X, y = training_data
        gp = GPSurrogate().fit(X[:40], y[:40])
        pred, _ = gp.predict(X[40:])
        ss_res = np.sum((pred - y[40:]) ** 2)
        ss_tot = np.sum((y[40:] - y[40:].mean()) ** 2)
        assert 1 - ss_res / ss_tot > 0.95

    def test_uncertainty_grows_away_from_data(self, training_data):
        _, X, y = training_data
        gp = GPSurrogate().fit(X, y)
        _, std_at_train = gp.predict(X[:5])
        far = np.array([[0.5, 0.05]])  # far outside the sampled box
        _, std_far = gp.predict(far)
        assert std_far[0] > 5 * std_at_train.mean()

    def test_predict_dict_matches_predict(self, training_data):
        ds, X, y = training_data
        gp = GPSurrogate().fit(X, y)
        x = ds.records[0]["x"]
        mean_d, std_d = gp.predict_dict(x, ds)
        mean_m, std_m = gp.predict(ds.encode(x).reshape(1, -1))
        assert mean_d == pytest.approx(float(mean_m[0]), rel=1e-12)
        assert std_d == pytest.approx(float(std_m[0]), rel=1e-12)

    def test_cv_diagnostics_finite_and_good(self, training_data):
        _, X, y = training_data
        gp = GPSurrogate().fit(X, y)
        rmse = gp.cv_rmse(k=5)
        r2 = gp.cv_r2(k=5)
        assert np.isfinite(rmse) and rmse < 0.1 * np.ptp(y)
        assert r2 > 0.9

    def test_deterministic(self, training_data):
        _, X, y = training_data
        p1, _ = GPSurrogate(random_state=3).fit(X, y).predict(X[:7])
        p2, _ = GPSurrogate(random_state=3).fit(X, y).predict(X[:7])
        np.testing.assert_array_equal(p1, p2)

    def test_unfitted_raises(self):
        with pytest.raises(RuntimeError, match="not fitted"):
            GPSurrogate().predict(np.zeros((1, 2)))


class TestRandomForestSurrogate:
    def test_protocol_and_shapes(self, training_data):
        _, X, y = training_data
        rf = RandomForestSurrogate(random_state=0).fit(X, y)
        assert isinstance(rf, Surrogate)
        mean, std = rf.predict(X[:9])
        assert mean.shape == (9,) and std.shape == (9,)
        assert np.all(std >= 0)
        # in-sample mean tracks the data reasonably
        full_mean, _ = rf.predict(X)
        assert np.corrcoef(full_mean, y)[0, 1] > 0.9
