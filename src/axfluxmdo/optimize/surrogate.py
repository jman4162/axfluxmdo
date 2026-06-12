"""Gaussian-process and ensemble surrogates over design datasets.

scikit-learn is imported at module level here, but this module is only
reachable lazily from ``axfluxmdo.optimize`` (PEP 562), so the base package
and ``import axfluxmdo.optimize`` stay sklearn-free (test-enforced).

The GP is the primary surrogate: Matern(nu=2.5) with PER-DIMENSION (ARD)
length scales — mandatory for the mixed ordinal/continuous feature encoding —
plus a WhiteKernel jitter. Trustworthiness is judged by cross-validation
(``cv_rmse``/``cv_r2``), not by sklearn's ConvergenceWarning (suppressed: it
fires routinely on small-n fits).
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
except ImportError as exc:  # pragma: no cover - exercised only without [opt]
    raise ImportError(
        "surrogates require scikit-learn; install with: pip install 'axfluxmdo[opt]'"
    ) from exc

if TYPE_CHECKING:
    from axfluxmdo.optimize.dataset import DesignDataset


@runtime_checkable
class Surrogate(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> Surrogate: ...

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...


class GPSurrogate:
    """Gaussian-process surrogate with ARD Matern kernel and CV diagnostics."""

    def __init__(
        self,
        *,
        nu: float = 2.5,
        n_restarts: int = 5,
        random_state: int = 0,
        noise_level: float = 1e-6,
    ):
        self.nu = nu
        self.n_restarts = n_restarts
        self.random_state = random_state
        self.noise_level = noise_level
        self._gp: GaussianProcessRegressor | None = None
        self._x_mean: np.ndarray | None = None
        self._x_std: np.ndarray | None = None
        self._X: np.ndarray | None = None
        self._y: np.ndarray | None = None

    def _make_gp(self, n_dims: int, n_restarts: int) -> GaussianProcessRegressor:
        kernel = ConstantKernel(1.0) * Matern(
            length_scale=np.ones(n_dims), nu=self.nu
        ) + WhiteKernel(self.noise_level, noise_level_bounds=(1e-10, 1e-1))
        return GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=n_restarts,
            random_state=self.random_state,
        )

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        return (X - self._x_mean) / self._x_std

    def fit(self, X: np.ndarray, y: np.ndarray) -> GPSurrogate:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self._x_mean = X.mean(axis=0)
        self._x_std = np.where(X.std(axis=0) > 0, X.std(axis=0), 1.0)
        self._X, self._y = X, y
        self._gp = self._make_gp(X.shape[1], self.n_restarts)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self._gp.fit(self._standardize(X), y)
        return self

    def fit_dataset(self, dataset: DesignDataset, y: str) -> GPSurrogate:
        X, target = dataset.to_arrays(y)
        return self.fit(X, target)

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._gp is None:
            raise RuntimeError("surrogate is not fitted")
        mean, std = self._gp.predict(self._standardize(np.asarray(X, dtype=float)), return_std=True)
        return mean, std

    def predict_dict(self, x: Mapping[str, object], dataset: DesignDataset) -> tuple[float, float]:
        """Predict at one design dict using the dataset's feature encoding."""
        mean, std = self.predict(dataset.encode(x).reshape(1, -1))
        return float(mean[0]), float(std[0])

    def cv_rmse(self, k: int = 5) -> float:
        """Seeded k-fold RMSE on the training data — the honest trust signal."""
        return self._cross_validate(k)[0]

    def cv_r2(self, k: int = 5) -> float:
        return self._cross_validate(k)[1]

    def _cross_validate(self, k: int) -> tuple[float, float]:
        if self._X is None:
            raise RuntimeError("surrogate is not fitted")
        X, y = self._X, self._y
        rng = np.random.default_rng(self.random_state)
        order = rng.permutation(len(y))
        folds = np.array_split(order, k)
        preds = np.empty_like(y)
        for fold in folds:
            train = np.setdiff1d(order, fold)
            sub = GPSurrogate(
                nu=self.nu,
                n_restarts=1,
                random_state=self.random_state,
                noise_level=self.noise_level,
            ).fit(X[train], y[train])
            preds[fold] = sub.predict(X[fold])[0]
        rmse = float(np.sqrt(np.mean((preds - y) ** 2)))
        ss_res = float(np.sum((preds - y) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return rmse, r2


class RandomForestSurrogate:
    """Ensemble surrogate: per-tree spread as the uncertainty estimate.

    The documented fallback for non-smooth responses; satisfies the same
    Surrogate protocol as the GP.
    """

    def __init__(self, *, n_estimators: int = 200, random_state: int = 0):
        self._rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> RandomForestSurrogate:
        self._rf.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=float))
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=float)
        per_tree = np.stack([tree.predict(X) for tree in self._rf.estimators_])
        return per_tree.mean(axis=0), per_tree.std(axis=0)
