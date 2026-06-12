"""Bayesian optimization for expensive design evaluations.

The loop reuses the Phase-3 :class:`~axfluxmdo.optimize.problem.DesignProblem`
wholesale (search space, objective/constraint parsing, geometry penalization).
The intended use case is an EXPENSIVE objective — e.g. a GetDP solve via
``expensive_fn`` — but the default path evaluates the cheap analytical model
so tests and examples run anywhere.

Mechanics (v1, documented tradeoffs):

- Initial design: scipy Latin hypercube over continuous/integer variables
  (integers rounded, deduped, topped up), seeded random draws for choices.
- Acquisition: closed-form expected improvement in MINIMIZE space, maximized
  over a seeded candidate pool (half uniform, half perturbations of the
  incumbent) — derivative-free and correct for mixed spaces; gradient
  multistart is future work.
- Constraints: evaluated by the cheap model's g-vector at every evaluated
  point. The incumbent is the best FEASIBLE point. The surrogate trains on
  all points, with infeasible ones assigned a SOFT penalty
  (worst feasible + 10% of the feasible range) — never the 1e9 geometry
  penalty, which would destroy GP smoothness. Probability-of-feasibility
  classifiers are future work.

Human-sign convention: ``BOStudy`` reports objective values with their
natural sign (same invariant as ``ParetoStudy.F``); minimize-space values
live only inside this module.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

try:
    from scipy.stats import norm, qmc
except ImportError as exc:  # pragma: no cover - exercised only without [opt]
    raise ImportError(
        "bayesian_optimize requires scipy; install with: pip install 'axfluxmdo[opt]'"
    ) from exc

from axfluxmdo.geometry.axial_flux import AxialFluxMotor
from axfluxmdo.models.analytical import AnalyticalResult
from axfluxmdo.models.base import Model
from axfluxmdo.operating_point import OperatingPoint
from axfluxmdo.optimize.dataset import DesignDataset
from axfluxmdo.optimize.problem import DesignProblem, Objective, UnknownKeyError
from axfluxmdo.optimize.surrogate import GPSurrogate, Surrogate

SOFT_PENALTY_FRACTION = 0.1


@dataclass
class BOStudy:
    """Result of one Bayesian-optimization run (human-sign objective values)."""

    objective: Objective
    dataset: DesignDataset
    X: list[dict]
    y: np.ndarray  # human sign, evaluation order
    feasible: np.ndarray  # bool mask
    history: np.ndarray  # best-feasible-so-far trace (NaN until first feasible)
    best_x: dict
    best_value: float
    best_result: AnalyticalResult | None
    best_motor: AxialFluxMotor | None
    surrogate: Surrogate
    n_initial: int
    n_iterations: int
    seed: int
    problem: DesignProblem = field(repr=False)

    def summary(self) -> str:
        n_feas = int(self.feasible.sum())
        shown = {
            k: round(float(v), 6) if isinstance(v, float) else v for k, v in self.best_x.items()
        }
        lines = [
            f"BOStudy: {len(self.X)} evaluations ({self.n_initial} initial + "
            f"{self.n_iterations} BO), {n_feas} feasible, seed={self.seed}",
            f"  {self.objective.label}: best = {self.best_value:.6g}",
            f"  best design: {shown}",
        ]
        recs = self.recommend(k=1)
        if recs:
            top = recs[0]
            lines.append(
                f"  top recommendation (risk-adjusted): observed {top['observed']:.6g}, "
                f"surrogate {top['predicted_mean']:.6g} ± {top['predicted_std']:.2g}"
            )
        return "\n".join(lines)

    def recommend(self, k: int = 5, *, risk_aversion: float = 1.0) -> list[dict]:
        """Top-k feasible evaluated designs ranked by surrogate-pessimistic value.

        Score = mean − risk_aversion·σ for maximize objectives (the reverse for
        minimize). Ranks only EVALUATED (verified-feasible) designs —
        uncertainty-aware without certifying unevaluated space.
        """
        entries = []
        for i, x in enumerate(self.X):
            if not self.feasible[i]:
                continue
            mean, std = self.surrogate.predict(self.dataset.encode(x).reshape(1, -1))
            mean_h = self.objective.sense * -float(mean[0])  # minimize-space -> human
            if self.objective.sense > 0:
                score = mean_h - risk_aversion * float(std[0])
            else:
                score = mean_h + risk_aversion * float(std[0])
            entries.append(
                {
                    "x": x,
                    "observed": float(self.y[i]),
                    "predicted_mean": mean_h,
                    "predicted_std": float(std[0]),
                    "score": score,
                }
            )
        reverse = self.objective.sense > 0
        entries.sort(key=lambda e: e["score"], reverse=reverse)
        return entries[:k]


def bayesian_optimize(
    motor: AxialFluxMotor,
    op: OperatingPoint,
    *,
    variables: Mapping[str, object],
    objective: str,
    constraints: Sequence[str] = (),
    model: Model | None = None,
    expensive_fn: Callable[[AxialFluxMotor, OperatingPoint], float | dict] | None = None,
    n_initial: int = 10,
    n_iterations: int = 25,
    seed: int = 1,
    xi: float = 0.01,
    n_candidates: int = 2048,
    surrogate: Surrogate | None = None,
    enforce_model_constraints: bool = True,
) -> BOStudy:
    """Single-objective Bayesian optimization over the motor design space."""
    rng = np.random.default_rng(seed)

    # Objective parsing: through the problem when the key is a model output;
    # manual prefix parsing when expensive_fn supplies a novel key.
    problem_objective, objective_obj = _resolve_objective(
        motor, op, variables, objective, constraints, model, enforce_model_constraints
    )
    problem = problem_objective
    obj = objective_obj

    dataset = DesignDataset(problem.variable_names, choices=problem.choices)

    xs: list[dict] = []
    y_min: list[float] = []  # minimize-space objective values
    g_ok: list[bool] = []
    results: list[AnalyticalResult | None] = []
    motors: list[AxialFluxMotor | None] = []

    def evaluate(x: dict) -> None:
        record = problem.evaluate(x)
        feasible = record.result is not None and bool(np.all(record.g <= 0))
        outputs = dict(record.result.to_dict()) if record.result is not None else {}
        if expensive_fn is not None and record.motor is not None:
            raw = expensive_fn(record.motor, op)
            extra = raw if isinstance(raw, Mapping) else {obj.name: float(raw)}
            outputs |= {k: float(v) for k, v in extra.items()}
        if obj.name in outputs:
            value_min = -outputs[obj.name] if obj.sense > 0 else outputs[obj.name]
        else:  # invalid geometry with expensive_fn never called
            value_min = math.inf
            feasible = False
        xs.append(x)
        y_min.append(value_min)
        g_ok.append(feasible)
        results.append(record.result)
        motors.append(record.motor)
        if outputs:
            dataset.append(x, outputs)

    for x in _initial_design(problem, n_initial, rng):
        evaluate(x)

    surr = surrogate or GPSurrogate(random_state=seed)

    for _ in range(n_iterations):
        X_train, y_train = _training_targets(dataset, xs, y_min, g_ok)
        surr.fit(X_train, y_train)
        incumbent = _incumbent_value(y_min, g_ok)
        candidates = _candidate_pool(problem, xs, y_min, g_ok, n_candidates, rng)
        if not candidates:
            break
        encoded = np.vstack([dataset.encode(c) for c in candidates])
        mean, std = surr.predict(encoded)
        evaluate(candidates[int(np.argmax(_expected_improvement(mean, std, incumbent, xi)))])

    # Final fit on everything for the returned surrogate / recommendations
    X_train, y_train = _training_targets(dataset, xs, y_min, g_ok)
    surr.fit(X_train, y_train)

    y_human = np.array([(-v if obj.sense > 0 else v) for v in y_min])
    feasible = np.array(g_ok)
    history = _history(y_min, g_ok, obj.sense)

    if not feasible.any():
        raise RuntimeError(
            "bayesian_optimize found no feasible designs; relax the constraints, "
            "widen the bounds, or increase n_initial/n_iterations"
        )
    best_idx = int(np.argmin(np.where(feasible, y_min, np.inf)))

    return BOStudy(
        objective=obj,
        dataset=dataset,
        X=xs,
        y=y_human,
        feasible=feasible,
        history=history,
        best_x=xs[best_idx],
        best_value=float(y_human[best_idx]),
        best_result=results[best_idx],
        best_motor=motors[best_idx],
        surrogate=surr,
        n_initial=n_initial,
        n_iterations=n_iterations,
        seed=seed,
        problem=problem,
    )


# -- helpers -------------------------------------------------------------------


def _resolve_objective(
    motor, op, variables, objective, constraints, model, enforce_model_constraints
) -> tuple[DesignProblem, Objective]:
    """Build the DesignProblem; allow expensive_fn objectives over novel keys."""
    if objective.startswith("maximize_"):
        sense, raw = +1, objective[len("maximize_") :]
    elif objective.startswith("minimize_"):
        sense, raw = -1, objective[len("minimize_") :]
    else:
        raise ValueError(f"objective {objective!r} must start with 'maximize_' or 'minimize_'")
    try:
        problem = DesignProblem(
            motor,
            op,
            variables=variables,
            objectives=[objective],
            constraints=constraints,
            model=model,
            enforce_model_constraints=enforce_model_constraints,
        )
        return problem, problem.objectives[0]
    except UnknownKeyError:
        # Novel key supplied by expensive_fn: anchor the problem on a benign
        # built-in objective for evaluation plumbing; the BO loop reads the
        # novel key from the merged outputs.
        problem = DesignProblem(
            motor,
            op,
            variables=variables,
            objectives=["maximize_torque_nm"],
            constraints=constraints,
            model=model,
            enforce_model_constraints=enforce_model_constraints,
        )
        return problem, Objective(name=raw, sense=sense, label=objective)


def _initial_design(problem: DesignProblem, n_initial: int, rng) -> list[dict]:
    cont_names = list(problem.continuous)
    int_names = list(problem.integer)
    box_names = cont_names + int_names
    lo = [problem.continuous[n][0] for n in cont_names] + [problem.integer[n][0] for n in int_names]
    hi = [problem.continuous[n][1] for n in cont_names] + [problem.integer[n][1] for n in int_names]

    designs: list[dict] = []
    seen: set[tuple] = set()

    def add(values: np.ndarray) -> None:
        x: dict = {}
        for name, value in zip(box_names, values, strict=True):
            x[name] = int(round(value)) if name in problem.integer else float(value)
        for name, options in problem.choices.items():
            x[name] = options[int(rng.integers(len(options)))]
        key = tuple(x[n] for n in problem.variable_names)
        if key not in seen:
            seen.add(key)
            designs.append(x)

    if box_names:
        sampler = qmc.LatinHypercube(d=len(box_names), seed=int(rng.integers(2**31)))
        for row in qmc.scale(sampler.random(n_initial), lo, hi):
            add(row)
        while len(designs) < n_initial:  # top up duplicates from integer rounding
            add(rng.uniform(lo, hi))
    else:  # pure-choice space
        while len(designs) < n_initial and len(seen) < _choice_space_size(problem):
            add(np.empty(0))
    return designs


def _choice_space_size(problem: DesignProblem) -> int:
    size = 1
    for options in problem.choices.values():
        size *= len(options)
    return size


def _training_targets(dataset, xs, y_min, g_ok) -> tuple[np.ndarray, np.ndarray]:
    """Soft-penalize infeasible points; never feed 1e9 into the GP."""
    X = np.vstack([dataset.encode(x) for x in xs])
    y = np.array(y_min, dtype=float)
    feasible = np.array(g_ok)
    finite = np.isfinite(y)
    if feasible.any():
        feas_vals = y[feasible & finite]
        penalty = feas_vals.max() + SOFT_PENALTY_FRACTION * max(np.ptp(feas_vals), 1e-9)
        y = np.where(feasible & finite, y, penalty)
    else:
        fallback = y[finite].max() if finite.any() else 0.0
        y = np.where(finite, y, fallback)
    return X, y


def _incumbent_value(y_min, g_ok) -> float:
    vals = [v for v, ok in zip(y_min, g_ok, strict=True) if ok and math.isfinite(v)]
    if vals:
        return min(vals)
    finite = [v for v in y_min if math.isfinite(v)]
    return min(finite) if finite else 0.0


def _expected_improvement(mean, std, incumbent, xi) -> np.ndarray:
    improvement = incumbent - mean - xi
    with np.errstate(divide="ignore", invalid="ignore"):
        z = improvement / std
        ei = improvement * norm.cdf(z) + std * norm.pdf(z)
    return np.where(std > 1e-12, ei, 0.0)


def _candidate_pool(problem, xs, y_min, g_ok, n_candidates, rng) -> list[dict]:
    evaluated = {tuple(x[n] for n in problem.variable_names) for x in xs}

    best_x = None
    best_val = math.inf
    for x, v, ok in zip(xs, y_min, g_ok, strict=True):
        if ok and v < best_val:
            best_x, best_val = x, v
    if best_x is None:
        best_x = xs[int(np.argmin(y_min))]

    candidates: list[dict] = []
    half = n_candidates // 2
    for i in range(n_candidates):
        x: dict = {}
        for name, (lo, hi) in problem.continuous.items():
            if i < half:
                x[name] = float(rng.uniform(lo, hi))
            else:
                sigma = 0.1 * (hi - lo)
                x[name] = float(np.clip(rng.normal(best_x[name], sigma), lo, hi))
        for name, (lo, hi) in problem.integer.items():
            if i < half:
                x[name] = int(rng.integers(lo, hi + 1))
            else:
                sigma = max(1.0, 0.1 * (hi - lo))
                x[name] = int(np.clip(round(rng.normal(best_x[name], sigma)), lo, hi))
        for name, options in problem.choices.items():
            if i < half or rng.random() < 0.2:
                x[name] = options[int(rng.integers(len(options)))]
            else:
                x[name] = best_x[name]
        key = tuple(x[n] for n in problem.variable_names)
        if key in evaluated:
            continue
        try:
            problem.apply(x)  # geometry-validity filter (free, no model evaluation)
        except ValueError:
            continue
        candidates.append(x)
    return candidates


def _history(y_min, g_ok, sense) -> np.ndarray:
    out = np.full(len(y_min), np.nan)
    best = math.inf
    for i, (v, ok) in enumerate(zip(y_min, g_ok, strict=True)):
        if ok and v < best:
            best = v
        if math.isfinite(best):
            out[i] = -best if sense > 0 else best
    return out
