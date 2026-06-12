"""Bayesian-optimization visualization (matplotlib-only; sklearn objects
arrive inside the study argument, so this module never imports sklearn)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from axfluxmdo.optimize.bayesopt import BOStudy


def plot_convergence(study: BOStudy, *, show: bool = False) -> Figure:
    """Best-feasible-so-far trace vs evaluation index."""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    idx = np.arange(len(study.y))
    feas = study.feasible
    ax.scatter(idx[feas], study.y[feas], s=28, label="feasible evaluation", zorder=3)
    if (~feas).any():
        ax.scatter(
            idx[~feas],
            study.y[~feas],
            s=36,
            marker="x",
            color="r",
            label="infeasible",
            zorder=3,
        )
    ax.plot(idx, study.history, drawstyle="steps-post", lw=1.8, label="best so far")
    ax.axvspan(-0.5, study.n_initial - 0.5, color="0.92", zorder=0, label="initial design")
    ax.set_xlabel("evaluation")
    ax.set_ylabel(study.objective.label)
    ax.set_title(f"BO convergence — best {study.best_value:.4g} after {len(study.y)} evaluations")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_surrogate_slice(study: BOStudy, x_var: str, *, n: int = 100, show: bool = False) -> Figure:
    """GP mean ± 2σ along one variable through the best design (the uncertainty view).

    Other variables are frozen at ``best_x``; evaluated points are projected
    onto the slice axis at their true objective values, so points far from
    the slice can sit away from the band — the band is the surrogate's
    uncertainty ALONG THIS SLICE only.
    """
    problem = study.problem
    tick_labels = None
    if x_var in problem.continuous:
        lo, hi = problem.continuous[x_var]
        sweep_values: list = list(np.linspace(lo, hi, n))
        sweep_axis = np.asarray(sweep_values, dtype=float)
    elif x_var in problem.integer:
        lo, hi = problem.integer[x_var]
        sweep_values = [int(v) for v in range(lo, hi + 1)]
        sweep_axis = np.asarray(sweep_values, dtype=float)
    elif x_var in problem.choices:
        options = problem.choices[x_var]
        sweep_values = list(options)
        if all(isinstance(o, (int, float)) for o in options):
            sweep_axis = np.asarray(options, dtype=float)
        else:  # non-numeric choices: plot over option index, label the ticks
            sweep_axis = np.arange(len(options), dtype=float)
            tick_labels = [str(o) for o in options]
    else:
        raise ValueError(f"unknown design variable {x_var!r}")

    sense = study.objective.sense
    means, stds = [], []
    for value in sweep_values:
        x = dict(study.best_x)
        if x_var in problem.continuous:
            x[x_var] = float(value)
        elif x_var in problem.integer:
            x[x_var] = int(value)
        else:
            x[x_var] = value  # choice option as-is; dataset.encode handles mapping
        row = study.dataset.encode(x).reshape(1, -1)
        mean, std = study.surrogate.predict(row)
        means.append(sense * -float(mean[0]))  # minimize-space -> human
        stds.append(float(std[0]))
    means = np.array(means)
    stds = np.array(stds)
    sweep = sweep_axis

    def to_axis(value) -> float:
        """Map a design value onto the slice axis (index for non-numeric choices)."""
        if tick_labels is not None:
            return float(problem.choices[x_var].index(value))
        return float(value)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(sweep, means, lw=1.8, label="GP mean (slice)")
    ax.fill_between(sweep, means - 2 * stds, means + 2 * stds, alpha=0.25, label="±2σ")
    evaluated = np.array([to_axis(x[x_var]) for x in study.X])
    ax.scatter(
        evaluated[study.feasible],
        study.y[study.feasible],
        s=26,
        color="k",
        label="evaluated (projected)",
        zorder=3,
    )
    best_v = to_axis(study.best_x[x_var])
    ax.scatter([best_v], [study.best_value], marker="*", s=240, color="r", label="best", zorder=4)
    if tick_labels is not None:
        ax.set_xticks(sweep)
        ax.set_xticklabels(tick_labels)
    ax.set_xlabel(x_var)
    ax.set_ylabel(study.objective.label)
    ax.set_title(f"Surrogate slice through the best design — {x_var}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if show:
        plt.show()
    return fig
