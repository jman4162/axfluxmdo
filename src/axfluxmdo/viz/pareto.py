"""Pareto-front visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from axfluxmdo.optimize.pymoo_runner import ParetoStudy


def plot_pareto(
    study: ParetoStudy,
    *,
    x: str = "torque_density",
    y: str = "efficiency",
    color: str | None = None,
    annotate_best: bool = False,
    show: bool = False,
) -> Figure:
    """Scatter the Pareto front; axes accept aliases, to_dict keys, or design variables.

    A three-objective study renders as x/y position plus the ``color`` channel.
    """
    from axfluxmdo.optimize.problem import resolve_key

    records = study.to_records()
    available = records[0].keys()

    def column(name: str) -> tuple[str, np.ndarray]:
        key = name if name in available else resolve_key(name, available)
        return key, np.array([rec[key] for rec in records])

    x_key, xs = column(x)
    y_key, ys = column(y)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    if color is not None:
        c_key, cs = column(color)
        sc = ax.scatter(xs, ys, c=cs, cmap="viridis", s=45, edgecolors="k", linewidths=0.4)
        fig.colorbar(sc, ax=ax, label=c_key)
    else:
        ax.scatter(xs, ys, s=45, edgecolors="k", linewidths=0.4)

    if annotate_best:
        for name, marker in ((x, "^"), (y, "s")):
            idx = study.best(name)
            ax.scatter(
                xs[idx],
                ys[idx],
                marker=marker,
                s=160,
                facecolors="none",
                edgecolors="r",
                linewidths=1.5,
                label=f"best {name}",
            )
        ax.legend(fontsize=8)

    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title(f"Pareto front ({len(study)} designs)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if show:
        plt.show()
    return fig
