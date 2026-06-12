"""Tornado chart for one-at-a-time design sensitivities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from axfluxmdo.optimize.sensitivity import SensitivityResult


def plot_tornado(sens: SensitivityResult, *, show: bool = False) -> Figure:
    """Horizontal diverging bars around the baseline output, largest swing on top."""
    entries = sens.entries
    fig, ax = plt.subplots(figsize=(8, 0.6 * max(4, len(entries)) + 1.2))

    y_positions = range(len(entries) - 1, -1, -1)  # largest swing at the top
    for y_pos, entry in zip(y_positions, entries, strict=True):
        low_delta = entry.low_output - sens.baseline
        high_delta = entry.high_output - sens.baseline
        ax.barh(y_pos, low_delta, left=sens.baseline, height=0.6, color="#1f77b4", alpha=0.85)
        ax.barh(y_pos, high_delta, left=sens.baseline, height=0.6, color="#d62728", alpha=0.85)
        ax.annotate(
            f"{entry.low_input:.4g}",
            xy=(sens.baseline + low_delta, y_pos),
            xytext=(-4, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=8,
        )
        ax.annotate(
            f"{entry.high_input:.4g}",
            xy=(sens.baseline + high_delta, y_pos),
            xytext=(4, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8,
        )

    ax.axvline(sens.baseline, color="k", lw=1)
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels([e.variable for e in entries])
    ax.set_xlabel(sens.output)
    ax.set_title(f"Sensitivity of {sens.output} (baseline {sens.baseline:.4g})")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    if show:
        plt.show()
    return fig
