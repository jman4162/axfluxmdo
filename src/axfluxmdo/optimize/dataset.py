"""Dataset of evaluated designs (numpy-only; usable without the [opt] extra).

Each record pairs a design vector ``x`` (dict over the declared variables)
with the flat ``outputs`` dict of its evaluation (``result.to_dict()``, plus
any extra keys such as an expensive-FEA objective).

Feature encoding for surrogates is **ordinal-as-float**: continuous and
integer variables are cast to float; Choice variables use the option VALUE
when numeric (e.g. ``pole_pairs`` — an ordered physical quantity) and the
option INDEX otherwise. One-hot encoding for genuinely unordered categoricals
is documented future work; the GP's per-dimension ARD length scales absorb
the differing column scales.

Persistence is JSON Lines with a versioned header
(``axfluxmdo-dataset-v1``) — dependency-free, append-friendly, git-diffable;
these datasets are O(10^2..10^3) rows, so columnar formats buy nothing.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from axfluxmdo.optimize.problem import resolve_key

if TYPE_CHECKING:
    from axfluxmdo.optimize.problem import DesignProblem
    from axfluxmdo.optimize.pymoo_runner import ParetoStudy

FORMAT_VERSION = "axfluxmdo-dataset-v1"


class DesignDataset:
    """Ordered collection of (design vector, evaluation outputs) records."""

    def __init__(
        self,
        variable_names: Sequence[str],
        *,
        choices: Mapping[str, list] | None = None,
    ):
        self.variable_names = list(variable_names)
        self.choices = {k: list(v) for k, v in (choices or {}).items()}
        self.records: list[dict] = []

    # -- construction ----------------------------------------------------------

    @classmethod
    def from_study(cls, study: ParetoStudy) -> DesignDataset:
        """Wrap a ParetoStudy's points without re-evaluating anything."""
        ds = cls(study.variables, choices=study.problem.choices)
        for x, result in zip(study.X, study.results, strict=True):
            ds.append(x, result.to_dict())
        return ds

    @classmethod
    def from_evaluations(
        cls, problem: DesignProblem, xs: Iterable[Mapping[str, object]]
    ) -> DesignDataset:
        """Evaluate each design with the problem's model and record it."""
        ds = cls(problem.variable_names, choices=problem.choices)
        for x in xs:
            record = problem.evaluate(x)
            if record.result is not None:
                ds.append(x, record.result.to_dict())
        return ds

    # -- mutation ----------------------------------------------------------------

    def append(self, x: Mapping[str, object], outputs: Mapping[str, float]) -> None:
        missing = set(self.variable_names) - set(x)
        if missing:
            raise ValueError(f"design vector missing variables: {sorted(missing)}")
        self.records.append({"x": dict(x), "outputs": dict(outputs)})

    def extend(self, records: Iterable[tuple[Mapping, Mapping]]) -> None:
        for x, outputs in records:
            self.append(x, outputs)

    # -- access ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    def encode(self, x: Mapping[str, object]) -> np.ndarray:
        """One design dict -> feature row (fixed variable order)."""
        row = []
        for name in self.variable_names:
            value = x[name]
            if name in self.choices and not isinstance(value, (int, float)):
                value = self.choices[name].index(value)
            row.append(float(value))
        return np.array(row)

    def feature_matrix(self) -> np.ndarray:
        """(n, d) float matrix, columns in ``variable_names`` order."""
        if not self.records:
            return np.empty((0, len(self.variable_names)))
        return np.vstack([self.encode(rec["x"]) for rec in self.records])

    def to_arrays(self, y: str) -> tuple[np.ndarray, np.ndarray]:
        """(X, y) for surrogate fitting; ``y`` accepts aliases or output keys."""
        if not self.records:
            raise ValueError("dataset is empty")
        available = self.records[0]["outputs"].keys()
        key = y if y in available else resolve_key(y, available)
        return self.feature_matrix(), np.array([rec["outputs"][key] for rec in self.records])

    def dedupe(self, *, tol: float = 1e-12) -> DesignDataset:
        """Drop records whose feature rows duplicate an earlier one (keep first)."""
        out = DesignDataset(self.variable_names, choices=self.choices)
        seen: list[np.ndarray] = []
        for rec in self.records:
            row = self.encode(rec["x"])
            if any(np.all(np.abs(row - s) <= tol) for s in seen):
                continue
            seen.append(row)
            out.records.append(rec)
        return out

    # -- persistence ---------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        with path.open("w") as fh:
            header = {
                "format": FORMAT_VERSION,
                "variables": self.variable_names,
                "choices": self.choices,
            }
            fh.write(json.dumps(header) + "\n")
            for rec in self.records:
                fh.write(json.dumps(rec) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> DesignDataset:
        path = Path(path)
        with path.open() as fh:
            header = json.loads(fh.readline())
            if header.get("format") != FORMAT_VERSION:
                raise ValueError(
                    f"{path}: unknown dataset format {header.get('format')!r}; "
                    f"expected {FORMAT_VERSION!r}"
                )
            ds = cls(header["variables"], choices=header.get("choices") or {})
            for line in fh:
                line = line.strip()
                if line:
                    ds.records.append(json.loads(line))
        return ds
