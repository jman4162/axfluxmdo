"""DesignDataset tests — deliberately import no sklearn (numpy-only layer)."""

import numpy as np
import pytest

from axfluxmdo.optimize import DesignDataset, DesignProblem


@pytest.fixture
def ds():
    d = DesignDataset(
        ["outer_radius", "pole_pairs", "winding_scheme"],
        choices={"pole_pairs": [8, 10, 12], "winding_scheme": ["wave", "lap"]},
    )
    d.append(
        {"outer_radius": 0.08, "pole_pairs": 10, "winding_scheme": "lap"},
        {"torque_nm": 8.5, "torque_density_nm_kg": 2.3, "efficiency": 0.95},
    )
    d.append(
        {"outer_radius": 0.09, "pole_pairs": 12, "winding_scheme": "wave"},
        {"torque_nm": 9.1, "torque_density_nm_kg": 2.1, "efficiency": 0.96},
    )
    return d


class TestBasics:
    def test_len_iter_order(self, ds):
        assert len(ds) == 2
        assert [rec["x"]["pole_pairs"] for rec in ds] == [10, 12]

    def test_missing_variable_rejected(self, ds):
        with pytest.raises(ValueError, match="missing"):
            ds.append({"outer_radius": 0.07}, {"torque_nm": 1.0})


class TestEncoding:
    def test_numeric_choice_uses_value_nonnumeric_uses_index(self, ds):
        row = ds.encode({"outer_radius": 0.08, "pole_pairs": 10, "winding_scheme": "lap"})
        np.testing.assert_allclose(row, [0.08, 10.0, 1.0])  # "lap" is index 1

    def test_feature_matrix_order(self, ds):
        X = ds.feature_matrix()
        assert X.shape == (2, 3)
        np.testing.assert_allclose(X[:, 0], [0.08, 0.09])  # variable_names order

    def test_to_arrays_with_alias(self, ds):
        X, y = ds.to_arrays("torque_density")  # alias resolves
        np.testing.assert_allclose(y, [2.3, 2.1])
        with pytest.raises(ValueError, match="unknown"):
            ds.to_arrays("nonsense_key")


class TestDedupe:
    def test_keeps_first_drops_duplicate(self, ds):
        ds.append(
            {"outer_radius": 0.08, "pole_pairs": 10, "winding_scheme": "lap"},
            {"torque_nm": 999.0, "torque_density_nm_kg": 9.9, "efficiency": 0.5},
        )
        deduped = ds.dedupe()
        assert len(deduped) == 2
        assert deduped.records[0]["outputs"]["torque_nm"] == 8.5  # first kept


class TestPersistence:
    def test_round_trip(self, ds, tmp_path):
        path = tmp_path / "designs.jsonl"
        ds.save(path)
        loaded = DesignDataset.load(path)
        assert loaded.variable_names == ds.variable_names
        assert loaded.choices == ds.choices
        assert loaded.records == ds.records

    def test_header_format_pinned(self, ds, tmp_path):
        import json

        path = tmp_path / "designs.jsonl"
        ds.save(path)
        header = json.loads(path.read_text().splitlines()[0])
        assert header["format"] == "axfluxmdo-dataset-v1"

    def test_unknown_format_rejected(self, tmp_path):
        bad = tmp_path / "bad.jsonl"
        bad.write_text('{"format": "v999", "variables": []}\n')
        with pytest.raises(ValueError, match="unknown dataset format"):
            DesignDataset.load(bad)


class TestFromEvaluations:
    def test_records_real_results(self, reference_motor, reference_op):
        problem = DesignProblem(
            reference_motor,
            reference_op,
            variables={"outer_radius": (0.06, 0.10)},
            objectives=["maximize_torque_density"],
        )
        ds = DesignDataset.from_evaluations(
            problem, [{"outer_radius": 0.07}, {"outer_radius": 0.09}]
        )
        assert len(ds) == 2
        X, y = ds.to_arrays("torque_density_nm_kg")
        assert np.all(y > 0)
