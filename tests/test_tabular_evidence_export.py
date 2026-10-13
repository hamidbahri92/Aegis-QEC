"""No silently dropped research fields in CSV and Parquet exports."""
from __future__ import annotations

import csv
import os

import pytest

from a3d.export import export_csv, export_parquet


def _records():
    return [
        {"distance": 3, "logical_failures": 4},
        {"distance": 5, "seed": 901, "physical_error_rate": 0.015},
    ]


def test_csv_export_preserves_all_late_appearing_columns(tmp_path):
    path = tmp_path / "report.csv"
    export_csv(str(path), _records())
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == [
            "distance", "logical_failures",
            "physical_error_rate", "seed",
        ]
        first, second = list(reader)
    assert first["logical_failures"] == "4"
    assert first["seed"] == ""
    assert second["physical_error_rate"] == "0.015"
    assert second["logical_failures"] == ""


def test_parquet_export_preserves_all_late_appearing_columns(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "report.parquet"
    export_parquet(str(path), _records())
    table = pq.read_table(str(path))
    assert table.column_names == [
        "distance", "logical_failures",
        "physical_error_rate", "seed",
    ]
    rows = table.to_pylist()
    assert rows[0]["logical_failures"] == 4
    assert rows[0]["seed"] is None
    assert rows[1]["physical_error_rate"] == pytest.approx(0.015)
    assert rows[1]["logical_failures"] is None


@pytest.mark.parametrize("kind", ("csv", "parquet"))
def test_invalid_records_cannot_overwrite_previous_results(tmp_path, kind):
    if kind == "parquet":
        pytest.importorskip("pyarrow")
    path = tmp_path / f"accepted.{kind}"
    path.write_bytes(b"accepted-record")
    exporter = export_csv if kind == "csv" else export_parquet
    with pytest.raises(TypeError, match="column names"):
        exporter(str(path), [{1: "bad"}])
    assert path.read_bytes() == b"accepted-record"


@pytest.mark.parametrize("kind", ("csv", "parquet"))
def test_interrupted_atomic_export_preserves_prior_result(tmp_path, kind, monkeypatch):
    if kind == "parquet":
        pytest.importorskip("pyarrow")
    path = tmp_path / f"accepted.{kind}"
    exporter = export_csv if kind == "csv" else export_parquet
    exporter(str(path), _records())
    previous = path.read_bytes()

    def fail_replace(src, dst):
        raise OSError("simulated storage failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated storage failure"):
        exporter(str(path), [{"distance": 9}])
    assert path.read_bytes() == previous
    assert not list(tmp_path.glob(".aegis-*.tmp"))
