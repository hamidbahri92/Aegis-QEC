"""Safety and genuine C-compatibility tests for fixed-point hardware exports."""
from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest

from a3d.export import export_fixed_point_header


def test_fixed_point_rounding_saturation_and_c_compilation(tmp_path):
    path = tmp_path / "weights.h"
    export_fixed_point_header(
        str(path),
        {"weights": np.array([[-3.0, -0.125], [0.125, 5e100]])},
        {"weights": 0.25},
        meta={"provenance": "unit-test"},
    )
    code = path.read_text(encoding="utf-8")
    assert "#define WEIGHTS_SCALE 0.25" in code
    assert "#define WEIGHTS_DIM0 2" in code
    assert "#define WEIGHTS_DIM1 2" in code
    assert "  -12," in code
    assert "  0," in code
    assert "  32767," in code
    assert "provenance: unit-test" in code

    compiler = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if compiler:
        source = tmp_path / "check.c"
        source.write_text(
            '#include "weights.h"\n'
            "int main(void) { return weights[0] == -12 ? 0 : 1; }\n",
            encoding="utf-8",
        )
        subprocess.run(
            [compiler, "-std=c11", "-fsyntax-only", str(source)],
            check=True, cwd=str(tmp_path), capture_output=True, text=True,
        )


@pytest.mark.parametrize(
    ("name", "values", "scale"),
    [
        ("bad-name", [1.0], 1.0),
        ("valid", [float("nan")], 1.0),
        ("valid", [float("inf")], 1.0),
        ("valid", [2.0], 0.0),
        ("valid", [2.0], float("nan")),
        ("valid", [2.0], -1.0),
        ("valid", [], 1.0),
    ],
)
def test_invalid_hardware_artifacts_do_not_replace_prior_evidence(
    tmp_path, name, values, scale
):
    path = tmp_path / "protected.h"
    path.write_text("KNOWN_GOOD\n", encoding="utf-8")
    with pytest.raises(ValueError):
        export_fixed_point_header(
            str(path), {name: np.asarray(values)}, {name: scale}
        )
    assert path.read_text(encoding="utf-8") == "KNOWN_GOOD\n"


def test_export_refuses_comment_injection(tmp_path):
    path = tmp_path / "unsafe.h"
    with pytest.raises(ValueError, match="unsafe C comment"):
        export_fixed_point_header(
            str(path),
            {"weights": np.array([1.0])},
            {"weights": 1.0},
            meta={"source": "*/\n#define PWNED 1"},
        )
    assert not path.exists()


def test_interrupted_atomic_header_publish_preserves_valid_original(tmp_path, monkeypatch):
    import os

    path = tmp_path / "valid.h"
    export_fixed_point_header(
        str(path), {"weights": np.array([1.0])},
        {"weights": 1.0},
    )
    accepted = path.read_bytes()

    def disk_failure(source, destination):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(os, "replace", disk_failure)
    with pytest.raises(OSError, match="simulated disk failure"):
        export_fixed_point_header(
            str(path), {"weights": np.array([2.0])},
            {"weights": 1.0},
        )
    assert path.read_bytes() == accepted
    assert not list(tmp_path.glob(".aegis-export-*.tmp"))
