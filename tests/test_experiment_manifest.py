from __future__ import annotations

import json
import zipfile

import pytest

pytest.importorskip("stim")

from aegis_qec.experiment import (
    create_research_bundle,
    load_experiment_manifest,
    run_experiment_manifest,
    verify_research_bundle,
)


def _write_study_manifest(path):
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "tiny-study",
                "operation": "study",
                "parameters": {
                    "distances": [3],
                    "physical_error_rates": [0.02],
                    "shots": 16,
                    "basis": "x",
                    "seed": 1234,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def test_manifest_run_creates_hashed_artifacts_and_valid_bundle(tmp_path):
    manifest_path = tmp_path / "experiment.json"
    _write_study_manifest(manifest_path)

    run = run_experiment_manifest(
        str(manifest_path),
        output_dir=str(tmp_path / "out"),
    )

    assert run["operation"] == "study"
    assert run["name"] == "tiny-study"
    assert len(run["manifest"]["sha256"]) == 64
    assert "study_json" in run["artifacts"]
    assert "study_csv" in run["artifacts"]
    assert "study_plot" in run["artifacts"]
    assert len(run["run_record"]["sha256"]) == 64

    bundle_path = tmp_path / "tiny-study.aegis.zip"
    bundle = create_research_bundle(
        manifest_path=str(manifest_path),
        run_record=run,
        bundle_path=str(bundle_path),
    )
    assert len(bundle["sha256"]) == 64

    verification = verify_research_bundle(str(bundle_path))
    assert verification["valid"] is True
    assert verification["failures"] == []


def test_bundle_verifier_detects_unexpected_content(tmp_path):
    manifest_path = tmp_path / "experiment.json"
    _write_study_manifest(manifest_path)
    run = run_experiment_manifest(
        str(manifest_path),
        output_dir=str(tmp_path / "out"),
    )
    bundle_path = tmp_path / "tiny-study.aegis.zip"
    create_research_bundle(
        manifest_path=str(manifest_path),
        run_record=run,
        bundle_path=str(bundle_path),
    )

    with zipfile.ZipFile(bundle_path, "a") as archive:
        archive.writestr("tampered.txt", b"unexpected")

    verification = verify_research_bundle(str(bundle_path))
    assert verification["valid"] is False
    assert any(
        "unexpected entry: tampered.txt" in failure
        for failure in verification["failures"]
    )


def test_manifest_rejects_unknown_operation(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "operation": "teleport",
                "parameters": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="operation must be one of"):
        load_experiment_manifest(str(path))



def test_bundle_builder_refuses_missing_recorded_evidence(tmp_path):
    path = tmp_path / "experiment.json"
    _write_study_manifest(path)
    run = run_experiment_manifest(str(path), output_dir=str(tmp_path / "out"))
    from pathlib import Path

    Path(run["artifacts"]["study_csv"]["path"]).unlink()
    with pytest.raises(FileNotFoundError, match="required artifacts evidence missing"):
        create_research_bundle(
            manifest_path=str(path),
            run_record=run,
            bundle_path=str(tmp_path / "should-not-exist.zip"),
        )
    assert not (tmp_path / "should-not-exist.zip").exists()


def test_bundle_builder_refuses_tampered_recorded_evidence(tmp_path):
    path = tmp_path / "experiment.json"
    _write_study_manifest(path)
    run = run_experiment_manifest(str(path), output_dir=str(tmp_path / "out"))
    from pathlib import Path

    Path(run["artifacts"]["study_json"]["path"]).write_bytes(b"replaced")
    with pytest.raises(ValueError, match="recorded artifacts evidence hash mismatch"):
        create_research_bundle(
            manifest_path=str(path),
            run_record=run,
            bundle_path=str(tmp_path / "untrusted.zip"),
        )


def test_bundle_verifier_refuses_self_consistent_but_incomplete_bundle(tmp_path):
    path = tmp_path / "experiment.json"
    _write_study_manifest(path)
    run = run_experiment_manifest(str(path), output_dir=str(tmp_path / "out"))
    valid = tmp_path / "complete.zip"
    create_research_bundle(
        manifest_path=str(path), run_record=run, bundle_path=str(valid)
    )
    with zipfile.ZipFile(valid, "r") as archive:
        payloads = {
            name: archive.read(name)
            for name in archive.namelist()
        }
    manifest = json.loads(payloads["BUNDLE_MANIFEST.json"])
    missing = next(
        item["path"] for item in manifest["entries"]
        if item["path"].startswith("artifacts/study_json/")
    )
    del payloads[missing]
    manifest["entries"] = [
        item for item in manifest["entries"] if item["path"] != missing
    ]
    payloads["BUNDLE_MANIFEST.json"] = json.dumps(manifest).encode()
    forged = tmp_path / "incomplete-but-rehashed.zip"
    with zipfile.ZipFile(forged, "w") as archive:
        for name, data in payloads.items():
            archive.writestr(name, data)
    result = verify_research_bundle(str(forged))
    assert result["valid"] is False
    assert any(
        "missing or inconsistent referenced evidence" in item
        for item in result["failures"]
    )
