"""Independent adversarial checks for package integrity and safe verification."""
from __future__ import annotations

import hashlib
import json
import os
import warnings
import zipfile
from pathlib import Path

import pytest

from aegis_qec.experiment import verify_research_bundle
from aegis_qec.paper import verify_submission_package


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _submission(tmp_path: Path) -> tuple[Path, Path]:
    directory = tmp_path / "submission"
    directory.mkdir()
    files = {
        "research-project.json": b'{"title":"Research"}\n',
        "audit.json": b'{"valid":true}\n',
    }
    payload = []
    for name, value in sorted(files.items()):
        (directory / name).write_bytes(value)
        payload.append({"path": name, "sha256": _sha(value), "bytes": len(value)})
    manifest = {
        "schema_version": 1,
        "package_type": "aegis_qec_submission_package",
        "payload": payload,
    }
    data = (json.dumps(manifest, sort_keys=True) + "\n").encode()
    (directory / "MANIFEST.json").write_bytes(data)
    checks = "".join(
        f"{item['sha256']}  {item['path']}\n" for item in payload
    ) + f"{_sha(data)}  MANIFEST.json\n"
    (directory / "checksums.sha256").write_text(checks, encoding="utf-8")
    zipped = tmp_path / "submission.zip"
    with zipfile.ZipFile(zipped, "w") as archive:
        for path in sorted(directory.iterdir()):
            archive.write(path, path.name)
    return directory, zipped


def _research_zip(tmp_path: Path) -> Path:
    zipped = tmp_path / "research.zip"
    items = {"experiment.json": b"{}", "aegis-run.json": b"{}"}
    manifest = {
        "schema_version": 1,
        "bundle_type": "aegis_qec_research_bundle",
        "entries": [
            {"path": key, "bytes": len(value), "sha256": _sha(value)}
            for key, value in sorted(items.items())
        ],
    }
    with zipfile.ZipFile(zipped, "w") as archive:
        for key, value in items.items():
            archive.writestr(key, value)
        archive.writestr("BUNDLE_MANIFEST.json", json.dumps(manifest))
    return zipped


def test_valid_submission_directory_and_zip(tmp_path):
    directory, zipped = _submission(tmp_path)
    assert verify_submission_package(str(directory))["valid"] is True
    assert verify_submission_package(str(zipped))["valid"] is True


def test_submission_rejects_modified_checksum_manifest(tmp_path):
    directory, _ = _submission(tmp_path)
    manifest = directory / "MANIFEST.json"
    manifest.write_bytes(manifest.read_bytes() + b" ")
    result = verify_submission_package(str(directory))
    assert result["valid"] is False
    assert "checksums.sha256 mismatch" in result["failures"]


def test_submission_rejects_traversal_even_with_matching_external_file(tmp_path):
    directory, _ = _submission(tmp_path)
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"secret")
    manifest_file = directory / "MANIFEST.json"
    manifest = json.loads(manifest_file.read_text())
    manifest["payload"].append({
        "path": "../outside.json",
        "bytes": 6,
        "sha256": _sha(b"secret"),
    })
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
    result = verify_submission_package(str(directory))
    assert result["valid"] is False
    assert any("unsafe or reserved" in x for x in result["failures"])
    assert outside.read_bytes() == b"secret"


def test_submission_rejects_duplicate_member_names(tmp_path):
    _, zipped = _submission(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(zipped, "a") as archive:
            archive.writestr("audit.json", b'{"valid":true}\n')
    result = verify_submission_package(str(zipped))
    assert result["valid"] is False
    assert "duplicate archive path: audit.json" in result["failures"]


def test_submission_rejects_duplicate_manifest_entries(tmp_path):
    directory, _ = _submission(tmp_path)
    file = directory / "MANIFEST.json"
    manifest = json.loads(file.read_text())
    manifest["payload"].append(dict(manifest["payload"][0]))
    file.write_text(json.dumps(manifest), encoding="utf-8")
    result = verify_submission_package(str(directory))
    assert result["valid"] is False
    assert any("duplicate manifest path" in x for x in result["failures"])


def test_submission_rejects_symlink_payload(tmp_path):
    directory, _ = _submission(tmp_path)
    if not hasattr(os, "symlink"):
        pytest.skip("no symbolic links")
    (directory / "audit.json").unlink()
    outside = tmp_path / "audit-real.json"
    outside.write_bytes(b'{"valid":true}\n')
    try:
        (directory / "audit.json").symlink_to(outside)
    except (NotImplementedError, OSError):
        pytest.skip("symlinks unavailable")
    result = verify_submission_package(str(directory))
    assert result["valid"] is False
    assert any("symlink" in item for item in result["failures"])


def test_research_zip_requires_run_and_experiment(tmp_path):
    zipped = _research_zip(tmp_path)
    assert verify_research_bundle(str(zipped))["valid"] is True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(zipped, "a") as archive:
            archive.writestr("experiment.json", b"{}")
    result = verify_research_bundle(str(zipped))
    assert result["valid"] is False
    assert "duplicate archive path: experiment.json" in result["failures"]


def test_research_zip_rejects_unsafe_backslash_path(tmp_path):
    zipped = _research_zip(tmp_path)
    with zipfile.ZipFile(zipped, "a") as archive:
        archive.writestr("artifacts\\..\\escape", b"unsafe")
    result = verify_research_bundle(str(zipped))
    assert result["valid"] is False
    assert any("unsafe archive path" in x for x in result["failures"])
