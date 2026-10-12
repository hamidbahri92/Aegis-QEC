"""Streaming, deterministic and crash-safe research submission archives."""
from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pytest

from aegis_qec.paper import _zip_directory


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_archive_streaming_does_not_call_read_bytes_on_large_payload(tmp_path, monkeypatch):
    source = tmp_path / "payload"
    source.mkdir()
    dataset = source / "dataset.h5"
    data = b"scientific-evidence" * 400_000
    dataset.write_bytes(data)
    expected_hash = hashlib.sha256(data).hexdigest()
    destination = tmp_path / "package.zip"
    read_original = Path.read_bytes

    def refuse_whole_file_read(path):
        if path == dataset:
            raise AssertionError("streaming archive loaded entire dataset")
        return read_original(path)

    monkeypatch.setattr(Path, "read_bytes", refuse_whole_file_read)
    _zip_directory(source, destination)

    with zipfile.ZipFile(destination) as archive:
        digest = hashlib.sha256()
        with archive.open("dataset.h5") as stream:
            for chunk in iter(lambda: stream.read(256 * 1024), b""):
                digest.update(chunk)
        assert digest.hexdigest() == expected_hash


def test_archives_are_byte_reproducible_across_repeated_invocations(tmp_path):
    source = tmp_path / "package"
    source.mkdir()
    (source / "data.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (source / "nested").mkdir()
    (source / "nested" / "record.json").write_text('{"valid":true}', encoding="utf-8")
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    _zip_directory(source, first)
    _zip_directory(source, second)
    assert _digest(first) == _digest(second)


def test_failed_archive_write_does_not_destroy_prior_success(tmp_path, monkeypatch):
    source = tmp_path / "contents"
    source.mkdir()
    (source / "dataset.bin").write_bytes(b"known-good" * 20)
    target = tmp_path / "protected.zip"
    _zip_directory(source, target)
    accepted_hash = _digest(target)

    import aegis_qec.paper as paper

    def simulated_disk_error(reader, writer, length):
        writer.write(reader.read(2))
        raise OSError("simulated interrupted archive")

    monkeypatch.setattr(paper.shutil, "copyfileobj", simulated_disk_error)
    with pytest.raises(OSError, match="interrupted archive"):
        _zip_directory(source, target)
    assert _digest(target) == accepted_hash
    assert not list(tmp_path.glob(".aegis-archive-*.tmp"))


def test_archive_refuses_recursive_output_and_symlinks(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "artifact.txt").write_text("evidence", encoding="utf-8")
    with pytest.raises(ValueError, match="nested inside"):
        _zip_directory(source, source / "bad.zip")
    outside = tmp_path / "other.txt"
    outside.write_text("external", encoding="utf-8")
    link = source / "sneaky.txt"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not available")
    with pytest.raises(ValueError, match="symlink artifact"):
        _zip_directory(source, tmp_path / "no-symlinks.zip")
