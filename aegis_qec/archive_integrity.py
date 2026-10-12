"""Strict integrity checks for portable Aegis research and submission bundles.

These checks authenticate consistency against a manifest, not the identity of
the manifest's author. External signatures or trusted hashes are needed for
authenticity against an attacker who replaces the whole package.
"""
from __future__ import annotations

import hashlib
import json
import re
import stat
import zipfile
from pathlib import Path
from typing import Any

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def _safe_name(value: Any) -> bool:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        return False
    if "\x00" in value:
        return False
    return all(part not in {"", ".", ".."} for part in value.split("/"))


def _manifest_inventory(
    manifest: Any,
    *,
    manifest_type: str,
    inventory_name: str,
    reserved: set[str],
    required: set[str],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    errors: list[str] = []
    entries: dict[str, dict[str, Any]] = {}
    if not isinstance(manifest, dict):
        return entries, ["manifest must be a JSON object"]
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        errors.append("unsupported manifest schema_version")
    kind = "package_type" if inventory_name == "payload" else "bundle_type"
    if manifest.get(kind) != manifest_type:
        errors.append("unexpected manifest type")
    inventory = manifest.get(inventory_name)
    if not isinstance(inventory, list) or not inventory:
        return entries, errors + ["manifest inventory must be a non-empty list"]
    for index, entry in enumerate(inventory):
        if not isinstance(entry, dict):
            errors.append(f"invalid inventory entry: {index}")
            continue
        name = entry.get("path")
        if not _safe_name(name) or name in reserved:
            errors.append(f"unsafe or reserved payload path: {name!r}")
            continue
        if name in entries:
            errors.append(f"duplicate manifest path: {name}")
            continue
        size = entry.get("bytes")
        digest = entry.get("sha256")
        if type(size) is not int or size < 0:
            errors.append(f"invalid payload size: {name}")
            continue
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            errors.append(f"invalid payload sha256: {name}")
            continue
        entries[name] = entry
    for name in sorted(required - set(entries)):
        errors.append("missing required payload: " + name)
    return entries, errors


def _checksum_bytes(entries: dict[str, dict[str, Any]], manifest_bytes: bytes, manifest_name: str) -> bytes:
    lines = [
        f"{entry['sha256']}  {name}\n"
        for name, entry in entries.items()
    ]
    lines.append(f"{hashlib.sha256(manifest_bytes).hexdigest()}  {manifest_name}\n")
    return "".join(lines).encode("utf-8")


def verify_manifested_payload(
    source: Path,
    *,
    manifest_name: str,
    manifest_type: str,
    inventory_name: str,
    required_names: tuple[str, ...],
    has_checksums: bool,
) -> list[str]:
    """Verify membership, unambiguous paths, content hashes and checksums.

    Never follow symlinks inside directory bundles or materialize archive entries.
    Stream ZIP contents so a manifest cannot force whole-file memory allocation.
    """
    reserved = {manifest_name}
    if has_checksums:
        reserved.add("checksums.sha256")
    errors: list[str] = []
    if source.is_dir():
        manifest_file = source / manifest_name
        if not manifest_file.is_file() or manifest_file.is_symlink():
            raise ValueError("submission directory is missing a safe " + manifest_name)
        manifest_bytes = manifest_file.read_bytes()
        manifest = json.loads(manifest_bytes)
        entries, problems = _manifest_inventory(
            manifest,
            manifest_type=manifest_type,
            inventory_name=inventory_name,
            reserved=reserved,
            required=set(required_names),
        )
        errors.extend(problems)
        actual = set()
        for candidate in source.rglob("*"):
            if candidate.is_symlink():
                errors.append("symlink in package: " + candidate.relative_to(source).as_posix())
            if candidate.is_file() or candidate.is_symlink():
                actual.add(candidate.relative_to(source).as_posix())
        for name in sorted(actual - set(entries) - reserved):
            errors.append(("unexpected entry: " if inventory_name == "entries" else "unexpected payload: ") + name)
        for name, item in entries.items():
            parts = name.split("/")
            target = source.joinpath(*parts)
            if any(source.joinpath(*parts[:i]).is_symlink() for i in range(1, len(parts) + 1)):
                errors.append("symlink payload: " + name)
                continue
            if not target.is_file():
                errors.append("missing payload: " + name)
                continue
            if target.stat().st_size != item["bytes"]:
                errors.append("size mismatch: " + name)
            digest = hashlib.sha256()
            with target.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != item["sha256"]:
                errors.append("hash mismatch: " + name)
        if has_checksums:
            checksums = source / "checksums.sha256"
            if not checksums.is_file() or checksums.is_symlink():
                errors.append("missing checksums.sha256")
            elif not errors and checksums.read_bytes().replace(b"\\r\\n", b"\\n") != _checksum_bytes(entries, manifest_bytes, manifest_name):
                errors.append("checksums.sha256 mismatch")
        return errors

    if not source.is_file() or source.suffix.lower() != ".zip":
        raise ValueError("bundle path must be a directory or .zip file")
    with zipfile.ZipFile(source, "r") as archive:
        files: dict[str, zipfile.ZipInfo] = {}
        for info in archive.infolist():
            name = info.filename
            if not _safe_name(name) or info.is_dir():
                errors.append("unsafe archive path: " + repr(name))
                continue
            mode = (info.external_attr >> 16) & 0o170000
            if mode == stat.S_IFLNK:
                errors.append("symlink archive entry: " + name)
                continue
            if name in files:
                errors.append("duplicate archive path: " + name)
                continue
            files[name] = info
        if manifest_name not in files:
            raise ValueError("bundle is missing " + manifest_name)
        manifest_bytes = archive.read(files[manifest_name])
        manifest = json.loads(manifest_bytes)
        entries, problems = _manifest_inventory(
            manifest,
            manifest_type=manifest_type,
            inventory_name=inventory_name,
            reserved=reserved,
            required=set(required_names),
        )
        errors.extend(problems)
        for name in sorted(set(files) - set(entries) - reserved):
            errors.append(("unexpected entry: " if inventory_name == "entries" else "unexpected payload: ") + name)
        for name, item in entries.items():
            info = files.get(name)
            if info is None:
                errors.append("missing payload: " + name)
                continue
            if info.file_size != item["bytes"]:
                errors.append("size mismatch: " + name)
            digest = hashlib.sha256()
            try:
                with archive.open(info) as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
            except (OSError, EOFError, zipfile.BadZipFile) as exc:
                errors.append(f"damaged payload {name}: {exc}")
                continue
            if digest.hexdigest() != item["sha256"]:
                errors.append("hash mismatch: " + name)
        if has_checksums:
            info = files.get("checksums.sha256")
            if info is None:
                errors.append("missing checksums.sha256")
            elif not errors and archive.read(info).replace(b"\\r\\n", b"\\n") != _checksum_bytes(entries, manifest_bytes, manifest_name):
                errors.append("checksums.sha256 mismatch")
    return errors
