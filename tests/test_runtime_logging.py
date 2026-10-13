"""Requested decoding evidence and algorithms must fail closed."""
from __future__ import annotations

import pytest

from a3d.config import AegisConfig
from a3d.graph import RotatedSurfaceLayout
from a3d.runtime import DecoderRuntime


class DummyModel:
    pass


def dummy_logits_fn(_graph):
    raise RuntimeError("model failed")


def _ready_runtime(**settings):
    cfg = AegisConfig(distance=3, rounds=3, **settings)
    rt = DecoderRuntime(cfg, RotatedSurfaceLayout(3))
    sx = [0] * len(rt.builder.node_order("X"))
    sz = [0] * len(rt.builder.node_order("Z"))
    return rt, sx, sz


def test_attached_ml_failure_is_not_silently_reported_as_baseline():
    rt, sx, sz = _ready_runtime()
    rt.attach_ml_model(DummyModel(), dummy_logits_fn)
    with pytest.raises(RuntimeError, match="model failed"):
        rt.decode_from_syndromes_calibrated(sx, sz)


def test_explicit_decode_event_logging_persists_both_axes(tmp_path):
    path = tmp_path / "events.csv"
    rt, sx, sz = _ready_runtime(
        log_events=True, log_path=str(path)
    )
    rt.decode_from_syndromes_uniform(sx, sz)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("X,mwpm,")
    assert lines[1].startswith("Z,mwpm,")


def test_requested_event_logging_write_failure_propagates(tmp_path):
    directory = tmp_path / "not-a-file"
    directory.mkdir()
    rt, sx, sz = _ready_runtime(
        log_events=True, log_path=str(directory)
    )
    with pytest.raises(OSError):
        rt.decode_from_syndromes_uniform(sx, sz)


def test_profile_path_without_directory_is_valid(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rt, sx, sz = _ready_runtime(
        profile=True, profile_path="local-profile.csv"
    )
    rt.decode_from_syndromes_uniform(sx, sz)
    assert (tmp_path / "local-profile.csv").is_file()


def test_requested_certificate_failure_propagates(monkeypatch):
    import a3d.certificates as certificates

    def failed_certificate(*args, **kwargs):
        raise RuntimeError("certificate unavailable")

    monkeypatch.setattr(certificates, "osd_polish", failed_certificate)
    rt, sx, sz = _ready_runtime(run_certificate=True)
    with pytest.raises(RuntimeError, match="certificate unavailable"):
        rt.decode_from_syndromes_uniform(sx, sz)


def test_unknown_requested_certificate_mode_is_rejected():
    rt, sx, sz = _ready_runtime(
        run_certificate=True, certificate_mode="unimplemented"
    )
    with pytest.raises(ValueError, match="certificate_mode"):
        rt.decode_from_syndromes_uniform(sx, sz)
