"""An experiment must never quietly execute a different decoding algorithm."""
from __future__ import annotations

import pytest

from a3d.config import AegisConfig
from a3d.graph import RotatedSurfaceLayout
from a3d.runtime import DecoderRuntime


def _runtime(**settings):
    cfg = AegisConfig(distance=3, rounds=3, **settings)
    rt = DecoderRuntime(cfg, RotatedSurfaceLayout(3))
    sx = [0] * len(rt.builder.node_order("X"))
    sz = [0] * len(rt.builder.node_order("Z"))
    return rt, sx, sz


@pytest.mark.parametrize("axis", ("X", "Z"))
def test_runtime_rejects_nonbinary_syndromes(axis):
    rt, sx, sz = _runtime()
    if axis == "X":
        sx[0] = 2
    else:
        sz[0] = -1
    with pytest.raises(ValueError, match="binary detector bits"):
        rt.decode_from_syndromes_uniform(sx, sz)


@pytest.mark.parametrize("axis", ("X", "Z"))
def test_runtime_rejects_nonfinite_soft_evidence(axis):
    rt, sx, sz = _runtime()
    llr_x = [1.0] * len(sx)
    llr_z = [1.0] * len(sz)
    if axis == "X":
        llr_x[0] = float("nan")
    else:
        llr_z[0] = float("inf")
    with pytest.raises(ValueError, match="nonfinite"):
        rt.decode_from_softinfo(llr_x, llr_z)


def test_runtime_refuses_to_drop_requested_leakage_flags():
    rt, sx, sz = _runtime()
    with pytest.raises(NotImplementedError, match="refusing to discard"):
        rt.decode_from_softinfo(
            [1.0] * len(sx), [1.0] * len(sz),
            leakage_flags=[1],
        )


def test_bp_invalid_reweighting_is_rejected(monkeypatch):
    from a3d.reweight_bp import BeliefPropagationReweighter

    rt, sx, sz = _runtime(reweighter_type="bp")
    calls = []

    def invalid_second_sector(self, graph):
        calls.append(len(graph.edges))
        if len(calls) == 1:
            return [1.0] * len(graph.edges)
        return [float("nan")] * len(graph.edges)

    monkeypatch.setattr(
        BeliefPropagationReweighter, "reweight",
        invalid_second_sector,
    )
    with pytest.raises(ValueError, match="invalid edge costs"):
        rt.decode_from_syndromes_uniform(sx, sz)
    assert len(calls) == 2


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("p_data", 1.0), ("p_meas", 5.0),
        ("p_leak", 1.0), ("time_weight_scale", 0.0),
    ),
)
def test_config_rejects_invalid_physical_parameters(field, value):
    cfg = AegisConfig()
    setattr(cfg, field, value)
    with pytest.raises(ValueError):
        cfg.validate()
