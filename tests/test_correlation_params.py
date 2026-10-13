
# FILE: tests/test_correlation_params.py
from a3d import AegisConfig, DecoderRuntime, RotatedSurfaceLayout
from a3d.correlation_model import CorrelationParams


def test_corr_params_integration(tmp_path):
    # write a small params file
    p = tmp_path/"corr.json"
    CorrelationParams(neighbor_bonus=0.2, boundary_penalty=0.01, time_space_bonus=0.07).to_json(str(p))
    cfg = AegisConfig(distance=3, rounds=3, decoder_type="mwpm_corr")
    cfg.correlation_params = str(p)
    lay = RotatedSurfaceLayout(cfg.distance)
    rt = DecoderRuntime(cfg, lay)
    sX = [0]*len(rt.builder.node_order("X"))
    sZ = [0]*len(rt.builder.node_order("Z"))
    resX, resZ = rt.decode_from_syndromes_uniform(sX, sZ)
    assert resX.avg_cost >= 0 and resZ.avg_cost >= 0


def test_invalid_explicit_correlation_file_fails_instead_of_using_defaults(tmp_path):
    import pytest

    cfg = AegisConfig(distance=3, rounds=3, decoder_type="mwpm_corr")
    cfg.correlation_params = str(tmp_path / "missing-correlation.json")
    rt = DecoderRuntime(cfg, RotatedSurfaceLayout(3))
    sx = [0] * len(rt.builder.node_order("X"))
    sz = [0] * len(rt.builder.node_order("Z"))
    with pytest.raises(FileNotFoundError):
        rt.decode_from_syndromes_uniform(sx, sz)
