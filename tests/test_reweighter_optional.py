import pytest

from a3d.config import AegisConfig
from a3d.graph import RotatedSurfaceLayout
from a3d.runtime import DecoderRuntime


@pytest.mark.parametrize("algorithm", ["transformer", "transformer_sota"])
def test_untrained_transformer_reweighter_fails_closed(algorithm):
    cfg = AegisConfig(distance=3, rounds=3, decoder_type="mwpm")
    cfg.reweighter_type = algorithm
    runtime = DecoderRuntime(cfg, RotatedSurfaceLayout(cfg.distance))
    sx = [0] * len(runtime.builder.node_order("X"))
    sz = [0] * len(runtime.builder.node_order("Z"))
    with pytest.raises(ValueError, match="trained checkpoint"):
        runtime.decode_from_syndromes_uniform(sx, sz)


@pytest.mark.parametrize("algorithm", ["transformer", "transformer_sota"])
def test_missing_transformer_checkpoint_does_not_fall_back(algorithm, tmp_path):
    cfg = AegisConfig(distance=3, rounds=3, decoder_type="mwpm")
    cfg.reweighter_type = algorithm
    cfg.reweighter_weights = str(tmp_path / "nonexistent.pt")
    runtime = DecoderRuntime(cfg, RotatedSurfaceLayout(cfg.distance))
    sx = [0] * len(runtime.builder.node_order("X"))
    sz = [0] * len(runtime.builder.node_order("Z"))
    with pytest.raises((ImportError, FileNotFoundError)):
        runtime.decode_from_syndromes_uniform(sx, sz)
