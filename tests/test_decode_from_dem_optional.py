import pytest


def test_decode_from_dem_requires_detection_events():
    pytest.importorskip("stim")
    from a3d.decoder_mwpm_pm import PyMatchingMWPMDecoder

    decoder = PyMatchingMWPMDecoder()
    dem = "error(0.01) D0 D1\n"
    with pytest.raises(ValueError, match="observed syndrome"):
        decoder.decode_from_dem(dem)


def test_decode_from_dem_accepts_explicit_syndrome():
    pytest.importorskip("stim")
    from a3d.decoder_mwpm_pm import PyMatchingMWPMDecoder

    decoder = PyMatchingMWPMDecoder()
    dem = "error(0.01) D0 D1\n"
    prediction = decoder.decode_from_dem(dem, [1, 1])
    assert prediction is not None



def test_dem_batch_rejects_nonbinary_bits_without_mutating_input():
    pytest.importorskip("stim")
    import numpy as np

    from a3d.decoder_mwpm_pm import PyMatchingMWPMDecoder

    decoder = PyMatchingMWPMDecoder()
    dem = "error(0.01) D0 D1\n"
    invalid = np.asarray([[2, 1], [0, 0]], dtype=np.uint8)
    original = invalid.copy()
    with pytest.raises(ValueError, match="binary detector bits"):
        decoder.decode_dem_batch(dem, invalid)
    np.testing.assert_array_equal(invalid, original)
    with pytest.raises(ValueError, match="binary detector bits"):
        decoder.decode_dem(dem, [1, 3])

    valid = np.asarray([[1, 1], [0, 0]], dtype=np.uint8)
    preserved = valid.copy()
    predicted = decoder.decode_dem_batch(dem, valid)
    assert predicted.shape[0] == 2
    np.testing.assert_array_equal(valid, preserved)
