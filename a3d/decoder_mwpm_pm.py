from __future__ import annotations

from typing import Sequence

import numpy as np
import pymatching

from .decoder_mwpm import MWPMDecoder


class PyMatchingMWPMDecoder(MWPMDecoder):
    """Compatibility wrapper for PyMatching-backed graph and DEM decoding.

    The ordinary MWPMDecoder already uses PyMatching sparse blossom. This
    class remains for callers that imported the historical adapter directly.
    """

    def decode_graph(self, graph, syndromes):
        return self.decode(graph, syndromes)

    @staticmethod
    def matching_from_dem(dem_text: str) -> pymatching.Matching:
        try:
            import stim
        except ImportError as exc:  # pragma: no cover - optional extra
            raise RuntimeError(
                "Stim DEM support requires the optional dependency: "
                "pip install 'aegis-qec[full]'"
            ) from exc
        dem = stim.DetectorErrorModel(dem_text)
        return pymatching.Matching.from_detector_error_model(dem)

    def decode_dem(self, dem_text: str, syndrome: Sequence[int]) -> np.ndarray:
        """Decode one explicit DEM detector shot and return predicted observables."""
        matching = self.matching_from_dem(dem_text)
        shot = np.asarray(syndrome)
        if np.any((shot != 0) & (shot != 1)):
            raise ValueError("DEM syndrome must contain only binary detector bits")
        shot = np.asarray(shot, dtype=np.uint8)
        if len(shot) != matching.num_detectors:
            raise ValueError(
                f"Expected {matching.num_detectors} DEM detector bits, got {len(shot)}"
            )
        return matching.decode(shot)

    def decode_dem_batch(
        self,
        dem_text: str,
        syndromes: Sequence[Sequence[int]],
    ) -> np.ndarray:
        """Decode a batch of explicit DEM detector shots with one compiled matcher."""
        matching = self.matching_from_dem(dem_text)
        shots = np.asarray(syndromes)
        if shots.ndim != 2:
            raise ValueError("DEM batch must be a 2D array-like of detector shots")
        if np.any((shots != 0) & (shots != 1)):
            raise ValueError("DEM batch must contain only binary detector bits")
        # Never mask bits in-place: callers often reuse this evidence matrix.
        shots = np.asarray(shots, dtype=np.uint8)
        if shots.shape[1] != matching.num_detectors:
            raise ValueError(
                f"Expected {matching.num_detectors} DEM detector bits per shot, "
                f"got {shots.shape[1]}"
            )
        return matching.decode_batch(shots)

    def decode_from_dem(
        self, dem_text: str, syndrome: Sequence[int] | None = None
    ):
        if syndrome is None:
            raise ValueError(
                "A detector error model does not contain an observed syndrome. "
                "Pass the detection-event bits as syndrome=..."
            )
        return self.decode_dem(dem_text, syndrome)
