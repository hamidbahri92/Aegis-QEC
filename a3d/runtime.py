# FILE: a3d/runtime.py
from __future__ import annotations

import logging
import time
from typing import Dict, List, Tuple

import numpy as np

from .config import AegisConfig
from .decoder_bposd import OSDDecoder
from .decoder_greedy import DecodeResult, GreedyMatchingDecoder
from .decoder_mwpm import MWPMDecoder
from .decoder_mwpm_corr import CorrelationMWPMDecoder
from .decoder_mwpm_pipeline import PipelinedMWPMDecoder
from .decoder_unionfind import UnionFindErasuresDecoder
from .graph import DecodingGraph, DecodingGraphBuilder, RotatedSurfaceLayout
from .logging_utils import log_decode_events

logger = logging.getLogger("a3d.runtime")


def _logodds(p: float) -> float:
    p = min(max(float(p), 1.0e-12), 1.0 - 1.0e-12)
    return -float(np.log(p / (1.0 - p)))


class DecoderRuntime:
    """Core runtime: build graphs, decode with chosen algorithm, optional ML rescore."""

    def __init__(self, cfg: AegisConfig, layout: RotatedSurfaceLayout):
        self.cfg = cfg
        self.layout = layout
        self.builder = DecodingGraphBuilder(layout, cfg.rounds, diagonal_adjacency=True)
        self.uf = GreedyMatchingDecoder()
        self.uf_e = UnionFindErasuresDecoder()
        self.bposd = OSDDecoder(self.uf, osd_order=1, k_candidates=32)
        self.mwpm = MWPMDecoder()
        self.mwpm2 = PipelinedMWPMDecoder()
        self.mwpm_corr = CorrelationMWPMDecoder()
        self._ml = None  # optional

        self.cfg.validate()

    def _validate_inputs(self, syndromes_X: List[int], syndromes_Z: List[int]) -> None:
        nX = len(self.builder.node_order("X"))
        nZ = len(self.builder.node_order("Z"))
        if len(syndromes_X) != nX:
            raise ValueError(f"Expected {nX} X-syndromes, got {len(syndromes_X)}")
        if len(syndromes_Z) != nZ:
            raise ValueError(f"Expected {nZ} Z-syndromes, got {len(syndromes_Z)}")
        if any(value not in (0, 1) for value in syndromes_X):
            raise ValueError("X syndrome must contain only binary detector bits")
        if any(value not in (0, 1) for value in syndromes_Z):
            raise ValueError("Z syndrome must contain only binary detector bits")

    def _weight_dicts_from_cfg(self, sector: str) -> Tuple[
        Dict[Tuple[Tuple[int, int], int], float],
        Dict[Tuple[Tuple[int, int], int], float],
        Dict[Tuple[Tuple[int, int], int], float],
    ]:
        order = self.builder.node_order(sector)
        T = self.cfg.rounds
        w_space = {(coord, t): _logodds(self.cfg.p_data) for (coord, t) in order}
        w_time = {
            (coord, t): _logodds(self.cfg.p_meas) * self.cfg.time_weight_scale
            for (coord, t) in order
            if t < T - 1
        }
        p_erase = {
            (coord, t): min(max(self.cfg.p_leak, 0.0), 0.99)
            for (coord, t) in order
            if t < T - 1
        }
        return w_space, w_time, p_erase

    def _decode_with_choice(
        self, graph: DecodingGraph, syn: List[int], axis: str
    ) -> DecodeResult:
        dtype = self.cfg.decoder_type
        if dtype == "greedy":
            res = self.uf.decode(graph, syn)
            if res.avg_cost > self.cfg.uf_confidence_threshold:
                logger.warning(
                    "UF %s avg_cost=%.3f > threshold=%.3f; OSD fallback",
                    axis,
                    res.avg_cost,
                    self.cfg.uf_confidence_threshold,
                )
                res = self.bposd.decode(graph, syn)
            return res
        elif dtype == "osd":
            return self.bposd.decode(graph, syn)
        elif dtype == "mwpm":
            return self.mwpm.decode(graph, syn)
        elif dtype in ("mwpm2","pipelined_mwpm","pmwpm"):
            return self.mwpm2.decode(graph, syn)
        elif dtype in ("mwpm_corr","mwpmc","mwpm3"):
            # Invalid correlation parameters must never silently substitute
            # the decoder defaults for a requested calibrated experiment.
            if self.cfg.correlation_params:
                from .correlation_model import CorrelationParams

                self.mwpm_corr._corr_params = CorrelationParams.from_json(
                    self.cfg.correlation_params
                )
            return self.mwpm_corr.decode(graph, syn)
        elif dtype in ("uf","unionfind","uf_e"):
            return self.uf_e.decode(graph, syn)
        else:
            return self.bposd.decode(graph, syn)

    def decode_window(
        self,
        w_space_X: Dict[Tuple[Tuple[int, int], int], float],
        w_time_X: Dict[Tuple[Tuple[int, int], int], float],
        p_erase_X: Dict[Tuple[Tuple[int, int], int], float],
        w_space_Z: Dict[Tuple[Tuple[int, int], int], float],
        w_time_Z: Dict[Tuple[Tuple[int, int], int], float],
        p_erase_Z: Dict[Tuple[Tuple[int, int], int], float],
        syndromes_X: List[int],
        syndromes_Z: List[int],
    ) -> Tuple[DecodeResult, DecodeResult]:
        self._validate_inputs(syndromes_X, syndromes_Z)
        graph_X = self.builder.build("X", w_space_X, w_time_X, p_erase_X)
        graph_Z = self.builder.build("Z", w_space_Z, w_time_Z, p_erase_Z)

        # An enabled reweighter is part of the experiment definition.
        # Resolve and validate *both* sectors before mutating either graph.
        # A bad checkpoint or unavailable PyTorch must fail the experiment.
        rtw = self.cfg.reweighter_type
        if rtw != "none":
            if rtw == "transformer":
                from .reweight_transformer import GraphEdgeTransformerReweighter

                rw = GraphEdgeTransformerReweighter(
                    weights_path=self.cfg.reweighter_weights
                )
            elif rtw == "transformer_sota":
                from .reweight_transformer_sota import GraphEdgeTransformerSOTA

                rw = GraphEdgeTransformerSOTA(
                    weights_path=self.cfg.reweighter_weights
                )
            elif rtw == "bp":
                from .reweight_bp import BeliefPropagationReweighter

                rw = BeliefPropagationReweighter()
            else:
                raise ValueError(f"unknown reweighter: {rtw}")

            new_costs_x = list(rw.reweight(graph_X))
            new_costs_z = list(rw.reweight(graph_Z))
            for axis, graph, costs in (
                ("X", graph_X, new_costs_x),
                ("Z", graph_Z, new_costs_z),
            ):
                if len(costs) != len(graph.edges):
                    raise ValueError(f"{axis} reweighter returned incorrect edge count")
                if not all(np.isfinite(cost) and float(cost) >= 0 for cost in costs):
                    raise ValueError(f"{axis} reweighter returned invalid edge costs")
            for graph, costs in (
                (graph_X, new_costs_x),
                (graph_Z, new_costs_z),
            ):
                for edge, cost in zip(graph.edges, costs, strict=True):
                    object.__setattr__(edge, "weight", float(cost))


        t0 = time.perf_counter()
        resX = self._decode_with_choice(graph_X, syndromes_X, axis="X")
        t1 = time.perf_counter()
        resZ = self._decode_with_choice(graph_Z, syndromes_Z, axis="Z")
        t2 = time.perf_counter()

        # Optional decode event logging
        try:
            if getattr(self.cfg, "log_events", False):
                log_decode_events(getattr(self.cfg, "log_path", "logs/decoding_events.csv"), "X", getattr(self.cfg, "decoder_type", "unknown"), len(resX.corrections))
                log_decode_events(getattr(self.cfg, "log_path", "logs/decoding_events.csv"), "Z", getattr(self.cfg, "decoder_type", "unknown"), len(resZ.corrections))
        except Exception:
            pass


        # Certificates (optional) and confidence scoring
        try:
            if getattr(self.cfg, "run_certificate", True) and getattr(self.cfg, "certificate_mode", "osd") == "osd":
                from .certificates import osd_polish
                from .decoder_greedy import GreedyMatchingDecoder
                # polish X
                gdec = GreedyMatchingDecoder()
                resX2, improvedX = osd_polish(graph_X, syndromes_X, gdec, resX)
                if improvedX: resX = resX2
                # polish Z
                resZ2, improvedZ = osd_polish(graph_Z, syndromes_Z, gdec, resZ)
                if improvedZ: resZ = resZ2
        except Exception:
            pass

        # Confidence
        try:
            from .confidence import confidence_from_cost
            resX.confidence = float(confidence_from_cost(resX.avg_cost, max(1, len(resX.corrections))))
            resZ.confidence = float(confidence_from_cost(resZ.avg_cost, max(1, len(resZ.corrections))))
        except Exception:
            pass

        # Profiling (optional)
        try:
            if getattr(self.cfg, "profile", False):
                import csv
                import os
                os.makedirs(os.path.dirname(getattr(self.cfg, "profile_path", "logs/profile.csv")), exist_ok=True)
                with open(getattr(self.cfg, "profile_path", "logs/profile.csv"), "a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["decode_X_s", t1 - t0, "decode_Z_s", t2 - t1])
        except Exception:
            pass

        if getattr(self, "_ml", None) is not None:
            try:
                from .decoder_ml import decode_with_gnn

                logitsX = self._ml_logits(graph_X)
                logitsZ = self._ml_logits(graph_Z)
                mresX = decode_with_gnn(graph_X, logitsX, threshold=0.6)
                mresZ = decode_with_gnn(graph_Z, logitsZ, threshold=0.6)
                if mresX.log_likelihood < resX.log_likelihood and mresX.avg_cost > 0:
                    resX = mresX
                if mresZ.log_likelihood < resZ.log_likelihood and mresZ.avg_cost > 0:
                    resZ = mresZ
            except Exception:
                logger.exception("ML fallback failed")

        return resX, resZ

    def decode_from_syndromes_calibrated(
        self,
        syndromes_X: List[int],
        syndromes_Z: List[int],
    ) -> Tuple[DecodeResult, DecodeResult]:
        w_space_X, w_time_X, p_erase_X = self._weight_dicts_from_cfg("X")
        w_space_Z, w_time_Z, p_erase_Z = self._weight_dicts_from_cfg("Z")
        return self.decode_window(
            w_space_X,
            w_time_X,
            p_erase_X,
            w_space_Z,
            w_time_Z,
            p_erase_Z,
            syndromes_X,
            syndromes_Z,
        )

    def decode_from_syndromes_uniform(
        self,
        syndromes_X: List[int],
        syndromes_Z: List[int],
        weight_space: float = 1.0,
        weight_time: float = 1.0,
        perase_time: float = 0.0,
    ) -> Tuple[DecodeResult, DecodeResult]:
        order_X = self.builder.node_order("X")
        order_Z = self.builder.node_order("Z")
        T = self.cfg.rounds

        w_space_X = {(coord, t): float(weight_space) for (coord, t) in order_X}
        w_time_X = {
            (coord, t): float(weight_time) for (coord, t) in order_X if t < T - 1
        }
        p_erase_X = {
            (coord, t): float(perase_time) for (coord, t) in order_X if t < T - 1
        }

        w_space_Z = {(coord, t): float(weight_space) for (coord, t) in order_Z}
        w_time_Z = {
            (coord, t): float(weight_time) for (coord, t) in order_Z if t < T - 1
        }
        p_erase_Z = {
            (coord, t): float(perase_time) for (coord, t) in order_Z if t < T - 1
        }

        return self.decode_window(
            w_space_X,
            w_time_X,
            p_erase_X,
            w_space_Z,
            w_time_Z,
            p_erase_Z,
            syndromes_X,
            syndromes_Z,
        )

    def attach_ml_model(self, model, logits_fn) -> None:
        self._ml = model
        self._ml_logits = logits_fn


    def decode_batch_from_syndromes_uniform(self, batch_X: List[List[int]], batch_Z: List[List[int]], **kwargs):
        """Decode a batch of X/Z syndrome pairs using the same runtime/builder.
        Returns list of (resX, resZ)."""
        out = []
        for sX, sZ in zip(batch_X, batch_Z, strict=True):
            out.append(self.decode_from_syndromes_uniform(sX, sZ, **kwargs))
        return out


    def decode_from_softinfo(self, soft_llr_X, soft_llr_Z, leakage_flags=None, **kwargs):
        from .llr_schema import SoftSyndrome, validate_soft

        validate_soft(
            SoftSyndrome(
                llr_X=soft_llr_X, llr_Z=soft_llr_Z,
                leakage_flags=leakage_flags,
            )
        )
        if leakage_flags is not None:
            raise NotImplementedError(
                "leakage_flags are not implemented by the binary-sign "
                "soft-information decoder; refusing to discard them"
            )
        # This path is intentionally sign-only, not an analog soft decoder.
        syndromes_x = [int(float(value) < 0) for value in soft_llr_X]
        syndromes_z = [int(float(value) < 0) for value in soft_llr_Z]
        return self.decode_from_syndromes_uniform(
            syndromes_x, syndromes_z, **kwargs
        )


    def decode_from_dem_text(self, dem_text: str, syndrome=None):
        """Decode observed detection events against a Stim detector error model.

        A detector error model contains probabilities and detector connectivity,
        not the observed detection events for a shot. Callers must therefore
        provide the syndrome explicitly. The returned NumPy vector contains the
        predicted observable/fault flips from PyMatching sparse blossom.
        """
        if syndrome is None:
            raise ValueError(
                "A detector error model does not contain an observed syndrome. "
                "Pass detection-event bits with syndrome=..."
            )
        from .decoder_mwpm_pm import PyMatchingMWPMDecoder

        self._last_dem_text = dem_text
        return PyMatchingMWPMDecoder().decode_dem(dem_text, syndrome)

    def decode_from_dem_file(self, path: str, syndrome=None):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        return self.decode_from_dem_text(text, syndrome=syndrome)
