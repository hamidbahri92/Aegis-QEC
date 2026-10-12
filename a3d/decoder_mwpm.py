from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pymatching

from .decoder_greedy import DecodeResult
from .graph import DecodingGraph, Edge
from .stats import effective_cost_from_edge


def _effective_cost(edge: Edge) -> float:
    return float(effective_cost_from_edge(edge))


def _is_boundary_node(graph: DecodingGraph, nid: int) -> bool:
    meta = graph.node_meta.get(nid)
    if meta and len(meta) >= 4:
        role = meta[3]
        return isinstance(role, str) and role.startswith("boundary")
    return False


@dataclass(frozen=True)
class _CompiledMatching:
    matching: pymatching.Matching
    detector_nodes: Tuple[int, ...]
    node_to_detector: Dict[int, int]
    graph_positions: Dict[int, int]


class MWPMDecoder:
    """Exact MWPM using PyMatching v2+'s sparse-blossom implementation.

    Aegis graphs contain explicit boundary nodes. PyMatching is slightly more
    efficient with a virtual boundary, so boundary edges are translated with
    Matching.add_boundary_edge and detector nodes are compactly re-indexed.

    Every Aegis graph edge is assigned its own PyMatching fault id. This lets
    PyMatching return the actual correction path, not just the paired defects,
    and gives Aegis a faithful mapping back to Edge objects.
    """

    backend = "pymatching-sparse-blossom"

    def __init__(self) -> None:
        self._cached_signature: Tuple[object, ...] | None = None
        self._cached_compiled: _CompiledMatching | None = None

    def _signature(self, graph: DecodingGraph) -> Tuple[object, ...]:
        detector_nodes = tuple(
            n for n in graph.nodes if not _is_boundary_node(graph, n)
        )
        edge_signature = tuple(
            (
                int(e.u),
                int(e.v),
                _effective_cost(e),
                _is_boundary_node(graph, e.u),
                _is_boundary_node(graph, e.v),
            )
            for e in graph.edges
        )
        return detector_nodes, edge_signature

    def _compile(self, graph: DecodingGraph) -> _CompiledMatching:
        signature = self._signature(graph)
        if signature == self._cached_signature and self._cached_compiled is not None:
            return self._cached_compiled

        detector_nodes = tuple(
            n for n in graph.nodes if not _is_boundary_node(graph, n)
        )
        node_to_detector = {nid: i for i, nid in enumerate(detector_nodes)}
        graph_positions = {nid: i for i, nid in enumerate(graph.nodes)}
        matching = pymatching.Matching()

        for edge_index, edge in enumerate(graph.edges):
            u_boundary = _is_boundary_node(graph, edge.u)
            v_boundary = _is_boundary_node(graph, edge.v)
            weight = _effective_cost(edge)

            if u_boundary and v_boundary:
                continue

            if u_boundary or v_boundary:
                detector = edge.v if u_boundary else edge.u
                if detector not in node_to_detector:
                    continue
                matching.add_boundary_edge(
                    node_to_detector[detector],
                    fault_ids={edge_index},
                    weight=weight,
                    merge_strategy="smallest-weight",
                )
                continue

            matching.add_edge(
                node_to_detector[edge.u],
                node_to_detector[edge.v],
                fault_ids={edge_index},
                weight=weight,
                merge_strategy="smallest-weight",
            )

        matching.ensure_num_fault_ids(len(graph.edges))

        compiled = _CompiledMatching(
            matching=matching,
            detector_nodes=detector_nodes,
            node_to_detector=node_to_detector,
            graph_positions=graph_positions,
        )
        self._cached_signature = signature
        self._cached_compiled = compiled
        return compiled

    @staticmethod
    def _prepare_syndrome(
        graph: DecodingGraph,
        compiled: _CompiledMatching,
        syndromes: Sequence[int],
    ) -> np.ndarray:
        if len(syndromes) == len(compiled.detector_nodes):
            values = syndromes
        elif len(syndromes) == len(graph.nodes):
            values = [
                syndromes[compiled.graph_positions[nid]]
                for nid in compiled.detector_nodes
            ]
        else:
            raise ValueError(
                "Syndrome length mismatch: expected "
                f"{len(compiled.detector_nodes)} detector bits or "
                f"{len(graph.nodes)} graph-node bits, got {len(syndromes)}"
            )
        return np.asarray([int(v) & 1 for v in values], dtype=np.uint8)

    @staticmethod
    def _result_from_fault_vector(
        graph: DecodingGraph,
        fault_vector: Iterable[int],
    ) -> DecodeResult:
        bits = list(fault_vector)
        chosen_edges = [
            graph.edges[i]
            for i, bit in enumerate(bits[: len(graph.edges)])
            if int(bit) & 1
        ]

        total_cost = sum(_effective_cost(edge) for edge in chosen_edges)
        matched_to_boundary: List[int] = []
        seen_boundary_detectors = set()
        for edge in chosen_edges:
            u_boundary = _is_boundary_node(graph, edge.u)
            v_boundary = _is_boundary_node(graph, edge.v)
            if u_boundary ^ v_boundary:
                detector = edge.v if u_boundary else edge.u
                if detector not in seen_boundary_detectors:
                    seen_boundary_detectors.add(detector)
                    matched_to_boundary.append(detector)

        avg_cost = total_cost / max(1, len(chosen_edges))
        return DecodeResult(
            corrections=chosen_edges,
            log_likelihood=total_cost,
            matched_to_boundary=matched_to_boundary,
            avg_cost=avg_cost,
        )

    def decode(
        self, graph: DecodingGraph, syndromes: Sequence[int]
    ) -> DecodeResult:
        compiled = self._compile(graph)
        syndrome = self._prepare_syndrome(graph, compiled, syndromes)
        if not np.any(syndrome):
            return DecodeResult([], 0.0, [], 0.0)
        correction = compiled.matching.decode(syndrome)
        return self._result_from_fault_vector(graph, correction)

    def decode_batch(
        self,
        graph: DecodingGraph,
        syndrome_batch: Sequence[Sequence[int]],
    ) -> List[DecodeResult]:
        """Decode many syndromes with one compiled sparse-blossom graph."""
        if len(syndrome_batch) == 0:
            return []
        compiled = self._compile(graph)
        shots = np.stack(
            [
                self._prepare_syndrome(graph, compiled, syn)
                for syn in syndrome_batch
            ],
            axis=0,
        ).astype(np.uint8, copy=False)
        corrections = compiled.matching.decode_batch(shots)
        return [
            self._result_from_fault_vector(graph, correction)
            for correction in corrections
        ]
