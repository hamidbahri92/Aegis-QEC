from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import List, Set, Tuple

from .graph import DecodingGraph, Edge
from .stats import effective_cost_from_edge


@dataclass
class DecodeResult:
    corrections: List[Edge]
    log_likelihood: float
    matched_to_boundary: List[int]
    avg_cost: float


class GreedyMatchingDecoder:
    @classmethod
    def _syndrome_by_node(
        cls, graph: DecodingGraph, syndromes: List[int]
    ) -> dict[int, int]:
        """Map compact detector or full-graph vectors to actual graph node IDs."""
        detectors = [
            node for node in graph.nodes
            if not cls._is_boundary_node(graph, node)
        ]
        if len(syndromes) == len(detectors):
            nodes = detectors
        elif len(syndromes) == len(graph.nodes):
            nodes = graph.nodes
        else:
            raise ValueError(
                "Syndrome length mismatch: expected "
                f"{len(detectors)} detector bits or "
                f"{len(graph.nodes)} graph node bits"
            )
        if any(value not in (0, 1) for value in syndromes):
            raise ValueError("Syndrome must contain only binary detector bits")
        values = {node: int(value) for node, value in zip(nodes, syndromes, strict=True)}
        if any(
            value for node, value in values.items()
            if cls._is_boundary_node(graph, node)
        ):
            raise ValueError("virtual boundary nodes cannot have detector events")
        return values

    @staticmethod
    def _defects(syndromes: List[int]) -> Set[int]:
        return {i for i, s in enumerate(syndromes) if s == 1}

    @staticmethod
    def _effective_cost(edge: Edge) -> float:
        return effective_cost_from_edge(edge)

    @staticmethod
    def _is_boundary_node(graph: DecodingGraph, nid: int) -> bool:
        role = graph.node_meta.get(nid, ("", None, 0, ""))[3]
        return role.startswith("boundary")

    def decode(self, graph: DecodingGraph, syndromes: List[int]) -> DecodeResult:
        values = self._syndrome_by_node(graph, syndromes)
        defects = {node for node, bit in values.items() if bit == 1}
        if not defects:
            return DecodeResult([], 0.0, [], 0.0)

        edge_heap: List[Tuple[float, int, Edge]] = [
            (self._effective_cost(e), idx, e) for idx, e in enumerate(graph.edges)
        ]
        heapq.heapify(edge_heap)

        matched: Set[int] = set()
        chosen: List[Edge] = []
        to_boundary: List[int] = []
        total_cost = 0.0

        while edge_heap and (len(matched) < len(defects)):
            cost, _idx, edge = heapq.heappop(edge_heap)
            u, v = edge.u, edge.v
            u_is_b = self._is_boundary_node(graph, u)
            v_is_b = self._is_boundary_node(graph, v)

            if not u_is_b and not v_is_b:
                if (
                    (u in defects and v in defects)
                    and (u not in matched)
                    and (v not in matched)
                ):
                    chosen.append(edge)
                    matched.add(u)
                    matched.add(v)
                    total_cost += cost
            else:
                s = v if u_is_b else u
                if (s in defects) and (s not in matched):
                    chosen.append(edge)
                    matched.add(s)
                    to_boundary.append(s)
                    total_cost += cost

        if len(matched) != len(defects):
            raise ValueError(
                "greedy decoding failed to resolve every detector event"
            )

        pair_count = max(1, len(chosen))
        avg_cost = total_cost / pair_count

        return DecodeResult(
            corrections=chosen,
            log_likelihood=total_cost,
            matched_to_boundary=to_boundary,
            avg_cost=avg_cost,
        )
