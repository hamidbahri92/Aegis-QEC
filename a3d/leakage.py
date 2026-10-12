"""Leakage pre-growth on explicit decoding-graph node identities."""
from __future__ import annotations

from typing import List

from .graph import DecodingGraph, Edge


def apply_leakage_pregrowth(
    graph: DecodingGraph, leakage_flags: List[int]
) -> List[Edge]:
    """Choose boundary edges incident on flagged nodes.

    Leakage flags are indexed by graph.nodes order, not by graph node ID.
    Non-contiguous node identifiers must therefore be preserved.
    """
    if len(leakage_flags) > len(graph.nodes):
        raise ValueError("leakage flags exceed decoding graph node count")
    leaked: set[int] = set()
    for index, flag in enumerate(leakage_flags):
        if flag not in (0, 1):
            raise ValueError("leakage flags must be binary")
        if flag == 1:
            leaked.add(graph.nodes[index])
    if not leaked:
        return []
    return [
        edge
        for edge in graph.edges
        if edge.etype == "boundary" and (
            edge.u in leaked or edge.v in leaked
        )
    ]
