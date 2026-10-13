"""Union-find heuristic must respect node identities and prove residual parity."""
from __future__ import annotations

import pytest

from a3d.decoder_unionfind import UnionFindErasuresDecoder
from a3d.graph import DecodingGraph, Edge


def _graph(boundary=False, erasure=False):
    nodes = [10, 20, 99] if boundary else [10, 20]
    edges = [Edge(10, 20, 2.0, "time" if erasure else "space", 1.0 if erasure else 0.0)]
    if boundary:
        edges.append(Edge(10, 99, 3.0, "boundary"))
    return DecodingGraph(
        nodes=nodes,
        edges=edges,
        node_meta={
            10: ("X", None, 0, "stab"),
            20: ("X", None, 0, "stab"),
            99: ("X", None, 0, "boundary-H-W"),
        },
    )


@pytest.mark.parametrize("syndrome", ([1, 1], [1, 1, 0]))
def test_unionfind_uses_real_noncontiguous_detector_node_ids(syndrome):
    graph = _graph(boundary=True)
    result = UnionFindErasuresDecoder().decode(graph, syndrome)
    assert result.corrections == [graph.edges[0]]
    assert result.log_likelihood == pytest.approx(2.0)


def test_unionfind_erasure_peeling_preserves_full_correction_cost():
    graph = _graph(erasure=True)
    result = UnionFindErasuresDecoder().decode(graph, [1, 1])
    assert result.corrections == graph.edges
    assert result.log_likelihood == pytest.approx(2.0)


def test_unionfind_single_defect_to_boundary_is_proven():
    graph = _graph(boundary=True)
    result = UnionFindErasuresDecoder().decode(graph, [1, 0])
    assert result.corrections == [graph.edges[1]]
    assert result.matched_to_boundary == [10]


def test_unionfind_rejects_incorrect_or_unresolved_syndromes():
    graph = _graph()
    with pytest.raises(ValueError, match="binary"):
        UnionFindErasuresDecoder().decode(graph, [2, 0])
    with pytest.raises(ValueError, match="length mismatch"):
        UnionFindErasuresDecoder().decode(graph, [1])
    with pytest.raises(ValueError, match="did not satisfy"):
        UnionFindErasuresDecoder().decode(graph, [1, 0])


def test_unionfind_rejects_edges_with_unknown_node_ids():
    graph = _graph()
    graph.edges.append(Edge(20, 987, 1.0, "space"))
    with pytest.raises(ValueError, match="unknown node ID"):
        UnionFindErasuresDecoder().decode(graph, [1, 1])
