from __future__ import annotations

from collections import Counter

from a3d.decoder_mwpm import MWPMDecoder, _is_boundary_node
from a3d.graph import DecodingGraph, Edge


def test_mwpm_backend_is_sparse_blossom():
    assert MWPMDecoder.backend == "pymatching-sparse-blossom"


def test_mwpm_returns_path_edges_between_defects():
    graph = DecodingGraph(
        nodes=[0, 1, 2],
        edges=[
            Edge(0, 1, 2.0, "space"),
            Edge(1, 2, 2.0, "space"),
        ],
        node_meta={
            0: ("X", None, 0, "stab"),
            1: ("X", None, 0, "stab"),
            2: ("X", None, 0, "stab"),
        },
    )
    result = MWPMDecoder().decode(graph, [1, 0, 1])
    assert result.corrections == graph.edges
    assert result.matched_to_boundary == []


def test_mwpm_maps_virtual_boundary_back_to_aegis_edge():
    boundary = 2
    graph = DecodingGraph(
        nodes=[0, 1, boundary],
        edges=[
            Edge(0, 1, 5.0, "space"),
            Edge(0, boundary, 1.0, "boundary"),
        ],
        node_meta={
            0: ("X", None, 0, "stab"),
            1: ("X", None, 0, "stab"),
            boundary: ("X", None, 0, "boundary-H-W"),
        },
    )
    result = MWPMDecoder().decode(graph, [1, 0])
    assert result.corrections == [graph.edges[1]]
    assert result.matched_to_boundary == [0]


def test_batch_reuses_sparse_blossom_graph_and_preserves_results():
    graph = DecodingGraph(
        nodes=[0, 1, 2],
        edges=[
            Edge(0, 1, 2.0, "space"),
            Edge(1, 2, 2.0, "space"),
        ],
        node_meta={
            0: ("X", None, 0, "stab"),
            1: ("X", None, 0, "stab"),
            2: ("X", None, 0, "stab"),
        },
    )
    decoder = MWPMDecoder()
    results = decoder.decode_batch(graph, [[1, 0, 1], [0, 0, 0]])
    assert results[0].corrections == graph.edges
    assert results[1].corrections == []



def _normalise_pm_edge(pair):
    u, v = int(pair[0]), int(pair[1])
    if v == -1:
        return (u, -1)
    return tuple(sorted((u, v)))


def _normalise_aegis_edge(graph, compiled, edge):
    u_boundary = _is_boundary_node(graph, edge.u)
    v_boundary = _is_boundary_node(graph, edge.v)
    if u_boundary or v_boundary:
        detector = edge.v if u_boundary else edge.u
        return (compiled.node_to_detector[detector], -1)
    return tuple(
        sorted(
            (
                compiled.node_to_detector[edge.u],
                compiled.node_to_detector[edge.v],
            )
        )
    )


def test_fault_vector_mapping_matches_pymatching_edge_oracle():
    left = 4
    right = 5
    graph = DecodingGraph(
        nodes=[0, 1, 2, 3, left, right],
        edges=[
            Edge(0, 1, 0.91, "space"),
            Edge(1, 2, 1.17, "space"),
            Edge(2, 3, 1.43, "space"),
            Edge(0, left, 1.83, "boundary"),
            Edge(3, right, 2.11, "boundary"),
            Edge(1, 3, 2.37, "space"),
        ],
        node_meta={
            0: ("X", None, 0, "stab"),
            1: ("X", None, 0, "stab"),
            2: ("X", None, 0, "stab"),
            3: ("X", None, 0, "stab"),
            left: ("X", None, 0, "boundary-H-W"),
            right: ("X", None, 0, "boundary-H-E"),
        },
    )
    decoder = MWPMDecoder()
    compiled = decoder._compile(graph)

    for syndrome in (
        [1, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 1],
    ):
        result = decoder.decode(graph, syndrome)
        prepared = decoder._prepare_syndrome(graph, compiled, syndrome)
        oracle_edges = compiled.matching.decode_to_edges_array(prepared)

        expected = Counter(_normalise_pm_edge(pair) for pair in oracle_edges)
        actual = Counter(
            _normalise_aegis_edge(graph, compiled, edge)
            for edge in result.corrections
        )
        assert actual == expected



def test_batch_accepts_numpy_detector_matrix_without_truth_ambiguity():
    import numpy as np

    graph = DecodingGraph(
        nodes=[0, 1],
        edges=[Edge(0, 1, 1.0, "space")],
        node_meta={
            0: ("X", None, 0, "stab"),
            1: ("X", None, 0, "stab"),
        },
    )
    samples = np.asarray([[1, 1], [0, 0]], dtype=np.uint8)
    results = MWPMDecoder().decode_batch(graph, samples)
    assert len(results) == 2
    assert results[0].corrections == graph.edges
    assert results[1].corrections == []
    assert MWPMDecoder().decode_batch(
        graph, np.empty((0, 2), dtype=np.uint8)
    ) == []
