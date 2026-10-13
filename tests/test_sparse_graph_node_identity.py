"""Identity-safe decoding regression tests on sparse, non-contiguous nodes."""
from __future__ import annotations

import pytest

from a3d.decoder_bposd import OSDDecoder
from a3d.decoder_greedy import GreedyMatchingDecoder
from a3d.graph import DecodingGraph, Edge


def _graph():
    return DecodingGraph(
        nodes=[10, 20, 99],
        edges=[
            Edge(10, 20, 1.0, "space"),
            Edge(10, 99, 3.0, "boundary"),
        ],
        node_meta={
            10: ("X", None, 0, "stab"),
            20: ("X", None, 0, "stab"),
            99: ("X", None, 0, "boundary-H-W"),
        },
    )


@pytest.mark.parametrize("input_bits", ([1, 1], [1, 1, 0]))
def test_greedy_maps_compact_and_expanded_syndromes_to_real_node_ids(input_bits):
    graph = _graph()
    result = GreedyMatchingDecoder().decode(graph, input_bits)
    assert result.corrections == [graph.edges[0]]
    assert result.matched_to_boundary == []


def test_greedy_uses_actual_node_id_for_boundary_correction():
    graph = _graph()
    result = GreedyMatchingDecoder().decode(graph, [1, 0])
    assert result.corrections == [graph.edges[1]]
    assert result.matched_to_boundary == [10]


def test_greedy_rejects_unreachable_or_nonbinary_detector():
    graph = _graph()
    with pytest.raises(ValueError, match="binary"):
        GreedyMatchingDecoder().decode(graph, [2, 0])
    with pytest.raises(ValueError, match="boundary"):
        GreedyMatchingDecoder().decode(graph, [0, 0, 1])
    with pytest.raises(ValueError, match="length mismatch"):
        GreedyMatchingDecoder().decode(graph, [1])
    disconnected = DecodingGraph(
        nodes=[10, 20], edges=[], node_meta=graph.node_meta
    )
    with pytest.raises(ValueError, match="failed to resolve"):
        GreedyMatchingDecoder().decode(disconnected, [1, 0])


def test_osd_preserves_original_syndrome_when_trial_unresolvable():
    graph = _graph()
    osd = OSDDecoder(GreedyMatchingDecoder(), osd_order=1)
    result = osd.decode(graph, [1, 1])
    assert result.corrections == [graph.edges[0]]
    assert result.avg_cost == pytest.approx(1.0)
