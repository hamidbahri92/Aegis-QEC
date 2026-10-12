"""Tests of auxiliary QEC evidence paths previously outside the CI suite."""
from __future__ import annotations

import numpy as np
import pytest

from a3d.analog_lut import GKPLUTBuilder
from a3d.graph import DecodingGraph, Edge
from a3d.leakage import apply_leakage_pregrowth
from a3d.llr_schema import SoftSyndrome, validate_soft


def test_leakage_flags_refer_to_node_order_not_node_id():
    graph = DecodingGraph(
        nodes=[10, 25, 99],
        edges=[
            Edge(10, 99, 1.0, "boundary"),
            Edge(25, 99, 2.0, "boundary"),
            Edge(10, 25, 3.0, "space"),
        ],
        node_meta={},
    )
    chosen = apply_leakage_pregrowth(graph, [1, 0, 0])
    assert chosen == [graph.edges[0]]
    assert apply_leakage_pregrowth(graph, [0, 1, 0]) == [graph.edges[1]]
    assert apply_leakage_pregrowth(graph, [0, 0, 0]) == []
    with pytest.raises(ValueError, match="binary"):
        apply_leakage_pregrowth(graph, [2, 0, 0])
    with pytest.raises(ValueError, match="exceed"):
        apply_leakage_pregrowth(graph, [1, 0, 0, 0])


def test_soft_syndrome_validates_numbers_even_under_python_optimization():
    validate_soft(
        SoftSyndrome(
            llr_X=[1.0, -3.0], llr_Z=[0.0, 2.0],
            p_erase_time=[0.0, 0.5, 1.0], leakage_flags=[0, 1],
        )
    )
    with pytest.raises(TypeError, match="must be lists"):
        validate_soft(SoftSyndrome(llr_X=(0.0,), llr_Z=[]))
    for bad in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError, match="nonfinite"):
            validate_soft(SoftSyndrome(llr_X=[bad], llr_Z=[]))
    for bad in (-0.01, 1.01, float("nan")):
        with pytest.raises(ValueError, match="erasure probability"):
            validate_soft(
                SoftSyndrome(llr_X=[], llr_Z=[], p_erase_time=[bad])
            )
    with pytest.raises(ValueError, match="binary"):
        validate_soft(
            SoftSyndrome(llr_X=[], llr_Z=[], leakage_flags=[0.5])
        )


def test_synthetic_gkp_lut_is_monotone_finite_and_explicitly_uncalibrated():
    lut = GKPLUTBuilder(-3.5, 3.5, 129).build_lut()
    assert lut["model"] == "synthetic_not_calibrated"
    assert lut["scientifically_validated"] is False
    assert np.all(np.diff(lut["x"]) > 0)
    assert len(lut["Lq"]) == len(lut["Lp"]) == 129
    assert np.all(np.isfinite(lut["Lq"]))
    assert np.all(np.isfinite(lut["Lp"]))
    for bounds in ((1.0, 0.0), (float("nan"), 2.0)):
        with pytest.raises(ValueError):
            GKPLUTBuilder(*bounds)
    with pytest.raises(ValueError, match="two samples"):
        GKPLUTBuilder(n=1)
