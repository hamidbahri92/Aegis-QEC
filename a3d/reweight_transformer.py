
# FILE: a3d/reweight_transformer.py
from __future__ import annotations

from typing import List

try:
    import torch
    import torch.nn as nn
    TORCH_OK = True
except Exception:  # pragma: no cover - optional dependency
    TORCH_OK = False
    torch = None
    nn = None  # type: ignore[assignment]

_ModuleBase = nn.Module if TORCH_OK else object

from .graph import DecodingGraph
from .stats import effective_cost_from_edge


def _edge_features(graph: DecodingGraph) -> List[List[float]]:
    """Simple structural features for each edge (u,v).
    - base_cost (LLR/NLL fused with erasure)
    - is_time, is_boundary, is_space (one-hot)
    - p_erase
    - degree(u), degree(v)
    - round(u), round(v) (from node_meta)
    - manhattan distance between coords if available else 0
    """
    deg = [0]*len(graph.nodes)
    for e in graph.edges:
        deg[e.u]+=1; deg[e.v]+=1
    feats: List[List[float]] = []
    for e in graph.edges:
        meta_u = graph.node_meta.get(e.u, ("", None, 0, ""))
        meta_v = graph.node_meta.get(e.v, ("", None, 0, ""))
        _, cu, tu, ru = meta_u
        _, cv, tv, rv = meta_v
        is_time = 1.0 if e.etype=="time" else 0.0
        is_bnd  = 1.0 if e.etype=="boundary" else 0.0
        is_space= 1.0 if e.etype=="space" else 0.0
        pe = float(e.p_erase)
        base = effective_cost_from_edge(e)
        du = deg[e.u]; dv = deg[e.v]
        if cu is not None and cv is not None:
            dist = abs(cu[0]-cv[0]) + abs(cu[1]-cv[1])
        else:
            dist = 0.0
        feats.append([base, is_time, is_bnd, is_space, pe, float(du), float(dv), float(tu), float(tv), float(dist)])
    return feats

class _TinyEdgeTransformer(_ModuleBase):
    def __init__(self, d_model: int = 32, nhead: int = 4, nlayers: int = 2):
        super().__init__()
        self.inp = nn.Linear(10, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.out = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (1, E, 10)
        h = self.inp(x)
        h = self.enc(h)
        y = self.out(h).squeeze(-1)  # (1, E)
        return y

class GraphEdgeTransformerReweighter:
    """Optional transformer that predicts **logit adjustments** per edge.
    Final cost := clamp( effective_cost + alpha * sigmoid(logit), min=1e-6 ).
    """
    def __init__(self, weights_path: str = "", alpha: float = 0.5):
        if not weights_path:
            raise ValueError(
                "transformer reweighting requires an explicitly trained checkpoint; "
                "untrained random weights are not scientific evidence"
            )
        if not TORCH_OK:
            raise ImportError("PyTorch is required for transformer reweighting")
        self.alpha = float(alpha)
        self.weights_path = weights_path
        self.model = _TinyEdgeTransformer()
        state = torch.load(
            weights_path, map_location="cpu", weights_only=True
        )
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    def reweight(self, graph: DecodingGraph) -> List[float]:
        feats = _edge_features(graph)
        costs = [effective_cost_from_edge(e) for e in graph.edges]
        if not TORCH_OK or self.model is None or len(feats)==0:
            return costs
        import torch
        with torch.no_grad():
            x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)  # (1,E,10)
            logits = self.model(x)[0]  # (E,)
            adj = torch.sigmoid(logits).tolist()
        new_costs: List[float] = []
        for c,a in zip(costs, adj, strict=True):
            new_costs.append(max(1e-6, c + self.alpha*float(a)))
        return new_costs
