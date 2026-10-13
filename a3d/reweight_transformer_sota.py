# FILE: a3d/reweight_transformer_sota.py
from __future__ import annotations

from typing import List

try:
    import torch
    import torch.nn as nn
    TORCH_OK=True
except Exception:
    TORCH_OK=False
    torch=None
    nn = None  # type: ignore[assignment]

_ModuleBase = nn.Module if TORCH_OK else object

from .graph import DecodingGraph
from .stats import effective_cost_from_edge


class SwiGLU(_ModuleBase):
    def __init__(self, d: int, mult: int = 4):
        super().__init__()
        self.w1 = nn.Linear(d, mult*d)
        self.w2 = nn.Linear(d, mult*d)
        self.w3 = nn.Linear(mult*d, d)
        self.act = nn.SiLU()
    def forward(self, x):
        return self.w3(self.act(self.w1(x)) * self.w2(x))

class Block(_ModuleBase):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.ff = SwiGLU(d)
    def forward(self, x):
        h = x
        x = self.norm1(x)
        x,_ = self.attn(x,x,x, need_weights=False)
        x = x + h
        h = x
        x = self.norm2(x)
        x = self.ff(x) + h
        return x

class SotaEdgeTransformer(_ModuleBase):
    def __init__(self, d_model: int = 64, heads: int = 8, layers: int = 4, in_dim: int = 10):
        super().__init__()
        self.enc = nn.Linear(in_dim, d_model)
        self.blocks = nn.ModuleList([Block(d_model, heads) for _ in range(layers)])
        self.out = nn.Linear(d_model, 1)
    def forward(self, x):
        h = self.enc(x)
        for blk in self.blocks:
            h = blk(h)
        return self.out(h).squeeze(-1)

def _edge_features(graph: DecodingGraph) -> List[List[float]]:
    deg = [0]*len(graph.nodes)
    for e in graph.edges:
        deg[e.u]+=1; deg[e.v]+=1
    feats = []
    for e in graph.edges:
        mu = graph.node_meta.get(e.u, ("",None,0,""))
        mv = graph.node_meta.get(e.v, ("",None,0,""))
        _, cu, tu, _ = mu
        _, cv, tv, _ = mv
        is_time = 1.0 if e.etype=="time" else 0.0
        is_bnd  = 1.0 if e.etype=="boundary" else 0.0
        is_space= 1.0 if e.etype=="space" else 0.0
        pe = float(e.p_erase)
        base = effective_cost_from_edge(e)
        du = deg[e.u]; dv = deg[e.v]
        dist = 0.0
        if cu is not None and cv is not None:
            dist = abs(cu[0]-cv[0]) + abs(cu[1]-cv[1])
        feats.append([base, is_time, is_bnd, is_space, pe, float(du), float(dv), float(tu), float(tv), float(dist)])
    return feats

class GraphEdgeTransformerSOTA:
    def __init__(self, weights_path: str = "", alpha: float = 0.75):
        if not weights_path:
            raise ValueError(
                "transformer reweighting requires an explicitly trained checkpoint; "
                "untrained random weights are not scientific evidence"
            )
        if not TORCH_OK:
            raise ImportError("PyTorch is required for transformer reweighting")
        self.alpha = float(alpha)
        self.weights_path = weights_path
        self.model = SotaEdgeTransformer()
        state = torch.load(
            weights_path, map_location="cpu", weights_only=True
        )
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    def reweight(self, graph: DecodingGraph):
        costs = [effective_cost_from_edge(e) for e in graph.edges]
        if not TORCH_OK or self.model is None or len(costs)==0:
            return costs
        import torch
        feats = _edge_features(graph)
        with torch.no_grad():
            x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)
            logits = self.model(x)[0]
            adj = torch.sigmoid(logits).tolist()
        return [max(1e-6, c + self.alpha*float(a)) for c,a in zip(costs, adj, strict=True)]
