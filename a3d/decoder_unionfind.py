# FILE: a3d/decoder_unionfind.py
from __future__ import annotations

from collections import defaultdict, deque
from typing import List, Set

from .accel import maybe_jit
from .decoder_greedy import DecodeResult, GreedyMatchingDecoder
from .graph import DecodingGraph, Edge
from .stats import effective_cost_from_edge


class _DSU:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.sz = [1]*n
        self.parity = [0]*n   # defects mod 2 in the component
        self.has_boundary = [False]*n

    def find(self, x:int)->int:
        while self.p[x]!=x:
            self.p[x]=self.p[self.p[x]]
            x=self.p[x]
        return x

    def unite(self,a:int,b:int):
        ra, rb = self.find(a), self.find(b)
        if ra==rb:
            return ra
        if self.sz[ra]<self.sz[rb]:
            ra,rb = rb,ra
        self.p[rb]=ra
        self.sz[ra]+=self.sz[rb]
        self.parity[ra]=(self.parity[ra]+self.parity[rb])%2
        self.has_boundary[ra]= self.has_boundary[ra] or self.has_boundary[rb]
        return ra


@maybe_jit
def _useful_pair(parity_ru: int, parity_rv: int, has_bnd_ru: int, has_bnd_rv: int) -> int:
    # returns 1 if edge is useful to connect components (pair odd or attach odd to boundary)
    if parity_ru==1 and parity_rv==1:
        return 1
    if (has_bnd_ru==1 or has_bnd_rv==1) and ((parity_ru==1) or (parity_rv==1)):
        return 1
    return 0
class UnionFindErasuresDecoder:
    """Union-Find style decoder with erasure-aware pre-peeling and edge ordering.

    Algorithm:
      1. **Peel erasures:** group nodes connected by time-like edges with high erasure probability
         and greedily connect odd defects along those chains.
      2. **Union-Find growth:** add remaining edges in ascending effective cost until all
         defects are neutralized (paired or absorbed by a boundary).
    """

    def _is_boundary(self, graph: DecodingGraph, nid:int)->bool:
        role = graph.node_meta.get(nid, ('',None,0,''))[3]
        return role.startswith('boundary')

    def _peel_erasures(self, graph: DecodingGraph, defects: Set[int], perase_threshold: float = 0.5) -> List[Edge]:
        """Connect defects along time-like erased edges first (p_erase >= threshold)."""
        adj = defaultdict(list)
        elig: List[Edge] = []
        for e in graph.edges:
            if e.etype == "time" and float(e.p_erase) >= perase_threshold:
                adj[e.u].append(e.v)
                adj[e.v].append(e.u)
                elig.append(e)
        visited: Set[int] = set()
        chosen: List[Edge] = []
        for d in list(defects):
            if d in visited or d not in adj:
                visited.add(d)
                continue
            # BFS component of erased-time connectivity
            comp: List[int] = []
            dq = deque([d]); visited.add(d)
            while dq:
                u = dq.popleft()
                comp.append(u)
                for v in adj[u]:
                    if v not in visited:
                        visited.add(v); dq.append(v)
            # connect odd nodes in this component along eligible edges (simple chain)
            comp_defs = [u for u in comp if u in defects]
            for i in range(1, len(comp_defs)):
                u, v = comp_defs[i-1], comp_defs[i]
                # pick any eligible edge between u and v if it exists
                ok = None
                for e in elig:
                    if (e.u==u and e.v==v) or (e.u==v and e.v==u):
                        ok = e; break
                if ok:
                    chosen.append(ok)
                    if u in defects: defects.remove(u)
                    if v in defects: defects.remove(v)
        return chosen

    def decode(self, graph: DecodingGraph, syndromes: List[int]) -> DecodeResult:
        """Decode graph-node detector events, rejecting incomplete corrections.

        This legacy union-find-style heuristic is not a substitute for a
        production sparse-blossom decoder. Only parity-consistent corrections
        may be returned as successful decoding results.
        """
        node_to_position = {
            node: index for index, node in enumerate(graph.nodes)
        }
        if len(node_to_position) != len(graph.nodes):
            raise ValueError("decoding graph contains duplicate node IDs")
        readings = GreedyMatchingDecoder._syndrome_by_node(
            graph, syndromes
        )
        original_defects = {
            node for node, bit in readings.items() if bit
        }
        defects = set(original_defects)

        # Erasure pre-peeling may consume some defects before DSU growth.
        chosen = self._peel_erasures(
            graph, defects, perase_threshold=0.5
        )
        matched_to_boundary: List[int] = []
        dsu = _DSU(len(graph.nodes))
        for node, position in node_to_position.items():
            dsu.parity[position] = int(node in defects)
            dsu.has_boundary[position] = self._is_boundary(graph, node)

        edges = sorted(graph.edges, key=effective_cost_from_edge)
        odd_components = sum(dsu.parity)

        for edge in edges:
            if edge.u not in node_to_position or edge.v not in node_to_position:
                raise ValueError("decoding edge references an unknown node ID")
            u = node_to_position[edge.u]
            v = node_to_position[edge.v]
            ru, rv = dsu.find(u), dsu.find(v)
            if ru == rv:
                continue
            before = dsu.parity[ru] + dsu.parity[rv]
            will_have_boundary = (
                dsu.has_boundary[ru] or dsu.has_boundary[rv]
            )
            useful = bool(
                _useful_pair(
                    int(dsu.parity[ru]), int(dsu.parity[rv]),
                    int(dsu.has_boundary[ru]), int(dsu.has_boundary[rv]),
                )
            )
            if not useful:
                continue

            dsu.unite(ru, rv)
            chosen.append(edge)
            if self._is_boundary(graph, edge.u) and edge.v in defects:
                matched_to_boundary.append(edge.v)
            elif self._is_boundary(graph, edge.v) and edge.u in defects:
                matched_to_boundary.append(edge.u)
            if before == 2:
                odd_components -= 2
            elif will_have_boundary and before == 1:
                odd_components -= 1
            if odd_components <= 0:
                break

        # DSU cluster parity is not sufficient to certify the final edge
        # correction. Verify the actual edge-boundary vector explicitly.
        residual = set(original_defects)
        for edge in chosen:
            for node in (edge.u, edge.v):
                if self._is_boundary(graph, node):
                    continue
                if node in residual:
                    residual.remove(node)
                else:
                    residual.add(node)
        if residual:
            raise ValueError(
                "union-find heuristic did not satisfy the observed syndrome"
            )

        total_cost = float(
            sum(effective_cost_from_edge(edge) for edge in chosen)
        )
        avg_cost = total_cost / max(1, len(chosen))
        return DecodeResult(
            chosen, total_cost, matched_to_boundary, avg_cost
        )
