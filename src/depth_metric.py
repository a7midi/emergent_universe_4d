"""
Depth‑scaled quasi‑metric  d_∞ defined in Paper III §2.7.

On a finite acyclic causal site depth(c) is just the layer index.
Because d_∞(u,v)=2^(−min(depth(u),depth(v))) depends only on v once reachability
is known, we pre‑compute the value per vertex and store a reachability bitset.
"""
from __future__ import annotations
from typing import Dict, Tuple
import networkx as nx
import numpy as np

class DepthMetric:
    def __init__(self, causal_site: "CausalSite"):
        self.site = causal_site
        # value table: v ↦ 2^(−depth(v))
        self._val = {v: 2.0 ** (-self.site.graph.nodes[v]["layer"])
                     for v in self.site.graph.nodes}
        # reachability matrix as bit‑array of sets for O(1) membership test
        self._reach: Dict[int, set[int]] = {}
        for u in self.site.graph.nodes:
            self._reach[u] = set(nx.descendants(self.site.graph, u)) | {u}

    # ------------------------------------------------------------------ #
    def d_infty(self, u: int, v: int) -> float:
        return self._val[v] if v in self._reach[u] else float("inf")

    def get_symmetric_radius(self, u: int, v: int) -> float:
        """r(u,v)=max{d(u,v),d(v,u)}  — a true metric."""
        return max(self.d_infty(u, v), self.d_infty(v, u))
