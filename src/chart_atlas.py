"""
ChartAtlas – FINAL hop‑based neighbourhood implementation
Matches the analysis: local “balls” are defined by hop distance, not by the
(symmetric) metric that is ∞ for almost every pair in a DAG.
"""

from __future__ import annotations
from collections import deque
from typing import Dict, List, Set

import itertools
import numpy as np
import networkx as nx
from sklearn.manifold import MDS
from scipy.linalg import orthogonal_procrustes
from tqdm import tqdm

from src.depth_metric import DepthMetric


class ChartAtlas:
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        causal_site: "CausalSite",
        dmetric: DepthMetric,
        chart_scale_k: int = 4,
        gh_tol: float = 0.05,
    ):
        self.site = causal_site
        self.metric = dmetric
        self.k0 = chart_scale_k

        self.node_ids = list(self.site.graph.nodes)
        self.id_map = {nid: i for i, nid in enumerate(self.node_ids)}
        self.global_coords = np.full((len(self.node_ids), 4), np.nan, np.float64)

        self._build_atlas()

    # ------------------------------------------------------------------ #
    def _max_hops(self) -> int:
        """Heuristic: 2 hops for k≥4, 3 for k=3, 4 for k≤2."""
        return max(2, 6 - self.k0)

    # ------------------------------------------------------------------ #
    def _ball_by_hops(self, centre: int) -> List[int]:
        """BFS up to max_hops in either causal direction."""
        max_h = self._max_hops()
        visited: Set[int] = {centre}
        ball = [centre]
        q = deque([(centre, 0)])
        while q:
            nid, h = q.popleft()
            if h >= max_h:
                continue
            for nb in itertools.chain(
                self.site.graph.successors(nid), self.site.graph.predecessors(nid)
            ):
                if nb not in visited:
                    visited.add(nb)
                    ball.append(nb)
                    q.append((nb, h + 1))
        return ball

    # ------------------------------------------------------------------ #
    def _embed_and_place(self, centre: int) -> Set[int]:
        nodes = self._ball_by_hops(centre)
        if len(nodes) < 4:
            return set()

        idxs = [self.id_map[n] for n in nodes]

        # pairwise symmetric radius inside the small ball
        D = np.zeros((len(idxs), len(idxs)))
        for i, u in enumerate(nodes):
            for j in range(i + 1, len(nodes)):
                d = self.metric.get_symmetric_radius(u, nodes[j])
                D[i, j] = D[j, i] = d if np.isfinite(d) else 1.0  # default large

        X3 = MDS(
            n_components=3,
            dissimilarity="precomputed",
            normalized_stress=False,
            random_state=0,
            max_iter=120,
        ).fit_transform(D)

        # align with already‑stitched coords if ≥3 in common
        existing = [i for i in idxs if not np.isnan(self.global_coords[i, 0])]
        if len(existing) >= 3:
            A = X3[[idxs.index(i) for i in existing]]
            B = self.global_coords[existing, :3]
            R, _ = orthogonal_procrustes(A, B)
            X3 = X3 @ R + B.mean(0) - (A @ R).mean(0)

        placed = set()
        for loc, row in zip(X3, idxs):
            if np.isnan(self.global_coords[row, 0]):
                τ = -self.site.graph.nodes[self.node_ids[row]]["layer"]
                self.global_coords[row] = np.append(loc, τ)
                placed.add(row)
        return placed

    # ------------------------------------------------------------------ #
    def _build_atlas(self):
        undirected = self.site.graph.to_undirected()
        comps = list(nx.connected_components(undirected))

        pbar = tqdm(total=len(self.node_ids), desc="Stitching Charts")
        stitched: Set[int] = set()

        for comp in comps:
            todo = deque([next(iter(comp))])
            while todo:
                centre = todo.popleft()
                if centre in stitched:
                    continue
                newly = self._embed_and_place(centre)
                if newly:
                    stitched |= newly
                    pbar.update(len(newly))
                    for nid in newly:
                        for nb in itertools.chain(
                            self.site.graph.successors(nid),
                            self.site.graph.predecessors(nid),
                        ):
                            if nb in comp and nb not in stitched:
                                todo.append(nb)
        pbar.close()

        # any remaining unplaced nodes → fallback (0,0,0,τ)
        for nid in self.node_ids:
            row = self.id_map[nid]
            if np.isnan(self.global_coords[row, 0]):
                τ = -self.site.graph.nodes[nid]["layer"]
                self.global_coords[row] = np.array([0.0, 0.0, 0.0, τ])

    # ------------------------------------------------------------------ #
    def position(self, nid: int) -> np.ndarray:
        return self.global_coords[self.id_map[nid]]
