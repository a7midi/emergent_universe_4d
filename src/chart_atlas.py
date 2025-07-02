"""
ChartAtlas
──────────
• builds local exponential charts B^(k)(c) → ℝ⁴ using classical MDS  
• stitches them with Procrustes alignment  
• exposes

    atlas.position(node_id)        → (x, y, z, τ)
    atlas.id_map                   → {node_id: row‑index}
    atlas.global_coords            → N×4 numpy array (rows ordered by id_map)

so that export_data.py and kinematics.py can do fast vectorised look‑ups.
"""

from __future__ import annotations
from typing import Dict, List

import numpy as np
from sklearn.manifold import MDS
from scipy.linalg import orthogonal_procrustes
from tqdm import tqdm

from src.depth_metric import DepthMetric


class ChartAtlas:
    # -------------------------------------------------- #
    def __init__(
        self,
        causal_site: "CausalSite",
        dmetric: DepthMetric,
        chart_scale_k: int = 4,
        gh_tolerance: float = 0.05,
    ) -> None:
        self.site = causal_site
        self.dmetric = dmetric
        self.k0 = chart_scale_k
        self.gh_tol = gh_tolerance

        # nid → 4‑vector   (filled during _build_atlas)
        self.coords: Dict[int, np.ndarray] = {}

        self._build_atlas()
        self._freeze()

    # -------------------------------------------------- #
    #  private helpers                                   #
    # -------------------------------------------------- #
    def _build_atlas(self) -> None:
        nodes = list(self.site.graph.nodes)
        rng = np.random.default_rng(42)

        for centre in tqdm(nodes, desc="Stitching Charts"):
            k = self.site.graph.nodes[centre]["layer"]
            if k < self.k0:
                continue
            L_k = 2.0 ** (-k)

            # collect ball
            ball: List[int] = [
                v
                for v in nodes
                if self.dmetric.get_symmetric_radius(centre, v) <= L_k
            ]
            if len(ball) < 5:
                continue

            # distance matrix (symmetric)
            dist = np.zeros((len(ball), len(ball)))
            for i, u in enumerate(ball):
                for j, v in enumerate(ball):
                    if i < j:
                        d = self.dmetric.get_symmetric_radius(u, v)
                        dist[i, j] = dist[j, i] = d

            # MDS embedding → ℝ³  (we append τ later)
            mds = MDS(
                n_components=3,
                dissimilarity="precomputed",
                random_state=rng,
                normalized_stress="auto",
            )
            X3 = mds.fit_transform(dist)

            # align with existing coords if overlap ≥ 4 points
            overlap = [v for v in ball if v in self.coords]
            if len(overlap) >= 4:
                A = X3[[ball.index(v) for v in overlap]]
                B = np.array([self.coords[v][:3] for v in overlap])
                R, _ = orthogonal_procrustes(A, B)
                X3 = X3 @ R + B.mean(0) - (A @ R).mean(0)

            for v, xyz in zip(ball, X3):
                τ = -self.site.graph.nodes[v]["layer"]
                self.coords[v] = np.append(xyz, τ)

    # freeze into fast structures
    def _freeze(self):
        sorted_ids = sorted(self.coords)
        self.id_map = {nid: i for i, nid in enumerate(sorted_ids)}
        self.global_coords = np.vstack([self.coords[n] for n in sorted_ids])

    # -------------------------------------------------- #
    #  public helper                                     #
    # -------------------------------------------------- #
    def position(self, v: int) -> np.ndarray:
        """
        Return 4‑vector (x,y,z,τ) for node v.  If v is outside any chart
        (shouldn’t happen) fall back to depth‑only τ.
        """
        if v in self.coords:
            return self.coords[v]

        # Fallback – put node at origin, only τ meaningful
        τ = -self.site.graph.nodes[v]["layer"]
        self.coords[v] = np.array([0.0, 0.0, 0.0, τ])
        return self.coords[v]
