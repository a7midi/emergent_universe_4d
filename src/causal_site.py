# src/causal_site.py
from __future__ import annotations
import random
from typing import Dict, List

import networkx as nx
import numpy as np

from src.depth_metric import DepthMetric
from src.chart_atlas import ChartAtlas


class CausalSite:
    """
    Generates a finite acyclic causal site (layered DAG) together with
    emergent‑geometry helpers:  self.metric  and  self.atlas.
    """

    # -------------------------------------------------------------- #
    def __init__(self, config: Dict):
        self.config = config
        self.graph: nx.DiGraph = nx.DiGraph()
        self.nodes_by_layer: Dict[int, List[int]] = {}
        self.node_positions: Dict[int, np.ndarray] = {}  # legacy fall‑back
        self.metric: DepthMetric | None = None
        self.atlas: ChartAtlas | None = None

    # -------------------------------------------------------------- #
    #  1. generate graph                                             #
    # -------------------------------------------------------------- #
    def generate_graph(self) -> None:
        cfg = self.config["causal_site"]
        layers = cfg["layers"]
        avg = cfg["avg_nodes_per_layer"]
        edge_p = cfg["edge_probability"]
        max_back = cfg["max_lookback_layers"]
        R = self.config["tags"]["max_out_degree_R"]

        print("Generating causal site graph...")
        node_counter = 0
        for layer_idx in range(layers):
            n_in_layer = np.random.poisson(avg)
            self.nodes_by_layer[layer_idx] = []
            for _ in range(n_in_layer):
                self.graph.add_node(node_counter, layer=layer_idx)
                self.nodes_by_layer[layer_idx].append(node_counter)
                node_counter += 1

            # connect to previous ≤ max_back layers
            for nid in self.nodes_by_layer[layer_idx]:
                for back in range(1, max_back + 1):
                    if layer_idx - back < 0:
                        break
                    for parent in self.nodes_by_layer[layer_idx - back]:
                        if random.random() < edge_p:
                            self.graph.add_edge(parent, nid)

        # Enforce out‑degree ≤ R
        print(f"Enforcing maximum successor count (R) of {R}...")
        for parent in list(self.graph.nodes):
            succ = list(self.graph.successors(parent))
            if len(succ) > R:
                for child in random.sample(succ, len(succ) - R):
                    self.graph.remove_edge(parent, child)

        # Convenience: ensure every visible node has at least one parent
        hidden_layer = self.config["simulation"]["hide_layer_index"]
        print("Safeguarding against isolated visible nodes...")
        for layer_idx, ids in self.nodes_by_layer.items():
            if layer_idx <= hidden_layer:
                continue
            for nid in ids:
                if self.graph.in_degree(nid) == 0:
                    parent = random.choice(self.nodes_by_layer[layer_idx - 1])
                    self.graph.add_edge(parent, nid)

        print(
            f"Graph generation complete. Total nodes: {self.graph.number_of_nodes()}"
        )

    # -------------------------------------------------------------- #
    #  2. build emergent geometry                                    #
    # -------------------------------------------------------------- #
    def build_emergent_geometry(self) -> None:
        """
        Creates:
            self.metric – DepthMetric   (fast quasi‑metric)
            self.atlas  – ChartAtlas    (4‑D positions)
        """
        gcfg = self.config.get("geometry", {})
        chart_k = gcfg.get("chart_scale_k", 4)
        gh_tol = gcfg.get("gh_tolerance", 0.05)

        print(f"Building chart atlas with radius L_k={2**(-chart_k)} (k={chart_k})...")

        # metric first
        self.metric = DepthMetric(self)
        # then atlas (needs metric)
        self.atlas = ChartAtlas(self, self.metric, chart_k, gh_tol)

        # store legacy 3‑D positions for fall‑back plotting
        for nid in self.graph.nodes:
            self.node_positions[nid] = self.atlas.position(nid)[:3]

        print("Chart atlas construction complete.")

    # -------------------------------------------------------------- #
    #  convenience helper used by ParticleDetector & GeometryStats   #
    # -------------------------------------------------------------- #
    def get_predecessors(self, nid: int):
        return self.graph.predecessors(nid)
