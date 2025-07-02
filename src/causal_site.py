# src/causal_site.py
# --- UPGRADED FOR EMERGENT GEOMETRY ---

import networkx as nx
import numpy as np
# --- UPGRADE: Import new geometry modules
from src.depth_metric import DepthMetric
from src.chart_atlas import ChartAtlas

class CausalSite:
    """
    Represents the finite, acyclic causal site of the universe and serves as
    the container for emergent geometric structures.
    """

    def __init__(self, config):
        """Initializes the CausalSite."""
        self.config = config
        self.graph = nx.DiGraph()
        self.nodes_by_layer = {}
        # --- UPGRADE: node_positions are now derived, not imposed.
        self.node_positions = {} 
        # --- UPGRADE: nodes_by_cell is deprecated.
        
        # --- UPGRADE: Geometry engine attributes
        self.metric = None
        self.atlas = None

    def generate_graph(self):
        """
        Procedurally generates the layered causal graph, correctly enforcing the
        out-degree (successor) bound and other theoretical constraints.
        """
        print("Generating causal site graph...")
        node_counter = 0
        
        cs_config = self.config.get('causal_site', {})
        layers = cs_config.get('layers', 50)
        avg_nodes = cs_config.get('avg_nodes_per_layer', 40)
        edge_prob = cs_config.get('edge_probability', 0.1)
        max_lookback = cs_config.get('max_lookback_layers', 1)
        
        self.num_layers = layers
        rng = np.random.default_rng(self.config['simulation']['seed'])
        
        for layer_index in range(self.num_layers):
            num_nodes_in_layer = rng.poisson(avg_nodes) if avg_nodes > 0 else 1
            if num_nodes_in_layer == 0: num_nodes_in_layer = 1
            
            self.nodes_by_layer[layer_index] = []

            for _ in range(num_nodes_in_layer):
                node_id = node_counter
                # --- UPGRADE: Remove ad-hoc 3D position assignment.
                # Positions will be derived from the metric.
                self.graph.add_node(node_id, layer=layer_index)
                self.nodes_by_layer[layer_index].append(node_id)
                node_counter += 1

                if layer_index > 0:
                    start_layer = max(0, layer_index - max_lookback) if max_lookback > 0 else layer_index - 1
                    if start_layer >= layer_index: continue

                    for lookback_idx in range(start_layer, layer_index):
                        previous_layer_nodes = self.nodes_by_layer.get(lookback_idx, [])
                        for potential_parent in previous_layer_nodes:
                            distance = layer_index - lookback_idx
                            if rng.random() < edge_prob / (distance**0.5):
                                self.graph.add_edge(potential_parent, node_id)
        
        max_r = self.config.get('tags', {}).get('max_out_degree_R', 2)
        print(f"Enforcing maximum successor count (R) of {max_r}...")
        for layer_index in range(self.num_layers - 1):
            for node_id in self.nodes_by_layer.get(layer_index, []):
                successors = list(self.graph.successors(node_id))
                if len(successors) > max_r:
                    children_to_prune = rng.choice(successors, size=len(successors) - max_r, replace=False)
                    for child_to_prune in children_to_prune:
                        self.graph.remove_edge(node_id, child_to_prune)

        print("Safeguarding against isolated visible nodes...")
        sim_config = self.config.get('simulation', {})
        hide_layer_index = sim_config.get('hide_layer_index', -1)
        for layer_index in range(1, self.num_layers):
            if layer_index == hide_layer_index: continue
            for node_id in self.nodes_by_layer.get(layer_index, []):
                if self.graph.in_degree(node_id) == 0:
                    start_layer = max(0, layer_index - max_lookback) if max_lookback > 0 else layer_index - 1
                    potential_parent_layer_idx = rng.integers(start_layer, layer_index) if start_layer < layer_index else 0
                    potential_parents = self.nodes_by_layer.get(potential_parent_layer_idx, [])
                    if potential_parents:
                        self.graph.add_edge(rng.choice(potential_parents), node_id)
        
        print(f"Graph generation complete. Total nodes: {self.graph.number_of_nodes()}")

    def build_emergent_geometry(self):
        """
        Builds the emergent metric and coordinate atlas for the site.
        """
        # --- UPGRADE: This method orchestrates the new geometry pipeline.
        if self.graph.number_of_nodes() == 0: return

        # 1. Calculate the emergent metric from causal structure.
        self.metric = DepthMetric(self)
        
        # 2. Build the 4D coordinate atlas from the metric.
        self.atlas = ChartAtlas(self, self.metric, self.config)
        self.node_positions = self.atlas.build_atlas()

    def get_predecessors(self, node_id):
        """Returns the immediate causal predecessors of a given site."""
        return list(self.graph.predecessors(node_id))