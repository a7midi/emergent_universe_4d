"""
state_manager.py

VERSION 43 (Theoretically Sound): This version implements all audit fixes.
- FIX: InjectiveFusionTable is pre-seeded to guarantee immediate surjectivity.
- FIX: Predecessor lists are sorted before fusion to ensure platform-independent determinism.
- Retains drifting clocks, causal consistency, and memory density calculations.
"""

import numpy as np
from math import gcd
from src.causal_site import CausalSite

class InjectiveFusionTable:
    """The 'injective' fusion mode: complex, history-dependent, and surjective."""
    def __init__(self, q, rng):
        self.q = q
        self.mapping = {}
        
        # This counter is used only AFTER the pre-seeding phase.
        self.next_tag_internal = 0
        
        # --- FIX: Pre-seed table to guarantee surjectivity from the start ---
        # Create a shuffled list of all possible output tags (0 to q-1).
        self.preseed_values = rng.permutation(q)
        self.preseed_idx = 0

    def fuse(self, pred_tags_tuple: tuple, receiver_id: int) -> int:
        """
        Creates a unique mapping based on the ordered tags and the receiving node.
        """
        # Key is order-sensitive and site-specific.
        key = pred_tags_tuple + (receiver_id,)

        if key not in self.mapping:
            # --- AMENDED: Use pre-seeded values first ---
            if self.preseed_idx < self.q:
                # For the first q unique encounters, assign a unique pre-shuffled tag.
                # This ensures the fusion map is surjective from the very beginning.
                self.mapping[key] = self.preseed_values[self.preseed_idx]
                self.preseed_idx += 1
            else:
                # After the pre-seed pool is exhausted, use the wrapping counter.
                self.mapping[key] = self.next_tag_internal
                self.next_tag_internal = (self.next_tag_internal + 1) % self.q
        
        return self.mapping[key]

class StateManager:
    """
    Manages the state of all sites and evolves them according to a selected
    deterministic tag-fusion update rule.
    """
    def __init__(self, causal_site: CausalSite, config: dict):
        if not config:
            raise ValueError("Configuration could not be loaded. Aborting.")
        
        self.causal_site = causal_site
        self.config = config
        
        self.num_nodes = self.causal_site.graph.number_of_nodes()
        self.q = self.config['tags']['alphabet_size_q']
        self.state = np.zeros(self.num_nodes, dtype=int)
        
        sim_config = self.config.get('simulation', {})
        self.rng = np.random.default_rng(sim_config.get("seed", 42))

        self.fusion_mode = self.config['tags'].get('fusion_mode', 'injective')
        self.fusion_table = None
        if self.fusion_mode == 'injective':
            # --- FIX: Pass the RNG to the fusion table for pre-seeding ---
            self.fusion_table = InjectiveFusionTable(self.q, self.rng)
        
        self.tick_counter = 0
        self.hidden_nodes = set()
        
        hide_layer_index = sim_config.get('hide_layer_index')

        if hide_layer_index is not None and hide_layer_index >= 0:
            self.hidden_nodes = set(self.causal_site.nodes_by_layer.get(hide_layer_index, []))
        
        self.hidden_params = {}
        if self.hidden_nodes:
            # Re-initialize RNG for this specific part to ensure consistency if needed
            hidden_rng = np.random.default_rng(sim_config.get("seed", 42))
            for n in self.hidden_nodes:
                mult = hidden_rng.integers(2, self.q)
                while gcd(mult, self.q) != 1:
                    mult = hidden_rng.integers(2, self.q)
                add = hidden_rng.integers(0, self.q)
                self.hidden_params[n] = (int(mult), int(add))
        
        detector_config = self.config.get('detector', {})
        self.mem_window = detector_config.get('memory_window', 1024)
        self.memory_density = np.zeros(self.num_nodes, dtype=np.uint16)

        self.initialize_state()

    def initialize_state(self):
        # Use the instance's RNG for initialization
        self.state = self.rng.integers(0, self.q, size=self.num_nodes)

    def _fusion(self, predecessor_tags: tuple, node_id: int):
        if not predecessor_tags:
            return None
        if self.fusion_mode == 'injective':
            return self.fusion_table.fuse(predecessor_tags, node_id)
        elif self.fusion_mode == 'sum_mod_q':
            return sum(predecessor_tags) % self.q
        elif self.fusion_mode == 'quadratic':
            s_mod = sum(predecessor_tags) % self.q
            return (s_mod * s_mod) % self.q
        raise ValueError(f"Unknown fusion_mode: '{self.fusion_mode}' in config.yaml")

    def tick(self):
        """Applies a single, causally consistent deterministic update."""
        self.tick_counter += 1
        state_at_t = self.state
        next_state = state_at_t.copy()

        # Phase 1: Update hidden layer
        for node_id in self.hidden_nodes:
            mult, add = self.hidden_params[node_id]
            if self.tick_counter > 0 and self.tick_counter % 11 == 0:
                add = (add + node_id) % self.q
                new_mult = (mult + 1 + node_id) % self.q or 1
                while gcd(new_mult, self.q) != 1:
                    new_mult = (new_mult + 1) % self.q or 1
                mult = new_mult
                self.hidden_params[node_id] = (mult, add)
            next_state[node_id] = (mult * state_at_t[node_id] + add) % self.q

        # Phase 2: Update observable nodes
        if self.causal_site.nodes_by_layer:
            max_layer = max(self.causal_site.nodes_by_layer.keys())
            for layer_index in range(1, max_layer + 1):
                for node_id in self.causal_site.nodes_by_layer.get(layer_index, []):
                    if node_id in self.hidden_nodes:
                        continue
                    
                    # --- FIX: Enforce canonical ordering of predecessors for determinism ---
                    predecessors = tuple(sorted(self.causal_site.get_predecessors(node_id)))
                    
                    if not predecessors:
                        continue
                        
                    predecessor_tags = tuple(state_at_t[p_id] for p_id in predecessors)
                    new_tag = self._fusion(predecessor_tags, node_id)
                    
                    if new_tag is not None:
                        next_state[node_id] = new_tag
                        
        # Update memory density
        changed = next_state != state_at_t
        self.memory_density[changed] = np.minimum(
            self.memory_density[changed] + 1, self.mem_window
        )
        self.memory_density[~changed] = np.maximum(
            self.memory_density[~changed] - 1, 0
        )
        
        self.state = next_state

    def get_current_state(self):
        return self.state

    def get_memory_density(self):
        return self.memory_density
