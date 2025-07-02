#!/usr/bin/env python3
"""
export_data.py
──────────────
▸ Generates two artefacts in the ./results folder

    1. static_universe.json   – graph topology + 4‑D coordinates
    2. simulation_log.jsonl   – per‑tick particle list

The script assumes you have already installed requirements.txt plus tqdm.
"""

from __future__ import annotations
import json, os, pathlib
from typing import Dict, Any

from tqdm import tqdm

from src.config import CONFIG
from src.causal_site import CausalSite
from src.state_manager import StateManager
from src.particle_detector import ParticleDetector


RESULTS = pathlib.Path("results")
RESULTS.mkdir(exist_ok=True)


def dump_static_universe(site: CausalSite) -> None:
    """Write nodes + edges with 4‑D coordinates (x,y,z,τ)."""
    out: Dict[str, Any] = {"nodes": {}, "edges": list(site.graph.edges())}

    for nid, data in site.graph.nodes(data=True):
        if nid in site.node_positions:
            # --- FIX: Get position from site.node_positions, not site.atlas ---
            x, y, z, t = site.node_positions[nid].tolist()
            out["nodes"][str(nid)] = {
                "id": nid, # Also include ID for robust parsing in JS
                "layer": data["layer"],
                "position": [x, y, z, t],
            }

    (RESULTS / "static_universe.json").write_text(
        json.dumps(out, indent=2)
    )
    print("✓ static_universe.json written")


def dump_dynamic_log(site: CausalSite, sm: StateManager, det: ParticleDetector) -> None:
    """Run the simulation and emit one JSON line per tick."""
    total = CONFIG["simulation"]["total_ticks"]
    with (RESULTS / "simulation_log.jsonl").open("w") as fp:
        for tick in tqdm(range(total), desc="ticks"):
            sm.tick()
            parts = det.detect(sm.get_current_state(), tick)
            frame = {
                "tick": tick,
                "particles": [p.__dict__ for p in parts.values()],
            }
            fp.write(json.dumps(frame) + "\n")
    print("✓ simulation_log.jsonl written")


def main() -> None:
    # ------------------------------------------------------------------ #
    #  Build causal site and geometry                                    #
    # ------------------------------------------------------------------ #
    site = CausalSite(CONFIG)
    site.generate_graph()
    site.build_emergent_geometry()      # ← builds site.atlas & site.metric

    # ------------------------------------------------------------------ #
    #  Dump static substrate                                             #
    # ------------------------------------------------------------------ #
    dump_static_universe(site)

    # ------------------------------------------------------------------ #
    #  Simulation                                                        #
    # ------------------------------------------------------------------ #
    state_mgr = StateManager(site, CONFIG)
    detector = ParticleDetector(site, state_mgr, CONFIG)

    dump_dynamic_log(site, state_mgr, detector)


if __name__ == "__main__":
    main()