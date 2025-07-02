#!/usr/bin/env python3
from __future__ import annotations
import json, pathlib, warnings, math
from typing import Any

from tqdm import tqdm
import numpy as np

from src.config import CONFIG
from src.causal_site import CausalSite
from src.state_manager import StateManager
from src.particle_detector import ParticleDetector

OUT = pathlib.Path("results"); OUT.mkdir(exist_ok=True)


# ───────── helpers ─────────────────────────────────────────────────────────
def to_py(obj: Any) -> Any:
    """Deep‑convert numpy scalars and kill non‑finite floats."""
    if isinstance(obj, dict):
        return {k: to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_py(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if not math.isfinite(obj) else float(obj)
    return obj


def serialisable_particle(p):
    d = to_py(p.__dict__)
    d["nodes"] = [int(n) for n in p.nodes]
    return d


# ───────── static substrate ───────────────────────────────────────────────
def dump_static(site: CausalSite):
    data = {
        "nodes": {
            str(n): {
                "layer": int(meta["layer"]),
                "position": list(map(float, site.atlas.position(n)))
            } for n, meta in site.graph.nodes(data=True)
        },
        "edges": [[int(u), int(v)] for u, v in site.graph.edges()],
    }
    (OUT / "static_universe.json").write_text(json.dumps(data, indent=2))
    print("✓ static_universe.json written")


# ───────── dynamic log ────────────────────────────────────────────────────
def dump_log(site: CausalSite, sm: StateManager, det: ParticleDetector):
    total       = CONFIG["simulation"]["total_ticks"]
    interval    = max(1, CONFIG["simulation"].get("log_interval", 1))  # safeguard
    verbose     = CONFIG["simulation"].get("verbose", False)

    path = OUT / "simulation_log.jsonl"
    with path.open("w") as fp:
        for t in tqdm(range(total), desc="ticks"):
            sm.tick()

            # detect particles every tick so lifetimes are correct
            live = det.detect(sm.get_current_state(), t)

            # -------- console logging -------------------------------------
            if verbose and t % interval == 0:
                print(f"[t={t:>6}] active: {len(live):>4}")

            # -------- write to JSONL only each `interval` ticks ------------
            if t % interval == 0 or t == total - 1:
                frame = {
                    "tick":            int(t),
                    "particles": [serialisable_particle(p) for p in live.values()],
                }
                fp.write(json.dumps(frame) + "\n")

    print(f"✓ simulation_log.jsonl written  (saved every {interval} ticks)")


# ───────── main ───────────────────────────────────────────────────────────
def main():
    warnings.filterwarnings("ignore", message="The default value of `n_init`", category=FutureWarning)

    site = CausalSite(CONFIG); site.generate_graph(); site.build_emergent_geometry()
    dump_static(site)

    sm = StateManager(site, CONFIG)
    det = ParticleDetector(site, sm, CONFIG)
    dump_log(site, sm, det)


if __name__ == "__main__":
    main()
