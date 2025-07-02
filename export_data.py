#!/usr/bin/env python3
from __future__ import annotations
import json, math, pathlib, warnings
from typing import Any

import numpy as np
from tqdm import tqdm

from src.config import CONFIG
from src.causal_site import CausalSite
from src.state_manager import StateManager
from src.particle_detector import ParticleDetector

OUT = pathlib.Path("results")
OUT.mkdir(exist_ok=True)

# ───────── helpers ────────────────────────────────────────────────────────
def to_py(obj: Any) -> Any:
    """Deep‑convert NumPy scalars and drop non‑finite floats."""
    if isinstance(obj, dict):
        return {k: to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_py(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if not math.isfinite(obj) else float(obj)
    return obj


def serialisable_particle(p) -> dict:
    d = to_py(p.__dict__)
    d["nodes"] = [int(n) for n in p.nodes]
    return d


# ───────── static substrate ───────────────────────────────────────────────
def dump_static(site: CausalSite) -> None:
    data = {
        "nodes": {
            str(n): {
                "layer": int(meta["layer"]),
                "position": list(map(float, site.atlas.position(n))),
            }
            for n, meta in site.graph.nodes(data=True)
        },
        "edges": [[int(u), int(v)] for u, v in site.graph.edges()],
    }
    (OUT / "static_universe.json").write_text(json.dumps(data, indent=2))
    print("✓ static_universe.json written")


# ───────── dynamic log ────────────────────────────────────────────────────
def dump_log(site: CausalSite, sm: StateManager, det: ParticleDetector) -> None:
    total     = CONFIG["simulation"]["total_ticks"]
    interval  = max(1, CONFIG["simulation"].get("log_interval", 1))
    verbose   = CONFIG["simulation"].get("verbose", False)

    path = OUT / "simulation_log.jsonl"
    with path.open("w") as fp, tqdm(range(total), desc="ticks") as bar:
        particles_in_window = 0
        for t in bar:
            sm.tick()
            live = det.detect(sm.get_current_state(), t)

            # ---------------- write every tick ---------------------------
            frame = {
                "tick": int(t),
                "particles": [serialisable_particle(p) for p in live.values()],
            }
            fp.write(json.dumps(frame) + "\n")

            # ---------------- status preview -----------------------------
            particles_in_window += len(live)
            if verbose and (t + 1) % interval == 0:
                tqdm.write(
                    f"[t={t:>6}] "
                    f"detected in last {interval:>3} ticks: {particles_in_window}"
                )
                particles_in_window = 0  # reset counter for next window

    print("✓ simulation_log.jsonl written  (contains full 0…{0} range)".format(total - 1))


# ───────── main ───────────────────────────────────────────────────────────
def main() -> None:
    warnings.filterwarnings(
        "ignore",
        message="The default value of `n_init`",
        category=FutureWarning,
    )

    site = CausalSite(CONFIG)
    site.generate_graph()
    site.build_emergent_geometry()
    dump_static(site)

    sm  = StateManager(site, CONFIG)
    det = ParticleDetector(site, sm, CONFIG)
    dump_log(site, sm, det)


if __name__ == "__main__":
    main()
