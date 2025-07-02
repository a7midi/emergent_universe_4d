import numpy as np
import math


def finite(x, default=0.0):
    """Return x if finite else default."""
    return x if math.isfinite(x) else default


def calculate_kinematics(particle, atlas, metric, last=None):
    """Return dict with centroid (4‑vec), radius, 3‑velocity – all finite."""
    if not particle.nodes:
        return {"centroid": [0, 0, 0, 0], "radius": 0.0, "velocity": [0, 0, 0]}

    nodes = list(particle.nodes)
    idxs = [atlas.id_map[n] for n in nodes if n in atlas.id_map]
    if not idxs:
        return {"centroid": [0, 0, 0, 0], "radius": 0.0, "velocity": [0, 0, 0]}

    P = atlas.global_coords[idxs]
    P = P[~np.isnan(P).any(axis=1)]
    if len(P) == 0:
        return {"centroid": [0, 0, 0, 0], "radius": 0.0, "velocity": [0, 0, 0]}

    centroid = P.mean(axis=0)

    # radius: max finite symmetric radius inside the cluster
    radii = [
        metric.get_symmetric_radius(nodes[np.argmin(((P[:, :3] - centroid[:3]) ** 2).sum(1))], n)
        for n in particle.nodes
    ]
    radii = [r for r in radii if math.isfinite(r)]
    radius = max(radii) if radii else 0.0

    vel = [0.0, 0.0, 0.0]
    if last and last["centroid"][3] != centroid[3]:
        dt = centroid[3] - last["centroid"][3]
        if dt:
            dx = centroid[:3] - np.array(last["centroid"][:3])
            vel = [finite(v) for v in (dx / dt)]

    return {
        "centroid": [finite(x) for x in centroid.tolist()],
        "radius": float(finite(radius)),
        "velocity": vel,
    }
