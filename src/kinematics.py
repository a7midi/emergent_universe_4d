import numpy as np


def calculate_kinematics(particle, atlas, metric, last=None):
    """Return centroid (4‑vec), radius, 3‑vel  in a dict."""
    if not particle.nodes:
        return {"centroid": [0, 0, 0, 0], "radius": 0.0, "velocity": [0, 0, 0]}

    node_list = list(particle.nodes)  # ← frozenset → list
    idxs = [atlas.id_map[n] for n in node_list if n in atlas.id_map]
    if not idxs:
        return {"centroid": [0, 0, 0, 0], "radius": 0.0, "velocity": [0, 0, 0]}

    P = atlas.global_coords[idxs]
    P = P[~np.isnan(P).any(axis=1)]
    if len(P) == 0:
        return {"centroid": [0, 0, 0, 0], "radius": 0.0, "velocity": [0, 0, 0]}

    centroid = P.mean(axis=0)

    nearest_id = node_list[np.argmin(((P[:, :3] - centroid[:3]) ** 2).sum(1))]
    radius = max(metric.get_symmetric_radius(nearest_id, n) for n in particle.nodes)

    vel = [0.0, 0.0, 0.0]
    if last and last["centroid"][3] != centroid[3]:
        dt = centroid[3] - last["centroid"][3]
        dx = centroid[:3] - np.array(last["centroid"][:3])
        if dt:
            vel = (dx / dt).tolist()

    return {
        "centroid": centroid.tolist(),
        "radius": float(radius),
        "velocity": vel,
    }
