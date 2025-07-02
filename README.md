Emergent‑Universe Simulation Suite
==================================

This software suite is a **computational laboratory** for exploring the
*Deterministic Causal‑Site* programme.  
Where the three accompanying papers prove that spacetime and stable,
particle‑like structures *can* emerge from a purely deterministic graph
substrate, the suite lets you **run experiments** and watch those predictions
play out.

---------------------------------------------------------------------------
1. Core architecture
---------------------------------------------------------------------------

Phase‑1  (emergent geometry)
    * causal_site.py      – creates a layered DAG (finite acyclic causal site)
    * depth_metric.py     – fast quasi‑metric  d∞(u,v)=2^‑depth(v) if  u⇝v
    * chart_atlas.py      – hop‑based local charts stitched into global 4‑vectors
                            (x, y, z, τ)  for every node

Phase‑2  (simulation & detection)
    * state_manager.py    – applies deterministic update functor **T** each tick
    * particle_detector.py– finds bounded + T‑invariant + indecomposable clusters
    * kinematics.py       – centroid, radius and 3‑velocity (finite values only)

All results stream to  

``results/static_universe.json``   (substrate + 4‑D coords)  
``results/simulation_log.jsonl``   (one JSON line per tick).

---------------------------------------------------------------------------
2. Module summary
---------------------------------------------------------------------------

file → concept → notes
---------------------------------
causal_site.py      | graph substrate *C*       | hosts metric & atlas
depth_metric.py     | d∞ quasi‑metric           | O(V + E) pre‑compute
chart_atlas.py      | 4‑D spacetime atlas       | hop BFS, MDS + Procrustes
state_manager.py    | update functor **T**      | fusion_mode ∈ injective / …
particle_detector.py| particle definition       | hash‑history + clustering
kinematics.py       | observables               | clamps NaN / ∞ → finite
export_data.py      | orchestration script      | writes both result files
visualizer.html     | 3‑D player (HUD)          | play/pause, scrub‑bar, FPS
infographic.html    | D3 dashboard              | drag‑and‑drop the .jsonl
build_scene.py      | optional glTF exporter    | substrate.glb for external apps

---------------------------------------------------------------------------
3. Installation & quick start
---------------------------------------------------------------------------

1. Create and activate a virtual‑env, then install deps

    cd <project‑folder>
    python -m venv venv
    venv\Scripts\activate          # on Windows
    pip install -r requirements.txt

   *requirements.txt* includes  
   numpy, networkx, tqdm, pyyaml, scikit‑learn, scipy, pygltflib.

2. Edit **config.yaml** to choose graph size, update rule, detector settings.

3. Run a full simulation

    python export_data.py

   Console shows a “Stitching Charts” bar, then tick progress.

4. Visualise

    python -m http.server
    # then open in your browser:
    #   http://localhost:8000/visualizer.html     (3‑D world‑lines)
    #   http://localhost:8000/infographic.html    (stats dashboard)

   Drag *results/simulation_log.jsonl* onto the dashboard page.

---------------------------------------------------------------------------
4. Understanding the output
---------------------------------------------------------------------------

• **Visualizer**  
  Grey graph = static causal network.  
  Coloured spheres = particles, sized by radius, coloured by period.  
  Use HUD buttons to pause, scrub, hide substrate, hide world‑lines.

• **Dashboard KPIs**  
  – *Total particles* number of distinct clusters detected  
  – *Unique periods* diversity of oscillation periods  
  – *Longest lifetime* ticks between first & last appearance  
  – *Total ticks* length of simulation

• **Charts**  
  – Doughnut: count per period  
  – Bar: average lifetime per period  
  – Scatter: speed (c) vs. cluster size  
  – Histogram: lifetime distribution

Empty scatter plot or lifetimes ≤ 3 indicates a “cold” universe (e.g.
`max_out_degree_R = 2`, short run).  
Increase `R` to 3 or 4 and run for ≥ 6000 ticks to see long‑lived, moving
particles.

---------------------------------------------------------------------------
Happy experimenting!
---------------------------------------------------------------------------


