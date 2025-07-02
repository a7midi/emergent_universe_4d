# main.py
# --- UPGRADED FOR EMERGENT GEOMETRY ---

import os
from tqdm import tqdm

from src.config import CONFIG
from src.causal_site import CausalSite
from src.state_manager import StateManager
from src.particle_detector import ParticleDetector
# visualization module is now optional and only used for data export
# from visualization.visualizer import Visualizer

def main():
    if CONFIG is None:
        return

    print("--- Emergent Universe Simulation (v2.0 with Emergent Spacetime) ---")
    
    # --- UPGRADE: Check for data export config instead of visualization
    export_config = CONFIG.get('export', {})
    is_exporting = export_config.get('enabled', False)
    if is_exporting:
        os.makedirs("results", exist_ok=True)

    print("1. Initializing Universe Substrate...")
    causal_site = CausalSite(CONFIG)
    causal_site.generate_graph()

    # --- UPGRADE: The crucial new step to derive geometry from causality.
    print("\n2. Building Emergent Spacetime Geometry...")
    causal_site.build_emergent_geometry()

    print("\n3. Initializing Dynamics and Detectors...")
    state_manager = StateManager(causal_site, CONFIG)
    particle_detector = ParticleDetector(causal_site, state_manager, CONFIG)
    
    print("\nInitialization Complete.\n")
    
    # --- UPGRADE: Logic is now focused on simulation and data export.
    if not is_exporting:
        print("Export is disabled in config.yaml. Simulation will run without producing output.")
        print("To generate data for the visualizer, set export.enabled to true.")
    
    print("4. Starting Simulation Loop...")
    
    total_ticks = CONFIG['simulation']['total_ticks']
    log_interval = CONFIG.get('simulation', {}).get('log_interval', 100)
    
    progress_bar = tqdm(range(total_ticks), desc="Simulating", mininterval=0.5)
    
    for tick in progress_bar:
        state_manager.tick()
        
        current_state = state_manager.get_current_state()
        active_particles = particle_detector.detect(current_state, tick)
        
        if tick > 0 and tick % log_interval == 0:
            num_looping = len(particle_detector.looping_nodes_last_tick)
            progress_bar.set_postfix({
                'looping_nodes': num_looping,
                'particles': len(active_particles)
            })

    print("\nSimulation Loop Complete.\n")

    print("5. Final Simulation Report...")
    final_particles = particle_detector.active
    archived_particles = particle_detector.archive
    print(f"Detected {len(final_particles)} active particle(s) at the end.")
    print(f"Archived {len(archived_particles)} total historical particle(s).")

    if is_exporting:
        print("\nNote: To visualize the results, run the export script:")
        print("python export_data.py")
    
    print("\n--- Simulation Finished ---")

if __name__ == '__main__':
    main()