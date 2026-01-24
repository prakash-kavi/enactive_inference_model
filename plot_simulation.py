"""
plot_simulation.py

Generate plots from simulation data (using recalibrated attractors):
- Attractor landscapes (2D and 3D)
- Diagnostic plots (hierarchy, free energy, radar, dwell times)
All plots saved to plots/simulation/
"""

import os
import json
import logging
import numpy as np
from pathlib import Path

# Import plotting modules
from viz.plot_diagnostics import (
    plot_hierarchy, plot_free_energy_bar, plot_network_radar, 
    plot_dwell_times, plot_cognitive_hierarchy,
    load_json_data, get_tail_stats
)
from viz.plot_attractors import (
    generate_plots as generate_attractor_plots,
    _load_cohort_series
)
from viz.plotting_utils import PLOT_DIR, save_figure

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

SIMULATION_PLOT_DIR = Path("plots") / "simulation"

def generate_simulation_plots():
    """Generate all plots from simulation data."""
    SIMULATION_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    
    logging.info("=" * 60)
    logging.info("GENERATING SIMULATION PLOTS")
    logging.info("=" * 60)
    logging.info("Reading data from data/simulation/")
    logging.info("Saving plots to %s", SIMULATION_PLOT_DIR)
    logging.info("=" * 60)
    
    # Load Data from simulation directory
    sim_data_dir = Path("data") / "simulation"
    
    def load_sim_json_data(cohort):
        """Load JSON data from simulation directory."""
        ts_path = sim_data_dir / f"thoughtseed_params_{cohort}.json"
        ai_path = sim_data_dir / f"active_inference_params_{cohort}.json"
        stats_path = sim_data_dir / f"transition_stats_{cohort}.json"
        
        ts_data = {}
        if ts_path.exists():
            with open(ts_path, 'r') as f:
                ts_data = json.load(f)
        
        ai_data = {}
        if ai_path.exists():
            with open(ai_path, 'r') as f:
                ai_data = json.load(f)
        
        stats_data = {}
        if stats_path.exists():
            with open(stats_path, 'r') as f:
                stats_data = json.load(f)
        
        # Merge time_series from thoughtseed_params if needed
        if "state_history" not in stats_data and "time_series" in ts_data:
            logging.info("Merging time_series from thoughtseed_params into stats for %s", cohort)
            stats_data.update(ts_data["time_series"])
        
        return ts_data, ai_data, stats_data
    
    nov_ts, nov_ai, nov_stats = load_sim_json_data("novice")
    exp_ts, exp_ai, exp_stats = load_sim_json_data("expert")
    
    # Slice to Tail
    nov_tail = get_tail_stats(nov_stats)
    exp_tail = get_tail_stats(exp_stats)
    
    # Inject experience level for plotting titles
    nov_tail['experience_level'] = 'novice'
    exp_tail['experience_level'] = 'expert'
    
    # ===== DIAGNOSTIC PLOTS =====
    logging.info("\nGenerating Diagnostic Plots...")
    
    # Panel A: Cognitive Hierarchy (4-Level Stack)
    plot_cognitive_hierarchy(
        nov_tail, exp_tail, 
        SIMULATION_PLOT_DIR / "FigS1C_Hierarchy_TimeSeries.png"
    )
    logging.info("✓ Cognitive Hierarchy")
    
    # Panel B: Free Energy Bar
    plot_free_energy_bar(
        nov_stats, exp_stats, 
        SIMULATION_PLOT_DIR / "Fig3B_FreeEnergy.png"
    )
    logging.info("✓ Free Energy Bar")
    
    # Panel D: Dwell Times
    plot_dwell_times(
        nov_stats, exp_stats, 
        SIMULATION_PLOT_DIR / "Fig3C_DwellTime.png"
    )
    logging.info("✓ Dwell Times")
    
    # Hierarchy Plots (Individual)
    plot_hierarchy(
        nov_tail, 
        SIMULATION_PLOT_DIR / "Fig4A_Hierarchy_Novice.png"
    )
    logging.info("✓ Hierarchy Novice")
    
    plot_hierarchy(
        exp_tail, 
        SIMULATION_PLOT_DIR / "Fig4B_Hierarchy_Expert.png"
    )
    logging.info("✓ Hierarchy Expert")
    
    # ===== ATTRACTOR PLOTS =====
    logging.info("\nGenerating Attractor Plots...")
    
    # Load data for attractor plots from simulation directory
    # Create custom loader that reads from simulation directory
    def load_sim_cohort_series(cohort, tail=500):
        """Load cohort series from simulation directory."""
        ts_data, _, stats_data = load_sim_json_data(cohort)
        tail_stats = get_tail_stats(stats_data, tail=tail)
        
        activations = np.asarray(tail_stats.get("activations_history", []), dtype=float)
        free_energy = np.asarray(tail_stats.get("free_energy_history", []), dtype=float)
        states = tail_stats.get("state_history", [])
        
        if activations.ndim != 2 or activations.shape[0] == 0:
            raise ValueError(f"Activation history missing or malformed for {cohort}")
        
        return {
            "cohort": cohort,
            "activations": activations,
            "free_energy": free_energy,
            "states": states,
            "activation_means": ts_data.get("activation_means_by_state", {}),
        }
    
    novice_series = load_sim_cohort_series("novice", tail=500)
    expert_series = load_sim_cohort_series("expert", tail=500)
    
    # Figure 5A: 2D Attractor
    from viz.plot_attractors import plot_attractor_2d
    plot_attractor_2d(
        novice_series, expert_series,
        save_path=SIMULATION_PLOT_DIR / "Fig5A_Attractor2D.png"
    )
    logging.info("✓ 2D Attractor")
    
    # Figure 5B: 3D Landscape
    from viz.plot_attractors import plot_attractor_landscape_3d
    plot_attractor_landscape_3d(
        novice_series, expert_series,
        save_path=SIMULATION_PLOT_DIR / "Fig5B_Attractor3D.png"
    )
    logging.info("✓ 3D Attractor Landscape")
    
    logging.info("\n" + "=" * 60)
    logging.info("All simulation plots generated in %s", SIMULATION_PLOT_DIR)
    logging.info("=" * 60)

if __name__ == "__main__":
    generate_simulation_plots()
