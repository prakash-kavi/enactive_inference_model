"""
plot_simulation.py

Generate plots from simulation data (using recalibrated attractors):
- Attractor landscapes (2D and 3D)
- Diagnostic plots (hierarchy, free energy, dwell times)
All plots saved to plots/simulation/
"""

import sys
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from viz.plot_diagnostics import (
    plot_hierarchy, plot_free_energy_bar,
    plot_dwell_times, plot_cognitive_hierarchy,
    get_tail_stats
)
from viz.plot_attractors import (
    plot_attractor_2d,
    plot_attractor_landscape_3d,
    load_cohort_series,
)
from viz.plot_fe_policy import generate_plots as generate_fe_policy_plots
from viz.plotting_utils import load_json_data_from

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

    sim_data_dir = Path("data") / "simulation"

    nov_ts, nov_ai, nov_stats = load_json_data_from(sim_data_dir, "novice")
    exp_ts, exp_ai, exp_stats = load_json_data_from(sim_data_dir, "expert")

    # Slice to tail and align
    nov_tail = get_tail_stats(nov_stats)
    exp_tail = get_tail_stats(exp_stats)

    # Inject experience level for plotting titles
    nov_tail['experience_level'] = 'novice'
    exp_tail['experience_level'] = 'expert'

    # ===== DIAGNOSTIC PLOTS =====
    logging.info("\nGenerating Diagnostic Plots...")

    plot_cognitive_hierarchy(
        nov_tail, exp_tail,
        SIMULATION_PLOT_DIR / "FigS1C_Hierarchy_TimeSeries.png"
    )
    logging.info("OK Cognitive Hierarchy")

    plot_free_energy_bar(
        nov_stats, exp_stats,
        SIMULATION_PLOT_DIR / "Fig3B_FreeEnergy.png"
    )
    logging.info("OK Free Energy Bar")

    plot_dwell_times(
        nov_stats, exp_stats,
        SIMULATION_PLOT_DIR / "Fig3C_DwellTime.png"
    )
    logging.info("OK Dwell Times")

    plot_hierarchy(
        nov_tail,
        SIMULATION_PLOT_DIR / "Fig4A_Hierarchy_Novice.png"
    )
    logging.info("OK Hierarchy Novice")

    plot_hierarchy(
        exp_tail,
        SIMULATION_PLOT_DIR / "Fig4B_Hierarchy_Expert.png"
    )
    logging.info("OK Hierarchy Expert")

    # ===== ATTRACTOR PLOTS =====
    logging.info("\nGenerating Attractor Plots...")

    novice_series = load_cohort_series("novice", tail=500, data_dir=sim_data_dir)
    expert_series = load_cohort_series("expert", tail=500, data_dir=sim_data_dir)

    plot_attractor_2d(
        novice_series, expert_series,
        save_path=SIMULATION_PLOT_DIR / "Fig5A_Attractor2D.png"
    )
    logging.info("OK 2D Attractor")

    plot_attractor_landscape_3d(
        novice_series, expert_series,
        save_path=SIMULATION_PLOT_DIR / "Fig5B_Attractor3D.png"
    )
    logging.info("OK 3D Attractor Landscape")

    # ===== FE + POLICY DIAGNOSTICS =====
    logging.info("\nGenerating FE/Policy Plots...")
    generate_fe_policy_plots(
        data_dir=sim_data_dir,
        output_dir=SIMULATION_PLOT_DIR,
    )
    logging.info("OK FE/Policy Diagnostics")

    logging.info("\n" + "=" * 60)
    logging.info("All simulation plots generated in %s", SIMULATION_PLOT_DIR)
    logging.info("=" * 60)


if __name__ == "__main__":
    generate_simulation_plots()
