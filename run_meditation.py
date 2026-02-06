"""Main entry point for lean meditation model training and simulation.

Usage:
    python run_meditation.py run      # Train expert + novice and generate plots
    python run_meditation.py plot     # Generate plots from saved results
    python run_meditation.py run --timesteps 5000  # Custom timesteps
"""

import argparse
import os
from pathlib import Path

from model.train import train_meditation
from viz.analysis import (
    plot_belief_about_belief,
    print_summary
)

# Import data processing utilities
from utils.analysis_utils import (
    get_tail_window,
    compute_network_profiles,
    compute_thoughtseed_means,
    compute_tail_statistics,
    TAIL_STEPS
)
from utils.config import STATES, NETWORKS, THOUGHTSEEDS

# Import publication plotting functions
from viz.convergence import plot_convergence
from viz.radar_plot import plot_comparison
from viz.hierarchy import plot_hierarchy
from viz.attractors import plot_attractor_2d, plot_attractor_3d
from viz.diagnostics import plot_fe_and_dwell, plot_transitions

# Fixed configuration for reproducibility
SEED = 42

# Get paths relative to current directory
CURRENT_DIR = Path(__file__).parent
OUTPUT_DIR = CURRENT_DIR / 'data'
PLOT_DIR = CURRENT_DIR / 'plots'

def run_training_and_simulation(timesteps: int):
    """Train both expert and novice, then run simulation for both."""
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Train expert
    print("\n" + "="*70)
    print("PHASE 1: TRAINING EXPERT PHENOTYPE")
    print("="*70)
    expert_results = train_meditation(
        experience_level='expert',
        timesteps=timesteps,
        seed=SEED,
        save_results=True,
        output_dir=str(OUTPUT_DIR)
    )
    print_summary(expert_results)
    
    # Train novice
    print("\n" + "="*70)
    print("PHASE 2: TRAINING NOVICE PHENOTYPE")
    print("="*70)
    novice_results = train_meditation(
        experience_level='novice',
        timesteps=timesteps,
        seed=SEED,
        save_results=True,
        output_dir=str(OUTPUT_DIR)
    )
    print_summary(novice_results)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Results saved to: {OUTPUT_DIR}/")
    print(f"  - training_results_expert_seed{SEED}.json")
    print(f"  - training_results_novice_seed{SEED}.json")
    print()
    
    return expert_results, novice_results


def generate_plots(expert_results, novice_results):
    """Generate all comparison and individual plots."""
    
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("GENERATING PLOTS")
    print("="*70)
    
    # ===== Prepare data for plotting =====
    print("  • Processing tail window data...")
    
    # Extract tail windows for simulation plots
    expert_tail = get_tail_window(expert_results, TAIL_STEPS)
    novice_tail = get_tail_window(novice_results, TAIL_STEPS)
    
    # Compute aggregated statistics from tail window
    expert_network_profiles = compute_network_profiles(expert_results, STATES, NETWORKS, TAIL_STEPS)
    novice_network_profiles = compute_network_profiles(novice_results, STATES, NETWORKS, TAIL_STEPS)
    
    expert_ts_means = compute_thoughtseed_means(expert_results, STATES, THOUGHTSEEDS, TAIL_STEPS)
    novice_ts_means = compute_thoughtseed_means(novice_results, STATES, THOUGHTSEEDS, TAIL_STEPS)
    
    expert_tail_stats = compute_tail_statistics(expert_results, STATES, TAIL_STEPS)
    novice_tail_stats = compute_tail_statistics(novice_results, STATES, TAIL_STEPS)
    
    # Add computed stats to tail data for plotting
    expert_tail['network_profiles_mean'] = expert_network_profiles
    novice_tail['network_profiles_mean'] = novice_network_profiles
    expert_tail['thoughtseed_means_per_state'] = expert_ts_means
    novice_tail['thoughtseed_means_per_state'] = novice_ts_means
    expert_tail.update(expert_tail_stats)
    novice_tail.update(novice_tail_stats)
    
    print(f"    Tail window: {TAIL_STEPS} steps (converged behavior)")
    print(f"    Full trajectory: {len(expert_results['state_history'])} steps")
    
    # ===== Publication-quality plots =====
    print("\n--- Publication Figures ---")
    
    # FigS1: Convergence plots (full trajectories)
    print("  • FigS1_Convergence_Expert.png")
    plot_convergence(expert_results, str(PLOT_DIR / "FigS1_Convergence_Expert.png"))
    
    print("  • FigS1_Convergence_Novice.png")
    plot_convergence(novice_results, str(PLOT_DIR / "FigS1_Convergence_Novice.png"))
    
    # Fig3: Comparison plot (radar + dwell + transitions from tail)
    print("  • Fig3A_Network_Radar.png")
    plot_comparison(novice_tail, expert_tail, str(PLOT_DIR / "Fig3A_Network_Radar.png"))
    
    # Fig4: Hierarchy plots (3-level dynamics from tail)
    print("  • Fig4A_Hierarchy_Novice.png")
    plot_hierarchy(novice_tail, str(PLOT_DIR / "Fig4A_Hierarchy_Novice.png"), "Novice")
    
    print("  • Fig4B_Hierarchy_Expert.png")
    plot_hierarchy(expert_tail, str(PLOT_DIR / "Fig4B_Hierarchy_Expert.png"), "Expert")
    
    # Fig5: Attractor plots (2D projection + 3D landscape from tail)
    print("  • Fig5A_Attractor2D.png")
    plot_attractor_2d(novice_tail, expert_tail, str(PLOT_DIR / "Fig5A_Attractor2D.png"))
    
    print("  • Fig5B_Attractor3D.png")
    plot_attractor_3d(novice_tail, expert_tail, str(PLOT_DIR / "Fig5B_Attractor3D.png"))
    
    # ===== Diagnostic plots =====
    print("\n--- Diagnostic Plots ---")
    print("  • Fig3B_FE_and_Dwell.png")
    plot_fe_and_dwell(novice_tail, expert_tail, str(PLOT_DIR / "Fig3B_FE_and_Dwell.png"))
    
    print("  • Fig3C_Transitions.png")
    plot_transitions(novice_tail, expert_tail, str(PLOT_DIR / "Fig3C_Transitions.png"))
    
    print("  • Fig3D.png")
    plot_belief_about_belief(novice_results, expert_results, str(PLOT_DIR / "Fig3D.png"))
    
    print("\n" + "="*70)
    print("PLOTTING COMPLETE")
    print("="*70)
    print(f"All plots: {PLOT_DIR}/")
    print(f"  - FigS1_Convergence_Expert.png")
    print(f"  - FigS1_Convergence_Novice.png")
    print(f"  - Fig3A_Network_Radar.png")
    print(f"  - Fig3B_FE_and_Dwell.png")
    print(f"  - Fig3C_Transitions.png")
    print(f"  - Fig3D.png")
    print(f"  - Fig4A_Hierarchy_Novice.png")
    print(f"  - Fig4B_Hierarchy_Expert.png")
    print(f"  - Fig5A_Attractor2D.png")
    print(f"  - Fig5B_Attractor3D.png")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Lean meditation model: training and visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  run   - Train both expert and novice, then generate plots
  plot  - Generate plots from previously saved results

Examples:
  # Run full pipeline (training + plots) with seed 42
  python run_meditation.py run
  
  # Custom timesteps (seed 42, outputs to data/ and plots/)
  python run_meditation.py run --timesteps 5000
  
  # Regenerate plots from saved results
  python run_meditation.py plot

Configuration (fixed for reproducibility):
  Seed: 42
  Data dir: data/
  Plot dir: plots/
        """
    )
    
    parser.add_argument('command', type=str, choices=['run', 'plot'],
                       help='Command: "run" for training+simulation, "plot" for visualization only')
    parser.add_argument('--timesteps', type=int, default=10000,
                       help='Training timesteps (default: 10000)')
    
    args = parser.parse_args()
    
    if args.command == 'run':
        # Run training and simulation
        expert_results, novice_results = run_training_and_simulation(
            timesteps=args.timesteps
        )
        
        # Generate plots
        generate_plots(expert_results, novice_results)
        
    elif args.command == 'plot':
        # Load saved results and generate plots
        import json
        
        expert_path = OUTPUT_DIR / f"training_results_expert_seed{SEED}.json"
        novice_path = OUTPUT_DIR / f"training_results_novice_seed{SEED}.json"
        
        if not expert_path.exists() or not novice_path.exists():
            print(f"ERROR: Could not find saved results:")
            print(f"  Looking for: {expert_path}")
            print(f"  Looking for: {novice_path}")
            print(f"\nRun 'python run_meditation.py run' first to generate results.")
            return
        
        print(f"\nLoading results from {OUTPUT_DIR}/...")
        with open(expert_path, 'r') as f:
            expert_results = json.load(f)
        with open(novice_path, 'r') as f:
            novice_results = json.load(f)
        
        generate_plots(expert_results, novice_results)


if __name__ == '__main__':
    main()
