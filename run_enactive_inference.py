"""Main entry point for lean meditation model learning and simulation.

Usage:
    python run_enactive_inference.py run
    python run_enactive_inference.py plot
"""

import argparse
from pathlib import Path

from model.training_loop import train_meditation_model, simulate_meditation
from model.phenotype import EXPERT_PHENOTYPE, NOVICE_PHENOTYPE
from viz.analysis import print_summary
from viz.analysis_utils import (
    get_tail_window,
    compute_network_profiles,
    compute_thoughtseed_means,
    compute_tail_statistics,
    TAIL_STEPS,
)
from utils.config import STATES, NETWORKS, THOUGHTSEEDS

from viz.convergence import plot_convergence, plot_convergence_comparison
from viz.radar_plot import plot_comparison
from viz.hierarchy import plot_hierarchy, plot_hierarchy_continuous
from viz.attractors import plot_attractor_pca
from viz.diagnostics import plot_fe_and_dwell, plot_transitions

SEED = 42
TIMESTEPS = 10000

CURRENT_DIR = Path(__file__).parent
OUTPUT_DIR = CURRENT_DIR / "data"
PLOT_DIR = CURRENT_DIR / "figures"


def run_training_and_simulation(timesteps: int):
    """Train both expert and novice, then run inference-only simulation."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("PHASE 1: LEARNING EXPERT PHENOTYPE")
    print("=" * 70)
    expert_trainer, expert_learning = train_meditation_model(
        phenotype=EXPERT_PHENOTYPE,
        timesteps=timesteps,
        seed=SEED,
        save_results=True,
        output_dir=str(OUTPUT_DIR),
    )
    print_summary(expert_learning)

    print("\n" + "=" * 70)
    print("PHASE 1b: SIMULATION EXPERT PHENOTYPE")
    print("=" * 70)
    expert_results = simulate_meditation(
        trainer=expert_trainer,
        timesteps=timesteps,
        reseed_rng=True,
        run_seed=SEED,
        save_results=True,
        output_dir=str(OUTPUT_DIR),
    )
    print_summary(expert_results)

    print("\n" + "=" * 70)
    print("PHASE 2: LEARNING NOVICE PHENOTYPE")
    print("=" * 70)
    novice_trainer, novice_learning = train_meditation_model(
        phenotype=NOVICE_PHENOTYPE,
        timesteps=timesteps,
        seed=SEED,
        save_results=True,
        output_dir=str(OUTPUT_DIR),
    )
    print_summary(novice_learning)

    print("\n" + "=" * 70)
    print("PHASE 2b: SIMULATION NOVICE PHENOTYPE")
    print("=" * 70)
    novice_results = simulate_meditation(
        trainer=novice_trainer,
        timesteps=timesteps,
        reseed_rng=True,
        run_seed=SEED,
        save_results=True,
        output_dir=str(OUTPUT_DIR),
    )
    print_summary(novice_results)

    print("\n" + "=" * 70)
    print("LEARNING + SIMULATION COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {OUTPUT_DIR}/")
    print(f"  - training_results_expert_seed{SEED}.json")
    print(f"  - training_results_novice_seed{SEED}.json")
    print(f"  - simulation_results_expert_seed{SEED}.json")
    print(f"  - simulation_results_novice_seed{SEED}.json")
    print()

    return expert_learning, expert_results, novice_learning, novice_results


def generate_plots(expert_results, novice_results, expert_learning=None, novice_learning=None):
    """Generate all comparison and individual plots."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)

    print("  - Processing tail window data...")
    expert_tail = get_tail_window(expert_results, TAIL_STEPS)
    novice_tail = get_tail_window(novice_results, TAIL_STEPS)
    expert_tail["tail_start"] = max(0, len(expert_results["state_history"]) - TAIL_STEPS)
    novice_tail["tail_start"] = max(0, len(novice_results["state_history"]) - TAIL_STEPS)

    expert_network_profiles = compute_network_profiles(expert_results, STATES, NETWORKS, TAIL_STEPS)
    novice_network_profiles = compute_network_profiles(novice_results, STATES, NETWORKS, TAIL_STEPS)

    expert_ts_means = compute_thoughtseed_means(expert_results, STATES, THOUGHTSEEDS, TAIL_STEPS)
    novice_ts_means = compute_thoughtseed_means(novice_results, STATES, THOUGHTSEEDS, TAIL_STEPS)

    expert_tail_stats = compute_tail_statistics(expert_results, STATES, TAIL_STEPS)
    novice_tail_stats = compute_tail_statistics(novice_results, STATES, TAIL_STEPS)

    expert_tail["network_profiles_mean"] = expert_network_profiles
    novice_tail["network_profiles_mean"] = novice_network_profiles
    expert_tail["thoughtseed_means_per_state"] = expert_ts_means
    novice_tail["thoughtseed_means_per_state"] = novice_ts_means
    expert_tail.update(expert_tail_stats)
    novice_tail.update(novice_tail_stats)

    print(f"    Tail window: {TAIL_STEPS} steps (converged behavior)")
    print(f"    Full trajectory: {len(expert_results['state_history'])} steps")

    print("\n--- Figures ---")
    print("  - FigS1_Convergence_Expert.pdf")
    if expert_learning is not None:
        plot_convergence_comparison(
            expert_learning,
            expert_results,
            str(PLOT_DIR / "FigS1_Convergence_Expert.pdf"),
        )
    else:
        plot_convergence(expert_results, str(PLOT_DIR / "FigS1_Convergence_Expert.pdf"))

    print("  - FigS1_Convergence_Novice.pdf")
    if novice_learning is not None:
        plot_convergence_comparison(
            novice_learning,
            novice_results,
            str(PLOT_DIR / "FigS1_Convergence_Novice.pdf"),
        )
    else:
        plot_convergence(novice_results, str(PLOT_DIR / "FigS1_Convergence_Novice.pdf"))

    print("  - fig3a.pdf")
    plot_comparison(novice_tail, expert_tail, str(PLOT_DIR / "fig3a.pdf"))

    print("  - fig3b.pdf")
    plot_fe_and_dwell(novice_tail, expert_tail, str(PLOT_DIR / "fig3b.pdf"))

    print("  - fig3c.pdf")
    plot_transitions(novice_tail, expert_tail, str(PLOT_DIR / "fig3c.pdf"))

    print("  - fig4a.pdf")
    plot_hierarchy(novice_tail, str(PLOT_DIR / "fig4a.pdf"), "Novice")

    print("  - fig4b.pdf")
    plot_hierarchy(expert_tail, str(PLOT_DIR / "fig4b.pdf"), "Expert")

    print("  - fig6a.pdf")
    plot_hierarchy_continuous(novice_tail, str(PLOT_DIR / "fig6a.pdf"), "Novice")

    print("  - fig6b.pdf")
    plot_hierarchy_continuous(expert_tail, str(PLOT_DIR / "fig6b.pdf"), "Expert")

    print("  - fig5.pdf")
    plot_attractor_pca(novice_tail, expert_tail, str(PLOT_DIR / "fig5.pdf"))

    print("\n" + "=" * 70)
    print("PLOTTING COMPLETE")
    print("=" * 70)
    print(f"All plots: {PLOT_DIR}/")
    print()


def _load_results():
    import json

    expert_path = OUTPUT_DIR / f"simulation_results_expert_seed{SEED}.json"
    novice_path = OUTPUT_DIR / f"simulation_results_novice_seed{SEED}.json"
    training_expert_path = OUTPUT_DIR / f"training_results_expert_seed{SEED}.json"
    training_novice_path = OUTPUT_DIR / f"training_results_novice_seed{SEED}.json"
    if not expert_path.exists() or not novice_path.exists():
        fallback_expert = OUTPUT_DIR / f"training_results_expert_seed{SEED}.json"
        fallback_novice = OUTPUT_DIR / f"training_results_novice_seed{SEED}.json"
        if fallback_expert.exists() and fallback_novice.exists():
            print("WARNING: Simulation results not found. Falling back to training results.")
            expert_path = fallback_expert
            novice_path = fallback_novice
        else:
            print("ERROR: Could not find saved results:")
            print(f"  Looking for: {expert_path}")
            print(f"  Looking for: {novice_path}")
            print("\nRun 'python run_enactive_inference.py run' first to generate results.")
            return None, None

    print(f"\nLoading results from {OUTPUT_DIR}/...")
    with open(expert_path, "r") as f:
        expert_results = json.load(f)
    with open(novice_path, "r") as f:
        novice_results = json.load(f)

    expert_learning = None
    novice_learning = None
    if training_expert_path.exists():
        with open(training_expert_path, "r") as f:
            expert_learning = json.load(f)
    if training_novice_path.exists():
        with open(training_novice_path, "r") as f:
            novice_learning = json.load(f)

    return expert_results, novice_results, expert_learning, novice_learning


def main():
    parser = argparse.ArgumentParser(
        description="Lean meditation model: learning, simulation, and visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  run   - Learn parameters, then simulate (inference-only), then plot
  plot  - Generate plots from previously saved results

Examples:
  python run_enactive_inference.py run
  python run_enactive_inference.py plot

Configuration (fixed for reproducibility):
  Seed: 42
  Data dir: data/
  Figures dir: figures/
        """,
    )

    parser.add_argument(
        "command",
        type=str,
        choices=["run", "plot"],
        help='Command: "run" for training+simulation, "plot" for visualization only',
    )

    args = parser.parse_args()

    if args.command == "run":
        expert_learning, expert_results, novice_learning, novice_results = run_training_and_simulation(
            timesteps=TIMESTEPS
        )
        generate_plots(
            expert_results,
            novice_results,
            expert_learning=expert_learning,
            novice_learning=novice_learning,
        )
    elif args.command == "plot":
        expert_results, novice_results, expert_learning, novice_learning = _load_results()
        if expert_results is None:
            return
        generate_plots(
            expert_results,
            novice_results,
            expert_learning=expert_learning,
            novice_learning=novice_learning,
        )


if __name__ == "__main__":
    main()
