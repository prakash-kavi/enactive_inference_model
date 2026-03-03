"""Main entry point for lean meditation model: train -> eval -> plot.

One run per phenotype: TRAIN_STEPS (learning), EVAL_STEPS (frozen, Fig S1), PLOT_STEPS (frozen, fig3-fig5).

Usage:
    python run_enactive_inference.py run
    python run_enactive_inference.py plot
"""

import argparse
import json
from pathlib import Path

from model.training_loop import train_meditation_model
from model.phenotype import EXPERT_PHENOTYPE, NOVICE_PHENOTYPE
from viz.analysis_utils import prepare_tail_data, plot_fe_and_dwell, plot_transitions
from utils.config import (
    STATES,
    NETWORKS,
    THOUGHTSEEDS,
    TRAIN_STEPS,
    EVAL_STEPS,
    PLOT_STEPS,
    TOTAL_STEPS,
)

from viz.convergence import plot_convergence
from viz.radar_plot import plot_comparison
from viz.hierarchy import plot_hierarchy_continuous
from viz.attractors import plot_attractor_pca

SEED = 42

CURRENT_DIR = Path(__file__).parent
OUTPUT_DIR = CURRENT_DIR / "data"
PLOT_DIR = CURRENT_DIR / "figures"

# List-valued keys to slice for eval/plot segments
_TIME_SERIES_KEYS = [
    "state_history",
    "free_energy_history",
    "loss_history",
    "meta_awareness_history",
    "network_activations_history",
    "thoughtseed_activations_history",
    "thoughtseed_prior_activations_history",
]


def _slice_results(results: dict, start: int, n_steps: int) -> dict:
    """Return a copy of results with time-series list fields sliced to [start:start+n_steps]."""
    out = dict(results)
    for key in _TIME_SERIES_KEYS:
        val = out.get(key)
        if isinstance(val, list) and len(val) >= start + n_steps:
            out[key] = val[start : start + n_steps]
    # Transitions: keep those in [start, start+n_steps), rebase timestamp to 0
    trans = out.get("transitions")
    if isinstance(trans, list):
        filtered = []
        for t in trans:
            if isinstance(t, dict):
                ts = t.get("timestamp", 0)
                if start <= ts < start + n_steps:
                    filtered.append({**t, "timestamp": ts - start})
        out["transitions"] = filtered
    return out


def run_training_and_simulation():
    """Run one train->eval->plot per phenotype; save one results file per phenotype."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("PHASE 1: EXPERT (train -> eval -> plot)")
    print("=" * 70)
    _, expert_results = train_meditation_model(
        phenotype=EXPERT_PHENOTYPE,
        timesteps=TOTAL_STEPS,
        train_steps=TRAIN_STEPS,
        seed=SEED,
        save_results=True,
        output_dir=str(OUTPUT_DIR),
    )
    n = len(expert_results["state_history"])
    print(f"    Done: {n} steps ({TRAIN_STEPS} train, {EVAL_STEPS} eval, {PLOT_STEPS} plot)")

    print("\n" + "=" * 70)
    print("PHASE 2: NOVICE (train -> eval -> plot)")
    print("=" * 70)
    _, novice_results = train_meditation_model(
        phenotype=NOVICE_PHENOTYPE,
        timesteps=TOTAL_STEPS,
        train_steps=TRAIN_STEPS,
        seed=SEED,
        save_results=True,
        output_dir=str(OUTPUT_DIR),
    )
    n = len(novice_results["state_history"])
    print(f"    Done: {n} steps ({TRAIN_STEPS} train, {EVAL_STEPS} eval, {PLOT_STEPS} plot)")

    print("\n" + "=" * 70)
    print("RUN COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {OUTPUT_DIR}/")
    print(f"  - training_results_expert_seed{SEED}.json")
    print(f"  - training_results_novice_seed{SEED}.json")
    print()

    return expert_results, novice_results


def generate_plots(expert_results, novice_results):
    """Generate plots: Fig S1 from EVAL_STEPS; fig3-fig5 from PLOT_STEPS (final tail)."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)

    # Plot window = last PLOT_STEPS (final tail)
    print("  - Processing plot window (final tail)...")
    expert_tail = prepare_tail_data(
        expert_results, STATES, NETWORKS, THOUGHTSEEDS, tail_steps=PLOT_STEPS
    )
    novice_tail = prepare_tail_data(
        novice_results, STATES, NETWORKS, THOUGHTSEEDS, tail_steps=PLOT_STEPS
    )
    print(f"    Plot window: {PLOT_STEPS} steps (final tail)")
    print(f"    Full run: {len(expert_results['state_history'])} steps")

    # Fig S1: convergence over full run, with eval and plot windows shaded
    eval_span = {
        "start": TRAIN_STEPS,
        "end": TRAIN_STEPS + EVAL_STEPS,
        "label": "Eval window",
        "color": "#c7d4f1",
        "alpha": 0.45,
    }
    plot_span = {
        "start": TRAIN_STEPS + EVAL_STEPS,
        "end": TRAIN_STEPS + EVAL_STEPS + PLOT_STEPS,
        "label": "Plot window",
        "color": "#f3d0c7",
        "alpha": 0.35,
    }
    spans = [eval_span, plot_span]

    print("\n--- Figures ---")
    print("  - FigS1_Convergence_Expert.pdf (full run with eval/plot shading)")
    plot_convergence(
        expert_results,
        str(PLOT_DIR / "FigS1_Convergence_Expert.pdf"),
        panel_title="Full run",
        tail_span=None,
        highlight_spans=spans,
    )
    print("  - FigS1_Convergence_Novice.pdf (full run with eval/plot shading)")
    plot_convergence(
        novice_results,
        str(PLOT_DIR / "FigS1_Convergence_Novice.pdf"),
        panel_title="Full run",
        tail_span=None,
        highlight_spans=spans,
    )

    print("  - fig3a.pdf")
    plot_comparison(novice_tail, expert_tail, str(PLOT_DIR / "fig3a.pdf"))

    print("  - fig3b.pdf")
    plot_fe_and_dwell(novice_tail, expert_tail, str(PLOT_DIR / "fig3b.pdf"))

    print("  - fig3c.pdf")
    plot_transitions(novice_tail, expert_tail, str(PLOT_DIR / "fig3c.pdf"))

    print("  - fig4a.pdf")
    plot_hierarchy_continuous(novice_tail, str(PLOT_DIR / "fig4a.pdf"), "Novice")

    print("  - fig4b.pdf")
    plot_hierarchy_continuous(expert_tail, str(PLOT_DIR / "fig4b.pdf"), "Expert")

    print("  - fig5.pdf")
    plot_attractor_pca(novice_tail, expert_tail, str(PLOT_DIR / "fig5.pdf"))

    print("\n" + "=" * 70)
    print("PLOTTING COMPLETE")
    print("=" * 70)
    print(f"All plots: {PLOT_DIR}/")
    print()


def _load_results():
    """Load single results file per phenotype (full train->eval->plot run)."""
    expert_path = OUTPUT_DIR / f"training_results_expert_seed{SEED}.json"
    novice_path = OUTPUT_DIR / f"training_results_novice_seed{SEED}.json"
    if not expert_path.exists() or not novice_path.exists():
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
    return expert_results, novice_results


def main():
    parser = argparse.ArgumentParser(
        description="Lean meditation model: train -> eval -> plot (one run per phenotype)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  run   - Train then eval+plot (frozen); save one file per phenotype; then plot
  plot  - Generate plots from previously saved results

Structure: TRAIN_STEPS (learning) -> EVAL_STEPS (Fig S1) -> PLOT_STEPS (fig3-fig5).
        """,
    )

    parser.add_argument(
        "command",
        type=str,
        choices=["run", "plot"],
        help='"run" for train+eval+plot, "plot" for visualization only',
    )

    args = parser.parse_args()

    if args.command == "run":
        expert_results, novice_results = run_training_and_simulation()
        generate_plots(expert_results, novice_results)
    elif args.command == "plot":
        expert_results, novice_results = _load_results()
        if expert_results is None:
            return
        generate_plots(expert_results, novice_results)


if __name__ == "__main__":
    main()
