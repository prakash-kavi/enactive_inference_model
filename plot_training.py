"""
plot_training.py

Streamlined visualization of training results:
1. FigS1: Convergence diagnostics (Free Energy & State Fraction) for Seed 42 (Novice & Expert).
2. Fig3B: Radar plot comparing Learner Attractors (Novice vs Expert Means).
"""

import os
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import from viz package
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from viz.plotting_utils import set_plot_style, STATE_COLORS, STATE_DISPLAY_NAMES, STATE_SHORT_NAMES, save_figure, load_time_series
from config.meditation_config import STATES

NETWORKS = ['DMN', 'VAN', 'DAN', 'FPN']

def clean_old_plots(output_dir):
    """Remove clutter from previous runs."""
    for f in os.listdir(output_dir):
        if f.startswith("FigTraining_") or f.startswith("Radar_") or f.startswith("Convergence_"):
            os.remove(os.path.join(output_dir, f))
    logging.info("Cleaned up old training plots.")

def load_summary(level, results_dir="data/training"):
    """Load convergence summary for a level."""
    path = Path(results_dir) / f"convergence_summary_{level}.json"
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Error loading convergence summary for {level}: {e}")
            return None
    return None

def rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling mean matching plot_convergence.py style."""
    if arr.size == 0:
        return np.array([])
    if window <= 1 or arr.size < window:
        return np.full(arr.shape, np.nan, dtype=float)
    cumsum = np.cumsum(np.insert(arr, 0, 0.0))
    means = (cumsum[window:] - cumsum[:-window]) / window
    pad = np.full(window - 1, np.nan)
    return np.concatenate([pad, means])

def rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling std matching plot_convergence.py style."""
    if arr.size == 0:
        return np.array([])
    if window <= 1 or arr.size < window:
        return np.full(arr.shape, np.nan, dtype=float)
    mean = rolling_mean(arr, window)
    squared = rolling_mean(arr ** 2, window)
    variance = squared - mean ** 2
    variance = np.clip(variance, 0.0, None)
    return np.sqrt(variance)

def cumulative_state_fraction(states: list) -> dict:
    """Compute cumulative state fractions matching plot_convergence.py style."""
    from config.meditation_config import STATES
    n = len(states)
    fractions = {state: np.zeros(n, dtype=float) for state in STATES}
    counts = {state: 0 for state in STATES}
    for idx, state in enumerate(states):
        counts[state] += 1
        denom = idx + 1
        for st in STATES:
            fractions[st][idx] = counts[st] / denom
    return fractions

def plot_convergence_history(level, results_dir="data/training", output_dir="plots/training"):
    """
    Plot convergence diagnostics (Free Energy & State Occupancy) for Seed 42.
    Matches exact style from viz/plot_convergence.py
    """
    set_plot_style()
    
    # Load History from Seed 42 output (in convergence_plots_data subdirectory)
    plots_data_dir = Path(results_dir) / "convergence_plots_data"
    history_file = plots_data_dir / f"thoughtseed_params_{level}.json"
    
    if not history_file.exists():
        logging.warning(f"No history found for {level} (checked {history_file}). "
                        f"This file is only created when seed 42 is run. Skipping FigS1.")
        return

    try:
        with open(history_file) as f:
            payload = json.load(f)
            data = payload.get("time_series", {})
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in {history_file}: {e}")
        return
    except Exception as e:
        logging.warning(f"Error loading {history_file}: {e}")
        return

    free_energy = np.asarray(data.get("free_energy_history", []), dtype=float)
    states = data.get("state_history", [])
    
    if free_energy.size == 0:
        logging.warning(f"No free energy data for {level}. Skipping plot.")
        return

    steps = np.arange(free_energy.size)
    window = 25  # Match plot_convergence.py default
    tail_span = 200  # Match plot_convergence.py default
    highlight_start = max(0, free_energy.size - tail_span)

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    # Panel 1: Free energy trend (exact match to plot_convergence.py)
    ax = axes[0]
    ax.plot(steps, free_energy, color="#cccccc", linewidth=1.0, label="Free energy (raw)")
    fe_mean = rolling_mean(free_energy, window)
    fe_std = rolling_std(free_energy, window)
    ax.plot(steps, fe_mean, color="#E74C3C", linewidth=2.0, label=f"Rolling mean (w={window})")
    valid = ~np.isnan(fe_mean)
    if np.any(valid):
        lower = (fe_mean - fe_std)[valid]
        upper = (fe_mean + fe_std)[valid]
        ax.fill_between(steps[valid], lower, upper, color="#E74C3C", alpha=0.18)
    ax.set_ylabel("Free energy")
    ax.set_title(f"Free-energy stabilisation ({level.title()})", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", frameon=True)

    # Panel 2: Cumulative state occupancy (exact match to plot_convergence.py)
    ax = axes[1]
    fractions = cumulative_state_fraction(states)
    for state in STATES:
        ax.plot(steps, fractions[state], color=STATE_COLORS[state], linewidth=1.8, label=STATE_SHORT_NAMES[state])
    ax.set_ylabel("Cumulative fraction")
    ax.set_xlabel("Timestep")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Cumulative state occupancy", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", frameon=True)

    # Add tail window highlight (matching plot_convergence.py)
    if highlight_start > 0:
        for ax in axes:
            ax.axvspan(highlight_start, free_energy.size, color="#f5f5f5", alpha=0.4, label="Tail window")
            handles, labels = ax.get_legend_handles_labels()
            dedup = {}
            for handle, label in zip(handles, labels):
                dedup[label] = handle
            ax.legend(list(dedup.values()), list(dedup.keys()), loc="best", frameon=True)

    fig.suptitle(f"Convergence diagnostics ({level.title()})", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    save_figure(fig, os.path.join(output_dir, f"FigS1_Convergence_{level.capitalize()}.png"), "Convergence")
    plt.close()

def plot_radar_comparison(novice_summary, expert_summary, output_dir="plots/training"):
    """
    Fig3B: Network Activation Profiles (radar plots showing network expectations for each state)
    Matches user's reference style:
    - 4 Subplots (one per state)
    - State-colored lines
    - Expert = Solid, Novice = Dashed
    - Unified legend at bottom
    - Reference Alphas (0.22 Novice, 0.16 Expert)
    """
    set_plot_style()
    
    # Map summaries to data access pattern
    nov_data = novice_summary['network_profiles_mean']
    exp_data = expert_summary['network_profiles_mean']

    fig = plt.figure(figsize=(14, 12))
    fig.suptitle('Learned Network Activation Profiles', fontsize=18, fontweight='bold')

    angles = np.linspace(0, 2*np.pi, len(NETWORKS), endpoint=False).tolist()
    angles += angles[:1]

    for i, state in enumerate(STATES):
        ax = fig.add_subplot(2, 2, i+1, polar=True)

        nov_state = nov_data.get(state, {})
        exp_state = exp_data.get(state, {})

        nov_vals = [float(nov_state.get(net, 0.0)) for net in NETWORKS]
        exp_vals = [float(exp_state.get(net, 0.0)) for net in NETWORKS]
        nov_vals += nov_vals[:1]
        exp_vals += exp_vals[:1]

        # Add concentric circle grid lines (one shade darker background)
        # Draw circles at 0.2, 0.4, 0.6, 0.8, 1.0
        circle_radii = [0.2, 0.4, 0.6, 0.8, 1.0]
        circle_angles = np.linspace(0, 2*np.pi, 100)
        for radius in circle_radii:
            ax.plot(circle_angles, [radius] * len(circle_angles), 
                   color='#999999', linewidth=0.8, alpha=0.4, zorder=0)

        # State-coloured lines/fills (Novice dashed, Expert solid) - Exact match to plot_diagnostics.py
        ax.plot(angles, nov_vals, color=STATE_COLORS[state], linewidth=2.6, linestyle='--', label="Novice", zorder=3)
        ax.fill(angles, nov_vals, color=STATE_COLORS[state], alpha=0.22, zorder=2)
        
        ax.plot(angles, exp_vals, color=STATE_COLORS[state], linewidth=2.8, linestyle='-', label="Expert", zorder=3)
        ax.fill(angles, exp_vals, color=STATE_COLORS[state], alpha=0.16, zorder=2)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(NETWORKS, fontsize=13, fontweight='bold')
        ax.tick_params(axis='y', labelsize=11)
        ax.set_ylim(0, 1)
        ax.set_title(STATE_DISPLAY_NAMES[state], fontsize=15, fontweight='bold', pad=18)
        ax.grid(True, linestyle='--', alpha=0.7)
        # Slightly increase label padding for polar plots
        for lbl in ax.get_xticklabels():
            lbl.set_y(0.02)

    labels = ["Expert", "Novice"]
    handles = [
        plt.Line2D([0], [0], color='black', linewidth=2.6, label=labels[0]),
        plt.Line2D([0], [0], color='black', linewidth=2.2, linestyle='--', label=labels[1])
    ]
    fig.legend(handles=handles, labels=labels, loc='upper center',
               bbox_to_anchor=(0.5, 0.08), ncol=2, fontsize=13)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "Fig3B_Radar_Comparison.png")
    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved Radar to {save_path}")
    except Exception as e:
        logging.error(f"Failed to save radar plot to {save_path}: {e}")
    finally:
        plt.close()

def generate_all(results_dir="data/training", output_dir="plots/training"):
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    os.makedirs(output_dir, exist_ok=True)
    
    clean_old_plots(output_dir)
    
    # 1. Plot Convergence History (FigS1)
    logging.info("Generating Convergence Plots (FigS1)...")
    plot_convergence_history('novice', results_dir, output_dir)
    plot_convergence_history('expert', results_dir, output_dir)
    
    # 2. Plot Radar Comparison (Fig3B)
    logging.info("Generating Radar Comparison (Fig3B)...")
    novice_summary = load_summary('novice', results_dir)
    expert_summary = load_summary('expert', results_dir)
    
    if novice_summary and expert_summary:
        plot_radar_comparison(novice_summary, expert_summary, output_dir)
    else:
        logging.warning("Could not load both summaries. Skipping Radar comparison.")
        
    logging.info("Visualizations complete.")

if __name__ == "__main__":
    generate_all()
