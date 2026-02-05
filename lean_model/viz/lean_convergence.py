"""
lean_convergence.py

Convergence diagnostic plots for lean meditation model.
Reuses plotting logic from viz/viz/plot_convergence.py.

Generates FigS1_Convergence_{Expert/Novice}.png:
- Panel 1: Free energy stabilization (raw + smoothed)
- Panel 2: Cumulative state occupancy over training
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List

from .plotting_utils import (
    set_plot_style,
    save_figure,
    STATE_COLORS,
    STATE_SHORT_NAMES
)
from ..config import STATES
from ..utils import TAIL_STEPS


def rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling mean matching reference implementation."""
    if arr.size == 0:
        return np.array([])
    if window <= 1 or arr.size < window:
        return np.full(arr.shape, np.nan, dtype=float)
    cumsum = np.cumsum(np.insert(arr, 0, 0.0))
    means = (cumsum[window:] - cumsum[:-window]) / window
    pad = np.full(window - 1, np.nan)
    return np.concatenate([pad, means])


def rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling std matching reference implementation."""
    if arr.size == 0:
        return np.array([])
    if window <= 1 or arr.size < window:
        return np.full(arr.shape, np.nan, dtype=float)
    mean = rolling_mean(arr, window)
    squared = rolling_mean(arr ** 2, window)
    variance = squared - mean ** 2
    variance = np.clip(variance, 0.0, None)
    return np.sqrt(variance)


def cumulative_state_fraction(states: List[str]) -> Dict[str, np.ndarray]:
    """Compute cumulative state fractions over time."""
    n = len(states)
    fractions = {state: np.zeros(n, dtype=float) for state in STATES}
    counts = {state: 0 for state in STATES}
    for idx, state in enumerate(states):
        counts[state] += 1
        denom = idx + 1
        for st in STATES:
            fractions[st][idx] = counts[st] / denom
    return fractions


def plot_convergence(results: Dict, save_path: str, window: int = 25):
    """Generate convergence diagnostic plot.
    
    Args:
        results: Training results dict with full trajectory
        save_path: Path to save figure
        window: Rolling window size for smoothing
    """
    tail_span = TAIL_STEPS  # Use global TAIL_STEPS (2000) for stable window shading
    set_plot_style()
    
    # Extract data
    free_energy = np.asarray(results['free_energy_history'], dtype=float)
    states = results['state_history']
    level = results['experience_level']
    
    steps = np.arange(free_energy.size)
    highlight_start = max(0, free_energy.size - tail_span)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    
    # Panel 1: Free energy trend
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
    
    # Panel 2: Cumulative state occupancy
    ax = axes[1]
    fractions = cumulative_state_fraction(states)
    for state in STATES:
        ax.plot(steps, fractions[state], color=STATE_COLORS[state], 
               linewidth=1.8, label=STATE_SHORT_NAMES[state])
    
    ax.set_ylabel("Cumulative fraction")
    ax.set_xlabel("Timestep")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Cumulative state occupancy", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", frameon=True)
    
    # Highlight tail window
    if highlight_start > 0:
        for ax in axes:
            ax.axvspan(highlight_start, free_energy.size, color="#d0d0d0", alpha=0.7, label="Tail window")
            handles, labels = ax.get_legend_handles_labels()
            dedup = {}
            for handle, label in zip(handles, labels):
                dedup[label] = handle
            ax.legend(list(dedup.values()), list(dedup.keys()), loc="best", frameon=True)
    
    fig.suptitle(f"Convergence diagnostics ({level.title()})", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    save_figure(fig, Path(save_path), "Convergence")
    plt.close(fig)
    
    # Log tail statistics
    if free_energy.size and highlight_start < free_energy.size:
        tail = free_energy[highlight_start:]
        print(f"    {level.title()} tail (last {tail_span} steps): mean F={tail.mean():.4f}, std={tail.std(ddof=0):.4f}")
