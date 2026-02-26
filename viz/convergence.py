"""Convergence diagnostic plots.

Generates FigS1_Convergence_{Expert/Novice}.pdf:
- Panel 1: Total loss (raw + smoothed)
- Panel 2: Cumulative state occupancy over training
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List

from viz.plotting_utils import (
    set_plot_style,
    save_figure,
    STATE_COLORS,
    STATE_SHORT_NAMES
)
from utils.config import STATES
from viz.analysis_utils import TAIL_STEPS


def rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling mean matching reference implementation."""
    if arr.size == 0:
        return np.array([])
    if window <= 1 or arr.size < window:
        return np.full(arr.shape, np.nan, dtype=float)
    series = pd.Series(arr, dtype=float)
    return series.rolling(window=window, min_periods=window).mean().to_numpy()


def rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling std matching reference implementation."""
    if arr.size == 0:
        return np.array([])
    if window <= 1 or arr.size < window:
        return np.full(arr.shape, np.nan, dtype=float)
    series = pd.Series(arr, dtype=float)
    return series.rolling(window=window, min_periods=window).std(ddof=0).to_numpy()


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
    tail_span = TAIL_STEPS  # Use global TAIL_STEPS for stable window shading
    set_plot_style()
    
    # Extract data
    free_energy = np.asarray(results['free_energy_history'], dtype=float)
    loss_history = np.asarray(results.get('loss_history', []), dtype=float)
    states = results['state_history']
    level = results['experience_level']
    
    use_loss = loss_history.size == free_energy.size and loss_history.size > 0
    primary = loss_history if use_loss else free_energy

    steps = np.arange(primary.size)
    highlight_start = max(0, primary.size - tail_span)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    
    # Panel 1: Free energy trend
    ax = axes[0]
    ax.plot(
        steps,
        primary,
        color="#cccccc",
        linewidth=1.0,
        label="Total loss (raw)" if use_loss else "Free energy (raw)"
    )
    
    fe_mean = rolling_mean(primary, window)
    fe_std = rolling_std(primary, window)
    ax.plot(steps, fe_mean, color="#E74C3C", linewidth=2.0, label=f"Rolling mean (w={window})")
    
    valid = ~np.isnan(fe_mean)
    if np.any(valid):
        lower = (fe_mean - fe_std)[valid]
        upper = (fe_mean + fe_std)[valid]
        ax.fill_between(steps[valid], lower, upper, color="#E74C3C", alpha=0.18)
    
    ax.set_ylabel("Loss" if use_loss else "Free energy", fontweight="bold")
    ax.set_title(f"{'Loss' if use_loss else 'Free energy'} convergence ({level.title()})",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", frameon=True)
    if use_loss and primary.size:
        y_min = np.nanmin(primary)
        y_max = np.nanmax(primary)
        if np.isfinite(y_min) and np.isfinite(y_max):
            span = y_max - y_min
            pad = 0.05 * span if span > 0 else 0.05 * max(abs(y_max), 1.0)
            ax.set_ylim(y_min - pad, y_max + pad)
    
    # Panel 2: Cumulative state occupancy
    ax = axes[1]
    fractions = cumulative_state_fraction(states)
    for state in STATES:
        ax.plot(steps, fractions[state], color=STATE_COLORS[state], 
               linewidth=1.8, label=STATE_SHORT_NAMES[state])
    
    ax.set_ylabel("Cumulative fraction", fontweight="bold")
    ax.set_xlabel("Timestep", fontweight="bold")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Cumulative state occupancy", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", frameon=True)
    
    # Highlight tail window
    if highlight_start > 0:
        for ax in axes:
            ax.axvspan(highlight_start, primary.size, color="#d0d0d0", alpha=0.7, label="Tail window")
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
    if primary.size and highlight_start < primary.size:
        tail = primary[highlight_start:]
        metric = "loss" if use_loss else "F"
        print(f"    {level.title()} tail (last {tail_span} steps): mean {metric}={tail.mean():.4f}, std={tail.std(ddof=0):.4f}")
