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


def _extract_primary_series(results: Dict):
    free_energy = np.asarray(results["free_energy_history"], dtype=float)
    loss_history = np.asarray(results.get("loss_history", []), dtype=float)
    use_loss = loss_history.size == free_energy.size and loss_history.size > 0
    primary = loss_history if use_loss else free_energy
    label = "Loss" if use_loss else "Free energy"
    return primary, label


def _plot_single_convergence_pair(
    ax_loss,
    ax_occ,
    results: Dict,
    window: int,
    panel_title: str,
    tail_span: int,
):
    primary, label = _extract_primary_series(results)
    states = results["state_history"]
    steps = np.arange(primary.size)
    highlight_start = max(0, primary.size - tail_span)

    ax_loss.plot(
        steps,
        primary,
        color="#cccccc",
        linewidth=1.0,
        label=f"{label} (raw)",
    )

    series_mean = rolling_mean(primary, window)
    series_std = rolling_std(primary, window)
    ax_loss.plot(
        steps,
        series_mean,
        color="#E74C3C",
        linewidth=2.0,
        label=f"Rolling mean (w={window})",
    )

    valid = ~np.isnan(series_mean)
    if np.any(valid):
        lower = (series_mean - series_std)[valid]
        upper = (series_mean + series_std)[valid]
        ax_loss.fill_between(steps[valid], lower, upper, color="#E74C3C", alpha=0.18)

    ax_loss.set_ylabel(label, fontweight="bold")
    ax_loss.set_title(f"{panel_title}: {label}", fontsize=12, fontweight="bold")
    ax_loss.legend(loc="upper right", frameon=True)
    if primary.size:
        y_min = np.nanmin(primary)
        y_max = np.nanmax(primary)
        if np.isfinite(y_min) and np.isfinite(y_max):
            span = y_max - y_min
            pad = 0.05 * span if span > 0 else 0.05 * max(abs(y_max), 1.0)
            ax_loss.set_ylim(y_min - pad, y_max + pad)

    fractions = cumulative_state_fraction(states)
    for state in STATES:
        ax_occ.plot(
            steps,
            fractions[state],
            color=STATE_COLORS[state],
            linewidth=1.6,
            label=STATE_SHORT_NAMES[state],
        )

    ax_occ.set_ylabel("Cumulative fraction", fontweight="bold")
    ax_occ.set_xlabel("Timestep", fontweight="bold")
    ax_occ.set_ylim(0.0, 1.0)
    ax_occ.set_title(f"{panel_title}: Cumulative state occupancy", fontsize=12, fontweight="bold")
    ax_occ.legend(loc="lower right", frameon=True)

    if highlight_start > 0:
        for ax in (ax_loss, ax_occ):
            ax.axvspan(
                highlight_start,
                primary.size,
                color="#d0d0d0",
                alpha=0.7,
                label="Tail window",
            )
            handles, labels = ax.get_legend_handles_labels()
            dedup = {}
            for handle, label in zip(handles, labels):
                dedup[label] = handle
            ax.legend(list(dedup.values()), list(dedup.keys()), loc="best", frameon=True)

    if primary.size and highlight_start < primary.size:
        tail = primary[highlight_start:]
        metric = "loss" if label == "Loss" else "F"
        level = results.get("experience_level", "unknown").title()
        print(
            f"    {level} {panel_title.lower()} tail (last {tail_span} steps): "
            f"mean {metric}={tail.mean():.4f}, std={tail.std(ddof=0):.4f}"
        )


def plot_convergence(results: Dict, save_path: str, window: int = 25):
    """Generate a single convergence diagnostic plot (legacy)."""
    tail_span = TAIL_STEPS
    set_plot_style()

    level = results.get("experience_level", "unknown")
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    _plot_single_convergence_pair(
        axes[0],
        axes[1],
        results,
        window,
        panel_title="Convergence",
        tail_span=tail_span,
    )

    fig.suptitle(f"Convergence diagnostics ({level.title()})", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_figure(fig, Path(save_path), "Convergence")
    plt.close(fig)


def plot_convergence_comparison(
    learning_results: Dict,
    simulation_results: Dict,
    save_path: str,
    window: int = 25,
):
    """Generate combined learning vs inference-only stability diagnostics."""
    tail_span = TAIL_STEPS
    set_plot_style()

    level = learning_results.get("experience_level", simulation_results.get("experience_level", "unknown"))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex="col")

    _plot_single_convergence_pair(
        axes[0, 0],
        axes[1, 0],
        learning_results,
        window,
        panel_title="Learning (E+M)",
        tail_span=tail_span,
    )
    _plot_single_convergence_pair(
        axes[0, 1],
        axes[1, 1],
        simulation_results,
        window,
        panel_title="Inference-only (E-step)",
        tail_span=tail_span,
    )

    fig.suptitle(
        f"Learning vs inference-only stability ({level.title()})",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, Path(save_path), "Convergence comparison")
    plt.close(fig)
