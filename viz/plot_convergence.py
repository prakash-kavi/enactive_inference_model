"""
plot_convergence.py

Generates convergence diagnostic figures:
- Figure S1: Convergence Diagnostics for Novice and Expert profiles
  (Free Energy, Precision, Complexity, Memory evolution over training)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import logging
import matplotlib.pyplot as plt
import numpy as np

from utils.meditation_config import STATES, NETWORKS
from .plotting_utils import (
    NETWORK_COLORS,
    PLOT_DIR,
    STATE_COLORS,
    STATE_SHORT_NAMES,
    load_time_series,
    save_figure,
    set_plot_style,
)


def rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    if arr.size == 0:
        return np.array([])
    if window <= 1 or arr.size < window:
        return np.full(arr.shape, np.nan, dtype=float)
    cumsum = np.cumsum(np.insert(arr, 0, 0.0))
    means = (cumsum[window:] - cumsum[:-window]) / window
    pad = np.full(window - 1, np.nan)
    return np.concatenate([pad, means])


def rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
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
    n = len(states)
    fractions = {state: np.zeros(n, dtype=float) for state in STATES}
    counts = {state: 0 for state in STATES}
    for idx, state in enumerate(states):
        counts[state] += 1
        denom = idx + 1
        for st in STATES:
            fractions[st][idx] = counts[st] / denom
    return fractions


def _network_matrix(history: List[Dict[str, float]]) -> np.ndarray:
    if not history:
        return np.zeros((0, len(NETWORKS)))
    return np.asarray([[step.get(net, 0.0) for net in NETWORKS] for step in history], dtype=float)


def plot_convergence_panels(cohort: str, window: int = 25, tail_span: int = 200, fe_ylim: tuple[float, float] | None = None) -> Path:
    set_plot_style()
    series = load_time_series(cohort)

    # Extract time-series produced by training outputs (thoughtseed_params.time_series)
    free_energy = np.asarray(series.get("free_energy_history", []), dtype=float)
    meta_awareness = np.asarray(series.get("meta_awareness_history", []), dtype=float)
    states = series.get("state_history", [])
    network_hist = _network_matrix(series.get("network_activations_history", []))

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
    ax.set_title(f"Free-energy stabilisation ({cohort.title()})", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", frameon=True)
    if fe_ylim:
        ax.set_ylim(fe_ylim[0], fe_ylim[1])  # set consistent y-axis range

    # Panel 2: Cumulative state occupancy
    ax = axes[1]
    fractions = cumulative_state_fraction(states)
    for state in STATES:
        ax.plot(steps, fractions[state], color=STATE_COLORS[state], linewidth=1.8, label=STATE_SHORT_NAMES[state])
    ax.set_ylabel("Cumulative fraction")
    ax.set_xlabel("Timestep")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Cumulative state occupancy", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", frameon=True)

    if highlight_start > 0:
        for ax in axes:
            ax.axvspan(highlight_start, free_energy.size, color="#f5f5f5", alpha=0.4, label="Tail window")
            handles, labels = ax.get_legend_handles_labels()
            dedup: Dict[str, tuple] = {}
            for handle, label in zip(handles, labels):
                dedup[label] = handle
            ax.legend(list(dedup.values()), list(dedup.keys()), loc="best", frameon=True)

    fig.suptitle(f"Convergence diagnostics ({cohort.title()})", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = Path(PLOT_DIR) / f"FigS1_Convergence_{cohort.title()}.png"
    save_figure(fig, out_path, "Convergence")
    plt.close(fig)

    if free_energy.size:
        tail = free_energy[highlight_start:]
        logging.info("%s tail (last %d steps): mean F=%.4f, std=%.4f", cohort.title(), tail_span, tail.mean(), tail.std(ddof=0))
    return out_path


def generate_all(window: int = 25, tail_span: int = 200) -> None:
    plot_dir = Path(PLOT_DIR)
    # Compute global min/max for free energy across cohorts to set consistent ylim
    global_fe_min = float('inf')
    global_fe_max = float('-inf')
    for cohort in ("novice", "expert"):
        series = load_time_series(cohort)
        free_energy = np.asarray(series.get("free_energy_history", []), dtype=float)
        if free_energy.size:
            global_fe_min = min(global_fe_min, free_energy.min())
            global_fe_max = max(global_fe_max, free_energy.max())
    for cohort in ("novice", "expert"):
        plot_convergence_panels(cohort, window=window, tail_span=tail_span, fe_ylim=(global_fe_min, global_fe_max))
    # save_figure already logs file paths

if __name__ == "__main__":
    generate_all()
