"""
plot_convergence.py

Generates convergence diagnostic figures:
- Figure S1: Convergence Diagnostics for Novice and Expert profiles
  (Free Energy, Precision, Complexity, Memory evolution over training)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import logging
import matplotlib.pyplot as plt
import numpy as np
import os

from config.meditation_config import STATES

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PLOT_DIR = BASE_DIR / "plots"

STATE_COLORS = {
    "breath_focus": "#2ca02c",
    "mind_wandering": "#1f77b4",
    "meta_awareness": "#d62728",
    "redirect_breath": "#ff7f0e",
}

STATE_SHORT = {
    "breath_focus": "BF",
    "mind_wandering": "MW",
    "meta_awareness": "MA",
    "redirect_breath": "RA",
}

NETWORK_KEYS = ["DMN", "VAN", "DAN", "FPN"]
NETWORK_COLORS = {
    "DMN": "#CA3542",
    "VAN": "#B77FB4",
    "DAN": "#2C8B4B",
    "FPN": "#E58429",
}


def set_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.linewidth"] = 0.5
    plt.rcParams["grid.linewidth"] = 0.5
    plt.rcParams["grid.alpha"] = 0.3


def _load_series(cohort: str) -> Dict[str, List]:
    path = DATA_DIR / f"thoughtseed_params_{cohort}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing training output: {path}")
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload.get("time_series", {})


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
        return np.zeros((0, len(NETWORK_KEYS)))
    return np.asarray([[step.get(net, 0.0) for net in NETWORK_KEYS] for step in history], dtype=float)


def plot_convergence_panels(cohort: str, window: int = 25, tail_span: int = 200, fe_ylim: tuple[float, float] | None = None) -> Path:
    set_plot_style()
    series = _load_series(cohort)

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
        ax.plot(steps, fractions[state], color=STATE_COLORS[state], linewidth=1.8, label=STATE_SHORT[state])
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

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOT_DIR / f"FigS1_Convergence_{cohort.title()}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    if free_energy.size:
        tail = free_energy[highlight_start:]
        logging.info("%s tail (last %d steps): mean F=%.4f, std=%.4f", cohort.title(), tail_span, tail.mean(), tail.std(ddof=0))
    return out_path


def generate_all(window: int = 25, tail_span: int = 200) -> None:
    # Compute global min/max for free energy across cohorts to set consistent ylim
    global_fe_min = float('inf')
    global_fe_max = float('-inf')
    for cohort in ("novice", "expert"):
        series = _load_series(cohort)
        free_energy = np.asarray(series.get("free_energy_history", []), dtype=float)
        if free_energy.size:
            global_fe_min = min(global_fe_min, free_energy.min())
            global_fe_max = max(global_fe_max, free_energy.max())
    for cohort in ("novice", "expert"):
        plot_convergence_panels(cohort, window=window, tail_span=tail_span, fe_ylim=(global_fe_min, global_fe_max))
    try:
        rel = os.path.relpath(str(PLOT_DIR), start=os.getcwd())
    except Exception:
        rel = str(PLOT_DIR)
    logging.info("Saved convergence plots to %s", rel)

if __name__ == "__main__":
    generate_all()


