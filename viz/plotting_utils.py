"""
plotting_utils.py

Shared utilities for plotting: data loading, style settings, and common constants.
This ensures consistency across all figure generation scripts.
"""

import os
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import copy

TAIL_STEPS = 200

STATE_DISPLAY_NAMES = {
    "breath_focus": "Breath Focus",
    "mind_wandering": "Mind Wandering",
    "meta_awareness": "Meta Awareness",
    "redirect_breath": "Redirect Attention"
}

STATE_SHORT_NAMES = {
    "breath_focus": "BF",
    "mind_wandering": "MW",
    "meta_awareness": "MA",
    "redirect_breath": "RA",
}

STATE_COLORS = {
    "breath_focus": "#2ca02c",
    "mind_wandering": "#1f77b4",
    "meta_awareness": "#d62728",
    "redirect_breath": "#ff7f0e",
}

NETWORK_COLORS = {
    'DMN': '#CA3542',
    'VAN': '#B77FB4',
    'DAN': '#2C8B4B',
    'FPN': '#E58429',
}

THOUGHTSEED_COLORS = {
    'attend_breath': '#f58231',
    'equanimity': '#3cb44b',
    'self_reflection': '#4363d8',
    'pain_discomfort': '#e6194B',
    'pending_tasks': '#911eb4'
}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
PLOT_DIR = os.path.join(BASE_DIR, "plots")

def set_plot_style():
    """Set a consistent publication-ready style for matplotlib."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.linewidth"] = 0.5
    plt.rcParams["grid.linewidth"] = 0.5
    plt.rcParams["grid.alpha"] = 0.3
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"

def load_json_data(cohort):
    """
    Load all JSON data for a specific cohort ('novice' or 'expert').
    Returns a tuple: (thoughtseed_params, active_inference_params, transition_stats)
    """
    ts_path = os.path.join(DATA_DIR, f"thoughtseed_params_{cohort}.json")
    ai_path = os.path.join(DATA_DIR, f"active_inference_params_{cohort}.json")
    stats_path = os.path.join(DATA_DIR, f"transition_stats_{cohort}.json")
    frozen_path = os.path.join(DATA_DIR, f"frozen_params_{cohort}.json")

    # Try loading frozen params first as it often contains the time series snapshot
    frozen_data = {}
    if os.path.exists(frozen_path):
        with open(frozen_path, 'r') as f:
            frozen_data = json.load(f)

    ts_data = {}
    if os.path.exists(ts_path):
        with open(ts_path, 'r') as f:
            ts_data = json.load(f)
            
    ai_data = {}
    if os.path.exists(ai_path):
        with open(ai_path, 'r') as f:
            ai_data = json.load(f)
            
    stats_data = {}
    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            stats_data = json.load(f)

    # Merge frozen time series into stats if missing (backwards compatibility)
    if "state_history" not in stats_data and "time_series_snapshot" in frozen_data:
        logging.info("Merging time_series_snapshot from frozen_params into stats for %s", cohort)
        stats_data.update(frozen_data["time_series_snapshot"])

    # Also check thoughtseed_params for time_series
    if "state_history" not in stats_data and "time_series" in ts_data:
        logging.info("Merging time_series from thoughtseed_params into stats for %s", cohort)
        stats_data.update(ts_data["time_series"])

    return ts_data, ai_data, stats_data

def slice_tail(sequence, tail=TAIL_STEPS):
    """Slice the last 'tail' elements from a list or array."""
    if tail is None or not sequence:
        return sequence
    if isinstance(sequence, list):
        return sequence[-tail:]
    if isinstance(sequence, np.ndarray):
        return sequence[-tail:]
    return sequence

def get_tail_stats(stats_data, tail=TAIL_STEPS):
    """Return a copy of stats_data with time-series fields sliced to the tail."""
    trimmed = copy.deepcopy(stats_data)
    keys = [
        'state_history',
        'meta_awareness_history',
        'network_activations_history',
        'free_energy_history',
        'dominant_ts_history',
        'activations_history',
        'van_hazard_history',
    ]
    for key in keys:
        if key in trimmed:
            trimmed[key] = slice_tail(trimmed[key], tail)
    return trimmed

def smooth_series(values, alpha=0.5):
    """Exponential moving average smoothing."""
    if values is None or len(values) == 0:
        return np.array([])
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    smoothed = np.empty_like(arr)
    smoothed[0] = arr[0]
    for idx in range(1, len(arr)):
        smoothed[idx] = (1 - alpha) * smoothed[idx - 1] + alpha * arr[idx]
    return smoothed
