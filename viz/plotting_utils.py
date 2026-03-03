"""
plotting_utils.py

Shared utilities for plotting: style settings and common constants.
Path constants (data/, figures/) live in run_enactive_inference.py as the single source.
"""

import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

STATE_DISPLAY_NAMES = {
    "breath_focus": "Breath Focus",
    "mind_wandering": "Mind Wandering",
    "meta_awareness": "Meta Awareness",
    "redirect_attention": "Redirect Attention"
}

STATE_SHORT_NAMES = {
    "breath_focus": "BF",
    "mind_wandering": "MW",
    "meta_awareness": "MA",
    "redirect_attention": "RA",
}

STATE_COLORS = {
    "breath_focus": "#2ca02c",
    "mind_wandering": "#1f77b4",
    "meta_awareness": "#d62728",
    "redirect_attention": "#ff7f0e",
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
    'aha_moment': '#4363d8',
    'pain_discomfort': '#e6194B',
    'pending_tasks': '#911eb4'
}


def set_plot_style():
    """Set a consistent publication-ready style for matplotlib."""
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.linewidth"] = 0.5
    plt.rcParams["grid.linewidth"] = 0.5
    plt.rcParams["grid.alpha"] = 0.3
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"

def save_figure(fig, save_path, label=None):
    """Save a figure and log a consistent message."""
    if not save_path:
        return
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    try:
        rel = os.path.relpath(str(path), start=os.getcwd())
    except Exception:
        rel = str(path)
    if label:
        logging.info("Saved %s to %s", label, rel)
    else:
        logging.info("Saved plot to %s", rel)

