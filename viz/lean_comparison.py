"""
lean_comparison.py

Radar comparison plot - EXACT copy from viz/viz/plot_training.py plot_radar_comparison()
Adapted to take data dicts instead of loading from file.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from viz.plotting_utils import (
    STATE_COLORS,
    STATE_DISPLAY_NAMES,
    save_figure,
    set_plot_style,
)
from utils.config import STATES, NETWORKS


def plot_comparison(novice_data: dict, expert_data: dict, save_path: str):
    """
    Fig3: Network Activation Profiles (radar plots showing network expectations for each state)
    EXACT COPY from plot_training.py plot_radar_comparison()
    Matches user's reference style:
    - 4 Subplots (one per state)
    - State-colored lines
    - Expert = Solid, Novice = Dashed
    - Unified legend at bottom
    - Reference Alphas (0.22 Novice, 0.16 Expert)
    """
    set_plot_style()
    
    # Map data to access pattern
    nov_data = novice_data['network_profiles_mean']
    exp_data = expert_data['network_profiles_mean']

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
    save_figure(fig, Path(save_path), "Radar Comparison")
    plt.close(fig)
