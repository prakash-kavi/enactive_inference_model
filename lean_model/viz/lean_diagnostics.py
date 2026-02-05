"""
lean_diagnostics.py

Diagnostic comparison plots - EXACT copy from viz/viz/plot_diagnostics.py
Combines Free Energy bars + Dwell times bars in 2-panel figure.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

from .plotting_utils import (
    STATE_COLORS,
    STATE_DISPLAY_NAMES,
    save_figure,
    set_plot_style,
)
from ..config import STATES, DEFAULTS


def get_dwell_times(stats):
    """Extract dwell times from state history - EXACT copy from plot_diagnostics.py"""
    state_history = stats.get("state_history", [])
    if not state_history: 
        return {s: [] for s in STATES}
    
    dwells = {s: [] for s in STATES}
    
    current_state = state_history[0]
    count = 0
    for s in state_history:
        if s == current_state:
            count += 1
        else:
            dwells[current_state].append(count)
            current_state = s
            count = 1
    dwells[current_state].append(count)
    return dwells


def plot_fe_and_dwell(novice_data: dict, expert_data: dict, save_path: str):
    """
    Combined 2-panel plot: Free Energy + Dwell Times
    EXACT copy from viz/viz/plot_diagnostics.py (plot_free_energy_bar + plot_dwell_times)
    """
    set_plot_style()
    
    fig = plt.figure(figsize=(16, 6))
    gs = GridSpec(1, 2, figure=fig, wspace=0.25)
    
    # ===== Panel A: Free Energy Bar Chart =====
    ax1 = fig.add_subplot(gs[0])
    
    def _collect_fe_by_state(stats):
        states = stats.get("state_history", [])
        fe = stats.get("free_energy_history", [])
        by_state = {s: [] for s in STATES}
        for s, f in zip(states, fe):
            if s in by_state:
                try:
                    by_state[s].append(float(f))
                except Exception:
                    pass
        return by_state

    nov_by_state = _collect_fe_by_state(novice_data)
    exp_by_state = _collect_fe_by_state(expert_data)

    x = np.arange(len(STATES))
    width = 0.35

    nov_vals = [np.mean(nov_by_state[s]) if nov_by_state[s] else 0.0 for s in STATES]
    exp_vals = [np.mean(exp_by_state[s]) if exp_by_state[s] else 0.0 for s in STATES]
    nov_err = [np.std(nov_by_state[s]) if nov_by_state[s] else 0.0 for s in STATES]
    exp_err = [np.std(exp_by_state[s]) if exp_by_state[s] else 0.0 for s in STATES]

    nov_bars = ax1.bar(x - width/2, nov_vals, width, yerr=nov_err, capsize=5,
                      label='Novice', color=[STATE_COLORS[s] for s in STATES], alpha=0.7,
                      edgecolor='black', linewidth=1)
    exp_bars = ax1.bar(x + width/2, exp_vals, width, yerr=exp_err, capsize=5,
                      label='Expert', color=[STATE_COLORS[s] for s in STATES], alpha=0.4,
                      hatch='//', edgecolor='black', linewidth=1)

    ax1.set_ylabel('Free Energy', fontsize=12, fontweight='bold')
    ax1.set_title('Free Energy: Mean and Variability Across States', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([STATE_DISPLAY_NAMES[s] for s in STATES], fontsize=11)
    ax1.legend(fontsize=11)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.3)
    ax1.set_axisbelow(True)
    
    # ===== Panel B: Dwell Times Bar Chart =====
    ax2 = fig.add_subplot(gs[1])
    
    nov_dwells = get_dwell_times(novice_data)
    exp_dwells = get_dwell_times(expert_data)

    dt = DEFAULTS['DEFAULT_DT']

    nov_means = [np.mean(nov_dwells[s]) * dt if nov_dwells[s] else 0 for s in STATES]
    exp_means = [np.mean(exp_dwells[s]) * dt if exp_dwells[s] else 0 for s in STATES]
    nov_err = [np.std(nov_dwells[s]) * dt if nov_dwells[s] else 0 for s in STATES]
    exp_err = [np.std(exp_dwells[s]) * dt if exp_dwells[s] else 0 for s in STATES]

    nov_bars = ax2.bar(x - width/2, nov_means, width, yerr=nov_err, capsize=5, label='Novice', 
                      color=[STATE_COLORS[s] for s in STATES], alpha=0.7, edgecolor='black', linewidth=1)
    exp_bars = ax2.bar(x + width/2, exp_means, width, yerr=exp_err, capsize=5, label='Expert', 
                      color=[STATE_COLORS[s] for s in STATES], alpha=0.4, hatch='//', edgecolor='black', linewidth=1)

    ax2.set_ylabel('Average Dwell Time (Seconds)', fontsize=12, fontweight='bold')
    ax2.set_title('Average Dwell Time per State', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([STATE_DISPLAY_NAMES[s] for s in STATES], fontsize=11)
    ax2.legend()
    ax2.grid(True, axis='y', linestyle='--', alpha=0.3)
    ax2.set_axisbelow(True)
    
    plt.tight_layout()
    save_figure(fig, Path(save_path), "FE and Dwell")
    plt.close(fig)


def plot_transitions(novice_data: dict, expert_data: dict, save_path: str):
    """
    Transition dynamics comparison: 2 heatmaps side-by-side (Novice | Expert)
    EXACT style from analysis.py plot_training_summary()
    """
    set_plot_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Get transition matrices
    nov_trans = novice_data.get('transition_matrix', {})
    exp_trans = expert_data.get('transition_matrix', {})
    
    # Build matrix arrays
    nov_matrix = np.zeros((len(STATES), len(STATES)))
    exp_matrix = np.zeros((len(STATES), len(STATES)))
    
    for i, from_state in enumerate(STATES):
        for j, to_state in enumerate(STATES):
            nov_matrix[i, j] = nov_trans.get(from_state, {}).get(to_state, 0.0)
            exp_matrix[i, j] = exp_trans.get(from_state, {}).get(to_state, 0.0)
    
    # Plot novice
    im1 = ax1.imshow(nov_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    ax1.set_xticks(range(len(STATES)))
    ax1.set_yticks(range(len(STATES)))
    ax1.set_xticklabels([s.replace('_', '\n') for s in STATES], fontsize=8)
    ax1.set_yticklabels([s.replace('_', ' ').title() for s in STATES], fontsize=8)
    ax1.set_xlabel('To State')
    ax1.set_ylabel('From State')
    ax1.set_title('Novice Transition Dynamics')
    
    # Add text annotations
    for i in range(len(STATES)):
        for j in range(len(STATES)):
            value = nov_matrix[i, j]
            if value > 0.01:  # Only show significant transitions
                ax1.text(j, i, f'{value:.2f}',
                       ha='center', va='center',
                       color='white' if value > 0.5 else 'black',
                       fontsize=8)
    
    plt.colorbar(im1, ax=ax1, label='Probability')
    
    # Plot expert
    im2 = ax2.imshow(exp_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    ax2.set_xticks(range(len(STATES)))
    ax2.set_yticks(range(len(STATES)))
    ax2.set_xticklabels([s.replace('_', '\n') for s in STATES], fontsize=8)
    ax2.set_yticklabels([s.replace('_', ' ').title() for s in STATES], fontsize=8)
    ax2.set_xlabel('To State')
    ax2.set_ylabel('From State')
    ax2.set_title('Expert Transition Dynamics')
    
    # Add text annotations
    for i in range(len(STATES)):
        for j in range(len(STATES)):
            value = exp_matrix[i, j]
            if value > 0.01:  # Only show significant transitions
                ax2.text(j, i, f'{value:.2f}',
                       ha='center', va='center',
                       color='white' if value > 0.5 else 'black',
                       fontsize=8)
    
    plt.colorbar(im2, ax=ax2, label='Probability')
    
    fig.suptitle('State Transition Dynamics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, Path(save_path), "Transitions")
    plt.close(fig)
