"""Diagnostic comparison plots.

Dwell time comparison (single-panel bar plot).
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
from utils.config import STATES


def get_dwell_times(stats):
    """Extract dwell times from state history."""
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


def _calc_significance(mean1, std1, n1, mean2, std2, n2):
    """Calculate statistical significance using Welch's t-test approximation."""
    if n1 <= 1 or n2 <= 1 or (std1 == 0 and std2 == 0):
        return ""
    
    # Welch's t-test
    se1 = (std1 ** 2) / n1
    se2 = (std2 ** 2) / n2
    se_diff = np.sqrt(se1 + se2)
    
    if se_diff == 0:
        return ""
        
    t_stat = abs(mean1 - mean2) / se_diff
    
    # Significance thresholds
    if t_stat > 3.291: return "***"  # p < 0.001
    if t_stat > 2.576: return "**"   # p < 0.01
    if t_stat > 1.960: return "*"    # p < 0.05
    return ""


def _add_significance_bracket(ax, x_start, x_end, y_start, y_end, text, color='black'):
    """Draw a bracket connecting two bars with significance text above."""
    if not text:
        return 0.0
        
    # Calculate dimensions
    span = max(abs(y_end - y_start), max(y_start, y_end, 1.0))
    h_bracket = 0.05 * span
    
    y_top = max(y_start, y_end) + h_bracket * 2
    y_text = y_top + h_bracket * 0.5
    
    # Draw bracket line: down-across-down
    line_x = [x_start, x_start, x_end, x_end]
    line_y = [y_start + h_bracket, y_top, y_top, y_end + h_bracket]
    
    ax.plot(line_x, line_y, lw=1.0, c=color)
    
    # Add text
    ax.text((x_start + x_end) * 0.5, y_text, text, ha='center', va='bottom', 
            color=color, fontweight='bold', fontsize=10)
            
    return y_text + h_bracket * 2  # Return total height used


def plot_fe_and_dwell(novice_data: dict, expert_data: dict, save_path: str):
    """
    Dwell time comparison with significance brackets.
    """
    set_plot_style()

    fig, ax2 = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)

    x = np.arange(len(STATES))
    width = 0.28

    nov_dwells = get_dwell_times(novice_data)
    exp_dwells = get_dwell_times(expert_data)

    # Dwell times already measured in timesteps

    # Calculate stats
    nov_means, exp_means = [], []
    nov_stds, exp_stds = [], []
    nov_ns, exp_ns = [], []

    for s in STATES:
        # Novice
        vals = nov_dwells[s]
        nov_means.append(np.mean(vals) if vals else 0)
        nov_stds.append(np.std(vals) if vals else 0)
        nov_ns.append(len(vals))

        # Expert
        vals = exp_dwells[s]
        exp_means.append(np.mean(vals) if vals else 0)
        exp_stds.append(np.std(vals) if vals else 0)
        exp_ns.append(len(vals))

    # Plot Bars
    ax2.bar(x - width/2, nov_means, width, yerr=nov_stds, capsize=4, label='Novice',
            color=[STATE_COLORS[s] for s in STATES], alpha=0.4, hatch='//', edgecolor='black', linewidth=1, error_kw={'alpha': 0.5})
    ax2.bar(x + width/2, exp_means, width, yerr=exp_stds, capsize=4, label='Expert',
            color=[STATE_COLORS[s] for s in STATES], alpha=0.7, edgecolor='black', linewidth=1, error_kw={'alpha': 0.5})

    # Add brackets
    max_y_used = 0
    current_max_y = max([m + s for m, s in zip(nov_means + exp_means, nov_stds + exp_stds)]) if nov_means else 1.0

    for i in range(len(STATES)):
        sig = _calc_significance(nov_means[i], nov_stds[i], nov_ns[i],
                                 exp_means[i], exp_stds[i], exp_ns[i])

        if sig:
            # Bar tops
            y1 = nov_means[i] + nov_stds[i]
            y2 = exp_means[i] + exp_stds[i]

            top_y = _add_significance_bracket(
                ax2,
                x[i] - width / 2,
                x[i] + width / 2,
                y1,
                y2,
                sig,
            )
            max_y_used = max(max_y_used, top_y)

    # Auto-scale Y with headroom
    top_limit = max(current_max_y, max_y_used) * 1.15
    ax2.set_ylim(0, top_limit)

    ax2.set_ylabel('Average Dwell Time (Timesteps)', fontsize=12, fontweight='bold')
    ax2.set_title('Dwell Times', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([STATE_DISPLAY_NAMES[s] for s in STATES], fontsize=11)
    ax2.legend()
    ax2.grid(True, axis='y', linestyle='--', alpha=0.3)
    ax2.set_axisbelow(True)

    save_figure(fig, Path(save_path), "Fig2B_Dwell")
    plt.close(fig)


def plot_transitions(novice_data: dict, expert_data: dict, save_path: str):
    """
    Transition dynamics comparison: 2 heatmaps side-by-side (Novice | Expert)
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
    ax1.set_xlabel('To State', fontweight='bold')
    ax1.set_ylabel('From State', fontweight='bold')
    ax1.set_title('Novice Transition Dynamics', fontsize=13, fontweight='bold')
    
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
    ax2.set_xlabel('To State', fontweight='bold')
    ax2.set_ylabel('From State', fontweight='bold')
    ax2.set_title('Expert Transition Dynamics', fontsize=13, fontweight='bold')
    
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
