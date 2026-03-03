"""Hierarchical dynamics visualization."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

from viz.plotting_utils import (
    STATE_COLORS,
    STATE_SHORT_NAMES,
    STATE_DISPLAY_NAMES,
    NETWORK_COLORS,
    THOUGHTSEED_COLORS,
    save_figure,
    set_plot_style,
)
from utils.config import NETWORKS, THOUGHTSEEDS, STATES


def plot_hierarchy_continuous(data, save_path: str, level_name: str):
    """Figure 4A/4B: hierarchical dynamics with continuous thoughtseed traces."""
    required_fields = [
        'state_history',
        'meta_awareness_history',
        'network_activations_history',
        'thoughtseed_activations_history',
    ]
    for field in required_fields:
        if field not in data:
            print(f"ERROR: Required data '{field}' missing for hierarchy plot")
            return

    n_steps = len(data['state_history'])
    time_steps = np.arange(n_steps) if n_steps > 0 else np.array([0.0])

    set_plot_style()

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 1, height_ratios=[1, 1.2, 1.5], figure=fig)

    # 1. Level 3: Meta-awareness and clarity (state confidence)
    ax1 = fig.add_subplot(gs[0])
    meta_awareness = data['meta_awareness_history']
    ax1.plot(time_steps, meta_awareness, color='#4363d8', linewidth=2, label='Meta-awareness')
    ax1.fill_between(time_steps, meta_awareness, alpha=0.2, color='#4363d8')
    clarity = data.get('state_confidence_history')
    if clarity is not None and len(clarity) == n_steps:
        clarity_smooth = pd.Series(clarity, dtype=float).ewm(alpha=0.2, adjust=False).mean().to_numpy()
        ax1.plot(time_steps, clarity_smooth, color='#0d7a4a', linewidth=1.2, linestyle='--', alpha=0.85, label='Clarity (state confidence)')
    ax1.set_ylabel('Level', fontsize=12, fontweight='bold')
    ax1.set_title('Level 3: Metacognition', fontsize=14, fontweight='bold', loc='left')
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc='upper right', fontsize=9, frameon=False)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax1.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    for spine in ax1.spines.values():
        spine.set_visible(False)
    ax1.patch.set_visible(False)

    # 2. Level 2: Thoughtseed trajectories (continuous)
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ts_history = data['thoughtseed_activations_history']
    ts_alpha = 0.2  # smoother thoughtseed traces (visual only)
    for i, ts in enumerate(THOUGHTSEEDS):
        ts_vals = [row[i] for row in ts_history]
        smoothed = pd.Series(ts_vals, dtype=float).ewm(alpha=ts_alpha, adjust=False).mean().to_numpy()
        ax2.plot(
            time_steps,
            smoothed,
            label=ts,
            color=THOUGHTSEED_COLORS[ts],
            linewidth=1.8,
        )
    ax2.set_ylabel('Thoughtseed Activation', fontsize=12, fontweight='bold')
    ax2.set_title('Level 2: Thoughtseed Trajectories', fontsize=14, fontweight='bold', pad=8, loc='left')
    ax2.set_ylim(0.0, 1.0)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax2.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    for spine in ax2.spines.values():
        spine.set_visible(False)
    ax2.patch.set_visible(False)
    ax2.legend(loc='upper right', fontsize=9, ncol=2, frameon=False)

    # 3. Level 1: Network Activations
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    for net in NETWORKS:
        net_acts = [n[net] for n in data['network_activations_history']]
        alpha = 0.3
        smoothed_acts = pd.Series(net_acts, dtype=float).ewm(alpha=alpha, adjust=False).mean().to_numpy()
        ax3.plot(time_steps, smoothed_acts, label=net, color=NETWORK_COLORS[net], linewidth=2)

    # Add L1 state labels at transition boundaries (no vertical lines)
    prev_state = None
    for i, state in enumerate(data['state_history']):
        if state != prev_state:
            x_val = time_steps[i]
            ax3.text(
                x_val,
                -0.08,
                STATE_SHORT_NAMES[state],
                rotation=90,
                fontsize=9,
                color=STATE_COLORS[state],
                transform=ax3.get_xaxis_transform(),
                ha='center',
                va='top',
            )
            prev_state = state

    state_legend_elements = [
        plt.Line2D([0], [0], color=STATE_COLORS[state], lw=4, label=f"{STATE_SHORT_NAMES[state]}: {STATE_DISPLAY_NAMES[state]}")
        for state in STATES
    ]
    fig.legend(handles=state_legend_elements, loc='lower center',
               fontsize=10, frameon=False, ncol=4, bbox_to_anchor=(0.5, 0.02))

    ax3.set_xlabel('Time (timesteps)', fontsize=12, fontweight='bold', labelpad=12)
    ax3.set_ylabel('Network Activation', fontsize=12, fontweight='bold')
    ax3.set_ylim(0.0, 1.0)
    ax3.set_title('Level 1: Network Dynamics', fontsize=14, fontweight='bold', loc='left')
    ax3.legend(loc='upper right', fontsize=10)
    ax3.grid(True, linestyle='--', alpha=0.5)
    for spine in ax3.spines.values():
        spine.set_visible(False)
    ax3.patch.set_visible(False)

    fig.suptitle(f'Hierarchical Dynamics ({level_name})',
                 fontsize=16, fontweight='bold', x=0.5, ha='center')

    plt.tight_layout(rect=[0, 0.08, 1, 0.99])
    save_figure(fig, Path(save_path), f"Hierarchy_Continuous_{level_name}")
    plt.close(fig)
