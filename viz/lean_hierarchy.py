"""
lean_hierarchy.py

Hierarchical dynamics visualization - EXACT copy from viz/viz/plot_diagnostics.py plot_hierarchy()
Adapted to take data dict instead of loading from file.
"""

import numpy as np
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
from config import NETWORKS, THOUGHTSEEDS, STATES, DEFAULTS


def plot_hierarchy(data, save_path: str, level_name: str):
    """
    Figure 4A/4B: Hierarchical Dynamics visualization showing:
    1. Level 3: Meta-awareness (Metacognition)
    2. Level 2: Dominant Thoughtseed transitions
    3. Level 1: Network activations (DMN, VAN, DAN, FPN)
    
    EXACT COPY from viz/viz/plot_diagnostics.py plot_hierarchy()
    """
    # Check for required data
    required_fields = ['state_history', 'meta_awareness_history', 'network_activations_history', 'dominant_ts_history']
    for field in required_fields:
        if field not in data:
            print(f"ERROR: Required data '{field}' missing for hierarchy plot")
            return
    
    dt = DEFAULTS['DEFAULT_DT']
    time_steps = np.arange(len(data['state_history'])) * dt
    
    set_plot_style()
    
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 1, height_ratios=[1, 1, 1.5], figure=fig)
    
    # 1. Level 3: Meta-awareness
    ax1 = fig.add_subplot(gs[0])
    meta_awareness = data['meta_awareness_history']
    
    # Smooth the data for better visualization
    smoothed_meta = np.zeros_like(meta_awareness)
    alpha = 0.3
    smoothed_meta[0] = meta_awareness[0]
    for j in range(1, len(meta_awareness)):
        smoothed_meta[j] = (1 - alpha) * smoothed_meta[j-1] + alpha * meta_awareness[j]
    
    ax1.plot(time_steps, smoothed_meta, color='#4363d8', linewidth=2)
    ax1.fill_between(time_steps, smoothed_meta, alpha=0.2, color='#4363d8')
    ax1.set_ylabel('Meta-Awareness Level', fontsize=12)
    ax1.set_title('Level 3: Metacognition', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.5)
    # Remove axis borders/spines for a cleaner look
    for spine in ax1.spines.values():
        spine.set_visible(False)
    ax1.patch.set_visible(False)
    
    # 2. Level 2: Dominant Thoughtseed
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    thoughtseeds = THOUGHTSEEDS
    ts_mapping = {ts: i for i, ts in enumerate(thoughtseeds)}
    
    # Create categorical scatter plot
    prev_ts = None
    prev_y = None
    prev_x = None
    for i, ts in enumerate(data['dominant_ts_history']):
        if ts not in ts_mapping:
            continue
        x_val = time_steps[i]
        ax2.scatter(x_val, ts_mapping[ts], color=THOUGHTSEED_COLORS[ts], s=25, 
                   edgecolors='white', linewidth=0.5, alpha=0.8)
        curr_y = ts_mapping[ts]
        if prev_ts is not None and ts != prev_ts:
            ax2.plot([prev_x, x_val], [prev_y, curr_y], color='#aaaaaa', 
                    linestyle='-', linewidth=0.5, alpha=0.4)
        prev_ts = ts
        prev_y = curr_y
        prev_x = x_val
    
    ax2.set_yticks(range(len(thoughtseeds)))
    ax2.set_yticklabels(thoughtseeds)
    ax2.invert_yaxis()
    ax2.set_ylabel('Dominant Thoughtseed', fontsize=12)
    ax2.set_title('Level 2: Dominant Thoughtseed', fontsize=14, fontweight='bold', pad=-15)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.5)
    # Remove axis borders/spines for consistency
    for spine in ax2.spines.values():
        spine.set_visible(False)
    ax2.patch.set_visible(False)
    
    # 3. Level 1: Network Activations
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    
    for net in NETWORKS:
        # Extract data for this network
        net_acts = [n[net] for n in data['network_activations_history']]
        
        # Smooth the data
        smoothed_acts = np.zeros_like(net_acts)
        alpha = 0.3
        smoothed_acts[0] = net_acts[0]
        for j in range(1, len(net_acts)):
            smoothed_acts[j] = (1 - alpha) * smoothed_acts[j-1] + alpha * net_acts[j]
        
        ax3.plot(time_steps, smoothed_acts, label=net, color=NETWORK_COLORS[net], linewidth=2)
    
    # Highlight state transitions across all plots
    prev_state = None
    state_boundaries = []
    
    for i, state in enumerate(data['state_history']):
        if state != prev_state:
            state_boundaries.append(i)
            x_val = time_steps[i]
            ax1.axvline(x=x_val, color='#bbbbbb', linestyle='--', alpha=0.5, zorder=0)
            ax2.axvline(x=x_val, color='#bbbbbb', linestyle='--', alpha=0.5, zorder=0)
            ax3.axvline(x=x_val, color='#bbbbbb', linestyle='--', alpha=0.5, zorder=0)
            
            # Add state label to top plot (ax1) instead of bottom plot
            ax1.text(x_val, -0.05, STATE_SHORT_NAMES[state], 
                rotation=90, fontsize=9, color=STATE_COLORS[state],
                transform=ax1.get_xaxis_transform(), ha='center', va='top')
            
            prev_state = state
            
    # Add state legend 
    state_legend_elements = [
        plt.Line2D([0], [0], color=STATE_COLORS[state], lw=4, label=f"{STATE_SHORT_NAMES[state]}: {STATE_DISPLAY_NAMES[state]}")
        for state in STATES
    ]
    
    # Create a separate legend for state abbreviations below the plot
    state_legend = fig.legend(handles=state_legend_elements, loc='lower center', 
                            fontsize=10, frameon=False, ncol=4, bbox_to_anchor=(0.5, 0.01))
    
    ax3.set_xlabel('Time (seconds)', fontsize=12)
    ax3.set_ylabel('Network Activation', fontsize=12)
    ax3.set_title('Level 1: Network Dynamics', fontsize=14, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=10)
    ax3.grid(True, linestyle='--', alpha=0.5)
    for spine in ax3.spines.values():
        spine.set_visible(False)
    ax3.patch.set_visible(False)
    
    fig.suptitle(f'Hierarchical Dynamics ({level_name})', fontsize=16, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    save_figure(fig, Path(save_path), f"Hierarchy_{level_name}")
    plt.close(fig)
