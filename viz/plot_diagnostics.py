"""
plot_diagnostics.py

Generates multiple diagnostic figures:
- Figure 3A: Free Energy Bar Chart (Tail Window)
- Figure 3B: Network Radar Plots (Tail Window)
- Figure 3C: Dwell Time Bar Chart (Tail Window)
- Figure 4A: Hierarchical Dynamics - Novice (Time Series)
- Figure 4B: Hierarchical Dynamics - Expert (Time Series)
- Figure S1C: Combined Time Series (Supplementary)
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from .plotting_utils import (
    set_plot_style, load_json_data, get_tail_stats, smooth_series,
    PLOT_DIR, TAIL_STEPS, STATE_COLORS, NETWORK_COLORS, 
    STATE_SHORT_NAMES, STATE_DISPLAY_NAMES, THOUGHTSEED_COLORS
)
from config.meditation_config import STATES, THOUGHTSEEDS

NETWORKS = ['DMN', 'VAN', 'DAN', 'FPN']

def plot_hierarchy(data, save_path=None):
    """
    Figure 4A/4B: Hierarchical Dynamics visualization showing:
    1. Level 3: Meta-awareness (Metacognition)
    2. Level 2: Dominant Thoughtseed transitions
    3. Level 1: Network activations (DMN, VAN, DAN, FPN)
    """
    # Check for required data
    required_fields = ['state_history', 'meta_awareness_history', 'network_activations_history', 'dominant_ts_history']
    for field in required_fields:
        if field not in data:
            print(f"ERROR: Required data '{field}' missing for hierarchy plot")
            return
    
    time_steps = np.arange(len(data['state_history']))
    
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
    for i, ts in enumerate(data['dominant_ts_history']):
        ax2.scatter(i, ts_mapping[ts], color=THOUGHTSEED_COLORS[ts], s=25, 
                   edgecolors='white', linewidth=0.5, alpha=0.8)
    
    # Connect dots with thin lines
    prev_ts = data['dominant_ts_history'][0]
    prev_y = ts_mapping[prev_ts]
    for i in range(1, len(data['dominant_ts_history'])):
        curr_ts = data['dominant_ts_history'][i]
        curr_y = ts_mapping[curr_ts]
        if curr_ts != prev_ts:
            ax2.plot([i-1, i], [prev_y, curr_y], color='#aaaaaa', 
                    linestyle='-', linewidth=0.5, alpha=0.4)
        prev_ts = curr_ts
        prev_y = curr_y
    
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
            ax1.axvline(x=i, color='#bbbbbb', linestyle='--', alpha=0.5, zorder=0)
            ax2.axvline(x=i, color='#bbbbbb', linestyle='--', alpha=0.5, zorder=0)
            ax3.axvline(x=i, color='#bbbbbb', linestyle='--', alpha=0.5, zorder=0)
            
            # Add state label to top plot (ax1) instead of bottom plot
            ax1.text(i, -0.05, STATE_SHORT_NAMES[state], 
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
    
    ax3.set_xlabel('Timestep', fontsize=12)
    ax3.set_ylabel('Network Activation', fontsize=12)
    ax3.set_title('Level 1: Network Dynamics', fontsize=14, fontweight='bold', pad =-25)
    ax3.set_ylim(0, 1.05)
    ax3.grid(True, linestyle='--', alpha=0.5)
    
    # Create a more elegant legend
    ax3.legend(loc='upper right', frameon=False, fancybox=False, fontsize=10)
    
    # Add overall title
    experience = data.get('experience_level', 'default')
    fig.suptitle(f'          {experience.title()}', 
               fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.25, bottom=0.12) 
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        try:
            rel = os.path.relpath(str(save_path), start=os.getcwd())
        except Exception:
            rel = save_path
        logging.info("Saved Hierarchy Plot to %s", rel)
    plt.close()

def plot_time_series(novice_stats, expert_stats, save_path=None):
    """Figure S1C: Combined Time Series showing Network Activations and Free Energy dynamics (Supplementary)"""
    set_plot_style()
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.5, 1])

    axes = []

    for col, (level, stats) in enumerate([('Novice', novice_stats), ('Expert', expert_stats)]):
        # Data is already sliced to tail by get_tail_stats
        state_history = stats['state_history']
        net_hist = stats['network_activations_history']
        fe_raw = np.array(stats.get('free_energy_history', []))
        time_steps = np.arange(len(state_history))

        # 1. Network Activations
        ax1 = fig.add_subplot(gs[0, col])
        for net in NETWORKS:
            net_acts = [n.get(net, 0.0) for n in net_hist]
            ax1.plot(time_steps, net_acts, label=f"{net}", color=NETWORK_COLORS[net], linewidth=1.5)

        # State markers
        prev_state = None
        for i, state in enumerate(state_history):
            if state != prev_state:
                state_label = STATE_SHORT_NAMES.get(state, state)
                ax1.axvline(x=i, color='gray', linestyle='--', alpha=0.5)
                if i < len(state_history) - 5: # Avoid label at very end
                    ax1.text(i, 1.05, state_label, rotation=90, fontsize=9, 
                             color=STATE_COLORS.get(state, '#000000'),
                             transform=ax1.get_xaxis_transform(), ha='center')
                prev_state = state

        ax1.set_ylim(0, 1.15)
        ax1.set_title(f"Network Activations ({level})", fontsize=14, fontweight='bold')
        ax1.set_ylabel('Activation', fontsize=12)
        if col == 1: ax1.legend(loc='upper right', framealpha=0.9, fancybox=True, fontsize=10)
        ax1.grid(True, alpha=0.3)

        # 2. Free Energy
        ax2 = fig.add_subplot(gs[1, col], sharex=ax1)
        fe_smooth = smooth_series(fe_raw)
        
        ax2.plot(time_steps, fe_smooth, color='#E74C3C', label="Free Energy (smoothed)", linewidth=2)
        if len(fe_raw) > 0:
            ax2.plot(time_steps, fe_raw, color='#E74C3C', alpha=0.25, linewidth=1)

        # State background shading
        prev_state = state_history[0] if len(state_history) > 0 else None
        start_idx = 0
        for i, state in enumerate(state_history):
            if state != prev_state or i == len(state_history)-1:
                ax2.axvspan(start_idx, i, alpha=0.1, color=STATE_COLORS.get(prev_state, '#cccccc'))
                start_idx = i
                prev_state = state

        ax2.set_title(f"Free Energy ({level})", fontsize=14, fontweight='bold')
        ax2.set_xlabel('Timestep (Tail Window)', fontsize=12)
        ax2.set_ylabel('Free Energy', fontsize=12)
        ax2.grid(True, alpha=0.3)
        axes.append(ax2)

    # Unify Y-axis for Free Energy
    y_mins = [ax.get_ylim()[0] for ax in axes]
    y_maxs = [ax.get_ylim()[1] for ax in axes]
    for ax in axes:
        ax.set_ylim(min(y_mins), max(y_maxs))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        try:
            rel = os.path.relpath(str(save_path), start=os.getcwd())
        except Exception:
            rel = save_path
        logging.info("Saved Time Series to %s", rel)
    plt.close()

def plot_cognitive_hierarchy(novice_stats, expert_stats, save_path=None):
    """Figure S1D: Combined Time Series showing 4-Level Stack (Cognitive Hierarchy)
    Rows:
    1. Network Activations (Level 1)
    2. Thoughtseed Activations (Level 2)
    3. Meta-awareness (Level 3)
    4. Free Energy (Diagnostics)
    """
    set_plot_style()
    # 4 rows, 2 columns (Novice vs Expert)
    fig = plt.figure(figsize=(18, 16))
    height_ratios = [1, 1, 1, 1] 
    gs = GridSpec(4, 2, figure=fig, height_ratios=height_ratios)

    axes = []

    for col, (level, stats) in enumerate([('Novice', novice_stats), ('Expert', expert_stats)]):
        # Data is already sliced to tail by get_tail_stats
        state_history = stats['state_history']
        net_hist = stats['network_activations_history']
        ts_hist = stats.get('activations_history', []) # Need activations_history for Level 2
        meta_hist = stats.get('meta_awareness_history', []) # Level 3
        fe_raw = np.array(stats.get('free_energy_history', []))
        
        time_steps = np.arange(len(state_history))

        # --- ROW 1: Diagnostics (Free Energy) ---
        ax1 = fig.add_subplot(gs[0, col])
        fe_smooth = smooth_series(fe_raw)
        
        ax1.plot(time_steps, fe_smooth, color='#E74C3C', label="Free Energy (smoothed)", linewidth=2)
        if len(fe_raw) > 0:
            ax1.plot(time_steps, fe_raw, color='#E74C3C', alpha=0.25, linewidth=1)

        ax1.set_title(f"System Diagnostics: Variational Free Energy ({level})", fontsize=14, fontweight='bold')
        ax1.set_ylabel('Free Energy', fontsize=11)
        ax1.grid(True, alpha=0.3)
        # Hide x-labels for top rows
        ax1.tick_params(labelbottom=False)
        axes.append(ax1) # For unify ylim

        # --- ROW 2: Level 3 (Meta-awareness) ---
        ax2 = fig.add_subplot(gs[1, col], sharex=ax1)
        if len(meta_hist) > 0:
            ax2.plot(time_steps, meta_hist, color='#4363d8', linewidth=2, label='Meta-awareness')
            ax2.fill_between(time_steps, meta_hist, alpha=0.1, color='#4363d8')
        
        ax2.set_ylim(0, 1.05)
        ax2.set_title(f"Level 3: Metacognition ({level})", fontsize=14, fontweight='bold')
        ax2.set_ylabel('Meta-Awareness Level', fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(labelbottom=False)

        # --- ROW 3: Level 2 (Thoughtseed Activations) ---
        ax3 = fig.add_subplot(gs[2, col], sharex=ax1)
        # Thoughtseeds need to be plotted as lines
        if ts_hist and len(ts_hist) > 0:
            # Transpose list of arrays to array of lists
            ts_array = np.array(ts_hist)
            for i, ts in enumerate(THOUGHTSEEDS):
                ts_vals = ts_array[:, i]
                ax3.plot(time_steps, ts_vals, label=ts.replace('_', ' ').title(), 
                         color=THOUGHTSEED_COLORS[ts], linewidth=1.5, alpha=0.8)
        
        ax3.set_ylim(0, 1.15)
        ax3.set_title(f"Level 2: Thoughtseed Dynamics ({level})", fontsize=14, fontweight='bold')
        ax3.set_ylabel('Activation', fontsize=11)
        if col == 1: ax3.legend(loc='upper right', framealpha=0.9, fancybox=True, fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(labelbottom=False)

        # --- ROW 4: Level 1 (Network Activations) ---
        ax4 = fig.add_subplot(gs[3, col], sharex=ax1)
        for net in NETWORKS:
            net_acts = [n.get(net, 0.0) for n in net_hist]
            ax4.plot(time_steps, net_acts, label=f"{net}", color=NETWORK_COLORS[net], linewidth=1.5)
        
        ax4.set_ylim(0, 1.15)
        ax4.set_title(f"Level 1: Network Dynamics ({level})", fontsize=14, fontweight='bold')
        ax4.set_xlabel('Timestep (Tail Window)', fontsize=12)
        ax4.set_ylabel('Activation', fontsize=11)
        if col == 1: ax4.legend(loc='upper right', framealpha=0.9, fancybox=True, fontsize=10)
        ax4.grid(True, alpha=0.3)

        # --- ADD STATE MARKERS TO ALL ROWS ---
        # We add vertical lines and background shading to all 4 subplots
        subplots = [ax1, ax2, ax3, ax4]
        
        prev_state = None
        for i, state in enumerate(state_history):
            if state != prev_state:
                state_label = STATE_SHORT_NAMES.get(state, state)
                for ax in subplots:
                    ax.axvline(x=i, color='gray', linestyle='--', alpha=0.3)
                
                # Only add text label to top plot
                if i < len(state_history) - 5: 
                    ax1.text(i, 1.15, state_label, rotation=90, fontsize=9, 
                             color=STATE_COLORS.get(state, '#000000'),
                             transform=ax1.get_xaxis_transform(), ha='center')
                prev_state = state

        # Add background shading
        prev_state = state_history[0] if len(state_history) > 0 else None
        start_idx = 0
        for i, state in enumerate(state_history):
            if state != prev_state or i == len(state_history)-1:
                color = STATE_COLORS.get(prev_state, '#cccccc')
                for ax in subplots:
                    ax.axvspan(start_idx, i, alpha=0.08, color=color)
                start_idx = i
                prev_state = state

    # Unify Y-axis for Free Energy
    y_mins = [ax.get_ylim()[0] for ax in axes]
    y_maxs = [ax.get_ylim()[1] for ax in axes]
    for ax in axes:
        ax.set_ylim(min(y_mins), max(y_maxs))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        try:
            rel = os.path.relpath(str(save_path), start=os.getcwd())
        except Exception:
            rel = save_path
        logging.info("Saved Time Series to %s", rel)
    plt.close()

def plot_free_energy_bar(novice_stats, expert_stats, save_path=None):
    """Figure 3A: Free Energy distribution per state (novice vs expert bar chart with error bars)."""
    set_plot_style()

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

    nov_by_state = _collect_fe_by_state(novice_stats)
    exp_by_state = _collect_fe_by_state(expert_stats)

    # Prepare data for bar chart (means + std)
    x = np.arange(len(STATES))
    width = 0.35

    nov_vals = [np.mean(nov_by_state[s]) if nov_by_state[s] else 0.0 for s in STATES]
    exp_vals = [np.mean(exp_by_state[s]) if exp_by_state[s] else 0.0 for s in STATES]
    nov_err = [np.std(nov_by_state[s]) if nov_by_state[s] else 0.0 for s in STATES]
    exp_err = [np.std(exp_by_state[s]) if exp_by_state[s] else 0.0 for s in STATES]

    fig, ax = plt.subplots(figsize=(10, 6))
    nov_bars = ax.bar(x - width/2, nov_vals, width, yerr=nov_err, capsize=5,
                      label='Novice', color=[STATE_COLORS[s] for s in STATES], alpha=0.7,
                      edgecolor='black', linewidth=1)
    exp_bars = ax.bar(x + width/2, exp_vals, width, yerr=exp_err, capsize=5,
                      label='Expert', color=[STATE_COLORS[s] for s in STATES], alpha=0.4,
                      hatch='//', edgecolor='black', linewidth=1)

    ax.set_ylabel('Free Energy', fontsize=12, fontweight='bold')
    ax.set_title('Free Energy: Mean and Variability Across States', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([STATE_DISPLAY_NAMES[s] for s in STATES], fontsize=11)

    ax.legend(fontsize=11)
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        try:
            rel = os.path.relpath(str(save_path), start=os.getcwd())
        except Exception:
            rel = save_path
        logging.info("Saved FE Bar to %s", rel)
    plt.close()

def plot_network_radar(novice_ts, expert_ts, save_path=None):
    """Figure 3B: Network Activation Profiles (radar plots showing network expectations for each state)"""
    set_plot_style()
    def _extract_net(ts_data):
        # Try to find network expectations
        if "network_expectations" in ts_data: # Sometimes in active_inference_params
             return ts_data["network_expectations"]
        # Or in thoughtseed params if structured that way
        return ts_data.get("network_expectations", {}) # Fallback

    # Network expectations are typically found in active_inference_params (ai_data)
    # We handle both locations for compatibility.
    nov_nets = _extract_net(novice_ts)
    exp_nets = _extract_net(expert_ts)

    fig = plt.figure(figsize=(14, 12))
    fig.suptitle('Network Activation Profiles', fontsize=18, fontweight='bold')

    angles = np.linspace(0, 2*np.pi, len(NETWORKS), endpoint=False).tolist()
    angles += angles[:1]

    for i, state in enumerate(STATES):
        ax = fig.add_subplot(2, 2, i+1, polar=True)

        nov_state = nov_nets.get(state, {})
        exp_state = exp_nets.get(state, {})

        nov_vals = [float(nov_state.get(net, 0.0)) for net in NETWORKS]
        exp_vals = [float(exp_state.get(net, 0.0)) for net in NETWORKS]
        nov_vals += nov_vals[:1]
        exp_vals += exp_vals[:1]

        # State-coloured lines/fills (Novice dashed, Expert solid) — restored previous scheme
        ax.plot(angles, nov_vals, color=STATE_COLORS[state], linewidth=2.6, linestyle='--', label="Novice")
        ax.fill(angles, nov_vals, color=STATE_COLORS[state], alpha=0.22)
        ax.plot(angles, exp_vals, color=STATE_COLORS[state], linewidth=2.8, linestyle='-', label="Expert")
        ax.fill(angles, exp_vals, color=STATE_COLORS[state], alpha=0.16)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(NETWORKS, fontsize=13, fontweight='bold')
        # Improve radial label size (y-axis labels)
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
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        try:
            rel = os.path.relpath(str(save_path), start=os.getcwd())
        except Exception:
            rel = save_path
        logging.info("Saved Radar to %s", rel)
    plt.close()

def get_dwell_times(stats):
    state_history = stats.get("state_history", [])
    if not state_history: return {s: [] for s in STATES}
    
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
    # Only append the last one if it's a complete dwell (which we can't know for sure in a tail slice)
    # But for visualization, it's better to include it than have nothing if the tail is all one state.
    dwells[current_state].append(count)
    return dwells

def plot_dwell_times(novice_stats, expert_stats, save_path=None):
    """Figure 3C: Average Dwell Time per State (bar chart comparing novice vs expert)"""
    set_plot_style()
    
    nov_dwells = get_dwell_times(novice_stats)
    exp_dwells = get_dwell_times(expert_stats)

    nov_means = [np.mean(nov_dwells[s]) if nov_dwells[s] else 0 for s in STATES]
    exp_means = [np.mean(exp_dwells[s]) if exp_dwells[s] else 0 for s in STATES]
    nov_err = [np.std(nov_dwells[s]) if nov_dwells[s] else 0 for s in STATES]
    exp_err = [np.std(exp_dwells[s]) if exp_dwells[s] else 0 for s in STATES]

    x = np.arange(len(STATES))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    nov_bars = ax.bar(x - width/2, nov_means, width, yerr=nov_err, capsize=5, label='Novice', 
                      color=[STATE_COLORS[s] for s in STATES], alpha=0.7, edgecolor='black', linewidth=1)
    exp_bars = ax.bar(x + width/2, exp_means, width, yerr=exp_err, capsize=5, label='Expert', 
                      color=[STATE_COLORS[s] for s in STATES], alpha=0.4, hatch='//', edgecolor='black', linewidth=1)

    ax.set_ylabel('Average Dwell Time (Timesteps)', fontsize=12, fontweight='bold')
    ax.set_title('Average Dwell Time per State', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([STATE_DISPLAY_NAMES[s] for s in STATES], fontsize=11)
    ax.legend()

    # Simple bar plot: no numeric annotations on bars (publication style)

    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        try:
            rel = os.path.relpath(str(save_path), start=os.getcwd())
        except Exception:
            rel = save_path
        logging.info("Saved Dwell Times to %s", rel)
    plt.close()

if __name__ == "__main__":
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    # Load Data
    nov_ts, nov_ai, nov_stats = load_json_data("novice")
    exp_ts, exp_ai, exp_stats = load_json_data("expert")
    
    # Slice to Tail
    nov_tail = get_tail_stats(nov_stats)
    exp_tail = get_tail_stats(exp_stats)
    
    # Inject experience level for plotting titles
    nov_tail['experience_level'] = 'novice'
    exp_tail['experience_level'] = 'expert'
    
    # Generate Panels
    # Panel A: Cognitive Hierarchy (4-Level Stack) - replaces old S1C
    plot_cognitive_hierarchy(nov_tail, exp_tail, os.path.join(PLOT_DIR, "FigS1C_Hierarchy_TimeSeries.png"))

    # Panel A2: Removed (redundant)
    
    # Panel B: Free Energy Bar (Use Full Stats for accurate distribution)
    plot_free_energy_bar(nov_stats, exp_stats, os.path.join(PLOT_DIR, "Fig3A_FreeEnergy.png"))
    
    # Panel C: Radar (Use AI params)
    plot_network_radar(nov_ai, exp_ai, os.path.join(PLOT_DIR, "Fig3B_Radar.png"))
    
    # Panel D: Dwell Times (Use Full Stats for accurate means/variance)
    plot_dwell_times(nov_stats, exp_stats, os.path.join(PLOT_DIR, "Fig3C_DwellTime.png"))
    
    # Generate Hierarchy Plots (Use Tail for visibility)
    plot_hierarchy(nov_tail, os.path.join(PLOT_DIR, "Fig4A_Hierarchy_Novice.png"))
    plot_hierarchy(exp_tail, os.path.join(PLOT_DIR, "Fig4B_Hierarchy_Expert.png"))
