"""Analysis utility functions for lean meditation model."""

from typing import Dict, List
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from utils.config import PLOT_STEPS, STATES
from viz.plotting_utils import (
    STATE_COLORS,
    STATE_DISPLAY_NAMES,
    save_figure,
    set_plot_style,
)

__all__ = [
    "PLOT_STEPS",
    "get_tail_window",
    "prepare_tail_data",
    "get_dwell_run_lengths",
    "get_dwell_times",
    "get_transition_matrix",
    "plot_fe_and_dwell",
    "plot_transitions",
    "compute_tail_statistics",
    "compute_network_profiles",
    "compute_thoughtseed_means",
]


def prepare_tail_data(
    results: Dict,
    states: List[str],
    networks: List[str],
    thoughtseeds: List[str],
    tail_steps: int = PLOT_STEPS,
) -> Dict:
    """Single entry point: prepare tail-window data with all derived stats for plotting.

    Returns a dict suitable for all plot functions: tail-sliced histories plus
    network_profiles_mean, thoughtseed_means_per_state, dwell_times, transition_matrix,
    and tail_start.
    """
    tail = get_tail_window(results, tail_steps)
    tail["tail_start"] = max(0, len(results.get("state_history", [])) - tail_steps)
    tail["network_profiles_mean"] = compute_network_profiles(results, states, networks, tail_steps)
    tail["thoughtseed_means_per_state"] = compute_thoughtseed_means(
        results, states, thoughtseeds, tail_steps
    )
    stats = compute_tail_statistics(results, states, tail_steps)
    tail["dwell_times"] = stats["dwell_times"]
    tail["transition_matrix"] = stats["transition_matrix"]
    return tail


def get_tail_window(results: Dict, tail_steps: int = PLOT_STEPS) -> Dict:
    """Return a view of results with all time-series fields sliced to the tail window.

    Slices each list in-place (no deepcopy) — callers must not mutate the returned lists.
    """
    time_series_keys = [
        'state_history',
        'free_energy_history',
        'loss_history',
        'meta_awareness_history',
        'network_activations_history',
        'thoughtseed_activations_history',
        'thoughtseed_prior_activations_history',
    ]
    tail_data = dict(results)  # shallow copy — only replaces sliced keys
    for key in time_series_keys:
        series = tail_data.get(key)
        if series:
            tail_data[key] = series[-tail_steps:]
    return tail_data


# ---------------------------------------------------------------------------
# Dwell and transition logic (single source of truth)
# ---------------------------------------------------------------------------

def get_dwell_run_lengths(state_history: List[str], states: List[str]) -> Dict[str, List[int]]:
    """Extract run-length sequences per state from state history.

    Returns:
        {state: [run_length_1, run_length_2, ...]} for bar plots with error bars.
    """
    from itertools import groupby
    dwells = {s: [] for s in states}
    for k, g in groupby(state_history):
        if k in dwells:
            dwells[k].append(sum(1 for _ in g))
    return dwells


def _build_transition_matrix(
    transitions: List[Dict],
    state_history: List[str],
    states: List[str],
    tail_start: int = None,
) -> Dict[str, Dict[str, float]]:
    """Build normalized transition matrix from transitions or state history."""
    trans_matrix = {fs: {ts: 0.0 for ts in states} for fs in states}
    if transitions:
        for tr in transitions:
            t, fs, ts = tr.get("timestamp"), tr.get('from'), tr.get('to')
            if (tail_start is None or (t is not None and t >= tail_start)) and fs in trans_matrix and ts in trans_matrix[fs]:
                trans_matrix[fs][ts] += 1
    elif state_history:
        for fs, ts in zip(state_history, state_history[1:]):
            if fs != ts and fs in trans_matrix and ts in trans_matrix[fs]:
                trans_matrix[fs][ts] += 1

    for fs in states:
        total = sum(trans_matrix[fs].values())
        if total > 0:
            trans_matrix[fs] = {ts: c / total for ts, c in trans_matrix[fs].items()}
    return trans_matrix


def get_dwell_times(stats: Dict, states: List[str] = None) -> Dict[str, List[int]]:
    """Extract dwell run-lengths from stats dict (state_history).

    Args:
        stats: Dict with 'state_history' (e.g. tail-windowed data).
        states: State names; if None, uses keys from empty-result template.

    Returns:
        {state: [run_length, ...]} for bar plots with std/error bars.
    """
    from utils.config import STATES
    states = states or STATES
    state_history = stats.get("state_history", [])
    return get_dwell_run_lengths(state_history, states)


def get_transition_matrix(stats: Dict, states: List[str] = None) -> Dict[str, Dict[str, float]]:
    """Compute transition matrix from tail-window transitions (state changes only).

    Args:
        stats: Dict with 'state_history', 'transitions', optional 'tail_start'.
        states: State names; if None, uses utils.config.STATES.
    """
    from utils.config import STATES
    states = states or STATES
    transitions = stats.get("transitions", [])
    state_history = stats.get("state_history", [])
    tail_start = stats.get("tail_start")
    return _build_transition_matrix(transitions, state_history, states, tail_start)


# ---------------------------------------------------------------------------
# Shared aggregation helper
# ---------------------------------------------------------------------------

def _state_conditional_means(
    value_history: list,
    state_history: list,
    states: List[str],
    n_features: int,
    extractor,
) -> Dict[str, list]:
    """Compute per-state mean feature vectors from aligned history lists.

    Args:
        value_history: List of per-timestep values (dict or array-like).
        state_history:  Parallel list of state labels.
        states:         List of state names to aggregate over.
        n_features:     Length of the feature vector for each timestep.
        extractor:      Callable(value) -> list/array of length n_features.

    Returns:
        {state: [mean_f0, mean_f1, ...]}
    """
    results = {}
    for state in states:
        indices = [i for i, s in enumerate(state_history) if s == state]
        if not indices:
            results[state] = [0.0] * n_features
            continue
        matrix = np.array([extractor(value_history[i]) for i in indices])
        results[state] = np.mean(matrix, axis=0).tolist()
    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_network_profiles(results: Dict, states: List[str], networks: List[str],
                             tail_steps: int = PLOT_STEPS) -> Dict:
    """Compute mean network activation profiles per state from tail window.

    Returns:
        {state: {network: mean_activation}}
    """
    tail_data = get_tail_window(results, tail_steps)
    means = _state_conditional_means(
        value_history=tail_data['network_activations_history'],
        state_history=tail_data['state_history'],
        states=states,
        n_features=len(networks),
        extractor=lambda v: [v.get(net, 0.0) for net in networks],
    )
    return {state: dict(zip(networks, vec)) for state, vec in means.items()}


def compute_thoughtseed_means(results: Dict, states: List[str], thoughtseeds: List[str],
                              tail_steps: int = PLOT_STEPS) -> Dict:
    """Compute mean thoughtseed activation vectors per state from tail window.

    Returns:
        {state: [mean_ts1, ..., mean_ts5]}
    """
    tail_data = get_tail_window(results, tail_steps)
    return _state_conditional_means(
        value_history=tail_data['thoughtseed_activations_history'],
        state_history=tail_data['state_history'],
        states=states,
        n_features=len(thoughtseeds),
        extractor=lambda v: list(v),
    )


def compute_tail_statistics(results: Dict, states: List[str],
                            tail_steps: int = PLOT_STEPS) -> Dict:
    """Compute dwell times and transition matrix from tail window.

    Returns:
        {'dwell_times': {state: mean}, 'transition_matrix': {from: {to: prob}}}
    """
    tail_data = get_tail_window(results, tail_steps)
    state_sequence = tail_data['state_history']

    dwell_run_lengths = get_dwell_run_lengths(state_sequence, states)
    avg_dwell = {
        state: float(np.mean(times)) if times else 0.0
        for state, times in dwell_run_lengths.items()
    }

    tail_start = max(0, len(results.get('state_history', [])) - tail_steps)
    trans_matrix = _build_transition_matrix(
        results.get('transitions', []),
        state_sequence,
        states,
        tail_start=tail_start,
    )

    return {'dwell_times': avg_dwell, 'transition_matrix': trans_matrix}


# ---------------------------------------------------------------------------
# Plot helpers (centralized)
# ---------------------------------------------------------------------------

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
    if t_stat > 3.291:
        return "***"  # p < 0.001
    if t_stat > 2.576:
        return "**"   # p < 0.01
    if t_stat > 1.960:
        return "*"    # p < 0.05
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

    nov_dwells = get_dwell_times(novice_data, STATES)
    exp_dwells = get_dwell_times(expert_data, STATES)

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
    ax2.bar(x - width / 2, nov_means, width, yerr=nov_stds, capsize=4, label='Novice',
            color=[STATE_COLORS[s] for s in STATES], alpha=0.4, hatch='//',
            edgecolor='black', linewidth=1, error_kw={'alpha': 0.5})
    ax2.bar(x + width / 2, exp_means, width, yerr=exp_stds, capsize=4, label='Expert',
            color=[STATE_COLORS[s] for s in STATES], alpha=0.7,
            edgecolor='black', linewidth=1, error_kw={'alpha': 0.5})

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

    # Get transition matrices from history
    nov_trans = get_transition_matrix(novice_data, STATES)
    exp_trans = get_transition_matrix(expert_data, STATES)

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
    ax1.set_xticklabels([STATE_DISPLAY_NAMES[s].replace(' ', '\n') for s in STATES], fontsize=8)
    ax1.set_yticklabels([STATE_DISPLAY_NAMES[s] for s in STATES], fontsize=8)
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
    ax2.set_xticklabels([STATE_DISPLAY_NAMES[s].replace(' ', '\n') for s in STATES], fontsize=8)
    ax2.set_yticklabels([STATE_DISPLAY_NAMES[s] for s in STATES], fontsize=8)
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
