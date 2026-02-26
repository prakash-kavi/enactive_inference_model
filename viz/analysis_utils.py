"""Analysis utility functions for lean meditation model."""

from typing import Dict, List
import numpy as np

TAIL_STEPS = 2000  # Last 2000 timesteps for converged behavior analysis


def get_tail_window(results: Dict, tail_steps: int = TAIL_STEPS) -> Dict:
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
        'dominant_ts_history',
        'action_errors_history',
        'efe_prag_history',
        'efe_epi_history',
    ]
    tail_data = dict(results)  # shallow copy — only replaces sliced keys
    for key in time_series_keys:
        series = tail_data.get(key)
        if series:
            tail_data[key] = series[-tail_steps:]
    return tail_data


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
                             tail_steps: int = TAIL_STEPS) -> Dict:
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
                              tail_steps: int = TAIL_STEPS) -> Dict:
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
                            tail_steps: int = TAIL_STEPS) -> Dict:
    """Compute dwell times and transition matrix from tail window.

    Returns:
        {'dwell_times': {state: mean}, 'transition_matrix': {from: {to: prob}}}
    """
    tail_data = get_tail_window(results, tail_steps)
    state_sequence = tail_data['state_history']

    # Dwell times
    dwell_times = {state: [] for state in states}
    if state_sequence:
        current = state_sequence[0]
        count = 1
        for s in state_sequence[1:]:
            if s == current:
                count += 1
            else:
                dwell_times[current].append(count)
                current = s
                count = 1
        dwell_times[current].append(count)

    avg_dwell = {
        state: float(np.mean(times)) if times else 0.0
        for state, times in dwell_times.items()
    }

    # Transition matrix (actual state changes only)
    trans_matrix = {fs: {ts: 0 for ts in states} for fs in states}
    transitions = results.get('transitions', [])
    if transitions:
        start_t = max(0, len(results.get('state_history', [])) - tail_steps)
        for tr in transitions:
            t = tr.get('timestamp')
            if t is not None and t < start_t:
                continue
            fs, ts = tr.get('from'), tr.get('to')
            if fs in trans_matrix and ts in trans_matrix[fs]:
                trans_matrix[fs][ts] += 1
    else:
        for i in range(len(state_sequence) - 1):
            fs, ts = state_sequence[i], state_sequence[i + 1]
            if fs != ts:
                trans_matrix[fs][ts] += 1

    for fs in states:
        total = sum(trans_matrix[fs].values())
        if total > 0:
            trans_matrix[fs] = {ts: c / total for ts, c in trans_matrix[fs].items()}

    return {'dwell_times': avg_dwell, 'transition_matrix': trans_matrix}


def compute_residual_scales(results: Dict, tail_steps: int = TAIL_STEPS) -> Dict:
    """Compute residual-based Gaussian scales from tail window histories.

    Currently only forward-model error history is retained.
    Returns:
        {'sigma_x2': 0.0, 'sigma_z2': 0.0, 'sigma_fwd2': ...}
    """
    tail = get_tail_window(results, tail_steps)

    def mean_or_zero(values: list) -> float:
        return float(np.mean(values)) if values else 0.0

    action_series = tail.get('action_errors_history', [])
    return {
        'sigma_x2': 0.0,
        'sigma_z2': 0.0,
        'sigma_fwd2': mean_or_zero(action_series),
    }
