"""Shared utility functions for lean meditation model."""

from typing import Dict, Iterable, Union, List
import copy

import numpy as np
import torch


TAIL_STEPS = 2000  # Last 2000 timesteps for converged behavior analysis (2000 * dt=0.2 = 400 seconds)


def to_float(value: Union[float, int, torch.Tensor]) -> float:
    """Convert scalar-like values (including 0-dim tensors) to float."""
    if isinstance(value, torch.Tensor):
        return float(value.detach().item())
    return float(value)


def clip_probability(value: Union[float, int, torch.Tensor]) -> float:
    """Clamp scalar-like values to [0, 1]."""
    return float(np.clip(to_float(value), 0.0, 1.0))


def clamp_for_log(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Clamp tensor to open interval used in log-probability terms."""
    return torch.clamp(x, eps, 1.0 - eps)


def clamp_activation(x: torch.Tensor, clip_min: float, clip_max: float) -> torch.Tensor:
    """Clamp activations to configured model bounds."""
    return torch.clamp(x, clip_min, clip_max)


def bernoulli_kl(q: torch.Tensor, p: torch.Tensor, eps: float) -> torch.Tensor:
    """Elementwise Bernoulli KL averaged over dimensions."""
    q_safe = clamp_for_log(q, eps)
    p_safe = clamp_for_log(p, eps)
    return torch.mean(
        q_safe * torch.log(q_safe / p_safe)
        + (1.0 - q_safe) * torch.log((1.0 - q_safe) / (1.0 - p_safe))
    )


def networks_to_tensor(
    network_values: Dict[str, Union[float, torch.Tensor]],
    networks: Iterable[str],
    device: torch.device = None,
    default: float = 0.0,
    detach: bool = False,
) -> torch.Tensor:
    """Convert ordered network dict values to a single tensor."""
    values = []
    for net in networks:
        value = network_values.get(net)
        if value is None:
            tensor = torch.tensor(default, dtype=torch.float32, device=device)
        elif isinstance(value, torch.Tensor):
            tensor = value.to(device) if device is not None else value
        else:
            tensor = torch.tensor(value, dtype=torch.float32, device=device)

        if detach:
            tensor = tensor.detach()
        values.append(tensor)

    return torch.stack(values)


# ============================================================================
# Post-Training Analysis Functions
# ============================================================================

def slice_tail(data, tail_steps=TAIL_STEPS):
    """Slice the last tail_steps from a list or array."""
    if data is None or len(data) == 0:
        return data
    return data[-tail_steps:]


def get_tail_window(results: Dict, tail_steps: int = TAIL_STEPS) -> Dict:
    """Extract tail window from training results for converged behavior analysis.
    
    Args:
        results: Full training results dictionary
        tail_steps: Number of steps from end to extract (default: 200)
        
    Returns:
        Dictionary with tail-sliced time series data
    """
    tail_data = copy.deepcopy(results)
    
    # Slice time series fields
    time_series_keys = [
        'state_history',
        'free_energy_history',
        'meta_awareness_history',
        'network_activations_history',
        'thoughtseed_activations_history',
        'dominant_ts_history',
        'action_errors'
    ]
    
    for key in time_series_keys:
        if key in tail_data:
            tail_data[key] = slice_tail(tail_data[key], tail_steps)
    
    return tail_data


def compute_network_profiles(results: Dict, states: List[str], networks: List[str], 
                             tail_steps: int = TAIL_STEPS) -> Dict:
    """Compute mean network activation profiles per state from tail window.
    
    Replicates logic from core/train/logger.py::compute_aggregates()
    
    Args:
        results: Training results with full history
        states: List of state names
        networks: List of network names
        tail_steps: Use last N steps (default: 200)
        
    Returns:
        Dict structure: {state: {network: mean_activation}}
    """
    tail_data = get_tail_window(results, tail_steps)
    
    state_history = tail_data['state_history']
    network_history = tail_data['network_activations_history']
    
    profiles = {}
    for state in states:
        indices = [i for i, s in enumerate(state_history) if s == state]
        if not indices:
            profiles[state] = {net: 0.0 for net in networks}
            continue
        
        # Extract network activations for this state
        network_matrix = np.array([
            [network_history[i].get(net, 0.0) for net in networks]
            for i in indices
        ])
        
        # Compute means
        means = np.mean(network_matrix, axis=0)
        profiles[state] = {
            net: float(means[j])
            for j, net in enumerate(networks)
        }
    
    return profiles


def compute_thoughtseed_means(results: Dict, states: List[str], thoughtseeds: List[str],
                              tail_steps: int = TAIL_STEPS) -> Dict:
    """Compute mean thoughtseed activation vectors per state from tail window.
    
    Args:
        results: Training results with full history
        states: List of state names
        thoughtseeds: List of thoughtseed names
        tail_steps: Use last N steps (default: 200)
        
    Returns:
        Dict structure: {state: [mean_ts1, mean_ts2, ..., mean_ts5]}
    """
    tail_data = get_tail_window(results, tail_steps)
    
    state_history = tail_data['state_history']
    ts_history = tail_data['thoughtseed_activations_history']
    
    means = {}
    for state in states:
        indices = [i for i, s in enumerate(state_history) if s == state]
        if not indices:
            means[state] = [0.0] * len(thoughtseeds)
            continue
        
        # Extract thoughtseed activations for this state
        ts_matrix = np.array([ts_history[i] for i in indices])
        
        # Compute means
        state_means = np.mean(ts_matrix, axis=0)
        means[state] = state_means.tolist()
    
    return means


def compute_tail_statistics(results: Dict, states: List[str], 
                            tail_steps: int = TAIL_STEPS) -> Dict:
    """Compute dwell times and transition matrix from tail window.
    
    Args:
        results: Training results with full history
        states: List of state names
        tail_steps: Use last N steps (default: 200)
        
    Returns:
        Dict with 'dwell_times' and 'transition_matrix'
    """
    tail_data = get_tail_window(results, tail_steps)
    state_sequence = tail_data['state_history']
    
    # Compute dwell times
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
    
    # Compute transition matrix (only count actual state changes, not self-transitions)
    trans_matrix = {from_state: {to_state: 0 for to_state in states} for from_state in states}
    for i in range(len(state_sequence) - 1):
        from_state = state_sequence[i]
        to_state = state_sequence[i + 1]
        # Only count if state actually changed
        if from_state != to_state:
            trans_matrix[from_state][to_state] += 1
    
    # Normalize
    for from_state in states:
        total = sum(trans_matrix[from_state].values())
        if total > 0:
            trans_matrix[from_state] = {
                to_state: count / total
                for to_state, count in trans_matrix[from_state].items()
            }
    
    return {
        'dwell_times': avg_dwell,
        'transition_matrix': trans_matrix
    }
