"""Small utility helpers used across the simulation."""

import os
import logging
import numpy as np
import json
import torch


def clip_array(x, vmin, vmax):
    """Clip a scalar or array to [vmin, vmax], preserving scalar return."""
    arr = np.asarray(x)
    clipped = np.clip(arr, vmin, vmax)
    if clipped.shape == ():
        return float(clipped)
    return clipped

def ensure_directories(base_dir=None):
    """Create `data/` and `plots/` under `base_dir` (or package root)."""
    if not base_dir:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.fspath(base_dir)
    os.makedirs(os.path.join(base_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "plots"), exist_ok=True)
    logging.info("Directories created/verified: data/, plots/")

class LeakyAccumulator:
    """Standardized Leaky Integrator for signal accumulation.
    Used for DMN spikes, FPN fatigue, VFE tracking, and Aha Detection.
    """
    def __init__(self, decay: float = 0.9, gain: float = 0.1, initial_value: float = 0.0, activation: str = 'linear'):
        self.decay = decay
        self.gain = gain
        self.value = initial_value
        self.activation = activation

    def update(self, input_val: float) -> float:
        """Update the accumulator: value = activation(decay * value + gain * input)."""
        new_val = self.decay * self.value + self.gain * input_val
        
        if self.activation == 'sigmoid':
            self.value = 1.0 / (1.0 + np.exp(-10.0 * (new_val - 0.5))) # Standard sigmoid centered at 0.5
        elif self.activation == 'tanh':
            self.value = np.tanh(new_val)
        elif self.activation == 'relu':
            self.value = max(0.0, new_val)
        else:
            self.value = new_val
            
        return self.value

    def reset(self, value: float = 0.0):
        """Reset the accumulator to a specific value."""
        self.value = value

def _save_json_outputs(learner, output_dir=None, aggregates=None):
    """Write learner parameters and time series to JSON files.

    Converts NumPy arrays to lists and computes per-state aggregates.
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(output_dir, exist_ok=True)
    logging.info("Generating consumer-ready JSON files...")

    def convert(obj):
        return to_json_serializable(obj)

    if aggregates is None:
        aggregates = compute_state_aggregates(learner)
    activations_history = learner.activations_history or []
    network_activations_history = learner.network_activations_history or []

    learned_profiles = {}
    if hasattr(learner, 'learned_network_profiles'):
        learned_profiles = convert(learner.learned_network_profiles)

    thoughtseed_params = {
        "agent_parameters": {},
        "activation_means_by_state": aggregates.get("activation_means_by_state", {}),
        "network_activations_by_state": aggregates.get("average_network_activations_by_state", {}),
        "learned_network_profiles": learned_profiles,
    }

    for i, ts in enumerate(learner.thoughtseeds):
        if activations_history:
            base_activation = float(np.mean([act[i] for act in activations_history]))
            responsiveness = float(max(0.5, 1.0 - np.std([act[i] for act in activations_history])))
        else:
            base_activation = 0.0
            responsiveness = 1.0

        # Extract network profile from VAE decoder (replaces old W matrix)
        # Decode a one-hot thoughtseed vector to get predicted network activations
        ts_idx = learner.thoughtseeds.index(ts)
        
        # Create one-hot vector for this thoughtseed
        device = next(learner.vae.parameters()).device
        one_hot = torch.zeros(1, len(learner.thoughtseeds), device=device)
        one_hot[0, ts_idx] = 1.0
        
        # Decode to get network profile (temporarily set to eval mode for consistent outputs)
        was_training = learner.vae.training
        learner.vae.eval()
        try:
            with torch.no_grad():
                network_pred = learner.vae.decode(one_hot)  # Shape: (1, 4)
                network_pred = network_pred.squeeze(0)  # Shape: (4,)
        finally:
            # Restore original training mode
            learner.vae.train(was_training)
        
        network_profile = {}
        for net_idx, net in enumerate(learner.networks):
            val = network_pred[net_idx].detach().cpu().item()
            network_profile[net] = float(val)
        
        thoughtseed_params["agent_parameters"][ts] = {
            "base_activation": base_activation,
            "responsiveness": responsiveness,
            "network_profile": network_profile,
        }

    thoughtseed_params["time_series"] = {
        "activations_history": convert(activations_history),
        "network_activations_history": convert(network_activations_history),
        "meta_awareness_history": learner.meta_awareness_history,
        "free_energy_history": learner.free_energy_history,
        "state_history": learner.state_history,
        "dominant_ts_history": learner.dominant_ts_history,
    }

    out_path_ts = os.path.join(output_dir, f"thoughtseed_params_{learner.experience_level}.json")
    with open(out_path_ts, "w", encoding="utf-8") as f:
        json.dump(thoughtseed_params, f, indent=2)

    active_inf_params = {
        "precision_weight": getattr(learner, "precision_weight", None),
        "complexity_penalty": getattr(learner, "complexity_penalty", None),
        "learning_rate": getattr(learner, "learning_rate", None),
        "average_free_energy_by_state": aggregates.get("average_free_energy_by_state", {}),
        "average_prediction_error_by_state": aggregates.get("average_prediction_error_by_state", {}),
        "average_precision_by_state": aggregates.get("average_precision_by_state", {}),
        "network_expectations": learned_profiles.get("state_network_expectations", {}),
    }

    out_path_ai = os.path.join(output_dir, f"active_inference_params_{learner.experience_level}.json")
    with open(out_path_ai, "w", encoding="utf-8") as f:
        json.dump(active_inf_params, f, indent=2)

    try:
        rel = os.path.relpath(output_dir, start=os.getcwd())
    except Exception:
        rel = output_dir
    logging.info("  - JSON parameter files saved to %s directory", rel)

def build_transition_stats(agent, state_transition_patterns, transition_timestamps, aggregates):
    """Build a serializable transition stats payload for logging/output."""
    serial_patterns = []
    for (frm, to, ts_dict, net_dict, fe) in state_transition_patterns:
        serial_patterns.append({
            'from': frm,
            'to': to,
            'thoughtseed_activations': {k: float(v) for k, v in ts_dict.items()},
            'network_acts': {k: float(v) for k, v in net_dict.items()},
            'free_energy': float(fe)
        })

    return {
        'transition_timestamps': [int(x) for x in transition_timestamps],
        'state_transition_patterns': serial_patterns,
        'distraction_buildup_rates': [float(x) for x in getattr(agent, "distraction_buildup_rates", [])],
        'average_network_activations_by_state': aggregates.get('average_network_activations_by_state', {}),
        'average_free_energy_by_state': aggregates.get('average_free_energy_by_state', {}),
    }

def to_json_serializable(obj):
    """Recursively convert NumPy arrays/lists/dicts to JSON-serializable forms."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_json_serializable(i) for i in obj]
    return obj

def compute_state_aggregates(learner):
    """Return per-state means for activations, networks, VFE and errors."""
    aggregates = {}
    states = learner.states
    activation_means = {}
    network_means = {}
    free_energy_means = {}
    pred_error_means = {}
    precision_means = {}

    state_indices = {
        state: [j for j, s in enumerate(learner.state_history) if s == state]
        for state in states
    }

    activations_history = learner.activations_history
    network_history = learner.network_activations_history
    free_energy_history = np.asarray(learner.free_energy_history, dtype=float)
    pred_error_history = np.asarray(learner.prediction_error_history, dtype=float)
    precision_history = np.asarray(learner.precision_history, dtype=float)

    for state in states:
        indices = state_indices.get(state, [])
        if not indices:
            continue

        if activations_history:
            acts = np.asarray([activations_history[j] for j in indices], dtype=float)
            act_means = np.mean(acts, axis=0)
            activation_means[state] = {
                ts: float(act_means[i])
                for i, ts in enumerate(learner.thoughtseeds)
            }

        if network_history:
            net_matrix = np.asarray(
                [[network_history[j].get(net, 0.0) for net in learner.networks] for j in indices],
                dtype=float
            )
            net_means = np.mean(net_matrix, axis=0)
            network_means[state] = {
                net: float(net_means[i])
                for i, net in enumerate(learner.networks)
            }

        free_energy_means[state] = float(np.mean(free_energy_history[indices]))
        pred_error_means[state] = float(np.mean(pred_error_history[indices]))
        precision_means[state] = float(np.mean(precision_history[indices]))

    aggregates["activation_means_by_state"] = activation_means
    aggregates["average_network_activations_by_state"] = network_means
    aggregates["average_free_energy_by_state"] = free_energy_means
    aggregates["average_prediction_error_by_state"] = pred_error_means
    aggregates["average_precision_by_state"] = precision_means

    return aggregates
