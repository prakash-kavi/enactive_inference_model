"""Small utility helpers used across the simulation."""

import os
import logging
import numpy as np
import json
import torch

from utils.meditation_config import (
    STATE_TRANSITION_PROBS,
    PREFERRED_TRANSITION_BIAS,
    STATES,
    THOUGHTSEED_BASE_ACTIVATIONS,
    THOUGHTSEED_TARGET_ADJUSTMENTS,
    THOUGHTSEED_LEVEL_OFFSETS,
    META_BASE_AWARENESS,
    META_THOUGHTSEED_INFLUENCES,
    ACTINF_DEFAULTS,
    ACTINF_EXPERT_OVERRIDES,
)


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
        base_dir = os.getcwd()
    base_dir = os.fspath(base_dir)
    os.makedirs(os.path.join(base_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "plots"), exist_ok=True)
    logging.info("Directories created/verified: data/, plots/")

def _normalize_probs(probs: dict, states: list) -> dict:
    vals = {s: max(0.0, float(probs.get(s, 0.0))) for s in states}
    total = sum(vals.values())
    if total <= 0.0:
        uniform = 1.0 / max(1, len(states))
        return {s: uniform for s in states}
    return {k: v / total for k, v in vals.items()}

def get_exit_transition_probs(experience_level: str, current_state: str) -> dict:
    """Return exit-only transition distribution for current_state."""
    base = STATE_TRANSITION_PROBS.get(experience_level, {}).get(current_state, {})
    return _normalize_probs(base, [s for s in STATES if s != current_state])

def get_preferred_transition_probs(experience_level: str, current_state: str) -> dict:
    """Return preferred next-state distribution P*(s'|s) as normalized base + bias."""
    base = get_exit_transition_probs(experience_level, current_state)
    bias = PREFERRED_TRANSITION_BIAS.get(experience_level, {}).get(current_state, {})
    if not base:
        return {}
    adjusted = {}
    for state, prob in base.items():
        adjusted[state] = max(0.0, float(prob) + float(bias.get(state, 0.0)))
    return _normalize_probs(adjusted, list(adjusted.keys()))

def get_thoughtseed_targets(state, meta_awareness, experience_level='novice'):
    """Get target activation values for each thoughtseed in the specified state."""
    activations = THOUGHTSEED_BASE_ACTIVATIONS[state].copy()
    for ts in activations:
        meta_mod = THOUGHTSEED_TARGET_ADJUSTMENTS[state][ts]
        activations[ts] += meta_mod * meta_awareness
        level_offset = THOUGHTSEED_LEVEL_OFFSETS.get(experience_level, {}).get(state, {}).get(ts, 0.0)
        activations[ts] += level_offset
    return activations

def compute_meta_awareness(state, thoughtseed_activations):
    """Compute meta-awareness from mediative state and thoughtseed activations."""
    base_awareness = META_BASE_AWARENESS[state]
    awareness_boost = 0
    for ts, influence in META_THOUGHTSEED_INFLUENCES.items():
        if ts in thoughtseed_activations:
            awareness_boost += thoughtseed_activations[ts] * influence
    return base_awareness + awareness_boost

def get_actinf_params(experience_level='novice'):
    params = dict(ACTINF_DEFAULTS)
    if experience_level == 'expert':
        params.update(ACTINF_EXPERT_OVERRIDES)
    return params

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

    thoughtseed_params = {
        "agent_parameters": {},
        "activation_means_by_state": aggregates.get("activation_means_by_state", {}),
        "network_activations_by_state": aggregates.get("average_network_activations_by_state", {}),
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
        "efe_history": getattr(learner, "efe_history", []),
        "transition_drive_history": getattr(learner, "transition_drive_history", []),
        "recon_loss_history": getattr(learner, "recon_loss_history", []),
        "kl_div_history": getattr(learner, "kl_div_history", []),
        "state_history": learner.state_history,
        "dominant_ts_history": learner.dominant_ts_history,
    }

    out_path_ts = os.path.join(output_dir, f"thoughtseed_params_{learner.experience_level}.json")
    with open(out_path_ts, "w", encoding="utf-8") as f:
        json.dump(thoughtseed_params, f, indent=2)

    params = getattr(learner, "params", {}) if hasattr(learner, "params") else {}
    active_inf_params = {
        "l3tol2_precision_min": params.get("l3tol2_precision_min"),
        "l3tol2_precision_max": params.get("l3tol2_precision_max"),
        "l2tol1_enactive_bias_min": params.get("l2tol1_enactive_bias_min"),
        "l2tol1_enactive_bias_max": params.get("l2tol1_enactive_bias_max"),
        "kl_beta": params.get("kl_beta"),
        "learning_rate": getattr(learner, "learning_rate", None),
        "average_free_energy_by_state": aggregates.get("average_free_energy_by_state", {}),
        "average_efe_by_state": aggregates.get("average_efe_by_state", {}),
        "average_prediction_error_by_state": aggregates.get("average_prediction_error_by_state", {}),
        "average_precision_by_state": aggregates.get("average_precision_by_state", {}),
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
    efe_means = {}

    state_indices = {
        state: [j for j, s in enumerate(learner.state_history) if s == state]
        for state in states
    }

    activations_history = learner.activations_history
    network_history = learner.network_activations_history
    free_energy_history = np.asarray(learner.free_energy_history, dtype=float)
    pred_error_history = np.asarray(learner.prediction_error_history, dtype=float)
    precision_history = np.asarray(learner.precision_history, dtype=float)
    efe_history = np.asarray(getattr(learner, "efe_history", []), dtype=float)

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
        if efe_history.size == len(learner.state_history):
            efe_means[state] = float(np.mean(efe_history[indices]))

    aggregates["activation_means_by_state"] = activation_means
    aggregates["average_network_activations_by_state"] = network_means
    aggregates["average_free_energy_by_state"] = free_energy_means
    aggregates["average_prediction_error_by_state"] = pred_error_means
    aggregates["average_precision_by_state"] = precision_means
    aggregates["average_efe_by_state"] = efe_means

    return aggregates
