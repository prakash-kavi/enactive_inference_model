"""Small utility helpers used across the simulation."""

import os
import logging
import numpy as np
import json


def clip_array(x, vmin, vmax):
    """Clip a scalar or array to [vmin, vmax], preserving scalar return."""
    arr = np.asarray(x)
    clipped = np.clip(arr, vmin, vmax)
    if clipped.shape == ():
        return float(clipped)
    return clipped


def ou_update(x_prev, mu, theta, sigma, dt=1.0):
    """Ornstein–Uhlenbeck update (mean reversion + Gaussian noise)."""
    x_prev_arr = np.asarray(x_prev)
    mu_arr = np.asarray(mu)
    noise = np.random.normal(0, 1, size=x_prev_arr.shape)
    dx = theta * (mu_arr - x_prev_arr) * dt + sigma * noise
    return x_prev_arr + dx


def ensure_directories(base_dir=None):
    """Create `data/` and `plots/` under `base_dir` (or package root)."""
    if not base_dir:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.fspath(base_dir)
    os.makedirs(os.path.join(base_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "plots"), exist_ok=True)
    logging.info("Directories created/verified: data/, plots/")


def _save_json_outputs(learner, output_dir=None):
    """Write learner parameters and time series to JSON files.

    Converts NumPy arrays to lists and computes per-state aggregates.
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(output_dir, exist_ok=True)
    logging.info("Generating consumer-ready JSON files...")

    def convert(obj):
        return to_json_serializable(obj)

    aggregates = compute_state_aggregates(learner)
    activations_history = learner.activations_history or []
    network_activations_history = learner.network_activations_history or []

    thoughtseed_params = {
        "agent_parameters": {},
        "activation_means_by_state": aggregates.get("activation_means_by_state", {}),
        "network_activations_by_state": aggregates.get("average_network_activations_by_state", {}),
        "learned_network_profiles": convert(learner.learned_network_profiles),
    }

    for i, ts in enumerate(learner.thoughtseeds):
        if activations_history:
            base_activation = float(np.mean([act[i] for act in activations_history]))
            responsiveness = float(max(0.5, 1.0 - np.std([act[i] for act in activations_history])))
        else:
            base_activation = 0.0
            responsiveness = 1.0

        thoughtseed_params["agent_parameters"][ts] = {
            "base_activation": base_activation,
            "responsiveness": responsiveness,
            "network_profile": learner.learned_network_profiles["thoughtseed_contributions"][ts],
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
        "network_expectations": learner.learned_network_profiles.get("state_network_expectations", {}),
    }

    out_path_ai = os.path.join(output_dir, f"active_inference_params_{learner.experience_level}.json")
    with open(out_path_ai, "w", encoding="utf-8") as f:
        json.dump(active_inf_params, f, indent=2)

    try:
        rel = os.path.relpath(output_dir, start=os.getcwd())
    except Exception:
        rel = output_dir
    logging.info("  - JSON parameter files saved to %s directory", rel)


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

    for state in states:
        indices = [j for j, s in enumerate(learner.state_history) if s == state]
        if not indices:
            continue

        activation_means[state] = {
            ts: float(np.mean([learner.activations_history[j][i] for j in indices]))
            for i, ts in enumerate(learner.thoughtseeds)
        }

        network_means[state] = {
            net: float(np.mean([learner.network_activations_history[j][net] for j in indices]))
            for net in learner.networks
        }

        free_energy_means[state] = float(np.mean([learner.free_energy_history[j] for j in indices]))
        pred_error_means[state] = float(np.mean([learner.prediction_error_history[j] for j in indices]))
        precision_means[state] = float(np.mean([learner.precision_history[j] for j in indices]))

    avg_acts_at_trans = {
        state: (np.mean(acts, axis=0).tolist() if len(acts) > 0 else np.zeros(learner.num_thoughtseeds).tolist())
        for state, acts in learner.transition_activations.items()
    }

    aggregates["activation_means_by_state"] = activation_means
    aggregates["average_network_activations_by_state"] = network_means
    aggregates["average_free_energy_by_state"] = free_energy_means
    aggregates["average_prediction_error_by_state"] = pred_error_means
    aggregates["average_precision_by_state"] = precision_means
    aggregates["average_activations_at_transition"] = avg_acts_at_trans

    return aggregates