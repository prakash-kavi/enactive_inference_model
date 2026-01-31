"""Simulation Logger: Decoupled history tracking for the meditation model."""

import os
import json
import logging
import numpy as np
import torch
from typing import Dict, List, Optional, Any

class SimulationLogger:
    """Handles logging of simulation steps, transitions, and metrics."""

    def __init__(self, experience_level: str):
        self.experience_level = experience_level
        self.reset()

    def reset(self):
        """Initialize empty history containers."""
        # Time Series
        self.state_history: List[str] = []
        self.activations_history: List[np.ndarray] = []
        self.meta_awareness_history: List[float] = []
        self.network_activations_history: List[Dict[str, float]] = []
        self.free_energy_history: List[float] = []
        self.prediction_error_history: List[float] = []
        self.precision_history: List[float] = []
        self.neural_efficiency_history: List[float] = []
        self.stability_indicators: List[float] = []
        self.dominant_ts_history: List[str] = []
        self.efe_history: List[float] = []
        self.transition_drive_history: List[float] = []
        self.recon_loss_history: List[float] = []
        self.kl_div_history: List[float] = []

        # Diagnostic Counters
        self.van_spike_detections: int = 0
        self.expert_mind_wandering_detections: int = 0

    def record_step(self, 
                    current_state: str,
                    activations: np.ndarray,
                    meta_awareness: float,
                    network_acts: Dict[str, float],
                    free_energy: float,
                    recon_loss: float,
                    kl_div: float,
                    transition_drive: float,
                    precision: float,
                    efe: float,
                    dominant_ts: str,
                    neural_efficiency: Optional[float],
                    stability_indicator: float,
                    is_expert_mw: bool = False,
                    is_van_spike: bool = False):
        """Log a single simulation timestep."""
        self.state_history.append(current_state)
        self.activations_history.append(activations)
        self.meta_awareness_history.append(meta_awareness)
        self.network_activations_history.append(network_acts)
        
        self.free_energy_history.append(free_energy)
        self.prediction_error_history.append(recon_loss) # recon_loss is used as approx for pred error
        self.recon_loss_history.append(recon_loss)
        self.kl_div_history.append(kl_div)
        
        self.transition_drive_history.append(transition_drive)
        self.precision_history.append(precision)
        self.efe_history.append(efe)
        self.dominant_ts_history.append(dominant_ts)
        
        if neural_efficiency is not None:
            self.neural_efficiency_history.append(neural_efficiency)
        
        self.stability_indicators.append(stability_indicator)

        if is_expert_mw:
            self.expert_mind_wandering_detections += 1
        
        if is_van_spike:
            self.van_spike_detections += 1

    def save_results(self, agent: Any, state_transition_patterns: List, transition_timestamps: List, output_dir: str = None):
        """Save simulation results to disk. Orchestrates aggregation and serialization."""
        out_dir = output_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
        os.makedirs(out_dir, exist_ok=True)
        logging.info("Generating consumer-ready JSON files...")

        # Compute Aggregates internally
        aggregates = self.compute_aggregates(agent)
        
        # Build Transition Stats
        transition_stats = self.build_transition_stats(
            state_transition_patterns, 
            transition_timestamps, 
            aggregates
        )
        
        out_path = os.path.join(out_dir, f"transition_stats_{self.experience_level}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(transition_stats, f, indent=2)

        # Save JSON Outputs
        self.save_json(agent, out_dir, aggregates)
        
        try:
            rel = os.path.relpath(out_dir, start=os.getcwd())
        except Exception:
            rel = out_dir
        logging.info("  - JSON parameter files saved to %s directory", rel)

    def compute_aggregates(self, agent: Any) -> Dict:
        """Compute per-state means for activations, networks, VFE and errors."""
        aggregates = {}
        states = agent.states
        activation_means = {}
        network_means = {}
        free_energy_means = {}
        pred_error_means = {}
        precision_means = {}
        efe_means = {}

        state_indices = {
            state: [j for j, s in enumerate(self.state_history) if s == state]
            for state in states
        }

        activations_history = self.activations_history
        network_history = self.network_activations_history
        free_energy_history = np.asarray(self.free_energy_history, dtype=float)
        pred_error_history = np.asarray(self.prediction_error_history, dtype=float)
        precision_history = np.asarray(self.precision_history, dtype=float)
        efe_history = np.asarray(self.efe_history, dtype=float)

        for state in states:
            indices = state_indices.get(state, [])
            if not indices:
                continue

            if activations_history:
                acts = np.asarray([activations_history[j] for j in indices], dtype=float)
                act_means = np.mean(acts, axis=0)
                activation_means[state] = {
                    ts: float(act_means[i])
                    for i, ts in enumerate(agent.thoughtseeds)
                }

            if network_history:
                net_matrix = np.asarray(
                    [[network_history[j].get(net, 0.0) for net in agent.networks] for j in indices],
                    dtype=float
                )
                net_means = np.mean(net_matrix, axis=0)
                network_means[state] = {
                    net: float(net_means[i])
                    for i, net in enumerate(agent.networks)
                }

            free_energy_means[state] = float(np.mean(free_energy_history[indices]))
            pred_error_means[state] = float(np.mean(pred_error_history[indices]))
            precision_means[state] = float(np.mean(precision_history[indices]))
            if efe_history.size == len(self.state_history):
                efe_means[state] = float(np.mean(efe_history[indices]))

        aggregates["activation_means_by_state"] = activation_means
        aggregates["average_network_activations_by_state"] = network_means
        aggregates["average_free_energy_by_state"] = free_energy_means
        aggregates["average_prediction_error_by_state"] = pred_error_means
        aggregates["average_precision_by_state"] = precision_means
        aggregates["average_efe_by_state"] = efe_means

        return aggregates

    def build_transition_stats(self, state_transition_patterns, transition_timestamps, aggregates):
        """Build a serializable transition stats payload."""
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

    def save_json(self, agent: Any, output_dir: str, aggregates: Dict):
        """Write parameter and time series JSONs."""
        def convert(obj):
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list): return [convert(i) for i in obj]
            return obj

        thoughtseed_params = {
            "agent_parameters": {},
            "activation_means_by_state": aggregates.get("activation_means_by_state", {}),
            "network_activations_by_state": aggregates.get("average_network_activations_by_state", {}),
        }

        for i, ts in enumerate(agent.thoughtseeds):
            if self.activations_history:
                base_activation = float(np.mean([act[i] for act in self.activations_history]))
                responsiveness = float(max(0.5, 1.0 - np.std([act[i] for act in self.activations_history])))
            else:
                base_activation = 0.0
                responsiveness = 1.0

            # Decode a one-hot thoughtseed vector to get predicted network activations
            ts_idx = agent.thoughtseeds.index(ts)
            device = next(agent.vae.parameters()).device
            one_hot = torch.zeros(1, len(agent.thoughtseeds), device=device)
            one_hot[0, ts_idx] = 1.0
            
            was_training = agent.vae.training
            agent.vae.eval()
            try:
                with torch.no_grad():
                    network_pred = agent.vae.decode(one_hot).squeeze(0)
            finally:
                agent.vae.train(was_training)
            
            network_profile = {}
            for net_idx, net in enumerate(agent.networks):
                val = network_pred[net_idx].detach().cpu().item()
                network_profile[net] = float(val)
            
            thoughtseed_params["agent_parameters"][ts] = {
                "base_activation": base_activation,
                "responsiveness": responsiveness,
                "network_profile": network_profile,
            }

        thoughtseed_params["time_series"] = {
            "activations_history": convert(self.activations_history),
            "network_activations_history": convert(self.network_activations_history),
            "meta_awareness_history": self.meta_awareness_history,
            "free_energy_history": self.free_energy_history,
            "efe_history": self.efe_history,
            "transition_drive_history": self.transition_drive_history,
            "recon_loss_history": self.recon_loss_history,
            "kl_div_history": self.kl_div_history,
            "state_history": self.state_history,
            "dominant_ts_history": self.dominant_ts_history,
        }

        with open(os.path.join(output_dir, f"thoughtseed_params_{self.experience_level}.json"), "w", encoding="utf-8") as f:
            json.dump(thoughtseed_params, f, indent=2)

        params = getattr(agent, "params", {}) if hasattr(agent, "params") else {}
        active_inf_params = {
            "l3tol2_precision_min": params.get("l3tol2_precision_min"),
            "l3tol2_precision_max": params.get("l3tol2_precision_max"),
            "kl_beta": params.get("kl_beta"),
            "learning_rate": getattr(agent, "learning_rate", None),
            "average_free_energy_by_state": aggregates.get("average_free_energy_by_state", {}),
            "average_efe_by_state": aggregates.get("average_efe_by_state", {}),
            "average_prediction_error_by_state": aggregates.get("average_prediction_error_by_state", {}),
            "average_precision_by_state": aggregates.get("average_precision_by_state", {}),
        }

        with open(os.path.join(output_dir, f"active_inference_params_{self.experience_level}.json"), "w", encoding="utf-8") as f:
            json.dump(active_inf_params, f, indent=2)
