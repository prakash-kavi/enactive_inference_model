"""Simulation Logger: Decoupled history tracking for the meditation model."""

import os
import json
import logging
import numpy as np
import torch
from typing import Dict, List, Optional, Any, Iterable, Tuple

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
        self.precision_history: List[float] = []
        self.dominant_ts_history: List[str] = []
        self.efe_history: List[float] = []
        self.efe_risk_history: List[float] = []
        self.efe_ambiguity_history: List[float] = []
        self.selected_policy_history: List[str] = []
        self.policy_confidence_history: List[float] = []
        self.policy_entropy_history: List[float] = []
        self.policy_posterior_history: List[Dict[str, float]] = []
        self.mw_burden_history: List[float] = []
        self.transition_hazard_history: List[float] = []
        self.activation_burden_component_history: List[float] = []
        self.coupling_burden_component_history: List[float] = []
        self.transition_drive_history: List[float] = []
        self.recon_loss_history: List[float] = []
        self.kl_div_history: List[float] = []
        self.latent_reconstruction_history: List[float] = []
        self.latent_prior_kl_history: List[float] = []
        self.latent_sensory_consistency_history: List[float] = []
        self.latent_temporal_consistency_history: List[float] = []
        self.latent_vfe_total_history: List[float] = []
        self.action_pred_error_history: List[float] = []  # Phase 4: forward prediction error
        self.transition_timestamps: List[int] = []
        self.state_transition_patterns: List[Tuple[str, str, Dict[str, float], Dict[str, float], float]] = []
        self.ra_reorienting_success_rate: float = 0.0


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
                    efe_risk: float,
                    efe_ambiguity: float,
                    selected_policy: str,
                    policy_confidence: float,
                    policy_entropy: float,
                    policy_posterior: Dict[str, float],
                    mw_burden: float,
                    transition_hazard: float,
                    activation_burden_component: float,
                    coupling_burden_component: float,
                    latent_reconstruction: float,
                    latent_prior_kl: float,
                    latent_sensory_consistency: float,
                    latent_temporal_consistency: float,
                    latent_vfe_total: float,
                    dominant_ts: str,
                    action_pred_error: float = 0.0):  # Phase 4: optional action error
        """Log a single simulation timestep."""
        self.state_history.append(current_state)
        self.activations_history.append(activations)
        self.meta_awareness_history.append(meta_awareness)
        self.network_activations_history.append(network_acts)
        
        self.free_energy_history.append(free_energy)
        self.recon_loss_history.append(recon_loss)
        self.kl_div_history.append(kl_div)
        self.action_pred_error_history.append(action_pred_error)  # Phase 4
        
        self.transition_drive_history.append(transition_drive)
        self.precision_history.append(precision)
        self.efe_history.append(efe)
        self.efe_risk_history.append(efe_risk)
        self.efe_ambiguity_history.append(efe_ambiguity)
        self.selected_policy_history.append(selected_policy)
        self.policy_confidence_history.append(policy_confidence)
        self.policy_entropy_history.append(policy_entropy)
        self.policy_posterior_history.append(policy_posterior)
        self.mw_burden_history.append(mw_burden)
        self.transition_hazard_history.append(transition_hazard)
        self.activation_burden_component_history.append(activation_burden_component)
        self.coupling_burden_component_history.append(coupling_burden_component)
        self.latent_reconstruction_history.append(latent_reconstruction)
        self.latent_prior_kl_history.append(latent_prior_kl)
        self.latent_sensory_consistency_history.append(latent_sensory_consistency)
        self.latent_temporal_consistency_history.append(latent_temporal_consistency)
        self.latent_vfe_total_history.append(latent_vfe_total)
        self.dominant_ts_history.append(dominant_ts)


    def record_transition(
        self,
        timestamp: int,
        from_state: str,
        to_state: str,
        thoughtseed_activations: Dict[str, float],
        network_acts: Dict[str, float],
        free_energy: float,
    ) -> None:
        """Log a state transition event."""
        self.transition_timestamps.append(int(timestamp))
        self.state_transition_patterns.append(
            (from_state, to_state, thoughtseed_activations, network_acts, float(free_energy))
        )

    def save_results(
        self,
        agent: Any,
        output_dir: str = None,
        state_transition_patterns: Optional[Iterable] = None,
        transition_timestamps: Optional[Iterable] = None,
    ):
        """Save simulation results to disk. Orchestrates aggregation and serialization."""
        out_dir = output_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
        os.makedirs(out_dir, exist_ok=True)
        logging.info("Generating consumer-ready JSON files...")

        # Compute Aggregates internally
        aggregates = self.compute_aggregates(agent)

        # Build Transition Stats
        transition_stats = self.build_transition_stats(
            list(state_transition_patterns) if state_transition_patterns is not None else self.state_transition_patterns,
            list(transition_timestamps) if transition_timestamps is not None else self.transition_timestamps,
            aggregates
        )
        
        out_path = os.path.join(out_dir, f"transition_stats_{self.experience_level}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(transition_stats, f, indent=2)

        # Save JSON Outputs
        self.save_json(agent, out_dir, aggregates)
        
        try:
            rel = os.path.relpath(out_dir, start=os.getcwd())
        except (ValueError, OSError):
            # ValueError on Windows with different drives, OSError for permission issues
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
        efe_risk_means = {}
        efe_ambiguity_means = {}
        policy_confidence_means = {}
        policy_entropy_means = {}
        mw_burden_means = {}
        transition_hazard_means = {}
        activation_burden_component_means = {}
        coupling_burden_component_means = {}
        latent_recon_means = {}
        latent_prior_kl_means = {}
        latent_sensory_means = {}
        latent_temporal_means = {}
        latent_total_means = {}

        state_indices = {
            state: [j for j, s in enumerate(self.state_history) if s == state]
            for state in states
        }

        activations_history = self.activations_history
        network_history = self.network_activations_history
        free_energy_history = np.asarray(self.free_energy_history, dtype=float)
        pred_error_history = np.asarray(self.recon_loss_history, dtype=float)
        precision_history = np.asarray(self.precision_history, dtype=float)
        efe_history = np.asarray(self.efe_history, dtype=float)
        efe_risk_history = np.asarray(self.efe_risk_history, dtype=float)
        efe_ambiguity_history = np.asarray(self.efe_ambiguity_history, dtype=float)
        policy_confidence_history = np.asarray(self.policy_confidence_history, dtype=float)
        policy_entropy_history = np.asarray(self.policy_entropy_history, dtype=float)
        mw_burden_history = np.asarray(self.mw_burden_history, dtype=float)
        transition_hazard_history = np.asarray(self.transition_hazard_history, dtype=float)
        activation_burden_component_history = np.asarray(self.activation_burden_component_history, dtype=float)
        coupling_burden_component_history = np.asarray(self.coupling_burden_component_history, dtype=float)
        latent_recon_history = np.asarray(self.latent_reconstruction_history, dtype=float)
        latent_prior_kl_history = np.asarray(self.latent_prior_kl_history, dtype=float)
        latent_sensory_history = np.asarray(self.latent_sensory_consistency_history, dtype=float)
        latent_temporal_history = np.asarray(self.latent_temporal_consistency_history, dtype=float)
        latent_total_history = np.asarray(self.latent_vfe_total_history, dtype=float)
        action_pred_error_history = np.asarray(self.action_pred_error_history, dtype=float)  # Phase 4

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
            if efe_risk_history.size == len(self.state_history):
                efe_risk_means[state] = float(np.mean(efe_risk_history[indices]))
            if efe_ambiguity_history.size == len(self.state_history):
                efe_ambiguity_means[state] = float(np.mean(efe_ambiguity_history[indices]))
            if policy_confidence_history.size == len(self.state_history):
                policy_confidence_means[state] = float(np.mean(policy_confidence_history[indices]))
            if policy_entropy_history.size == len(self.state_history):
                policy_entropy_means[state] = float(np.mean(policy_entropy_history[indices]))
            if mw_burden_history.size == len(self.state_history):
                mw_burden_means[state] = float(np.mean(mw_burden_history[indices]))
            if transition_hazard_history.size == len(self.state_history):
                transition_hazard_means[state] = float(np.mean(transition_hazard_history[indices]))
            if activation_burden_component_history.size == len(self.state_history):
                activation_burden_component_means[state] = float(np.mean(activation_burden_component_history[indices]))
            if coupling_burden_component_history.size == len(self.state_history):
                coupling_burden_component_means[state] = float(np.mean(coupling_burden_component_history[indices]))
            if latent_recon_history.size == len(self.state_history):
                latent_recon_means[state] = float(np.mean(latent_recon_history[indices]))
            if latent_prior_kl_history.size == len(self.state_history):
                latent_prior_kl_means[state] = float(np.mean(latent_prior_kl_history[indices]))
            if latent_sensory_history.size == len(self.state_history):
                latent_sensory_means[state] = float(np.mean(latent_sensory_history[indices]))
            if latent_temporal_history.size == len(self.state_history):
                latent_temporal_means[state] = float(np.mean(latent_temporal_history[indices]))
            if latent_total_history.size == len(self.state_history):
                latent_total_means[state] = float(np.mean(latent_total_history[indices]))

        # Phase 4: Add action prediction error aggregates
        action_pred_error_means = {}
        for state in states:
            indices = state_indices.get(state, [])
            if indices and action_pred_error_history.size == len(self.state_history):
                action_pred_error_means[state] = float(np.mean(action_pred_error_history[indices]))

        aggregates["activation_means_by_state"] = activation_means
        aggregates["average_network_activations_by_state"] = network_means
        aggregates["average_free_energy_by_state"] = free_energy_means
        aggregates["average_prediction_error_by_state"] = pred_error_means
        aggregates["average_action_pred_error_by_state"] = action_pred_error_means  # Phase 4
        aggregates["average_precision_by_state"] = precision_means
        aggregates["average_efe_by_state"] = efe_means
        aggregates["average_efe_risk_by_state"] = efe_risk_means
        aggregates["average_efe_ambiguity_by_state"] = efe_ambiguity_means
        aggregates["average_policy_confidence_by_state"] = policy_confidence_means
        aggregates["average_policy_entropy_by_state"] = policy_entropy_means
        aggregates["average_mw_burden_by_state"] = mw_burden_means
        aggregates["average_transition_hazard_by_state"] = transition_hazard_means
        aggregates["average_activation_burden_component_by_state"] = activation_burden_component_means
        aggregates["average_coupling_burden_component_by_state"] = coupling_burden_component_means
        aggregates["average_latent_reconstruction_by_state"] = latent_recon_means
        aggregates["average_latent_prior_kl_by_state"] = latent_prior_kl_means
        aggregates["average_latent_sensory_consistency_by_state"] = latent_sensory_means
        aggregates["average_latent_temporal_consistency_by_state"] = latent_temporal_means
        aggregates["average_latent_vfe_total_by_state"] = latent_total_means

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
        ra_exits = [row for row in serial_patterns if row.get('from') == 'redirect_attention']
        if ra_exits:
            ra_to_bf = [row for row in ra_exits if row.get('to') == 'breath_focus']
            ra_success_rate = float(len(ra_to_bf) / max(1, len(ra_exits)))
        else:
            ra_success_rate = 0.0
        self.ra_reorienting_success_rate = ra_success_rate

        return {
            'transition_timestamps': [int(x) for x in transition_timestamps],
            'state_transition_patterns': serial_patterns,
            'average_network_activations_by_state': aggregates.get('average_network_activations_by_state', {}),
            'average_free_energy_by_state': aggregates.get('average_free_energy_by_state', {}),
            'ra_reorienting_success_rate': ra_success_rate,
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
            "efe_risk_history": self.efe_risk_history,
            "efe_ambiguity_history": self.efe_ambiguity_history,
            "selected_policy_history": self.selected_policy_history,
            "policy_confidence_history": self.policy_confidence_history,
            "policy_entropy_history": self.policy_entropy_history,
            "policy_posterior_history": convert(self.policy_posterior_history),
            "mw_burden_history": self.mw_burden_history,
            "transition_hazard_history": self.transition_hazard_history,
            "activation_burden_component_history": self.activation_burden_component_history,
            "coupling_burden_component_history": self.coupling_burden_component_history,
            "transition_drive_history": self.transition_drive_history,
            "recon_loss_history": self.recon_loss_history,
            "kl_div_history": self.kl_div_history,
            "latent_reconstruction_history": self.latent_reconstruction_history,
            "latent_prior_kl_history": self.latent_prior_kl_history,
            "latent_sensory_consistency_history": self.latent_sensory_consistency_history,
            "latent_temporal_consistency_history": self.latent_temporal_consistency_history,
            "latent_vfe_total_history": self.latent_vfe_total_history,
            "action_pred_error_history": self.action_pred_error_history,  # Phase 4
            "state_history": self.state_history,
            "dominant_ts_history": self.dominant_ts_history,
        }

        with open(os.path.join(output_dir, f"thoughtseed_params_{self.experience_level}.json"), "w", encoding="utf-8") as f:
            json.dump(thoughtseed_params, f, indent=2)

        params = getattr(agent, "params", {}) if hasattr(agent, "params") else {}
        active_inf_params = {
            "l3tol2_precision_range": params.get("l3tol2_precision_range"),
            "kl_beta": params.get("kl_beta"),
            "l2_vi_steps": params.get("l2_vi_steps"),
            "l2_vi_lr": params.get("l2_vi_lr"),
            "l2_vi_obs_weight": params.get("l2_vi_obs_weight"),
            "l2_vi_prior_weight": params.get("l2_vi_prior_weight"),
            "l2_vi_sensory_weight": params.get("l2_vi_sensory_weight"),
            "l2_vi_temporal_weight": params.get("l2_vi_temporal_weight"),
            "l2_vi_grad_clip": params.get("l2_vi_grad_clip"),
            "efe_ambiguity_weight": params.get("efe_ambiguity_weight"),
            "efe_cycle_strength": params.get("efe_cycle_strength"),
            "efe_gain": params.get("efe_gain"),
            "policy_horizon": params.get("policy_horizon"),
            "policy_temperature": params.get("policy_temperature"),
            "policy_temperature_by_state": params.get("policy_temperature_by_state"),
            "policy_horizon_discount": params.get("policy_horizon_discount"),
            "learning_rate": getattr(agent, "learning_rate", None),
            "average_free_energy_by_state": aggregates.get("average_free_energy_by_state", {}),
            "average_efe_by_state": aggregates.get("average_efe_by_state", {}),
            "average_efe_risk_by_state": aggregates.get("average_efe_risk_by_state", {}),
            "average_efe_ambiguity_by_state": aggregates.get("average_efe_ambiguity_by_state", {}),
            "average_policy_confidence_by_state": aggregates.get("average_policy_confidence_by_state", {}),
            "average_policy_entropy_by_state": aggregates.get("average_policy_entropy_by_state", {}),
            "average_mw_burden_by_state": aggregates.get("average_mw_burden_by_state", {}),
            "average_transition_hazard_by_state": aggregates.get("average_transition_hazard_by_state", {}),
            "average_activation_burden_component_by_state": aggregates.get("average_activation_burden_component_by_state", {}),
            "average_coupling_burden_component_by_state": aggregates.get("average_coupling_burden_component_by_state", {}),
            "average_action_pred_error_by_state": aggregates.get("average_action_pred_error_by_state", {}),  # Phase 4
            "ra_reorienting_success_rate": self.ra_reorienting_success_rate,
            "average_latent_reconstruction_by_state": aggregates.get("average_latent_reconstruction_by_state", {}),
            "average_latent_prior_kl_by_state": aggregates.get("average_latent_prior_kl_by_state", {}),
            "average_latent_sensory_consistency_by_state": aggregates.get("average_latent_sensory_consistency_by_state", {}),
            "average_latent_temporal_consistency_by_state": aggregates.get("average_latent_temporal_consistency_by_state", {}),
            "average_latent_vfe_total_by_state": aggregates.get("average_latent_vfe_total_by_state", {}),
            "average_prediction_error_by_state": aggregates.get("average_prediction_error_by_state", {}),
            "average_precision_by_state": aggregates.get("average_precision_by_state", {}),
        }

        with open(os.path.join(output_dir, f"active_inference_params_{self.experience_level}.json"), "w", encoding="utf-8") as f:
            json.dump(active_inf_params, f, indent=2)
