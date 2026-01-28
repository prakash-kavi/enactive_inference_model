"""Layer 2: GNWBottleneck (attentional agents + torch learning)."""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Tuple, List

from config.meditation_config import (
    THOUGHTSEEDS, STATES, NETWORKS,
    ActInfParams, ThoughtseedParams, MetacognitionParams,
    DEFAULTS, NETWORK_MODULATION
)
from .markov_blanket_l1_l2 import MarkovBlanketL1L2
from .markov_blanket_l2_l3 import MarkovBlanketL2L3
from .layer3_phenomenological_monitor import WitnessingLayer


class GNWBottleneck(nn.Module):
    """
    Layer 2: attentional agents + learnable dynamics.
    """
    
    def __init__(self, experience_level: str = 'novice', timesteps_per_cycle: int = 200, seed: Optional[int] = None):
        super().__init__()
        
        self.experience_level = experience_level
        self.timesteps = timesteps_per_cycle
        self.thoughtseeds = THOUGHTSEEDS
        self.states = STATES
        self.num_thoughtseeds = len(self.thoughtseeds)
        self.networks = NETWORKS
        self.ts_index = {ts: i for i, ts in enumerate(self.thoughtseeds)}
        self.rng = np.random.RandomState(seed)
        
        self.params = ActInfParams.expert() if experience_level == 'expert' else ActInfParams.novice()
        
        self.register_buffer('aha_accum_val', torch.tensor(0.0))
        
        self.activations_history = []
        self.state_history = []
        self.meta_awareness_history = []
        self.network_activations_history = []
        self.free_energy_history = []
        self.prediction_error_history = []
        self.precision_history = []
        self.neural_efficiency_history = []
        self.stability_indicators = []
        self.dominant_ts_history = []
        
        self.van_history: List[float] = []
        self.van_spike_detections = 0
        self.expert_mind_wandering_detections = 0

        from .meditation_vae import MeditationVAE
        self.vae = MeditationVAE(
            input_dim=len(self.networks),
            latent_dim=self.num_thoughtseeds
        )
        
        self.theta_params = nn.ParameterDict()
        self.mu_params = nn.ParameterDict()
        
        for state in self.states:
            mu_dict = ThoughtseedParams.get_target_activations(state, 0.5, self.experience_level)
            mu_vec = [mu_dict[ts] for ts in self.thoughtseeds]
            self.mu_params[state] = nn.Parameter(torch.tensor(mu_vec, dtype=torch.float32))
            
            theta_mat = self._init_competitive_theta(state)
            self.theta_params[state] = nn.Parameter(torch.tensor(theta_mat, dtype=torch.float32))

        self.blanket = MarkovBlanketL1L2(smoothing=0.7)
        self.blanket_l2l3 = MarkovBlanketL2L3(smoothing=0.7)
        
        self.monitor = WitnessingLayer(
            networks=self.networks,
            thoughtseeds=self.thoughtseeds,
            sensory_precision_base=self.params.sensory_precision_base,
            prior_precision_base=self.params.prior_precision_base,
            precision_weight=self.params.precision_weight,
            complexity_penalty=self.params.complexity_penalty,
            get_meta_awareness_fn=self.get_meta_awareness,
            blanket_l2l3=self.blanket_l2l3
        )
        
        self.noise_level = self.params.noise_level
        self.learning_rate = self.params.learning_rate


    def _init_competitive_theta(self, state: str) -> np.ndarray:
        """Initialize theta matrix (numpy)."""
        if self.experience_level == 'expert':
            diag_base = 0.30
            inhibition_strength = 0.25
            synergy_strength = -0.15
        else:
            diag_base = 0.20
            inhibition_strength = 0.15
            synergy_strength = -0.05
        
        theta = np.eye(5) * diag_base
        idx_attend, idx_pain, idx_tasks, idx_aha, idx_equanimity = 0, 1, 2, 3, 4
        
        if state == 'breath_focus':
            theta[idx_attend, idx_tasks] = theta[idx_tasks, idx_attend] = inhibition_strength * 0.7
            theta[idx_attend, idx_pain] = theta[idx_pain, idx_attend] = inhibition_strength * 0.7
            theta[idx_attend, idx_equanimity] = theta[idx_equanimity, idx_attend] = synergy_strength
            theta[idx_equanimity, idx_tasks] = theta[idx_tasks, idx_equanimity] = inhibition_strength * 0.3
            theta[idx_equanimity, idx_pain] = theta[idx_pain, idx_equanimity] = inhibition_strength * 0.3
        elif state == 'mind_wandering':
            theta[idx_tasks, idx_attend] = theta[idx_attend, idx_tasks] = inhibition_strength * 0.7
            theta[idx_tasks, idx_equanimity] = theta[idx_equanimity, idx_tasks] = inhibition_strength * 0.7
            theta[idx_tasks, idx_pain] = theta[idx_pain, idx_tasks] = synergy_strength
            theta[idx_pain, idx_attend] = theta[idx_attend, idx_pain] = inhibition_strength * 0.3
            theta[idx_tasks, idx_tasks] = diag_base * 0.7
            theta[idx_pain, idx_pain] = diag_base * 0.7
        elif state == 'meta_awareness':
            for i in [idx_attend, idx_pain, idx_tasks, idx_equanimity]:
                theta[idx_aha, i] = theta[i, idx_aha] = inhibition_strength * 0.9
            theta[idx_aha, idx_aha] = diag_base * 1.1
        elif state == 'redirect_breath':
            theta[idx_attend, idx_tasks] = theta[idx_tasks, idx_attend] = inhibition_strength * 0.8
            theta[idx_attend, idx_pain] = theta[idx_pain, idx_attend] = inhibition_strength * 0.8
            theta[idx_attend, idx_equanimity] = theta[idx_equanimity, idx_attend] = synergy_strength
            theta[idx_equanimity, idx_tasks] = theta[idx_tasks, idx_equanimity] = inhibition_strength * 0.4
            theta[idx_equanimity, idx_pain] = theta[idx_pain, idx_equanimity] = inhibition_strength * 0.4
            theta[idx_attend, idx_attend] = diag_base * 1.05

        # Gershgorin stability (initial pass)
        epsilon = 0.05
        row_sums = np.sum(np.abs(theta), axis=1) - np.diag(theta)
        for i in range(5):
            if theta[i, i] <= row_sums[i]:
                theta[i, i] = row_sums[i] + epsilon
        return theta

    def get_meta_awareness(self, current_state: str, activations: torch.Tensor) -> float:
        """Compute meta-awareness (float)."""
        act_dict = {ts: activations[i].item() for i, ts in enumerate(self.thoughtseeds)}
        return MetacognitionParams.calculate_meta_awareness(
            state=current_state,
            thoughtseed_activations=act_dict,
            experience_level=self.experience_level
        )

    def detect_van_spike(self, current_van: float) -> bool:
        """Detect VAN spike."""
        self.van_history.append(current_van)
        if len(self.van_history) > 10:
            self.van_history.pop(0)
        
        if len(self.van_history) < 2:
            return False
            
        prev_van = self.van_history[-2]
        spike_threshold = 0.7
        spike_delta = 0.15
        
        is_spike = (current_van > spike_threshold) and ((current_van - prev_van) > spike_delta)
        if is_spike:
            self.van_spike_detections += 1
        return is_spike

    def get_network_modulation(self, network_acts: Dict[str, torch.Tensor], current_state: str) -> Tuple[torch.Tensor, bool, float]:
        """Compute network-driven modulation of attentional agents."""
        device = self.vae.encoder_net[0].weight.device
        modulations = torch.zeros(self.num_thoughtseeds, device=device)
        
        mods = NETWORK_MODULATION
        
        dmn = network_acts.get('DMN', torch.tensor(0.0)).item()
        van = network_acts.get('VAN', torch.tensor(0.0)).item()
        dan = network_acts.get('DAN', torch.tensor(0.0)).item()
        fpn = network_acts.get('FPN', torch.tensor(0.0)).item()

        idx = self.ts_index

        def add_mod(ts, val):
            modulations[idx[ts]] += val

        # DMN
        add_mod('pending_tasks', mods['DMN']['pending_tasks'] * dmn)
        add_mod('aha_moment', mods['DMN']['aha_moment'] * dmn)
        add_mod('attend_breath', mods['DMN']['attend_breath'] * dmn)
        
        # VAN
        add_mod('pain_discomfort', mods['VAN']['pain_discomfort'] * van)
        
        # VAN Spike Logic
        van_spike_detected = self.detect_van_spike(van)
        accum_update = 0.0
        if van_spike_detected:
            accum_update = 1.0
        elif van > 0.6:
            accum_update = 0.3
        else:
            accum_update = 0.0
        
        new_accum = self.params.aha_accum_decay * self.aha_accum_val + self.params.aha_accum_inc * accum_update
        self.aha_accum_val.copy_(new_accum.detach()) # Detached state
        
        if current_state == "meta_awareness":
            add_mod('aha_moment', mods['VAN']['aha_moment_meta_awareness'] * van)
        elif van_spike_detected:
             add_mod('aha_moment', 0.2 * van)
             
        # DAN
        add_mod('attend_breath', mods['DAN']['attend_breath'] * dan)
        add_mod('pending_tasks', mods['DAN']['pending_tasks'] * dan)
        add_mod('pain_discomfort', mods['DAN']['pain_discomfort'] * dan)
        
        # FPN
        fpn_enh = self.params.fpn_enhancement
        add_mod('aha_moment', mods['FPN']['aha_moment'] * fpn * fpn_enh)
        add_mod('equanimity', mods['FPN']['equanimity'] * fpn * fpn_enh)
        
        return modulations, van_spike_detected, van

    def perceptual_inference(self) -> torch.Tensor:
        """Perceptual inference (networks -> thoughtseed strengths)."""
        network_acts = self.blanket.sensory_states
        device = self.vae.encoder_net[0].weight.device
        
        # Convert inputs to tensor
        net_vec = torch.stack([network_acts.get(net, torch.tensor(0.0, device=device)) for net in self.networks])
        
        # Encoder (q(z|x))
        if net_vec.dim() == 1:
            x_in = net_vec.unsqueeze(0)
            logits = self.vae.encode(x_in)
            # Independent strengths: sigmoid per thoughtseed (no simplex constraint)
            inferred = torch.sigmoid(logits).squeeze(0)
        else:
            logits = self.vae.encode(net_vec)
            inferred = torch.sigmoid(logits)
        
        return torch.clamp(inferred, DEFAULTS['ACTIVATION_CLIP_MIN'], DEFAULTS['ACTIVATION_CLIP_MAX'])

    def update_thoughtseed_dynamics(self, current_activations: torch.Tensor, 
                                   current_state: str, current_dwell: int, dwell_limit: int,
                                   observed_networks: Dict[str, torch.Tensor],
                                   sensory_inference: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Update attentional-agent strengths."""
        dt = DEFAULTS['DEFAULT_DT']
        device = current_activations.device
        
        # 1. PRIOR: Learnable Mu
        mu_prior = self.mu_params[current_state].clone()
        
        # Network Modulations
        modulations, van_spike_detected, current_van = self.get_network_modulation(observed_networks, current_state)
        
        mu_prior = mu_prior + modulations
        
        # Aha accumulator boost
        if self.aha_accum_val > 0.01:
            mu_prior[self.ts_index['aha_moment']] += self.aha_accum_val * self.params.aha_target_gain
            
        # 2. Likelihood & precision weighting
        if sensory_inference is not None:
            ma_factor = self.blanket_l2l3.sensory_states.get('meta_awareness', 0.5)
            if isinstance(ma_factor, torch.Tensor):
                ma_factor = ma_factor.item()
            
            w_prior = 0.4 + 0.5 * ma_factor
            w_sensory = 1.0 - w_prior
            
            mu = (w_prior * mu_prior) + (w_sensory * sensory_inference)
        else:
            mu = mu_prior
            
        # 3. Distraction & fatigue
        fatigue_buffer = self.blanket.active_states.get('fatigue_buffer', 1.0)
        if isinstance(fatigue_buffer, torch.Tensor):
            fatigue_buffer = fatigue_buffer.item()
        
        if current_state in ["breath_focus", "redirect_breath"]:
             progress = min(1.5, current_dwell / max(5, dwell_limit))
             distraction_buildup = self.params.distraction_pressure * fatigue_buffer * progress
             
             mu[self.ts_index["pain_discomfort"]] += distraction_buildup
             mu[self.ts_index["pending_tasks"]] += distraction_buildup
             
             idx_bf = self.ts_index["attend_breath"]
             mu[idx_bf] = torch.max(torch.tensor(0.1, device=device), mu[idx_bf] - (self.params.fatigue_rate * fatigue_buffer * progress))

        # 4. Competitive dynamics
        theta_matrix = self.theta_params[current_state]
        sigma_matrix = torch.eye(5, device=device) * self.params.base_sigma 
        
        # Grip
        prec_mod = self.blanket_l2l3.active_states.get('precision_modulation', 1.0)
        if isinstance(prec_mod, torch.Tensor):
            prec_mod = prec_mod.item()
        
        pi_vec = torch.ones(self.num_thoughtseeds, device=device) * prec_mod
        dom_idx = torch.argmax(current_activations)
        pi_vec[dom_idx] *= 1.25
        
        theta_eff = theta_matrix * pi_vec.unsqueeze(1)
        
        # Update
        z = current_activations
        damping = 0.7
        drift = -torch.matmul(theta_eff, (z - mu)) * dt * damping
        
        noise_scale = 0.7 if self.experience_level == 'expert' else 1.0
        dW = torch.randn(self.num_thoughtseeds, device=device)
        noise = torch.matmul(sigma_matrix, dW) * np.sqrt(dt) * noise_scale
        
        updated_activations = torch.clamp(z + drift + noise, DEFAULTS['ACTIVATION_CLIP_MIN'], DEFAULTS['ACTIVATION_CLIP_MAX'])
        
        self.blanket_l2l3.update_sensory_states({
            'dominant_thoughtseed': self.thoughtseeds[int(torch.argmax(updated_activations))],
            'dominant_activation': updated_activations.max().detach().item(),
            'van_spike_detected': van_spike_detected,
            'aha_accumulator_value': float(self.aha_accum_val),
            'current_van': float(current_van)
        })
        
        return updated_activations

    def get_target_activations(self, state: str, meta_awareness: float) -> np.ndarray:
        """Helper to get target activations (Numpy) for initialization compatibility."""
        targets_dict = ThoughtseedParams.get_target_activations(state, meta_awareness, self.experience_level)
        target_activations = np.zeros(self.num_thoughtseeds)
        for i, ts in enumerate(self.thoughtseeds):
            target_activations[i] = targets_dict[ts]
        # Add noise using numpy rng
        target_activations += self.rng.normal(0, self.noise_level, size=self.num_thoughtseeds)
        return np.clip(target_activations, DEFAULTS['ACTIVATION_CLIP_MIN'], DEFAULTS['ACTIVATION_CLIP_MAX'])

    def prescriptive_action(self, z: torch.Tensor, vfe: torch.Tensor, current_state: str) -> dict:
        """Delegate policy to Layer 3."""
        
        # Agent bias: decode current mu into brain-network space
        mu_goal = self.mu_params[current_state]
        
        # agent_bias = Decoder(mu_goal)
        if mu_goal.dim() == 1:
            z_in = mu_goal.unsqueeze(0)
            bias_batch = self.vae.decode(z_in)
            agent_bias = bias_batch.squeeze(0)
        else:
            agent_bias = self.vae.decode(mu_goal)
        
        # Recognition signal ties aha + current VFE proxy
        vfe_val = vfe.item() if isinstance(vfe, torch.Tensor) else float(vfe)
        vfe_sig = 1.0 / (1.0 + np.exp(-vfe_val))
        recognition = (0.7 * float(self.aha_accum_val)) + (0.3 * vfe_sig)

        self.blanket_l2l3.update_sensory_states({
            'thoughtseed_activations': z,
            'current_state': current_state,
            'recognition_signal': recognition
        })
        
        prescription_l1l2 = self.monitor.evaluate_policies()
        
        # Inject Agent Bias into the prescription for Layer 1
        prescription_l1l2['agent_bias'] = agent_bias
        
        self.blanket.update_active_states(prescription_l1l2)
        
        return prescription_l1l2
