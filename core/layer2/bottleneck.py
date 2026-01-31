"""Layer 2: attentional agent bottleneck (VAE + policies)."""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, List, Tuple

from utils.meditation_config import THOUGHTSEEDS, STATES, NETWORKS, DEFAULTS
from utils.meditation_utils import get_actinf_params, get_thoughtseed_targets, compute_meta_awareness
from ..blankets.l1_l2 import MarkovBlanketL1L2
from ..blankets.l2_l3 import MarkovBlanketL2L3
from ..layer3.monitor import Layer3Monitor

class Layer2AttentionalModel(nn.Module):
    """Layer 2: attentional agents + learnable dynamics."""
    
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
        
        self.params = get_actinf_params(experience_level)
        
        self.register_buffer('aha_accum_val', torch.tensor(0.0))
        
        self.van_history: List[float] = []

        from .vae import MeditationVAE
        self.vae = MeditationVAE(
            input_dim=len(self.networks),
            latent_dim=self.num_thoughtseeds
        )
        
        self.mu_params = nn.ParameterDict()
        self.state_embed_dim = 2
        self.state_embeddings = nn.ParameterDict()
        self.state_embed_to_bias = nn.Linear(self.state_embed_dim, len(self.networks), bias=False)
        
        for state in self.states:
            mu_dict = get_thoughtseed_targets(state, 0.5, self.experience_level)
            mu_vec = [mu_dict[ts] for ts in self.thoughtseeds]
            self.mu_params[state] = nn.Parameter(torch.tensor(mu_vec, dtype=torch.float32))
            embed = torch.randn(self.state_embed_dim, dtype=torch.float32) * 0.05
            self.state_embeddings[state] = nn.Parameter(embed)
            
        
        self.register_buffer('z_ema', torch.zeros(self.num_thoughtseeds))
        self.z_ema_initialized = False

        self.blanket = MarkovBlanketL1L2(smoothing=0.7)
        self.blanket_l2l3 = MarkovBlanketL2L3(smoothing=0.7)
        
        self.monitor = Layer3Monitor(
            thoughtseeds=self.thoughtseeds,
            experience_level=self.experience_level,
            efe_ambiguity_weight=self.params.get('efe_ambiguity_weight', 0.4),
            l3tol2_precision_range=self.params.get('l3tol2_precision_range', (0.4, 0.6)),
            get_meta_awareness_fn=self.get_meta_awareness,
            blanket_l2l3=self.blanket_l2l3,
            vfe_ema_alpha=self.params['vfe_ema_alpha']
        )
        
        self.init_noise_sigma = float(self.params.get('z_noise_sigma', 0.0))
        self.learning_rate = self.params['learning_rate']
        self.z_ema_alpha = 0.75


    def get_meta_awareness(self, current_state: str, activations: torch.Tensor) -> float:
        """Compute meta-awareness (float)."""
        act_dict = {ts: activations[i].item() for i, ts in enumerate(self.thoughtseeds)}
        return compute_meta_awareness(
            state=current_state,
            thoughtseed_activations=act_dict
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
        return is_spike

    def _update_van_signals(self, network_acts: Dict[str, torch.Tensor], current_state: str) -> Tuple[bool, float]:
        """Update VAN spike and aha accumulator signals."""
        van = network_acts.get('VAN', torch.tensor(0.0)).item()
        van_spike_detected = self.detect_van_spike(van)
        if van_spike_detected:
            accum_update = 1.0
        elif van > 0.6:
            accum_update = 0.3
        else:
            accum_update = 0.0
        new_accum = self.params['aha_accum_decay'] * self.aha_accum_val + self.params['aha_accum_inc'] * accum_update
        self.aha_accum_val.copy_(new_accum.detach())
        return van_spike_detected, van

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

    def decode_with_state(self, z: torch.Tensor, current_state: str) -> torch.Tensor:
        """Decode thoughtseeds into networks with a learned state bias."""
        if z.dim() == 1:
            z_in = z.unsqueeze(0)
            decoded = self.vae.decode(z_in).squeeze(0)
        else:
            decoded = self.vae.decode(z)
        embed = self.state_embeddings.get(current_state)
        if embed is not None:
            bias = self.state_embed_to_bias(embed.unsqueeze(0)).squeeze(0)
            decoded = decoded + bias
        return torch.clamp(decoded, 0.0, 1.0)

    def update_thoughtseed_dynamics(self, current_activations: torch.Tensor, 
                                   current_state: str, current_dwell: int, dwell_limit: int,
                                   observed_networks: Dict[str, torch.Tensor],
                                   sensory_inference: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Update attentional-agent strengths."""
        van_spike_detected, _current_van = self._update_van_signals(observed_networks, current_state)

        mu_prior = self.mu_params[current_state]
        progress = 0.0
        if dwell_limit > 0:
            progress = min(1.0, current_dwell / max(1, dwell_limit))
        prec_mod = self.blanket_l2l3.active_states.get('precision_modulation', 0.5)
        if isinstance(prec_mod, torch.Tensor):
            prec_mod = prec_mod.item()
        prec_mod = float(np.clip(prec_mod, 0.0, 1.0))

        if sensory_inference is None:
            z_inferred = mu_prior
        else:
            w_prior = 0.2 + 0.6 * prec_mod + 0.2 * progress
            w_prior = max(0.15, min(w_prior, 0.85))
            z_inferred = (w_prior * mu_prior) + ((1 - w_prior) * sensory_inference)
        z_inferred = torch.clamp(z_inferred, DEFAULTS['ACTIVATION_CLIP_MIN'], DEFAULTS['ACTIVATION_CLIP_MAX'])

        # Dwell-scheduled latent noise for richer dynamics
        noise_sigma = self.params['z_noise_sigma']
        if dwell_limit > 0:
            noise_scale = 1.0 - min(1.0, current_dwell / max(1, dwell_limit))
        else:
            noise_scale = 1.0
        if noise_sigma > 0.0:
            noise = torch.randn_like(z_inferred) * noise_sigma * noise_scale
            z_inferred = torch.clamp(z_inferred + noise, DEFAULTS['ACTIVATION_CLIP_MIN'], DEFAULTS['ACTIVATION_CLIP_MAX'])

        # Soft WTA + global workspace broadcast (precision-gated).
        tau = max(0.25, 0.9 - (0.7 * prec_mod))
        winner = torch.softmax(z_inferred / tau, dim=0)
        workspace_gain = float(np.clip(prec_mod, 0.0, 1.0))
        z_inferred = torch.clamp(
            z_inferred + (workspace_gain * winner),
            DEFAULTS['ACTIVATION_CLIP_MIN'],
            DEFAULTS['ACTIVATION_CLIP_MAX']
        )

        alpha = self.z_ema_alpha
        aha_idx = self.ts_index.get('aha_moment')
        if aha_idx is not None and (z_inferred[aha_idx].item() > 0.6 or van_spike_detected):
            alpha = min(alpha, 0.6)
        alpha = min(alpha, 0.4 + 0.4 * progress)

        if current_dwell == 0:
            self.z_ema = z_inferred.clone()
            self.z_ema_initialized = True
        elif not self.z_ema_initialized or self.z_ema.shape != z_inferred.shape:
            self.z_ema = z_inferred.clone()
            self.z_ema_initialized = True
        else:
            self.z_ema = (alpha * self.z_ema) + ((1 - alpha) * z_inferred)

        updated_activations = torch.clamp(self.z_ema, DEFAULTS['ACTIVATION_CLIP_MIN'], DEFAULTS['ACTIVATION_CLIP_MAX'])
        
        self.blanket_l2l3.update_sensory_states({
            'aha_accumulator_value': float(self.aha_accum_val)
        })
        
        return updated_activations

    def prescriptive_action(self, z: torch.Tensor, vfe: torch.Tensor, current_state: str) -> dict:
        """Delegate policy to Layer 3."""
        
        # Agent bias: decode current mu into brain-network space (state-aware)
        mu_goal = self.mu_params[current_state]
        agent_bias = self.decode_with_state(mu_goal, current_state)
        
        # Recognition signal ties aha + current VFE proxy
        vfe_val = vfe.item() if isinstance(vfe, torch.Tensor) else float(vfe)
        vfe_sig = 1.0 / (1.0 + np.exp(-vfe_val))
        recognition = (0.7 * float(self.aha_accum_val)) + (0.3 * vfe_sig)

        self.blanket_l2l3.update_sensory_states({
            'thoughtseed_activations': z,
            'vfe': vfe_val,
            'current_state': current_state,
            'recognition_signal': recognition
        })
        
        prescription_l1l2 = self.monitor.evaluate_policies()
        
        # Inject Agent Bias into the prescription for Layer 1
        prescription_l1l2['agent_bias'] = agent_bias
        
        self.blanket.update_active_states(prescription_l1l2)
        
        return prescription_l1l2
