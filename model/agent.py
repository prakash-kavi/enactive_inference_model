"""Layer 2: Attentional agent with VAE, thoughtseeds, and forward dynamics.

Implements perception-action loop:
- Perception: Encode L1 networks → thoughtseed activations (recognition model)
- Dynamics: Update thoughtseeds via variational inference 
- Action: Select policy via forward-informed EFE minimization
- Forward model: Predict sensory consequences of actions (enactive)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional

from utils.config import (
    STATES, NETWORKS, THOUGHTSEEDS, DEFAULTS, EPS,
    get_params, get_thoughtseed_priors
)
from .blankets import MarkovBlanketL1L2, MarkovBlanketL2L3
from utils.math_utils import (
    bernoulli_kl,
    clamp_activation,
    clip_probability,
    networks_to_tensor,
)

class MeditationVAE(nn.Module):
    """VAE for network-thoughtseed mapping + forward dynamics."""
    
    def __init__(self, input_dim=4, latent_dim=5, hidden_dim=32):
        super().__init__()
        
        # Encoder: Networks → Thoughtseed logits
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Decoder: Thoughtseeds → Networks
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
        
        # Forward model: (Networks, Thoughtseeds) → Next Networks
        self.forward_net = nn.Sequential(
            nn.Linear(input_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Networks → Thoughtseed logits."""
        logits = self.encoder(x)
        return torch.sigmoid(logits)  # Independent strengths (no softmax)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Thoughtseeds → Networks."""
        return self.decoder(z)
    
    def predict_next(self, x_t: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        """Forward dynamics: predict next networks given current (x,z)."""
        if x_t.dim() == 1:
            x_t = x_t.unsqueeze(0)
            z_t = z_t.unsqueeze(0)
            combined = torch.cat([x_t, z_t], dim=-1)
            return self.forward_net(combined).squeeze(0)
        else:
            combined = torch.cat([x_t, z_t], dim=-1)
            return self.forward_net(combined)


class Layer2Agent(nn.Module):
    """Attentional agent: thoughtseeds + VAE + forward-informed action."""
    
    def __init__(self, experience_level: str = 'expert', 
                 blanket_l1l2: Optional[MarkovBlanketL1L2] = None,
                 blanket_l2l3: Optional[MarkovBlanketL2L3] = None):
        super().__init__()
        
        self.level = experience_level
        self.params = get_params(experience_level)
        
        # Markov blankets
        self.blanket_l1l2 = blanket_l1l2 or MarkovBlanketL1L2(smoothing=0.7)
        self.blanket_l2l3 = blanket_l2l3 or MarkovBlanketL2L3(smoothing=0.7)
        
        # VAE
        self.vae = MeditationVAE(
            input_dim=len(NETWORKS),
            latent_dim=len(THOUGHTSEEDS),
            hidden_dim=32
        )
        
        # Thoughtseed priors (state-dependent)
        self.mu_params = nn.ParameterDict()
        for state in STATES:
            priors = get_thoughtseed_priors(state, experience_level)
            mu_vec = [priors[ts] for ts in THOUGHTSEEDS]
            self.mu_params[state] = nn.Parameter(torch.tensor(mu_vec, dtype=torch.float32))
        
        # EMA for thoughtseed activations
        self.register_buffer('z_ema', torch.zeros(len(THOUGHTSEEDS)))
        self.z_ema_initialized = False
    
    def perceptual_inference(self) -> torch.Tensor:
        """Bottom-up: Encode L1 networks → thoughtseed activations."""
        network_acts = self.blanket_l1l2.sensory_states
        device = next(self.vae.parameters()).device
        
        # Stack networks
        x = networks_to_tensor(network_acts, NETWORKS, device=device)
        
        # Encode
        z = self.vae.encode(x.unsqueeze(0) if x.dim() == 1 else x)
        if z.dim() > 1:
            z = z.squeeze(0)
        
        return clamp_activation(z, DEFAULTS['ACTIVATION_CLIP_MIN'], DEFAULTS['ACTIVATION_CLIP_MAX'])
    
    def decode_with_state(self, z: torch.Tensor) -> torch.Tensor:
        """Top-down: Decode thoughtseeds → networks."""
        if z.dim() == 1:
            z_in = z.unsqueeze(0)
            decoded = self.vae.decode(z_in).squeeze(0)
        else:
            decoded = self.vae.decode(z)
        
        return torch.clamp(decoded, 0.0, 1.0)
    
    def update_thoughtseeds(self, 
                            current_state: str,
                            activations: torch.Tensor,
                            observed_networks: Dict[str, torch.Tensor],
                            sensory_inference: torch.Tensor) -> torch.Tensor:
        """Variational inference update for thoughtseed dynamics."""
        device = activations.device
        clip_min = DEFAULTS['ACTIVATION_CLIP_MIN']
        clip_max = DEFAULTS['ACTIVATION_CLIP_MAX']
        
        # Observed networks as tensor
        observed_vec = networks_to_tensor(observed_networks, NETWORKS, device=device)
        
        # Targets
        z_prev = clamp_activation(activations.detach(), clip_min, clip_max)
        sensory_target = clamp_activation(sensory_inference.detach(), clip_min, clip_max)
        # Clone prior to avoid gradient issues in BPTT
        prior_target = clamp_activation(self.mu_params[current_state].detach().clone(), clip_min, clip_max)
        
        # Precision modulation from L3
        precision = clip_probability(self.blanket_l2l3.active_states.get('precision_modulation', 0.5))
        
        # Initialize via blend
        z_init = 0.5 * z_prev + 0.5 * sensory_target
        z_init = (1.0 - 0.5 * precision) * z_init + (0.5 * precision) * prior_target
        z_var = clamp_activation(z_init, clip_min, clip_max).requires_grad_(True)
        
        # Variational optimization (2 steps)
        vi_lr = self.params['l2_vi_lr']
        obs_w = self.params['l2_vi_obs_weight']
        prior_w = self.params['l2_vi_prior_weight']
        sensory_w = self.params['l2_vi_sensory_weight'] * (1.0 - precision)
        temporal_w = self.params['l2_vi_temporal_weight']
        
        for _ in range(self.params['l2_vi_steps']):
            # Decode to networks
            recon_x = self.decode_with_state(z_var)
            
            # VFE terms
            recon_loss = torch.nn.functional.mse_loss(recon_x, observed_vec)
            
            # KL with prior (Bernoulli)
            kl_div = bernoulli_kl(z_var, prior_target, EPS)
            
            # Sensory + temporal consistency
            sensory_loss = torch.nn.functional.mse_loss(z_var, sensory_target)
            temporal_loss = torch.nn.functional.mse_loss(z_var, z_prev)
            
            # Total objective
            loss = (obs_w * recon_loss + 
                    prior_w * precision * kl_div + 
                    sensory_w * sensory_loss + 
                    temporal_w * temporal_loss)
            
            # Gradient descent
            grad = torch.autograd.grad(loss, z_var, create_graph=False)[0]
            grad = torch.clamp(grad, -5.0, 5.0)  # Clip gradients
            
            # Update (create new tensor, not in-place)
            z_var = clamp_activation(z_var - vi_lr * grad, clip_min, clip_max)
            z_var = z_var.detach().requires_grad_(True)
        
        # Finalize (detach from graph - VI is not part of BPTT)
        updated = z_var.detach()
        
        # EMA smoothing
        if not self.z_ema_initialized:
            self.z_ema = updated.clone()
            self.z_ema_initialized = True
        else:
            self.z_ema = 0.75 * self.z_ema + 0.25 * updated
        
        return clamp_activation(updated, clip_min, clip_max)
    
    def prescriptive_action(self, z: torch.Tensor, current_state: str) -> Dict:
        """Forward-informed action selection: evaluate candidates via prediction.
        
        Returns dict with:
            - selected_action_mu: Thoughtseed target for selected action
            - agent_bias: Network target for L1
            - l2tol1_enactive_bias: Precision modulation
            - transition_drive: Pressure to change state
        """
        # Get current networks for forward prediction
        x_current = networks_to_tensor(self.blanket_l1l2.sensory_states, NETWORKS, device=z.device)
        
        # Candidate actions (state attractors)
        candidates = [current_state]  # Always include "stay"
        if current_state == 'mind_wandering':
            candidates.append('meta_awareness')
        elif current_state == 'meta_awareness':
            candidates.append('redirect_attention')
        elif current_state == 'redirect_attention':
            candidates.append('breath_focus')
        
        # Evaluate each candidate via forward model
        best_mu = None
        min_error = float('inf')
        
        for candidate_state in candidates:
            mu_candidate = self.mu_params[candidate_state]
            
            # Forward prediction: what networks will result from this action?
            with torch.no_grad():
                x_next_pred = self.vae.predict_next(x_current, mu_candidate)
            
            # Expected networks from this action
            action_networks = self.decode_with_state(mu_candidate)
            
            # Prediction error
            error = torch.mean((x_next_pred - action_networks)**2).item()
            
            if error < min_error:
                min_error = error
                best_mu = mu_candidate
        
        # Selected action
        selected_mu = best_mu if best_mu is not None else self.mu_params[current_state]
        agent_bias = self.decode_with_state(selected_mu)
        
        # Precision from L3
        precision = clip_probability(self.blanket_l2l3.active_states.get('precision_modulation', 0.5))
        
        # Transition drive from L3
        transition_drive = clip_probability(self.blanket_l2l3.active_states.get('transition_drive', 0.0))
        
        # Package for L1 via Markov blanket
        return {
            'selected_action_mu': selected_mu,
            'agent_bias': agent_bias,
            'l2tol1_enactive_bias': precision,
            'noise_reduction': float(np.clip(1.0 - 0.6 * precision, 0.4, 1.0)),
            'transition_drive': transition_drive,
        }
