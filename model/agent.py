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
    get_params, get_thoughtseed_priors, get_exit_transition_probs
)
from .blankets import MarkovBlanketL1L2, MarkovBlanketL2L3
from utils.math_utils import (
    bernoulli_kl,
    bernoulli_entropy,
    clamp_activation,
    clip_probability,
    policy_posterior,
    policy_confidence,
    to_float,
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
        
        # Thoughtseed priors (state-dependent, FROZEN for both phenotypes)
        self.mu_params = nn.ParameterDict()
        for state in STATES:
            priors = get_thoughtseed_priors(state)
            mu_vec = [priors[ts] for ts in THOUGHTSEEDS]
            # Prior beliefs are fixed structural knowledge (Goal State)
            self.mu_params[state] = nn.Parameter(torch.tensor(mu_vec, dtype=torch.float32), requires_grad=False)
        
        # Expert: Unfrozen VAE (Learns amortized inference)
        # Novice: Frozen VAE (Stuck with random initialization)
        if self.level == 'novice':
            for param in self.vae.parameters():
                param.requires_grad = False
        
        # EMA for thoughtseed activations
        self.register_buffer('z_ema', torch.zeros(len(THOUGHTSEEDS)))
        self.z_ema_initialized = False
    
    def infer_z_from_x(self) -> torch.Tensor:
        """Bottom-up inference: encode networks (x) → posterior latent (z) aka thoughtseeds."""
        network_acts = self.blanket_l1l2.sensory_states
        device = next(self.vae.parameters()).device
        
        # Stack networks
        x = networks_to_tensor(network_acts, NETWORKS, device=device)
        
        # Encode
        # q(z|x): recognition density
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

    def compute_vfe(self, state: str, z: torch.Tensor, observed_networks: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute variational free energy F = reconstruction + KL(q||p)."""
        recon_x = self.decode_with_state(z)
        observed_x = networks_to_tensor(observed_networks, NETWORKS, device=z.device, detach=True)

        recon_loss = torch.nn.functional.mse_loss(recon_x, observed_x)
        prior = self.mu_params[state].detach()
        kl_div = bernoulli_kl(z, prior, EPS)

        return recon_loss + kl_div
    
    def update_posterior_z(
        self,
        current_state: str,
        activations: torch.Tensor,
        observed_networks: Dict[str, torch.Tensor],
        z_recognition: torch.Tensor,
    ) -> torch.Tensor:
        """Posterior update q(z): variational inference over thoughtseeds.

        Fixed-step VI for stability; precision modulates sensory vs prior influence.
        """
        device = activations.device
        clip_min = DEFAULTS['ACTIVATION_CLIP_MIN']
        clip_max = DEFAULTS['ACTIVATION_CLIP_MAX']

        # Observed networks as tensor
        observed_vec = networks_to_tensor(observed_networks, NETWORKS, device=device)
        
        # Targets
        z_prev = clamp_activation(activations.detach(), clip_min, clip_max)
        sensory_target = clamp_activation(z_recognition.detach(), clip_min, clip_max)
        prior_target = clamp_activation(self.mu_params[current_state].detach().clone(), clip_min, clip_max)
        
        # Sensory precision from L3 (bounded weight)
        precision = to_float(self.blanket_l2l3.active_states.get('precision_sensory', 1.0))
        precision_weight = clip_probability(precision)
        
        # Initialization: blend previous and sensory target, then bias toward prior by precision.
        z_init = 0.5 * z_prev + 0.5 * sensory_target
        z_init = (1.0 - precision_weight) * z_init + precision_weight * prior_target
        z_var = clamp_activation(z_init, clip_min, clip_max).requires_grad_(True)
        
        # Weights for F-energy terms
        # Low precision -> trust sensory/temporal terms
        # High precision -> trust prior
        sensory_w = 1.0 - precision_weight
        temporal_w = precision_weight

        # Variational optimization loop (fixed-step VI for stability)
        vi_steps = 2
        vi_lr = 0.2
        with torch.enable_grad():
            for _ in range(vi_steps):
                # Decode to networks (Generative Model)
                recon_x = self.decode_with_state(z_var)
                
                # Component Losses
                recon_loss = torch.nn.functional.mse_loss(recon_x, observed_vec)
                kl_div = bernoulli_kl(z_var, prior_target, EPS)
                sensory_loss = torch.nn.functional.mse_loss(z_var, sensory_target)
                temporal_loss = torch.nn.functional.mse_loss(z_var, z_prev)
                
                # Total Free Energy F(z)
                loss = (recon_loss + 
                        precision * kl_div + 
                        sensory_w * sensory_loss + 
                        temporal_w * temporal_loss)
                
                # Gradient Descent
                grad = torch.autograd.grad(loss, z_var, create_graph=False)[0]
                grad = torch.clamp(grad, -5.0, 5.0)
                
                z_var = clamp_activation(z_var - vi_lr * grad, clip_min, clip_max)
                z_var = z_var.detach().requires_grad_(True)
        
        # Finalize
        updated = z_var.detach()
        
        # EMA smoothing for stability
        if not self.z_ema_initialized:
            self.z_ema = updated.clone()
            self.z_ema_initialized = True
        else:
            self.z_ema = 0.75 * self.z_ema + 0.25 * updated
        
        return clamp_activation(updated, clip_min, clip_max)
    
    def infer_pi(self, z: torch.Tensor, current_state: str) -> Dict:
        """Policy inference via softmax posterior over candidate policies.
        
        Returns dict with:
            - selected_action_mu: posterior-weighted thoughtseed target
            - mu_x: posterior-weighted network target for L1
            - precision_gain: Precision modulation (weight)
            - policy_confidence: 1 - normalized entropy of q(pi)
        """
        # Get current networks for forward prediction
        x_current = networks_to_tensor(self.blanket_l1l2.sensory_states, NETWORKS, device=z.device)
        
        # Candidate policies (stay + next)
        candidates = [current_state]
        if current_state == 'mind_wandering':
            candidates.append('meta_awareness')
        elif current_state == 'meta_awareness':
            candidates.append('redirect_attention')
        elif current_state == 'redirect_attention':
            candidates.append('breath_focus')
        
        exit_probs = get_exit_transition_probs(self.level, current_state)
        avg_exit = (sum(exit_probs.values()) / len(exit_probs)) if exit_probs else (1.0 / max(len(candidates), 1))
        
        priors = []
        g_vals = []
        mu_candidates = []
        
        with torch.no_grad():
            for candidate_state in candidates:
                mu_candidate = self.mu_params[candidate_state]
                x_pred = self.vae.predict_next(x_current, mu_candidate)
                x_pref = self.decode_with_state(mu_candidate)
                
                # Risk for G(pi) (ambiguity disabled for stability)
                risk = 1.5 * bernoulli_kl(x_pred, x_pref, EPS)
                g_vals.append(float(risk.item()))
                
                mu_candidates.append(mu_candidate)
                
                if candidate_state == current_state:
                    prior = avg_exit
                else:
                    prior = exit_probs.get(candidate_state, avg_exit)
                priors.append(max(EPS, float(prior)))
        
        log_prior = np.log(np.array(priors, dtype=float))
        gamma = max(EPS, to_float(self.blanket_l2l3.active_states.get('policy_precision', 1.0)))
        q_pi = policy_posterior(log_prior, np.array(g_vals, dtype=float), gamma)
        pi_conf = policy_confidence(q_pi)
        policy_drive = float(1.0 - q_pi[0]) if len(q_pi) > 0 else 0.0
        
        weights = torch.tensor(q_pi, dtype=mu_candidates[0].dtype, device=mu_candidates[0].device)
        mu_stack = torch.stack(mu_candidates, dim=0)
        selected_mu = torch.sum(weights.unsqueeze(-1) * mu_stack, dim=0)
        mu_x = self.decode_with_state(selected_mu)
        
        precision = to_float(self.blanket_l2l3.active_states.get('precision_sensory', 1.0))
        precision_gain = clip_probability(precision)
        
        return {
            'selected_action_mu': selected_mu,
            'mu_x': mu_x,
            'precision_gain': precision_gain,
            'noise_reduction': float(np.clip(1.0 - 0.6 * precision_gain, 0.4, 1.0)),
            'policy_confidence': pi_conf,
            'policy_drive': policy_drive,
        }
