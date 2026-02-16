"""Layer 2: Attentional agent with VAE, thoughtseeds, and forward dynamics.

Perception-action loop (Eq. 2-8):
- Perception: q_phi(z|x) encodes L1 networks -> thoughtseeds
- Dynamics: fixed-step VI updates z (Eq. 3) under F(z) (Eq. 2)
- Action: policy posterior via expected free energy (Eq. 6-7)
- Forward model: predicts next networks for forward surprisal (Eq. 4)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional

from utils.config import (
    STATES, NETWORKS, THOUGHTSEEDS, DEFAULTS, EPS,
    get_params, get_thoughtseed_priors, get_exit_transition_probs
)
from .markov_blankets import MarkovBlanketL1L2, MarkovBlanketL2L3
from utils.math_utils import (
    bernoulli_kl,
    bernoulli_entropy,
    bernoulli_nll,
    mse_error,
    clamp_activation,
    clip_probability,
    policy_posterior,
    policy_precision,
    policy_confidence,
    normalize_scores,
    to_float,
    networks_to_tensor,
)

class MeditationVAE(nn.Module):
    """VAE for network-thoughtseed mapping + forward dynamics."""
    
    def __init__(self, input_dim=4, latent_dim=5, hidden_dim=32):
        super().__init__()
        
        # Encoder: Networks -> Thoughtseed logits
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Decoder: Thoughtseeds -> Networks
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
        
        # Forward model: (networks, thoughtseeds) -> next networks
        self.forward_net = nn.Sequential(
            nn.Linear(input_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode networks to thoughtseed activations q_phi(z|x)."""
        logits = self.encoder(x)
        return torch.sigmoid(logits)  # Independent strengths (no softmax)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode thoughtseeds to networks p_theta(x|z)."""
        return self.decoder(z)
    
    def predict_next(self, x_t: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        """Forward dynamics: predict next networks given (x_t, z_t)."""
        if x_t.dim() == 1:
            x_t = x_t.unsqueeze(0)
            z_t = z_t.unsqueeze(0)
            combined = torch.cat([x_t, z_t], dim=-1)
            return self.forward_net(combined).squeeze(0)
        else:
            combined = torch.cat([x_t, z_t], dim=-1)
            return self.forward_net(combined)


class Layer2Agent(nn.Module):
    """Attentional agent: thoughtseeds + VAE + EFE policy selection."""
    
    def __init__(self, experience_level: str = 'expert', 
                 blanket_l1l2: Optional[MarkovBlanketL1L2] = None,
                 blanket_l2l3: Optional[MarkovBlanketL2L3] = None):
        super().__init__()
        
        self.level = experience_level
        self.params = get_params(experience_level)
        
        # Markov blankets
        self.blanket_l1l2 = blanket_l1l2 or MarkovBlanketL1L2(smoothing=0.0)
        self.blanket_l2l3 = blanket_l2l3 or MarkovBlanketL2L3(smoothing=0.0)
        
        # VAE
        self.vae = MeditationVAE(
            input_dim=len(NETWORKS),
            latent_dim=len(THOUGHTSEEDS),
            hidden_dim=32
        )
        
        # Thoughtseed priors mu_z(s) (state-dependent, frozen)
        self.mu_params = nn.ParameterDict()
        for state in STATES:
            priors = get_thoughtseed_priors(state)
            mu_vec = [priors[ts] for ts in THOUGHTSEEDS]
            self.mu_params[state] = nn.Parameter(torch.tensor(mu_vec, dtype=torch.float32), requires_grad=False)
        
        
    
    def infer_z_from_x(self) -> torch.Tensor:
        """Bottom-up inference: encode networks (x) -> latent z (thoughtseeds)."""
        network_acts = self.blanket_l1l2.sensory_states
        x = networks_to_tensor(network_acts, NETWORKS)
        
        z = self.vae.encode(x.unsqueeze(0) if x.dim() == 1 else x)
        if z.dim() > 1:
            z = z.squeeze(0)
        
        return clamp_activation(z, DEFAULTS['CLIP_MIN'], DEFAULTS['CLIP_MAX'])
    
    def decode_with_state(self, z: torch.Tensor) -> torch.Tensor:
        """Top-down: decode thoughtseeds -> networks."""
        if z.dim() == 1:
            z_in = z.unsqueeze(0)
            decoded = self.vae.decode(z_in).squeeze(0)
        else:
            decoded = self.vae.decode(z)
        
        return torch.clamp(decoded, 0.0, 1.0)

    def compute_vfe(self, state: str, z: torch.Tensor, observed_networks: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute VFE (Eq. 2): reconstruction surprisal + KL complexity."""
        recon_x = self.decode_with_state(z)
        observed_x = networks_to_tensor(observed_networks, NETWORKS, detach=True)

        recon_loss = bernoulli_nll(recon_x, observed_x, EPS)
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
        """Posterior update q(z): fixed-step VI (Eq. 3) over thoughtseeds."""
        clip_min = DEFAULTS['CLIP_MIN']
        clip_max = DEFAULTS['CLIP_MAX']

        observed_vec = networks_to_tensor(observed_networks, NETWORKS)
        
        def clamp_z(t: torch.Tensor) -> torch.Tensor:
            return clamp_activation(t, clip_min, clip_max)

        z_prev = clamp_z(activations.detach())
        sensory_target = clamp_z(z_recognition.detach())
        prior_target = clamp_z(self.mu_params[current_state].detach().clone())
        
        # Sensory precision lambda_sens (Eq. 4)
        precision = to_float(self.blanket_l2l3.active_states.get('precision_sensory', 1.0))
        precision_weight_val = clip_probability(precision)
        
        # Initialization: blend previous and sensory target, then bias toward sensory when precision is high.
        z_init = 0.5 * z_prev + 0.5 * sensory_target
        z_init = precision_weight_val * z_init + (1.0 - precision_weight_val) * prior_target
        z_var = clamp_z(z_init).requires_grad_(True)
        
        sensory_w = precision_weight_val
        temporal_w = 1.0 - precision_weight_val

        # Variational optimization loop (fixed-step VI)
        vi_steps = 2
        vi_lr = 0.2
        with torch.enable_grad():
            for _ in range(vi_steps):
                recon_x = self.decode_with_state(z_var)
                
                recon_loss = bernoulli_nll(recon_x, observed_vec, EPS)
                kl_div = bernoulli_kl(z_var, prior_target, EPS)
                sensory_loss = mse_error(z_var, sensory_target)
                temporal_loss = mse_error(z_var, z_prev)
                
                loss = (recon_loss +
                        kl_div +
                        sensory_w * sensory_loss +
                        temporal_w * temporal_loss)
                
                grad = torch.autograd.grad(loss, z_var, create_graph=False)[0]
                grad = torch.clamp(grad, -5.0, 5.0)
                
                z_var = clamp_z(z_var - vi_lr * grad)
                z_var = z_var.detach().requires_grad_(True)
        
        return z_var.detach()

    def infer_z_step(
        self,
        current_state: str,
        activations: torch.Tensor,
        observed_networks: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Infer recognition and VI-refined thoughtseeds for the current step."""
        z_recognition = self.infer_z_from_x()
        z_posterior = self.update_posterior_z(
            current_state=current_state,
            activations=activations,
            observed_networks=observed_networks,
            z_recognition=z_recognition,
        )
        return z_posterior, z_recognition
    
    def infer_pi(self, z: torch.Tensor, current_state: str) -> Dict:
        """Policy inference via softmax posterior over candidate policies (Eq. 5-8).
        
        Returns dict with:
            - selected_action_mu: posterior-weighted thoughtseed target
            - mu_x: posterior-weighted network target for L1
            - precision_gain: Precision modulation (weight)
            - policy_confidence: 1 - normalized entropy of q(pi)
        """
        # Get current networks for forward prediction
        x_current = networks_to_tensor(self.blanket_l1l2.sensory_states, NETWORKS)
        
        candidates = [current_state] + [s for s in STATES if s != current_state]
        
        exit_probs = get_exit_transition_probs(self.level, current_state)
        avg_exit = (sum(exit_probs.values()) / len(exit_probs)) if exit_probs else (1.0 / max(len(candidates), 1))
        dwell_progress = to_float(self.blanket_l1l2.sensory_states.get('dwell_progress', 0.0))
        hazard = clip_probability(dwell_progress ** 2)
        
        priors = []
        g_vals = []
        mu_candidates = []
        
        with torch.no_grad():
            for candidate_state in candidates:
                mu_candidate = self.mu_params[candidate_state]
                x_pred = self.vae.predict_next(x_current, mu_candidate)
                x_pref = self.decode_with_state(mu_candidate)
                
                risk = bernoulli_kl(x_pred, x_pref, EPS)
                ambiguity = bernoulli_entropy(x_pred, EPS)
                g_vals.append(float((risk + ambiguity).item()))
                
                mu_candidates.append(mu_candidate)
                
                if candidate_state == current_state:
                    prior = 1.0 - hazard
                else:
                    prior = hazard * exit_probs.get(candidate_state, avg_exit)
                priors.append(max(EPS, float(prior)))
        
        log_prior = np.log(np.array(priors, dtype=float))
        g_array = normalize_scores(np.array(g_vals, dtype=float), EPS)

        gamma_prev = max(EPS, to_float(self.blanket_l2l3.active_states.get('policy_precision', 1.0)))
        q_pi = policy_posterior(log_prior, g_array, gamma_prev)
        gamma = policy_precision(q_pi, EPS)
        q_pi = policy_posterior(log_prior, g_array, gamma)
        pi_conf = policy_confidence(q_pi)
        policy_drive = float(1.0 - q_pi[0]) if len(q_pi) > 0 else 0.0
        
        weights = torch.tensor(q_pi, dtype=mu_candidates[0].dtype)
        mu_stack = torch.stack(mu_candidates, dim=0)
        selected_mu = torch.sum(weights.unsqueeze(-1) * mu_stack, dim=0)
        mu_current = self.mu_params[current_state].detach()
        selected_mu = (1.0 - hazard) * mu_current + hazard * selected_mu
        mu_x = self.decode_with_state(selected_mu)
        
        precision_sensory = to_float(self.blanket_l2l3.active_states.get('precision_sensory', 0.5))
        precision_gain = clip_probability(precision_sensory)
        self.blanket_l2l3.update_active_states({
            'policy_precision': gamma,
        })
        
        return {
            'selected_action_mu': selected_mu,
            'mu_x': mu_x,
            'precision_gain': precision_gain,
            'noise_reduction': float(np.clip(1.0 - 0.6 * precision_gain, 0.4, 1.0)),
            'policy_confidence': pi_conf,
            'policy_drive': policy_drive,
            'policy_precision': gamma,
            'precision_sensory': precision_sensory,
        }
