"""Layer 2: Attentional agent with recognition/decoder, thoughtseeds, and forward dynamics.

Perception-action loop:
- Perception: encode L1 networks -> thoughtseeds
- Dynamics: fixed-step VI updates z under F(z)
- Action: policy posterior via expected free energy
- Forward model: predicts next networks for forward surprisal
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional

from utils.config import (
    STATES, NETWORKS, THOUGHTSEEDS, DEFAULTS, EPS,
    THOUGHTSEED_STATE_PRIORS,
    VI_REFINEMENT_STATES,
    get_exit_transition_probs, get_policy_candidate_order,
    VI_STEPS, VI_LR,
)
from .markov_blankets import MarkovBlanketL1L2, MarkovBlanketL2L3
from .phenotype import PhenotypeConfig, EXPERT_PHENOTYPE, POLICY_GAMMA
from utils.math_utils import (
    mse_error,
    recon_error,
    prior_error,
    forward_error,
    clamp_activation,
    clip_probability,
    policy_posterior,
    normalize_scores,
    to_float,
    networks_to_tensor,
)

class MeditationVAE(nn.Module):
    """Recognition/decoder for network-thoughtseed mapping + forward dynamics."""
    
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
    """Attentional agent: thoughtseeds + recognition/decoder + EFE policy selection."""
    
    def __init__(self, phenotype: PhenotypeConfig = None,
                 blanket_l1l2: Optional[MarkovBlanketL1L2] = None,
                 blanket_l2l3: Optional[MarkovBlanketL2L3] = None):
        super().__init__()

        self.phenotype = phenotype if phenotype is not None else EXPERT_PHENOTYPE
        self.level = self.phenotype.level   # used for config table lookups
        
        # Markov blankets
        self.blanket_l1l2 = blanket_l1l2 or MarkovBlanketL1L2(smoothing=0.0)
        self.blanket_l2l3 = blanket_l2l3 or MarkovBlanketL2L3(smoothing=0.0)
        
        # Recognition/decoder
        self.vae = MeditationVAE(
            input_dim=len(NETWORKS),
            latent_dim=len(THOUGHTSEEDS),
            hidden_dim=32
        )
        
        # Thoughtseed priors mu_z(s) (state-dependent, frozen)
        self.mu_params = nn.ParameterDict()
        for state in STATES:
            priors = THOUGHTSEED_STATE_PRIORS[state].copy()
            mu_vec = [priors[ts] for ts in THOUGHTSEEDS]
            self.mu_params[state] = nn.Parameter(torch.tensor(mu_vec, dtype=torch.float32), requires_grad=False)
        
    
    def decode_with_state(self, z: torch.Tensor) -> torch.Tensor:
        """Top-down: decode thoughtseeds -> networks."""
        if z.dim() == 1:
            z_in = z.unsqueeze(0)
            decoded = self.vae.decode(z_in).squeeze(0)
        else:
            decoded = self.vae.decode(z)
        
        return torch.clamp(decoded, 0.0, 1.0)

    def compute_vfe(self, state: str, z: torch.Tensor, observed_networks: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute VFE: reconstruction error + prior matching."""
        recon_x = self.decode_with_state(z)
        observed_x = networks_to_tensor(observed_networks, NETWORKS, detach=True)

        recon_loss = recon_error(recon_x, observed_x)
        prior = self.mu_params[state].detach()
        prior_match = prior_error(z, prior)

        return recon_loss + prior_match
    
    def update_posterior_z(
        self,
        current_state: str,
        activations: torch.Tensor,
        z_recognition: torch.Tensor,
    ) -> torch.Tensor:
        """Posterior update q(z): fixed-step VI over thoughtseeds."""
        clip_min = DEFAULTS['CLIP_MIN']
        clip_max = DEFAULTS['CLIP_MAX']

        observed_vec = networks_to_tensor(self.blanket_l1l2.sensory_states, NETWORKS)
        
        def clamp_z(t: torch.Tensor) -> torch.Tensor:
            return clamp_activation(t, clip_min, clip_max)

        z_prev = clamp_z(activations.detach())
        sensory_target = clamp_z(z_recognition.detach())
        prior_target = clamp_z(self.mu_params[current_state].detach().clone())
        
        # Sensory precision lambda_sens
        precision = to_float(self.blanket_l2l3.active_states.get('precision_sensory', 1.0))
        precision_weight_val = clip_probability(precision)
        
        # Initialization: blend previous and sensory target, then bias toward sensory when precision is high.
        z_init = 0.5 * z_prev + 0.5 * sensory_target
        z_init = precision_weight_val * z_init + (1.0 - precision_weight_val) * prior_target
        z_var = clamp_z(z_init).requires_grad_(True)
        
        sensory_w = precision_weight_val
        temporal_w = 1.0 - precision_weight_val

        # Variational optimization loop (fixed-step VI)
        vi_steps = int(VI_STEPS) if current_state in VI_REFINEMENT_STATES else 0
        vi_lr = float(VI_LR)
        with torch.enable_grad():
            for _ in range(vi_steps):
                recon_x = self.decode_with_state(z_var)
                
                recon_loss = recon_error(recon_x, observed_vec)
                prior_match = prior_error(z_var, prior_target)
                sensory_loss = mse_error(z_var, sensory_target)
                temporal_loss = mse_error(z_var, z_prev)

                loss = (recon_loss +
                        prior_match +
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Infer recognition and VI-refined thoughtseeds for the current step."""
        # Bottom-up inference: encode networks (x) -> latent z (thoughtseeds)
        network_acts = self.blanket_l1l2.sensory_states
        x = networks_to_tensor(network_acts, NETWORKS)
        z = self.vae.encode(x.unsqueeze(0) if x.dim() == 1 else x)
        if z.dim() > 1:
            z = z.squeeze(0)
        z_recognition = clamp_activation(z, DEFAULTS['CLIP_MIN'], DEFAULTS['CLIP_MAX'])

        z_posterior = self.update_posterior_z(
            current_state=current_state,
            activations=activations,
            z_recognition=z_recognition,
        )
        return z_posterior, z_recognition
    
    # ------------------------------------------------------------------
    # Policy inference sub-steps
    # ------------------------------------------------------------------

    def _compute_dwell_prior(self, current_state: str, candidates: list,
                             hazard: float, exit_probs: dict) -> list:
        """Dwell-aware prior E(pi) over candidate policies (heuristic; no paper equation).

        Stay-prior = 1-h; exit-prior distributed over transition probabilities.
        h = dwell_progress^2 (quadratic hazard). Returns list of prior floats.
        """
        avg_exit = (sum(exit_probs.values()) / len(exit_probs)) if exit_probs else (
            1.0 / max(len(candidates), 1))
        priors = []
        for s in candidates:
            if s == current_state:
                priors.append(max(EPS, 1.0 - hazard))
            else:
                priors.append(max(EPS, hazard * exit_probs.get(s, avg_exit)))
        return priors

    def _evaluate_efe(self, x_current: torch.Tensor,
                      candidates: list):
        """Risk-only EFE G(pi) as MSE between predicted and preferred networks.

        FA meditation is convergent: ambiguity term omitted.
        Returns (g_vals: list[float], mu_candidates: list[Tensor]).
        """
        g_vals, mu_candidates = [], []
        with torch.no_grad():
            for s in candidates:
                mu_c = self.mu_params[s]
                x_pred = self.vae.predict_next(x_current, mu_c)
                x_pref = self.decode_with_state(mu_c)
                g_vals.append(float(forward_error(x_pred, x_pref).item()))
                mu_candidates.append(mu_c)
        return g_vals, mu_candidates

    def _compute_posterior(self, priors: list, g_vals: list):
        """Fixed policy precision gamma and softmax posterior q(pi).

        Gamma is a fixed scalar (POLICY_GAMMA). L3 learned log-prior is added
        to log E(pi) if available.
        Returns (q_pi, gamma, policy_drive).
        """
        log_prior = np.log(np.array(priors, dtype=float))
        l3_prior = self.blanket_l2l3.active_states.get('policy_prior')
        if l3_prior is not None and len(l3_prior) == len(log_prior):
            log_prior = log_prior + np.array(l3_prior, dtype=float)

        g_array = normalize_scores(np.array(g_vals, dtype=float), EPS)

        gamma = max(EPS, float(POLICY_GAMMA))
        q_pi = policy_posterior(log_prior, g_array, gamma)
        policy_drive = float(1.0 - q_pi[0]) if len(q_pi) > 0 else 0.0
        return q_pi, gamma, policy_drive

    def _select_attractor(self, q_pi, mu_candidates: list):
        """Posterior-weighted latent prediction.

        mu = sum_pi q(pi) mu_z(s_pi)
        Returns (selected_mu, mu_x).
        """
        weights = torch.tensor(q_pi, dtype=mu_candidates[0].dtype)
        mu = torch.sum(weights.unsqueeze(-1) * torch.stack(mu_candidates, 0), dim=0)
        return mu, self.decode_with_state(mu)

    # ------------------------------------------------------------------

    def infer_pi(self, current_state: str) -> Dict:
        """Policy inference: evaluate risk and select action."""
        x_current  = networks_to_tensor(self.blanket_l1l2.sensory_states, NETWORKS)
        candidates = get_policy_candidate_order(current_state)
        exit_probs = get_exit_transition_probs(self.level, current_state)
        hazard     = clip_probability(
            to_float(self.blanket_l1l2.sensory_states.get('dwell_progress', 0.0)) ** 2)

        priors                           = self._compute_dwell_prior(current_state, candidates, hazard, exit_probs)  # dwell prior (heuristic)
        g_vals, mu_candidates            = self._evaluate_efe(x_current, candidates)
        q_pi, gamma, pol_drive  = self._compute_posterior(priors, g_vals)
        selected_mu, mu_x       = self._select_attractor(q_pi, mu_candidates)

        policy_state_probs = {
            state: float(q_pi[i]) for i, state in enumerate(candidates)
        }

        return {
            'selected_action_mu': selected_mu,
            'mu_x':               mu_x,
            'policy_drive':       pol_drive,
            'q_pi':               np.array(q_pi, dtype=np.float64),
            'policy_state_probs': policy_state_probs,
        }
