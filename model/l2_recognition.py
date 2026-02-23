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
    THOUGHTSEED_STATE_PRIORS,
    get_exit_transition_probs, get_policy_candidate_order,
)
from .markov_blankets import MarkovBlanketL1L2, MarkovBlanketL2L3
from .phenotype import PhenotypeConfig, EXPERT_PHENOTYPE
from utils.math_utils import (
    bernoulli_kl,
    bernoulli_nll,
    mse_error,
    clamp_activation,
    clip_probability,
    policy_posterior,
    policy_precision,
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
    
    def __init__(self, phenotype: PhenotypeConfig = None,
                 blanket_l1l2: Optional[MarkovBlanketL1L2] = None,
                 blanket_l2l3: Optional[MarkovBlanketL2L3] = None):
        super().__init__()

        self.phenotype = phenotype if phenotype is not None else EXPERT_PHENOTYPE
        self.level = self.phenotype.level   # used for config table lookups
        self.params = {'learning_rate': self.phenotype.learning_rate}
        
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
            observed_networks=observed_networks,
            z_recognition=z_recognition,
        )
        return z_posterior, z_recognition
    
    # ------------------------------------------------------------------
    # Policy inference sub-steps (Eqs. 5–8)
    # ------------------------------------------------------------------

    def _compute_dwell_prior(self, current_state: str, candidates: list,
                             hazard: float, exit_probs: dict) -> list:
        """Eq. 5 — Dwell-aware prior E(π) over candidate policies.

        Stay-prior = 1-h; exit-prior distributed over transition probabilities.
        h = dwell_progress² (quadratic hazard). Returns list of prior floats.
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
        """Eq. 6 — Risk-only EFE G(π) = D_KL(x̂_π ‖ C_{s_π}) per policy.

        FA meditation is convergent: ambiguity term omitted.
        Returns (g_vals: list[float], mu_candidates: list[Tensor]).
        """
        g_vals, mu_candidates = [], []
        with torch.no_grad():
            for s in candidates:
                mu_c = self.mu_params[s]
                x_pred = self.vae.predict_next(x_current, mu_c)
                x_pref = self.decode_with_state(mu_c)
                g_vals.append(float(bernoulli_kl(x_pred, x_pref, EPS).item()))
                mu_candidates.append(mu_c)
        return g_vals, mu_candidates

    def _compute_posterior(self, priors: list, g_vals: list):
        """Eq. 7 — Policy precision γ and softmax posterior q(π).

        γ = 1 - H(E(π)) / log|Π| computed from prior entropy (no iteration).
        L3 learned log-prior ℓ_π(s) is added to log E(π) if available.
        Returns (q_pi, gamma, pi_conf, policy_drive).
        """
        log_prior = np.log(np.array(priors, dtype=float))
        l3_prior = self.blanket_l2l3.active_states.get('policy_prior')
        if l3_prior is not None and len(l3_prior) == len(log_prior):
            log_prior = log_prior + np.array(l3_prior, dtype=float)

        g_array = normalize_scores(np.array(g_vals, dtype=float), EPS)

        prior_arr = np.array(priors, dtype=float)
        prior_arr = prior_arr / (prior_arr.sum() + EPS)
        prior_entropy = float(-np.sum(prior_arr * np.log(prior_arr + EPS)))
        gamma = float(np.clip(
            1.0 - prior_entropy / np.log(max(len(priors), 2)), 0.0, 1.0))

        q_pi = policy_posterior(log_prior, g_array, gamma)
        pi_conf = policy_precision(q_pi)
        policy_drive = float(1.0 - q_pi[0]) if len(q_pi) > 0 else 0.0
        return q_pi, gamma, pi_conf, policy_drive

    def _select_attractor(self, q_pi, mu_candidates: list,
                          current_state: str, hazard: float):
        """Eq. 8 — Posterior-weighted latent target with dwell blending.

        μ = Σ_π q(π) μ_z(s_π)
        μ_sel = (1-h) μ_z(s_t) + h μ  — graded transition, preserves stochasticity.
        Returns (selected_mu, mu_x).
        """
        weights = torch.tensor(q_pi, dtype=mu_candidates[0].dtype)
        mu = torch.sum(weights.unsqueeze(-1) * torch.stack(mu_candidates, 0), dim=0)
        mu_current = self.mu_params[current_state].detach()
        selected_mu = (1.0 - hazard) * mu_current + hazard * mu
        return selected_mu, self.decode_with_state(selected_mu)

    # ------------------------------------------------------------------

    def infer_pi(self, z: torch.Tensor, current_state: str) -> Dict:
        """Policy inference — orchestrates Eqs. 5 → 6 → 7 → 8."""
        x_current  = networks_to_tensor(self.blanket_l1l2.sensory_states, NETWORKS)
        candidates = get_policy_candidate_order(current_state)
        exit_probs = get_exit_transition_probs(self.level, current_state)
        hazard     = clip_probability(
            to_float(self.blanket_l1l2.sensory_states.get('dwell_progress', 0.0)) ** 2)

        priors                           = self._compute_dwell_prior(current_state, candidates, hazard, exit_probs)  # Eq. 5
        g_vals, mu_candidates            = self._evaluate_efe(x_current, candidates)                                 # Eq. 6
        q_pi, gamma, pi_conf, pol_drive  = self._compute_posterior(priors, g_vals)                                   # Eq. 7
        selected_mu, mu_x                = self._select_attractor(q_pi, mu_candidates, current_state, hazard)        # Eq. 8

        return {
            'selected_action_mu': selected_mu,
            'mu_x':               mu_x,
            'policy_confidence':  pi_conf,
            'policy_drive':       pol_drive,
            'policy_precision':   gamma,
            'q_pi':               np.array(q_pi, dtype=np.float64),
        }
