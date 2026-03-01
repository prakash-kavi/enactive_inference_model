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
    STATES, NETWORKS, THOUGHTSEEDS, CLIP_MIN, CLIP_MAX, EPS,
    THOUGHTSEED_STATE_PRIORS,
    get_exit_transition_probs, get_policy_candidate_order,
    VI_STEPS, VI_LR, VI_MISMATCH_THRESHOLD,
    Z_NOISE_STD_BY_STATE,
)
from .markov_blankets import MarkovBlanketL1L2, MarkovBlanketL2L3
from .phenotype import PhenotypeConfig, EXPERT_PHENOTYPE
from utils.math_utils import (
    mse_error,
    recon_error,
    prior_error,
    forward_error,
    clamp_activation,
    clip_probability,
    to_float,
    networks_to_tensor,
)

class ThoughtseedModel(nn.Module):
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
        self.blanket_l1l2 = blanket_l1l2 or MarkovBlanketL1L2()
        self.blanket_l2l3 = blanket_l2l3 or MarkovBlanketL2L3()
        
        # Recognition/decoder
        self.thoughtseed_model = ThoughtseedModel(
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
        
    def _precision_params(self, precision_sensory: Optional[float] = None) -> tuple[float, float]:
        """Return (precision_blend, precision_weight) from a scalar precision signal."""
        if precision_sensory is None:
            precision_sensory = to_float(self.blanket_l2l3.active_states.get('precision_sensory', 1.0))
        precision_blend = float(np.clip(precision_sensory, CLIP_MIN, CLIP_MAX))
        precision_weight = precision_blend
        return precision_blend, precision_weight

    def _init_vi_state(
        self,
        z_prev: torch.Tensor,
        z_recognition: torch.Tensor,
        prior_target: torch.Tensor,
        precision_blend: float,
    ) -> torch.Tensor:
        """Precision-weighted initialization for VI refinement."""
        z_init = 0.5 * z_prev + 0.5 * z_recognition
        z_init = precision_blend * z_init + (1.0 - precision_blend) * prior_target
        return clamp_activation(z_init, CLIP_MIN, CLIP_MAX)
    
    def decode_with_state(self, z: torch.Tensor) -> torch.Tensor:
        """Top-down: decode thoughtseeds -> networks."""
        if z.dim() == 1:
            z_in = z.unsqueeze(0)
            decoded = self.thoughtseed_model.decode(z_in).squeeze(0)
        else:
            decoded = self.thoughtseed_model.decode(z)
        
        return torch.clamp(decoded, 0.0, 1.0)

    def compute_vfe(
        self,
        state: str,
        z: torch.Tensor,
        observed_x: torch.Tensor,
        precision_sensory: Optional[float] = None,
    ) -> torch.Tensor:
        """Compute VFE: precision-weighted reconstruction + prior matching."""
        recon_x = self.decode_with_state(z)
        observed_x = observed_x.detach()

        _, precision_weight = self._precision_params(precision_sensory)

        recon_loss = precision_weight * recon_error(recon_x, observed_x)
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
        clip_min = CLIP_MIN
        clip_max = CLIP_MAX

        observed_vec = networks_to_tensor(self.blanket_l1l2.sensory_states, NETWORKS)
        
        def clamp_z(t: torch.Tensor) -> torch.Tensor:
            return clamp_activation(t, clip_min, clip_max)

        z_prev = clamp_z(activations.detach())
        sensory_target = clamp_z(z_recognition.detach())
        prior_target = clamp_z(self.mu_params[current_state].detach().clone())
        
        # Sensory precision (used as observation weight + blending factor)
        precision_blend, precision_weight = self._precision_params()
        
        # Initialization: blend previous and sensory target, then bias toward sensory when precision is high.
        z_init = self._init_vi_state(z_prev, sensory_target, prior_target, precision_blend)
        z_var = z_init.requires_grad_(True)

        # Variational optimization loop (mismatch-triggered, precision-modulated VI)
        mismatch = float(mse_error(sensory_target, prior_target).item())
        if mismatch > VI_MISMATCH_THRESHOLD:
            # Fewer VI steps when predictability (precision) is high.
            uncertainty = max(0.0, 1.0 - float(precision_blend))
            vi_steps = int(np.ceil(float(VI_STEPS) * uncertainty))
            vi_steps = max(0, min(int(VI_STEPS), vi_steps))
        else:
            vi_steps = 0
        vi_lr = float(VI_LR)
        with torch.enable_grad():
            for _ in range(vi_steps):
                recon_x = self.decode_with_state(z_var)
                
                recon_loss = precision_weight * recon_error(recon_x, observed_vec)
                prior_match = prior_error(z_var, prior_target)
                loss = recon_loss + prior_match
                
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
        z = self.thoughtseed_model.encode(x.unsqueeze(0) if x.dim() == 1 else x)
        if z.dim() > 1:
            z = z.squeeze(0)
        z_mean = clamp_activation(z, CLIP_MIN, CLIP_MAX)
        noise_std = float(Z_NOISE_STD_BY_STATE.get(current_state, 0.0))
        if noise_std > 0.0:
            z_sample = z_mean + torch.randn_like(z_mean) * noise_std
        else:
            z_sample = z_mean
        z_recognition = clamp_activation(z_sample, CLIP_MIN, CLIP_MAX)

        z_posterior = self.update_posterior_z(
            current_state=current_state,
            activations=activations,
            z_recognition=z_recognition,
        )
        return z_posterior, z_recognition

    def infer_state_belief(self, z: torch.Tensor) -> Dict[str, float]:
        """Infer state belief q(s|z) from thoughtseed activations."""
        with torch.no_grad():
            dists = []
            for state in STATES:
                prior = self.mu_params[state].detach()
                dists.append(mse_error(z, prior))
            dist_vec = torch.stack(dists)
            if dist_vec.numel() > 1:
                mean = dist_vec.mean()
                std = dist_vec.std(unbiased=False)
                if float(std.item()) > EPS:
                    dist_vec = (dist_vec - mean) / (std + EPS)
            scale = 1.0
            logits = -dist_vec / scale
            probs = torch.softmax(logits, dim=0)
            return {
                state: float(probs[i].item())
                for i, state in enumerate(STATES)
            }
    
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
        """EFE G(pi) with pragmatic + epistemic components.

        Pragmatic: MSE between predicted and preferred networks.
        Epistemic: information-gain proxy in thoughtseed space.
        Returns (g_vals: list[float], mu_candidates: list[Tensor], stats: dict).
        """
        g_vals, mu_candidates = [], []
        g_prag_vals, i_epi_vals = [], []
        with torch.no_grad():
            for s in candidates:
                mu_c = self.mu_params[s]
                x_pred = self.thoughtseed_model.predict_next(x_current, mu_c)
                x_pref = self.decode_with_state(mu_c)

                g_prag = forward_error(x_pred, x_pref)
                g_prag_val = float(g_prag.item())

                # Epistemic value proxy: KL-like term in thoughtseed space
                z_pred = self.thoughtseed_model.encode(x_pred.unsqueeze(0) if x_pred.dim() == 1 else x_pred)
                if z_pred.dim() > 1:
                    z_pred = z_pred.squeeze(0)
                z_pred = clamp_activation(z_pred, CLIP_MIN, CLIP_MAX)
                info_gain = mse_error(z_pred, mu_c)
                info_gain_val = float(info_gain.item())

                g_prag_vals.append(g_prag_val)
                i_epi_vals.append(info_gain_val)
                mu_candidates.append(mu_c)
        # Normalize per policy set (z-score) to make terms comparable
        g_prag_arr = np.array(g_prag_vals, dtype=float)
        i_epi_arr = np.array(i_epi_vals, dtype=float)
        g_prag_std = float(np.std(g_prag_arr)) if g_prag_arr.size else 0.0
        i_epi_std = float(np.std(i_epi_arr)) if i_epi_arr.size else 0.0
        if g_prag_arr.size and g_prag_std > EPS:
            g_prag_norm = (g_prag_arr - float(np.mean(g_prag_arr))) / (g_prag_std + EPS)
        else:
            g_prag_norm = g_prag_arr
        if i_epi_arr.size and i_epi_std > EPS:
            i_epi_norm = (i_epi_arr - float(np.mean(i_epi_arr))) / (i_epi_std + EPS)
        else:
            i_epi_norm = i_epi_arr

        for g_p, i_e in zip(g_prag_norm, i_epi_norm):
            g_vals.append(float(g_p) - float(i_e))

        stats = {
            'g_prag_mean': float(np.mean(g_prag_vals)) if g_prag_vals else 0.0,
            'i_epi_mean': float(np.mean(i_epi_vals)) if i_epi_vals else 0.0,
        }
        return g_vals, mu_candidates, stats

    def _select_attractor(self, q_pi, mu_candidates: list):
        """Posterior-weighted latent prediction.

        mu = sum_pi q(pi) mu_z(s_pi)
        Returns (selected_mu, mu_x).
        """
        weights = torch.tensor(q_pi, dtype=mu_candidates[0].dtype)
        mu = torch.sum(weights.unsqueeze(-1) * torch.stack(mu_candidates, 0), dim=0)
        return mu, self.decode_with_state(mu)

    # ------------------------------------------------------------------

    def evaluate_policies(self, current_state: str) -> Dict:
        """Evaluate policy evidence G(pi) and return candidates + priors."""
        x_current  = networks_to_tensor(self.blanket_l1l2.sensory_states, NETWORKS)
        candidates = get_policy_candidate_order(current_state)
        exit_probs = get_exit_transition_probs(self.level, current_state)
        hazard     = clip_probability(
            to_float(self.blanket_l1l2.sensory_states.get('dwell_progress', 0.0)) ** 2)

        priors                           = self._compute_dwell_prior(current_state, candidates, hazard, exit_probs)  # dwell prior (heuristic)
        g_vals, mu_candidates, efe_stats = self._evaluate_efe(x_current, candidates)
        return {
            'candidates':     candidates,
            'priors':         priors,
            'g_vals':         g_vals,
            'mu_candidates':  mu_candidates,
            'efe_prag_mean':  efe_stats.get('g_prag_mean', 0.0),
            'efe_epi_mean':   efe_stats.get('i_epi_mean', 0.0),
        }

    def action_from_policy(self, q_pi, candidates: list, mu_candidates: list) -> Dict:
        """Compute action outputs from an externally selected policy posterior."""
        selected_mu, mu_x = self._select_attractor(q_pi, mu_candidates)
        transition_drive = float(1.0 - q_pi[0]) if len(q_pi) > 0 else 0.0
        policy_state_probs = {
            state: float(q_pi[i]) for i, state in enumerate(candidates)
        }
        return {
            'selected_action_mu': selected_mu,
            'mu_x':               mu_x,
            'transition_drive':   transition_drive,
            'policy_state_probs': policy_state_probs,
        }
