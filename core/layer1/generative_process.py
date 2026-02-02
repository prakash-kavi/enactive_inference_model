"""Layer 1: generative process for brain network dynamics (MVOU)."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional
from utils.meditation_config import DEFAULTS, STATES, NETWORKS, EPS
from .layer1_config import THETA_BASE, NETWORK_PROFILES, L1_GENERATIVE_COSTS
from .state_machine import StateMachine

IDX = {n: i for i, n in enumerate(NETWORKS)}
N_NETWORKS = len(NETWORKS)

class Layer1Process(nn.Module):
    """Layer 1 MVOU dynamics for brain networks."""
    
    # MVOU constants
    MAX_STIFFNESS = 2.5
    STIFFNESS_MARGIN = 0.1
    N_SUBSTEPS = 2
    INIT_ACTIVATION = 0.5
    
    # Expertise-specific variance
    BASE_VARIANCE = {'expert': 0.002, 'novice': 0.005}
    MIN_VARIANCE = 0.0005
    
    # Learned attractor smoothing
    SMOOTHING = {'learned': 0.9, 'default': 0.7}

    def __init__(self, experience_level: str = 'expert', seed: Optional[int] = None,
                 learned_attractors: Optional[Dict] = None):
        super().__init__()
        
        self.level = experience_level.lower()
        self.dt = DEFAULTS['DEFAULT_DT']
        self.networks = NETWORKS
        self.generative_costs = L1_GENERATIVE_COSTS[self.level]
        self.sm = StateMachine(self.level, self.dt, seed, generative_costs=self.generative_costs)
        self.learned_attractors = learned_attractors

        self.x = torch.full((N_NETWORKS,), self.INIT_ACTIVATION, dtype=torch.float32)
        self.smoothed_x = self.x.clone()
        self.smoothing_alpha = self.SMOOTHING['learned' if learned_attractors else 'default']
        self.mw_burden_ema = 0.0
        self.last_network_burden = 0.0
        self.last_transition_hazard = 0.0
        self.last_activation_burden_component = 0.0
        self.last_coupling_burden_component = 0.0

    def get_dynamics(self, state: str, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.learned_attractors and state in self.learned_attractors:
            prof = self.learned_attractors[state]
            fallback = NETWORK_PROFILES[state][self.level]
            mu_np = np.array([prof.get(n, fallback[n]) for n in self.networks])
        else:
            prof = NETWORK_PROFILES[state][self.level]
            mu_np = np.array([prof[n] for n in self.networks])

        mu = torch.tensor(mu_np, dtype=torch.float32, device=device)
        coupling_map = THETA_BASE[state]
        base_diag = 0.50 if state == 'mind_wandering' else 0.15

        theta_np = np.eye(4) * base_diag
        for (row, col), val in coupling_map.items():
            r_idx = IDX[row]
            c_idx = IDX[col]
            theta_np[r_idx, c_idx] = val

        theta = torch.tensor(theta_np, dtype=torch.float32, device=device)

        if self.level == 'expert':
            if state == 'breath_focus':
                theta = theta + torch.eye(N_NETWORKS, device=device) * 0.4
            elif state == 'redirect_attention':
                r, c = IDX['DMN'], IDX['DAN']
                if theta[r, c] > 0:
                    theta[r, c] *= 1.5
                if theta[c, r] > 0:
                    theta[c, r] *= 1.5
            elif state == 'meta_awareness':
                r, c = IDX['VAN'], IDX['FPN']
                if theta[r, c] < 0:
                    theta[r, c] *= 1.4
                if theta[c, r] < 0:
                    theta[c, r] *= 1.4

        return mu, self._clamp_theta_stiffness(theta, device)
    
    def _clamp_theta_stiffness(self, theta: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Clamp theta matrix to prevent excessive volatility."""
        off_diag_mask = 1.0 - torch.eye(N_NETWORKS, device=device)
        off_diag_sum = torch.sum(torch.abs(theta) * off_diag_mask, dim=1)
        
        scaling_factors = torch.ones(N_NETWORKS, device=device)
        high_volatility_mask = off_diag_sum > (self.MAX_STIFFNESS - self.STIFFNESS_MARGIN)
        if torch.any(high_volatility_mask):
            target = self.MAX_STIFFNESS - self.STIFFNESS_MARGIN
            scaling_factors[high_volatility_mask] = target / (off_diag_sum[high_volatility_mask] + EPS)
        
        theta_off = (theta * off_diag_mask) * scaling_factors.unsqueeze(1).expand(N_NETWORKS, N_NETWORKS)
        new_off_sum = torch.sum(torch.abs(theta_off), dim=1)
        min_diag = new_off_sum + self.STIFFNESS_MARGIN
        final_diag = torch.clamp(torch.maximum(torch.diag(theta), min_diag), max=self.MAX_STIFFNESS)
        
        return theta_off + torch.diag(final_diag)

    def set_rng(self, rng: np.random.RandomState):
        """Attach a shared RNG for dwell sampling."""
        if rng is not None:
            self.sm.rng = rng

    def update(self, active_states: Dict) -> Tuple[Dict[str, torch.Tensor], str]:
        """Advance one step and return network activations + current state."""
        # Extract scalars once
        drive = active_states['transition_drive'].item() if isinstance(active_states['transition_drive'], torch.Tensor) else active_states['transition_drive']
        noise_gain = active_states['noise_reduction'].item() if isinstance(active_states['noise_reduction'], torch.Tensor) else active_states['noise_reduction']
        
        current_state = self.sm.check_transition(drive, mw_burden=self.mw_burden_ema)
        self.current_state = current_state
        self.current_max_dwell = self.sm.current_max_steps
        self.last_transition_hazard = self.sm.last_transition_hazard

        device = self.x.device
        mu, theta = self.get_dynamics(current_state, device)

        # Apply agent bias if present
        agent_bias = active_states.get('agent_bias')
        if agent_bias is not None:
            bias_strength = active_states['l2tol1_enactive_bias'].item() if isinstance(active_states['l2tol1_enactive_bias'], torch.Tensor) else active_states['l2tol1_enactive_bias']
            bias_strength = np.clip(bias_strength, 0.0, 1.0)
            if not isinstance(agent_bias, torch.Tensor):
                agent_bias = torch.tensor(agent_bias, device=device, dtype=torch.float32)
            mu = (1 - bias_strength) * mu + bias_strength * agent_bias

        variance = max(self.MIN_VARIANCE, self.BASE_VARIANCE[self.level] * noise_gain)
        if current_state == 'redirect_attention':
            ra_diff = self.generative_costs['ra_diffusion_scale']
            ra_pull = self.generative_costs['ra_bf_pull_strength']
            variance *= ra_diff
            bf_mu, _ = self.get_dynamics('breath_focus', device)
            pull_mix = ra_pull / (1.0 + ra_pull)
            mu = (1.0 - pull_mix) * mu + pull_mix * bf_mu
            theta = theta + torch.eye(N_NETWORKS, device=device) * (0.12 * ra_pull)

        sigma = np.sqrt(variance)
        dt_sub = self.dt / self.N_SUBSTEPS
        curr_x = self.x

        for _ in range(self.N_SUBSTEPS):
            drift = -torch.matmul(theta, (curr_x - mu)) * dt_sub
            diffusion = torch.randn_like(curr_x) * sigma * np.sqrt(dt_sub)
            curr_x = curr_x + drift + diffusion
            curr_x = torch.clamp(curr_x, DEFAULTS['ACTIVATION_CLIP_MIN'], DEFAULTS['ACTIVATION_CLIP_MAX'])

        self.x = curr_x
        self.smoothed_x = self.smoothing_alpha * self.x + (1 - self.smoothing_alpha) * self.smoothed_x
        self._update_mw_burden(current_state, self.x, mu, theta)

        return dict(zip(self.networks, self.smoothed_x)), current_state

    def _update_mw_burden(self, current_state: str, x: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor) -> None:
        """Update MW detection burden from activation and coupling costs."""
        activation_cost_raw = torch.mean((x - mu) ** 2).item()
        offdiag_mask = 1.0 - torch.eye(N_NETWORKS, device=x.device)
        coupling_load_raw = torch.mean(torch.abs(theta * offdiag_mask)).item()

        # Normalized and weighted costs
        costs = self.generative_costs
        activation_cost = min(2.5, activation_cost_raw / costs['cost_activation_scale'])
        coupling_load = min(2.5, coupling_load_raw / costs['cost_coupling_scale'])
        activation_component = costs['cost_activation_weight'] * activation_cost
        coupling_component = costs['cost_coupling_weight'] * coupling_load
        raw_burden = activation_component + coupling_component

        # EMA update
        alpha = np.clip(costs['mw_detection_alpha'], 0.0, 1.0)
        decay = np.clip(costs['mw_detection_decay'], 0.0, 1.0)
        if current_state == 'mind_wandering':
            self.mw_burden_ema = (1.0 - alpha) * self.mw_burden_ema + alpha * raw_burden
        else:
            self.mw_burden_ema = (1.0 - decay) * self.mw_burden_ema
        
        self.mw_burden_ema = max(0.0, self.mw_burden_ema)
        self.last_network_burden = self.mw_burden_ema
        self.last_activation_burden_component = activation_component
        self.last_coupling_burden_component = coupling_component

    def reset(self, state: str = STATES[0]):
        self.sm.current_state = state
        self.sm._sample_next_dwell()
        self.current_max_dwell = self.sm.current_max_steps
        self.current_state = state
        self.x = torch.full((N_NETWORKS,), self.INIT_ACTIVATION, dtype=torch.float32, device=self.x.device)
        self.smoothed_x = self.x.clone()
        self.mw_burden_ema = 0.0
        self.last_network_burden = 0.0
        self.last_transition_hazard = 0.0
        self.last_activation_burden_component = 0.0
        self.last_coupling_burden_component = 0.0
