"""Layer 1: Multivariate Ornstein-Uhlenbeck generative process for brain networks.

Implements state-dependent attractor dynamics with:
- 4 brain networks (DMN, VAN, DAN, FPN)
- 4 meditation states (breath_focus, mind_wandering, meta_awareness, redirect_attention)
- Coupling matrices (Theta) defining network interactions
- State machine with dwell times and transition probabilities
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

from utils.config import (
    NETWORKS, DEFAULTS, EPS,
    THETA_BASE, NETWORK_PROFILES, DWELL_TIMES, STATE_TRANSITION_PROBS
)
from utils.math_utils import clip_probability, to_float

class Layer1Process(nn.Module):
    """MVOU generative process for brain network dynamics."""
    
    # MVOU integration constants
    MAX_STIFFNESS = 2.5
    N_SUBSTEPS = 2
    INIT_ACTIVATION = 0.5
    
    # Global noise variance (shared across phenotypes)
    NOISE_LEVEL = 0.004
    
    def __init__(self, experience_level: str = 'expert', seed: Optional[int] = None):
        super().__init__()
        
        self.level = experience_level
        self.dt = DEFAULTS['DEFAULT_DT']
        self.rng = np.random.RandomState(seed)
        
        # Network state
        self.x = torch.full((len(NETWORKS),), self.INIT_ACTIVATION, dtype=torch.float32)
        self.smoothed_x = self.x.clone()
        
        # State machine
        self.current_state = 'breath_focus'
        self.current_dwell = 0
        self.current_max_dwell = 0
        self._sample_next_dwell()
        
        # Network indexing
        self.net_idx = {n: i for i, n in enumerate(NETWORKS)}
        
    def _sample_next_dwell(self) -> None:
        """Sample dwell time for current state."""
        dwell_min, dwell_max = DWELL_TIMES[self.level][self.current_state]
        dwell_seconds = self.rng.uniform(dwell_min, dwell_max)
        self.current_max_dwell = int(dwell_seconds / self.dt)
        self.current_dwell = 0
        
    def _check_transition(self, policy_drive: float) -> str:
        """Check if state should transition based on dwell + drive."""
        self.current_dwell += 1
        
        # Dwell not elapsed: stay
        if self.current_dwell < self.current_max_dwell:
            return self.current_state
        
        # Dwell elapsed: compute transition hazard
        drive = clip_probability(policy_drive)
        base_hazard = 0.3
        drive_boost = 0.5 * drive
        hazard = base_hazard + drive_boost
        
        if self.rng.rand() < hazard:
            # Transition: sample next state
            probs = STATE_TRANSITION_PROBS[self.level][self.current_state]
            states = list(probs.keys())
            p_array = np.array([probs[s] for s in states])
            p_array /= p_array.sum()  # Normalize
            
            next_state = self.rng.choice(states, p=p_array)
            self.current_state = next_state
            self._sample_next_dwell()
        
        return self.current_state
    
    def _get_attractor(self, state: str) -> torch.Tensor:
        """Get attractor mean for state."""
        profile = NETWORK_PROFILES[state][self.level]
        mu_np = np.array([profile[n] for n in NETWORKS])
        return torch.tensor(mu_np, dtype=torch.float32, device=self.x.device)
    
    def _get_coupling(self, state: str) -> torch.Tensor:
        """Build Theta coupling matrix for state."""
        # Start with diagonal (self-inhibition)
        base_diag = 0.50 if state == 'mind_wandering' else 0.15
        theta_np = np.eye(len(NETWORKS)) * base_diag
        
        # Add off-diagonal couplings
        coupling_map = THETA_BASE[state]
        for (row_net, col_net), value in coupling_map.items():
            r_idx = self.net_idx[row_net]
            c_idx = self.net_idx[col_net]
            theta_np[r_idx, c_idx] = value
        
        theta = torch.tensor(theta_np, dtype=torch.float32, device=self.x.device)
        
        # Expert-specific modifications
        if self.level == 'expert':
            if state == 'breath_focus':
                # Stronger self-stabilization
                theta = theta + torch.eye(len(NETWORKS), device=self.x.device) * 0.4
            elif state == 'redirect_attention':
                # Amplify DMN-DAN inhibition
                r, c = self.net_idx['DMN'], self.net_idx['DAN']
                if theta[r, c] > 0:
                    theta[r, c] *= 1.5
                    theta[c, r] *= 1.5
            elif state == 'meta_awareness':
                # Amplify VAN-FPN synergy
                r, c = self.net_idx['VAN'], self.net_idx['FPN']
                if theta[r, c] < 0:
                    theta[r, c] *= 1.4
                    theta[c, r] *= 1.4
        
        return theta
    
    def _clamp_theta(self, theta: torch.Tensor) -> torch.Tensor:
        """Prevent excessive stiffness in coupling matrix."""
        n = len(NETWORKS)
        device = theta.device
        off_diag_mask = 1.0 - torch.eye(n, device=device)
        off_diag_sum = torch.sum(torch.abs(theta) * off_diag_mask, dim=1)
        
        # Scale down rows with high volatility
        scaling = torch.ones(n, device=device)
        high_mask = off_diag_sum > (self.MAX_STIFFNESS - 0.1)
        if torch.any(high_mask):
            target = self.MAX_STIFFNESS - 0.1
            scaling[high_mask] = target / (off_diag_sum[high_mask] + EPS)
        
        theta_off = (theta * off_diag_mask) * scaling.unsqueeze(1)
        
        # Ensure diagonal dominance
        new_off_sum = torch.sum(torch.abs(theta_off), dim=1)
        min_diag = new_off_sum + 0.1
        final_diag = torch.clamp(torch.maximum(torch.diag(theta), min_diag), max=self.MAX_STIFFNESS)
        
        return theta_off + torch.diag(final_diag)
    
    def update(self, active_states: Dict) -> Tuple[Dict[str, torch.Tensor], str]:
        """Advance one timestep and return network activations + current state.
        
        Args:
            active_states: Control from L2 via Markov blanket
                - policy_drive: float (0-1) (transition urge)
                - policy_confidence: float (0-1) (posterior confidence)
                - precision_gain: float (0-1) (action precision gain)
                - mu_x: Optional[torch.Tensor] (target network activations)
        
        Returns:
            network_acts: {network_name: activation}
            current_state: str
        """
        # Extract control signals
        drive = active_states.get('policy_drive', active_states.get('policy_confidence', 0.0))
        drive = to_float(drive)
        
        # Check for state transition
        self.current_state = self._check_transition(drive)
        
        # Get dynamics for current state
        mu = self._get_attractor(self.current_state)
        theta = self._get_coupling(self.current_state)
        theta = self._clamp_theta(theta)
        
        # Apply agent bias if present (L2 -> L1 active inference)
        mu_x = active_states.get('mu_x')
        if mu_x is not None:
            bias_strength = active_states.get('precision_gain', 0.0)
            bias_strength = 0.5 * clip_probability(bias_strength)
            
            if not isinstance(mu_x, torch.Tensor):
                mu_x = torch.tensor(mu_x, device=mu.device, dtype=torch.float32)
            mu = (1 - bias_strength) * mu + bias_strength * mu_x
        
        # Expertise-dependent noise, modulated by L3 precision via L2
        base_variance = self.NOISE_LEVEL
        noise_reduction = active_states.get('noise_reduction', 1.0)
        variance = base_variance * to_float(noise_reduction)  # L3 precision reduces L1 noise
        sigma = np.sqrt(variance)
        
        # MVOU integration (multiple substeps for stability)
        dt_sub = self.dt / self.N_SUBSTEPS
        curr_x = self.x
        
        for _ in range(self.N_SUBSTEPS):
            # dX = -Theta * (X - mu) * dt + sigma * dW
            drift = -torch.matmul(theta, (curr_x - mu)) * dt_sub
            diffusion = torch.randn_like(curr_x) * sigma * np.sqrt(dt_sub)
            curr_x = curr_x + drift + diffusion
            curr_x = torch.clamp(curr_x, DEFAULTS['ACTIVATION_CLIP_MIN'], DEFAULTS['ACTIVATION_CLIP_MAX'])
        
        self.x = curr_x
        self.smoothed_x = 0.4 * self.smoothed_x + 0.6 * self.x  # EMA smoothing
        
        # Return as dict for Markov blanket
        network_acts = {net: self.smoothed_x[i] for i, net in enumerate(NETWORKS)}
        return network_acts, self.current_state
    
    def reset(self, state: str = 'breath_focus') -> None:
        """Reset to initial state."""
        self.current_state = state
        self._sample_next_dwell()
        self.x = torch.full((len(NETWORKS),), self.INIT_ACTIVATION, dtype=torch.float32)
        self.smoothed_x = self.x.clone()
