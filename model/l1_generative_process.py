"""Layer 1: MVOU generative process for brain networks (Eq. 1).

Implements state-dependent attractor dynamics with:
- 4 brain networks (DMN, VAN, DAN, FPN)
- 4 meditation states (BF, MW, MA, RA)
- Coupling matrices Theta(s) defining network interactions
- Dwell-based state machine with transition priors
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

from utils.config import (
    NETWORKS, DEFAULTS, EPS, NOISE_LEVEL,
    THETA_BASE, NETWORK_PROFILES, DWELL_TIMES, STATE_TRANSITION_PROBS,
    L1_BASE_HAZARD,
)
from utils.math_utils import clip_probability, to_float
from model.phenotype import PhenotypeConfig, EXPERT_PHENOTYPE

class Layer1Process(nn.Module):
    """MVOU generative process for brain network dynamics (Eq. 1)."""

    # MVOU integration constants
    MAX_STIFFNESS = 2.5
    N_SUBSTEPS = 2
    INIT_ACTIVATION = 0.5

    def __init__(self, phenotype: PhenotypeConfig = None, seed: Optional[int] = None):
        super().__init__()

        self.phenotype = phenotype if phenotype is not None else EXPERT_PHENOTYPE
        self.level = self.phenotype.level   # used for config table lookups
        self.dt = DEFAULTS['DEFAULT_DT']
        self.rng = np.random.RandomState(seed)

        # Network state
        self.x = torch.full((len(NETWORKS),), self.INIT_ACTIVATION, dtype=torch.float32)
        
        # State machine
        self.current_state = 'breath_focus'
        self.current_dwell = 0
        self.current_max_dwell = 0
        self._sample_next_dwell()
        
        # Network indexing
        self.net_idx = {n: i for i, n in enumerate(NETWORKS)}
        
    def _sample_next_dwell(self) -> None:
        """Sample dwell duration for current state from config ranges."""
        dwell_min, dwell_max = DWELL_TIMES[self.level][self.current_state]
        dwell_seconds = self.rng.uniform(dwell_min, dwell_max)
        self.current_max_dwell = int(dwell_seconds / self.dt)
        self.current_dwell = 0
        
    def _check_transition(self, policy_drive: float) -> str:
        """Check if state should transition based on dwell + policy drive."""
        self.current_dwell += 1
        
        # Dwell not elapsed: stay
        if self.current_dwell < self.current_max_dwell:
            return self.current_state
        
        # Dwell elapsed: compute transition hazard (policy drive increases hazard; Eq. 5)
        policy_drive = clip_probability(policy_drive)
        drive_boost = 0.5 * policy_drive
        hazard = L1_BASE_HAZARD + drive_boost
        
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
        """Get attractor mean mu_x(s) for current state."""
        profile = NETWORK_PROFILES[state][self.level]
        mu_np = np.array([profile[n] for n in NETWORKS])
        return torch.tensor(mu_np, dtype=torch.float32)
    
    def _get_coupling(self, state: str) -> torch.Tensor:
        """Build Theta(s) coupling matrix for state (Eq. 1)."""
        # Start with diagonal (self-inhibition)
        base_diag = 0.50 if state == 'mind_wandering' else 0.15
        theta_np = np.eye(len(NETWORKS)) * base_diag
        
        # Add off-diagonal couplings
        coupling_map = THETA_BASE[state]
        for (row_net, col_net), value in coupling_map.items():
            r_idx = self.net_idx[row_net]
            c_idx = self.net_idx[col_net]
            theta_np[r_idx, c_idx] = value
        
        theta = torch.tensor(theta_np, dtype=torch.float32)

        # Phenotype-specific adjustments to Theta(s): experts have stronger
        # self-stabilisation and inter-network coupling. (theta_boost flag)
        if self.phenotype.theta_boost:
            if state == 'breath_focus':
                # Stronger self-stabilization
                theta = theta + torch.eye(len(NETWORKS)) * 0.4
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
        """Prevent excessive stiffness in Theta(s)."""
        n = len(NETWORKS)
        off_diag_mask = 1.0 - torch.eye(n)
        off_diag_sum = torch.sum(torch.abs(theta) * off_diag_mask, dim=1)

        # Scale down rows with high volatility
        scaling = torch.ones(n)
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
                - policy_drive: float (0-1) transition drive
                - policy_confidence: float (0-1) posterior confidence (used if drive absent)
                - precision_gain: float (0-1) gain on L2 target influence
                - mu_x: Optional[torch.Tensor] L2 target in network space
        
        Returns:
            network_acts: {network_name: activation}
            current_state: str
        """
        # Extract control signals
        policy_drive = active_states.get('policy_drive', active_states.get('policy_confidence', 0.0))
        policy_drive = to_float(policy_drive)
        
        # Check for state transition
        self.current_state = self._check_transition(policy_drive)
        
        # Get dynamics for current state
        mu = self._get_attractor(self.current_state)
        theta = self._get_coupling(self.current_state)
        theta = self._clamp_theta(theta)
        
        # Apply L2 attractor target mu_x directly (fixed bias_strength=0.5). (Eq. 1)
        # L2->L1 active state is mu_x only; precision_gain/noise_reduction removed.
        mu_x = active_states.get('mu_x')
        if mu_x is not None:
            if not isinstance(mu_x, torch.Tensor):
                mu_x = torch.tensor(mu_x, dtype=torch.float32)
            mu = 0.5 * mu + 0.5 * mu_x

        # Global process noise variance (Eq. 1) — fixed, not modulated by L2.
        sigma = np.sqrt(NOISE_LEVEL)
        
        # MVOU integration (Eq. 1) with substeps for stability
        dt_sub = self.dt / self.N_SUBSTEPS
        curr_x = self.x
        
        for _ in range(self.N_SUBSTEPS):
            # dX = -Theta(s) · (X - mu) dt + sigma dW
            drift = -torch.matmul(theta, (curr_x - mu)) * dt_sub
            diffusion = torch.randn_like(curr_x) * sigma * np.sqrt(dt_sub)
            curr_x = curr_x + drift + diffusion
            curr_x = torch.clamp(curr_x, DEFAULTS['CLIP_MIN'], DEFAULTS['CLIP_MAX'])
        
        self.x = curr_x
        
        # Return as dict for Markov blanket
        network_acts = {net: self.x[i] for i, net in enumerate(NETWORKS)}
        return network_acts, self.current_state
    
    def reset(self, state: str = 'breath_focus') -> None:
        """Reset to initial state."""
        self.current_state = state
        self._sample_next_dwell()
        self.x = torch.full((len(NETWORKS),), self.INIT_ACTIVATION, dtype=torch.float32)
