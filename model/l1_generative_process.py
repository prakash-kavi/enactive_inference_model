"""Layer 1: MVOU generative process for brain networks.

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
    NETWORKS, DEFAULT_DT, CLIP_MIN, CLIP_MAX, EPS, NOISE_LEVEL,
    THETA_BASE, NETWORK_PROFILES, DWELL_TIMES, STATE_TRANSITION_PROBS,
    THETA_MW_DIAG, THETA_DEFAULT_DIAG, THETA_BOOST_BF,
)
from utils.math_utils import clip_probability
from model.phenotype import PhenotypeConfig, EXPERT_PHENOTYPE

class Layer1Process(nn.Module):
    """MVOU generative process for brain network dynamics."""

    # MVOU integration constants
    MAX_STIFFNESS = 2.5
    N_SUBSTEPS = 2
    INIT_ACTIVATION = 0.5

    def __init__(self, phenotype: PhenotypeConfig = None, seed: Optional[int] = None):
        super().__init__()

        self.phenotype = phenotype if phenotype is not None else EXPERT_PHENOTYPE
        self.level = self.phenotype.level   # used for config table lookups
        self.dt = DEFAULT_DT
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
        self.current_max_dwell = max(1, int(dwell_seconds / self.dt))
        self.current_dwell = 0
        
    def _check_transition(
        self,
        policy_state_probs: Optional[Dict[str, float]] = None,
    ) -> str:
        """Check if state should transition based on dwell + policy posterior."""
        self.current_dwell += 1
        
        # Dwell not elapsed: stay
        if self.current_dwell < self.current_max_dwell:
            return self.current_state
        
        # Dwell elapsed: transition probability driven by policy posterior
        floor = 1.0 / max(self.current_max_dwell, 1)
        p_stay = None
        if policy_state_probs:
            p_stay = float(policy_state_probs.get(self.current_state, 0.0))
            p_stay = clip_probability(p_stay)
        transition_prob = max((1.0 - p_stay) if p_stay is not None else 0.0, floor)
        if self.rng.rand() < transition_prob:
            # Transition: sample next state (policy posterior if provided)
            probs = STATE_TRANSITION_PROBS[self.level][self.current_state]
            states = list(probs.keys())
            base_array = np.array([probs[s] for s in states], dtype=float)
            p_array = base_array.copy()

            if policy_state_probs:
                policy_weights = np.array(
                    [float(policy_state_probs.get(s, 0.0)) for s in states],
                    dtype=float
                )
                if policy_weights.sum() > 0:
                    p_array = policy_weights
            else:
                policy_weights = None

            if p_array.sum() <= 0:
                p_array = base_array.copy()
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
        """Build Theta(s) coupling matrix for state."""
        # Start with diagonal (self-inhibition)
        base_diag = THETA_MW_DIAG if state == 'mind_wandering' else THETA_DEFAULT_DIAG
        theta_np = np.eye(len(NETWORKS)) * base_diag
        
        # Add off-diagonal couplings
        coupling_map = THETA_BASE[state]
        for (row_net, col_net), value in coupling_map.items():
            r_idx = self.net_idx[row_net]
            c_idx = self.net_idx[col_net]
            theta_np[r_idx, c_idx] = value
        
        theta = torch.tensor(theta_np, dtype=torch.float32)

        # Phenotype-specific adjustment to Theta(s): experts get stronger
        # self-stabilisation in breath_focus, plus a global theta_scale.
        if self.phenotype.theta_boost:
            if state == 'breath_focus':
                # Stronger self-stabilization
                theta = theta + torch.eye(len(NETWORKS)) * THETA_BOOST_BF
        
        theta = theta * float(self.phenotype.theta_scale)
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
                - mu_x: Optional[torch.Tensor] L2 descending prediction in network space
                - policy_state_probs: Optional[Dict[str, float]] policy posterior over candidate states
        
        Returns:
            network_acts: {network_name: activation}
            current_state: str
        """
        # Extract control signals
        policy_state_probs = active_states.get('policy_state_probs')
        
        # Check for state transition
        self.current_state = self._check_transition(policy_state_probs)
        
        # Get dynamics for current state
        mu = self._get_attractor(self.current_state)
        theta = self._get_coupling(self.current_state)
        theta = self._clamp_theta(theta)
        
        # Apply descending prediction μ_x if provided (blend is formed upstream in L2/L3).
        # L2->L1 control signals are mu_x and policy_state_probs.
        mu_x = active_states.get('mu_x')
        if mu_x is not None:
            if not isinstance(mu_x, torch.Tensor):
                mu_x = torch.tensor(mu_x, dtype=torch.float32)
            mu = mu_x

        # Global process noise variance - fixed, not modulated by L2.
        sigma = np.sqrt(NOISE_LEVEL)
        
        # MVOU integration with substeps for stability
        dt_sub = self.dt / self.N_SUBSTEPS
        curr_x = self.x
        
        for _ in range(self.N_SUBSTEPS):
            # dX = -Theta(s) · (X - mu) dt + sigma dW
            drift = -torch.matmul(theta, (curr_x - mu)) * dt_sub
            diffusion = torch.randn_like(curr_x) * sigma * np.sqrt(dt_sub)
            curr_x = curr_x + drift + diffusion
            curr_x = torch.clamp(curr_x, CLIP_MIN, CLIP_MAX)
        
        self.x = curr_x
        
        # Return as dict for Markov blanket
        network_acts = {net: self.x[i] for i, net in enumerate(NETWORKS)}
        return network_acts, self.current_state
    
    def reset(self, state: str = 'breath_focus') -> None:
        """Reset to initial state."""
        self.current_state = state
        self._sample_next_dwell()
        self.x = torch.full((len(NETWORKS),), self.INIT_ACTIVATION, dtype=torch.float32)
