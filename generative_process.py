"""
Layer 1: Generative Process (Ground Truth)

This module implements the "ground truth" biological reality of meditation via 
Multivariate Ornstein-Uhlenbeck (MVOU) dynamics. It simulates the continuous 
evolution of brain network activations as a piecewise non-linear dynamical system.

The process follows the Stochastic Differential Equation:
    dx_t = -Θ(x_t - μ(t))dt + ΣdW_t

Key Features:
- dt = 1.0 second for empirical alignment.
- State-specific Attractor Dynamics (MVOU).
- Probabilistic Dwell Times (Weibull Distribution).
"""

import numpy as np
from typing import Dict, Tuple, Optional

class MeditationGenerativeProcess:
    """
    Layer 1: Generative Process (Ground Truth).
    Simulates unconscious network signals [DMN, VAN, DAN, FPN].
    """
    
    def __init__(self, experience_level: str = 'expert', seed: Optional[int] = None):
        self.networks = ['DMN', 'VAN', 'DAN', 'FPN']
        self.level = experience_level.lower()
        self.dt = 0.5  # Smaller time step for better numerical stability (2 substeps = 1 second)
        self.rng = np.random.RandomState(seed)
        
        # Initial activation state (Neutral baseline)
        self.x = np.array([0.5, 0.5, 0.5, 0.5])
        
        # Exponential Moving Average smoothing for smooth network trajectories
        # Alpha = 0.7 means 70% current value, 30% previous (smooth but responsive)
        self.smoothing_alpha = 0.7
        self.smoothed_x = self.x.copy()
        
        # Canonical Cycle Logic: BF -> MW -> MA -> RA -> BF
        self.current_state = 'BF'
        self.state_timer = 0
        
        # Empirical Dwell Time Ranges (seconds)
        self.dwell_configs = {
            'expert': {'BF': (15, 30), 'MW': (10, 20), 'MA': (1, 4), 'RA': (1, 4)},
            'novice': {'BF': (5, 15), 'MW': (20, 40), 'MA': (2, 6), 'RA': (2, 5)}
        }[self.level]
        
        self.current_max_dwell = self._sample_weibull_dwell()
        # Noise variance: Expert uses lower noise (50% reduction) for less jitter
        self.noise_variance = 0.001 if self.level == 'expert' else 0.002

    def _sample_weibull_dwell(self) -> int:
        """
        Samples a duration using a Weibull distribution for realistic 
        temporal jitter/heavy tails.
        """
        min_d, max_d = self.dwell_configs[self.current_state]
        mean_d = (min_d + max_d) / 2
        
        # Shape k=1.5 provides a realistic heavy-tail for cognitive states
        shape_k = 1.5
        # Scale lambda = mean / Gamma(1 + 1/k)
        scale_l = mean_d / 0.9027  # Gamma(1 + 1/1.5) approx 0.9027
        
        sample = scale_l * self.rng.weibull(shape_k)
        # Clamp to ensure we stay within plausible biological bounds
        return int(np.clip(sample, min_d, max_d))

    def _transition_state(self, active_states: Optional[dict] = None):
        """Cyclical transition: Breath Focus -> Mind Wandering -> Meta-Awareness -> Redirect.
        
        Args:
            active_states: Optional modulation from Markov Blanket (for dwell modifier)
        """
        flow = {'BF': 'MW', 'MW': 'MA', 'MA': 'RA', 'RA': 'BF'}
        self.current_state = flow[self.current_state]
        self.state_timer = 0
        
        # Sample new dwell time
        self.current_max_dwell = self._sample_weibull_dwell()
        
        # Apply volitional control (dwell modifier) only at transition to avoid mid-cycle disruption
        if active_states and 'dwell_modifier' in active_states:
            # Only apply to distraction states (MW) - allow agent to "short-circuit" wandering
            if self.current_state == 'MW':
                self.current_max_dwell = int(self.current_max_dwell * active_states['dwell_modifier'])
                self.current_max_dwell = max(1, self.current_max_dwell)  # Ensure at least 1 timestep

    def get_dynamics(self) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieves exact state-specific Mean Targets (mu) and Drift Matrices (Theta)."""
        # 1. MEAN TARGETS (mu)
        mu_map = {
            'expert': {
                'BF': [0.30, 0.45, 0.60, 0.45], 'MW': [0.60, 0.65, 0.40, 0.65],
                'MA': [0.40, 0.80, 0.50, 0.70], 'RA': [0.35, 0.55, 0.65, 0.55]
            },
            'novice': {
                'BF': [0.55, 0.50, 0.60, 0.65], 'MW': [0.75, 0.40, 0.35, 0.40],
                'MA': [0.50, 0.60, 0.50, 0.55], 'RA': [0.45, 0.50, 0.65, 0.70]
            }
        }
        mu = np.array(mu_map[self.level][self.current_state])

        # 2. DRIFT MATRICES (Theta) 
        # Expert stability base: 0.50. Meta-Awareness 'Aha' Spike: 0.80
        diag_val = 0.80 if (self.level == 'expert' and self.current_state == 'MA') else (0.50 if self.level == 'expert' else 0.15)
        theta = np.eye(4) * diag_val

        # Exact Off-diagonal Coupling
        # Convention: Negative coupling (anti-correlation) → positive Θ (competition/inhibition)
        #              Positive coupling (synergistic) → negative Θ (synchronization/excitation)
        # NOTE: Values are hardcoded based on empirical neuroscience, not from COUPLING_MATRICES.
        # This allows fine-tuning beyond the config values.
        if self.level == 'expert':
            if self.current_state == 'BF':
                theta[0, 2] = theta[2, 0] = 0.70  # DMN-DAN competition
                theta[0, 3] = theta[3, 0] = 0.60  # DMN-FPN anti-correlation
                theta[2, 3] = theta[3, 2] = -0.50 # DAN-FPN coordination
            elif self.current_state == 'MW':
                theta[0, 1] = theta[1, 0] = -0.30 # DMN-VAN monitoring
                theta[0, 3] = theta[3, 0] = -0.20 # DMN-FPN positive coupling
                theta[1, 3] = theta[3, 1] = -0.40 # VAN-FPN monitoring
            elif self.current_state == 'MA':
                theta[1, 3] = theta[3, 1] = -0.70 # Peak VAN-FPN salience peak
            elif self.current_state == 'RA':
                # RA = Focus-Reacquisition: Positive off-diagonals suppress DMN, driving return to focus
                theta[0, 2] = theta[2, 0] = 0.60  # DMN-DAN suppression (Crucial for Return)
                theta[0, 3] = theta[3, 0] = 0.50  # DMN-FPN suppression
                theta[2, 3] = theta[3, 2] = -0.70 # DAN-FPN coordination
        else:  # Novice coupling (weaker stability, higher competition effort)
            if self.current_state == 'BF':
                theta[0, 2] = theta[2, 0] = 0.50
                theta[2, 3] = theta[3, 2] = -0.40
            elif self.current_state == 'MW':
                theta[0, 2] = theta[2, 0] = 0.60
                theta[0, 3] = theta[3, 0] = 0.60
            elif self.current_state == 'MA':
                theta[1, 3] = theta[3, 1] = -0.50
            elif self.current_state == 'RA':
                # RA = Focus-Reacquisition: Positive off-diagonals suppress DMN
                theta[0, 2] = theta[2, 0] = 0.55
                theta[0, 3] = theta[3, 0] = 0.45
                theta[2, 3] = theta[3, 2] = -0.60

        # Stability Enforcement (Gershgorin Disk Theorem adaptation)
        epsilon = 0.01 
        row_sums = np.sum(np.abs(theta), axis=1) - np.diag(theta)
        for i in range(4):
            if theta[i, i] <= row_sums[i]:
                # Set diagonal just above the sum of off-diagonals
                theta[i, i] = row_sums[i] + epsilon                
        return mu, theta

    def update(self, active_states: Optional[dict] = None) -> Tuple[Dict[str, float], str]:
        """
        Core MVOU update step (Piecewise Linear Dynamics) with downward causation.
        
        Args:
            active_states: Optional modulation from Markov Blanket for attentional gain and volitional control
        
        Returns:
            Tuple of (network_activations_dict, state_abbreviation)
        """
        self.state_timer += 1
        if self.state_timer >= self.current_max_dwell:
            self._transition_state(active_states)

        mu, theta = self.get_dynamics()
        
        # Apply Attentional Gain (Noise Reduction) from Markov Blanket
        effective_noise_variance = self.noise_variance
        if active_states and 'noise_reduction' in active_states:
            effective_noise_variance = self.noise_variance * active_states['noise_reduction']
            effective_noise_variance = max(0.0005, effective_noise_variance)  # Safety: never zero
        
        # Correlated Noise: K mirrors coupling structure
        K = np.eye(4) * effective_noise_variance
        for i in range(4):
            for j in range(i + 1, 4):
                if theta[i, j] != 0:
                    corr = -0.5 * (theta[i, j] / np.max(np.abs(theta)))
                    K[i, j] = K[j, i] = corr * effective_noise_variance
        
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(K + np.eye(4) * 1e-6)
        
        # Sub-stepping: 2 steps with dt=0.5 = 1.0 second total (better numerical stability)
        n_substeps = 2
        dt_sub = self.dt
        
        for _ in range(n_substeps):
            # dx_t = -Theta(x_t - mu)dt
            drift = -theta @ (self.x - mu) * dt_sub
            
            # Generate noise for this substep
            noise = L @ self.rng.standard_normal(4) * np.sqrt(dt_sub)
            
            # Stochastic Update + Biological Saturation
            self.x = np.clip(self.x + drift + noise, 0.05, 0.9)
        
        # Apply exponential moving average smoothing for smooth network trajectories
        # This reduces jitter while maintaining responsiveness to state changes
        self.smoothed_x = self.smoothing_alpha * self.x + (1 - self.smoothing_alpha) * self.smoothed_x
        
        return dict(zip(self.networks, self.smoothed_x)), self.current_state

    def reset(self, state: str = 'BF'):
        """Reset the unconscious process."""
        self.current_state = state
        self.state_timer = 0
        self.current_max_dwell = self._sample_weibull_dwell()
        self.x = np.array([0.5, 0.5, 0.5, 0.5])
        self.smoothed_x = self.x.copy()  # Reset smoothed state