"""Layer 1: Generative process for brain network dynamics (MVOU)."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional
from config.meditation_config import DEFAULTS, NETWORK_PROFILES, DWELL_TIMES, STATES, NETWORKS

IDX = {n: i for i, n in enumerate(NETWORKS)}

THETA_NOVICE = {
    'breath_focus': {
        (IDX['DMN'], IDX['DAN']): 0.50, (IDX['DAN'], IDX['DMN']): 0.50,
        (IDX['DAN'], IDX['FPN']): -0.40, (IDX['FPN'], IDX['DAN']): -0.40
    },
    'mind_wandering': {
        (IDX['DMN'], IDX['VAN']): -0.30, (IDX['VAN'], IDX['DMN']): -0.30,
        (IDX['DMN'], IDX['FPN']): -0.15, (IDX['FPN'], IDX['DMN']): -0.15
    },
    'meta_awareness': {
        (IDX['VAN'], IDX['FPN']): -0.50, (IDX['FPN'], IDX['VAN']): -0.50
    },
    'redirect_breath': {
        (IDX['DMN'], IDX['DAN']): -0.30, (IDX['DAN'], IDX['DMN']): -0.30,
        (IDX['DMN'], IDX['FPN']): -0.20, (IDX['FPN'], IDX['DMN']): -0.20,
        (IDX['DAN'], IDX['FPN']): 0.40, (IDX['FPN'], IDX['DAN']): 0.40
    }
}


class StateMachine:
    """Markovian state transitions with dwell times."""

    def __init__(self, level: str, dt: float, seed: int):
        self.level = level
        self.dt = dt
        self.rng = np.random.RandomState(seed)

        self.current_state = STATES[0]
        self.timer_steps = 0
        self.current_max_steps = 0

        self.dwell_ranges_sec = DWELL_TIMES.get(level, DWELL_TIMES['novice'])
        self._sample_next_dwell()

    def _sample_next_dwell(self):
        min_s, max_s = self.dwell_ranges_sec[self.current_state]
        duration_s = self.rng.uniform(min_s, max_s)
        self.current_max_steps = int(duration_s / self.dt)
        self.timer_steps = 0

    def check_transition(self, dwell_modifier: float = 1.0) -> str:
        self.timer_steps += 1

        min_s, _ = self.dwell_ranges_sec[self.current_state]
        min_steps = int(min_s / self.dt)

        effective_limit_steps = int(self.current_max_steps * dwell_modifier)
        effective_limit_steps = max(effective_limit_steps, min_steps)

        if self.timer_steps >= effective_limit_steps:
            self.transition()

        return self.current_state

    def transition(self):
        try:
            current_idx = STATES.index(self.current_state)
            next_idx = (current_idx + 1) % len(STATES)
            self.current_state = STATES[next_idx]
        except ValueError:
            self.current_state = STATES[0]
        self._sample_next_dwell()


class MeditationGenerativeProcess(nn.Module):
    """Layer 1 MVOU dynamics for brain networks."""

    def __init__(self, experience_level: str = 'expert', seed: Optional[int] = None,
                 learned_attractors: Optional[Dict] = None):
        super().__init__()

        self.level = experience_level.lower()
        self.dt = DEFAULTS['DEFAULT_DT']
        self.networks = NETWORKS
        self.seed = seed

        self.sm = StateMachine(self.level, self.dt, seed)
        self.current_max_dwell = self.sm.current_max_steps

        self.x = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float32)
        self.smoothed_x = self.x.clone()
        self.smoothing_alpha = 0.9 if learned_attractors else 0.7

        self.learned_attractors = learned_attractors

    def get_dynamics(self, state: str, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.learned_attractors and state in self.learned_attractors:
            prof = self.learned_attractors[state]
            mu_np = np.array([prof.get(n, 0.5) for n in self.networks])
        else:
            prof = NETWORK_PROFILES[state][self.level]
            mu_np = np.array([prof[n] for n in self.networks])

        mu = torch.tensor(mu_np, dtype=torch.float32, device=device)

        coupling_map = THETA_NOVICE.get(state, {})
        base_diag = 0.50 if state == 'mind_wandering' else 0.15

        theta_np = np.eye(4) * base_diag
        for (row, col), val in coupling_map.items():
            theta_np[row, col] = val

        theta = torch.tensor(theta_np, dtype=torch.float32, device=device)

        if self.level == 'expert':
            if state == 'breath_focus':
                theta = theta + torch.eye(4, device=device) * 0.4
            elif state == 'redirect_breath':
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

        max_stiffness = 2.5
        margin = 0.1

        off_diag_mask = 1.0 - torch.eye(4, device=device)
        off_diag_sum = torch.sum(torch.abs(theta) * off_diag_mask, dim=1)

        scaling_factors = torch.ones(4, device=device)
        high_volatility_mask = off_diag_sum > (max_stiffness - margin)
        if torch.any(high_volatility_mask):
            target_sum = max_stiffness - margin
            scaling_factors[high_volatility_mask] = target_sum / (off_diag_sum[high_volatility_mask] + 1e-6)

        scaling_mat = scaling_factors.unsqueeze(1).expand(4, 4)
        theta_off = (theta * off_diag_mask) * scaling_mat

        new_off_sum = torch.sum(torch.abs(theta_off), dim=1)
        min_required_diag = new_off_sum + margin
        current_diag = torch.diag(theta)
        final_diag = torch.maximum(current_diag, min_required_diag)
        final_diag = torch.minimum(final_diag, torch.tensor(max_stiffness, device=device))

        theta = theta_off + torch.diag(final_diag)

        return mu, theta

    def update(self, active_states: Dict) -> Tuple[Dict[str, torch.Tensor], str]:
        dwell_mod = active_states.get('dwell_modifier', 1.0)
        if isinstance(dwell_mod, torch.Tensor):
            dwell_mod = dwell_mod.item()

        current_state = self.sm.check_transition(dwell_mod)
        self.current_state = current_state
        self.current_max_dwell = self.sm.current_max_steps

        device = self.x.device
        mu, theta = self.get_dynamics(current_state, device)

        agent_bias = active_states.get('agent_bias')
        if agent_bias is not None:
            alpha = 0.6 if self.level == 'expert' else 0.1
            if not isinstance(agent_bias, torch.Tensor):
                agent_bias = torch.tensor(agent_bias, device=device, dtype=torch.float32)
            mu = (1 - alpha) * mu + alpha * agent_bias

        noise_gain = active_states.get('noise_reduction', 1.0)
        if isinstance(noise_gain, torch.Tensor):
            noise_gain = noise_gain.item()

        base_variance = 0.001 if self.level == 'expert' else 0.002
        variance = max(0.0005, base_variance * noise_gain)
        sigma = np.sqrt(variance)

        n_substeps = 2
        dt_sub = self.dt
        curr_x = self.x

        for _ in range(n_substeps):
            drift = -torch.matmul(theta, (curr_x - mu)) * dt_sub
            diffusion = torch.randn_like(curr_x) * sigma * np.sqrt(dt_sub)
            curr_x = curr_x + drift + diffusion
            curr_x = torch.clamp(curr_x, DEFAULTS['ACTIVATION_CLIP_MIN'], DEFAULTS['ACTIVATION_CLIP_MAX'])

        self.x = curr_x
        self.smoothed_x = self.smoothing_alpha * self.x + (1 - self.smoothing_alpha) * self.smoothed_x

        return dict(zip(self.networks, self.smoothed_x)), current_state

    def reset(self, state: str = STATES[0]):
        self.sm.current_state = state
        self.sm._sample_next_dwell()
        self.current_max_dwell = self.sm.current_max_steps
        self.current_state = state
        self.x = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float32, device=self.x.device)
        self.smoothed_x = self.x.clone()
