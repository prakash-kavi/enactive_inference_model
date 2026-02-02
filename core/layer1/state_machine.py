"""Layer-1 State Machine: Markovian state transitions with dwell times and hazard-based switching."""

import numpy as np
from typing import Dict, Optional
from utils.meditation_config import STATES, DEFAULTS, EPS
from .layer1_config import DWELL_TIMES, STATE_TRANSITION_PROBS


class StateMachine:
    """Markovian state transitions with dwell times."""
    
    # Transition constants
    CYCLE_STRENGTH = 0.35
    MW_PERSISTENCE_THRESHOLD = 0.6

    def __init__(self, level: str, dt: float, seed: int, generative_costs: Optional[Dict] = None):
        self.level = level
        self.dt = dt
        self.rng = np.random.RandomState(seed)
        self.generative_costs = generative_costs
        self.current_state = STATES[0]
        self.timer_steps = 0
        self.current_max_steps = 0
        self.refractory_steps = max(1, int(DEFAULTS['REFRACTORY_SEC'] / self.dt))
        self.dwell_ranges_sec = DWELL_TIMES[level]
        self.transition_probs = STATE_TRANSITION_PROBS[level]
        self.last_transition_hazard = 0.0
        self._sample_next_dwell()

    def _sample_next_dwell(self):
        """Sample dwell duration with U-shaped distribution for variability."""
        min_s, max_s = self.dwell_ranges_sec[self.current_state]
        min_steps = max(1.0, min_s / self.dt)
        max_steps = max(min_steps, max_s / self.dt)
        if max_steps == min_steps:
            self.current_max_steps = min_steps
        else:
            u = float(self.rng.beta(2.0, 2.0))
            self.current_max_steps = min_steps + u * (max_steps - min_steps)
        self.timer_steps = 0

    def check_transition(self, transition_drive: float = 0.0, mw_burden: float = 0.0) -> str:
        """Check if state transition should occur based on dwell progress and hazard."""
        self.timer_steps += 1

        min_s, _ = self.dwell_ranges_sec[self.current_state]
        min_steps = min_s / self.dt
        self.last_transition_hazard = 0.0

        # Refractory period: no transitions
        if self.timer_steps < self.refractory_steps:
            return self.current_state

        drive = max(0.0, min(transition_drive, 1.0))
        
        # Min dwell not satisfied
        if self.timer_steps < min_steps:
            return self.current_state

        # Max dwell reached: force transition
        if self.timer_steps >= self.current_max_steps:
            self.transition(drive)
            return self.current_state

        # Probabilistic transition based on hazard
        progress = min(1.0, self.timer_steps / max(1, self.current_max_steps))
        
        if self.current_state == 'mind_wandering':
            # MW-specific: burden-driven detection hazard
            costs = self.generative_costs
            burden_excess = max(0.0, mw_burden - costs['mw_detection_threshold'])
            hazard = 0.015 + 0.20 * progress + 0.18 * drive + costs['mw_detection_gain'] * burden_excess
        else:
            hazard = 0.008 + 0.14 * progress + 0.15 * drive
        
        self.last_transition_hazard = np.clip(hazard, 0.0, 0.95)
        
        if self.rng.rand() < self.last_transition_hazard:
            self.transition(drive)
        
        return self.current_state

    def _normalize(self, weights: np.ndarray) -> np.ndarray:
        """Normalize weights, handling zero sum."""
        return weights / max(weights.sum(), EPS)
    
    def _apply_bias(self, weights: np.ndarray, states: list, target_state: str, bias_strength: float) -> np.ndarray:
        """Apply bias toward a target state."""
        if target_state not in states or bias_strength <= 0:
            return weights
        target = np.array([1.0 if s == target_state else 0.0 for s in states])
        return (1.0 - bias_strength) * weights + bias_strength * target

    def transition(self, transition_drive: float = 0.0):
        """Execute state transition with drive-modulated and state-specific biases."""
        probs = self.transition_probs.get(self.current_state)
        if probs:
            states = list(probs.keys())
            weights = self._normalize(np.array([probs[s] for s in states]))
            drive = np.clip(transition_drive, 0.0, 1.0)
            
            # Cycle-forward bias
            if drive > 0.0:
                try:
                    next_state = STATES[(STATES.index(self.current_state) + 1) % len(STATES)]
                    weights = self._apply_bias(weights, states, next_state, drive * self.CYCLE_STRENGTH)
                except ValueError:
                    pass
            
            # MW persistence bias toward meta-awareness
            if self.current_state == 'mind_wandering' and self.current_max_steps > 0:
                progress = min(1.0, self.timer_steps / max(1, self.current_max_steps))
                if progress > self.MW_PERSISTENCE_THRESHOLD:
                    bias = (progress - self.MW_PERSISTENCE_THRESHOLD) / (1.0 - self.MW_PERSISTENCE_THRESHOLD)
                    weights = self._apply_bias(weights, states, 'meta_awareness', np.clip(bias, 0.0, 1.0))
            
            # RA reorienting bias toward BF
            if self.current_state == 'redirect_attention':
                bf_bias = np.clip(self.generative_costs['ra_to_bf_transition_bias'], 0.0, 0.8)
                weights = self._apply_bias(weights, states, 'breath_focus', bf_bias)
            
            self.current_state = self.rng.choice(states, p=self._normalize(weights))
        else:
            # Fallback: cycle forward
            try:
                current_idx = STATES.index(self.current_state)
                next_idx = (current_idx + 1) % len(STATES)
                self.current_state = STATES[next_idx]
            except ValueError:
                self.current_state = STATES[0]
        
        self._sample_next_dwell()
