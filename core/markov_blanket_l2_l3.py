"""Markov Blanket L2-L3: attentional-agent monitoring interface.

Mediates between Layer 2 (GNW bottleneck) and Layer 3 (monitor).
- Sensory (L2 -> L3): thoughtseed activations, meta-awareness, state, event flags
- Active (L3 -> L2): precision modulation
"""

import numpy as np
from typing import Dict


class MarkovBlanketL2L3:
    """Markov Blanket for the Layer 2 <-> Layer 3 interface."""

    VALID_SENSORY_KEYS = {
        'thoughtseed_activations',
        'meta_awareness',
        'current_state',
        'dominant_thoughtseed',
        'dominant_activation',
        'van_spike_detected',
        'aha_accumulator_value',
        'current_van',
        'recognition_signal'
    }

    def __init__(self, smoothing: float = 0.7):
        self.active_states = {
            'precision_modulation': 1.0,
        }
        self.smoothing = smoothing
        self.sensory_states: Dict = {}

    def update_sensory_states(self, observations: Dict):
        """Update sensory states with data from Layer 2."""
        if not hasattr(self, 'sensory_states') or self.sensory_states is None:
            self.sensory_states = {}

        provided_keys = set(observations.keys())
        invalid_keys = provided_keys - self.VALID_SENSORY_KEYS
        if invalid_keys:
            raise ValueError(
                f"Invalid keys in observations: {invalid_keys}. "
                f"Valid keys: {self.VALID_SENSORY_KEYS}"
            )

        self.sensory_states.update(observations)

    def update_active_states(self, prescriptions: Dict[str, float]):
        """Update active states with precision modulation from Layer 3."""
        for key, target in prescriptions.items():
            if key in self.active_states:
                self.active_states[key] = (
                    self.smoothing * self.active_states[key] + (1 - self.smoothing) * target
                )

        self.active_states['precision_modulation'] = np.clip(self.active_states['precision_modulation'], 0.5, 2.0)

    def reset(self):
        """Reset to default values for new simulation runs."""
        self.active_states = {
            'precision_modulation': 1.0,
        }
        self.sensory_states = {}
