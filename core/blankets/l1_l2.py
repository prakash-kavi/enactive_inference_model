"""Markov Blanket L1-L2: brain-network interface.

Mediates between Layer 1 (generative process) and Layer 2 (attentional agents).
- Sensory (L1 -> L2): network activations [DMN, VAN, DAN, FPN]
- Active (L2 -> L1): modulation signals [noise_reduction, transition_drive]
"""

import numpy as np
from typing import Dict
from utils.meditation_config import NETWORKS



class MarkovBlanketL1L2:
    """Markov Blanket for the Layer 1 <-> Layer 2 interface."""

    def __init__(self, smoothing: float = 0.7):
        self.active_states = {
            'noise_reduction': 1.0,
            'l2tol1_enactive_bias': 0.0,
            'transition_drive': 0.0,
            'agent_bias': None,
        }
        self.smoothing = smoothing
        self.sensory_states: Dict[str, float] = {}

    def update_sensory_states(self, observations: Dict[str, float]):
        """Update sensory states with network activations from Layer 1."""
        if not hasattr(self, 'sensory_states') or self.sensory_states is None:
            self.sensory_states = {}

        self.sensory_states.update(observations)

    def update_active_states(self, prescriptions: Dict[str, float]):
        """Update active states with modulation signals from Layer 2."""
        for key, target in prescriptions.items():
            if key not in self.active_states:
                continue

            if key == 'agent_bias':
                self.active_states[key] = target
                continue

            self.active_states[key] = (
                self.smoothing * self.active_states[key] + (1 - self.smoothing) * target
            )

        self.active_states['noise_reduction'] = np.clip(self.active_states['noise_reduction'], 0.2, 1.2)
        self.active_states['l2tol1_enactive_bias'] = np.clip(self.active_states['l2tol1_enactive_bias'], 0.0, 1.0)
        self.active_states['transition_drive'] = np.clip(self.active_states['transition_drive'], 0.0, 1.0)

    def reset(self):
        """Reset to default values for new simulation runs."""
        self.active_states = {
            'noise_reduction': 1.0,
            'l2tol1_enactive_bias': 0.0,
            'transition_drive': 0.0,
            'agent_bias': None,
        }
        self.sensory_states = {}
