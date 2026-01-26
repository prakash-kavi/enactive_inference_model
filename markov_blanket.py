import numpy as np
from typing import Dict, Optional

class MarkovBlanket:
    """
    Generic Markov Blanket interface for mediating between layers.
    
    Can be used for:
    - L1 ↔ L2: Network activations (sensory) ↔ Prescriptions (active)
    - L2 ↔ L3: Thoughtseed activations (sensory) ↔ Precision modulations (active)
    """
    def __init__(self, smoothing: float = 0.9, active_states_template: Optional[Dict[str, float]] = None):
        """
        Initialize Markov Blanket.
        
        Args:
            smoothing: Temporal smoothing factor for active states (0.9 = 90% previous, 10% new)
            active_states_template: Template for active states. If None, uses L1-L2 defaults.
        """
        if active_states_template is None:
            # Default: L1-L2 interface (biological modulations)
            self.active_states = {
                'noise_reduction': 1.0,  # Attentional Gain
                'dwell_modifier': 1.0,   # Volitional Control
                'fatigue_buffer': 1.0    # Equanimity Buffer
            }
        else:
            # Custom template (e.g., for L2-L3 interface)
            self.active_states = active_states_template.copy()
        
        self.smoothing = smoothing
        # Sensory States (s): Observations from the lower layer
        self.sensory_states = {}

    def update_sensory_states(self, observations: dict):
        """Update the sensory boundary with new data from Layer 1."""
        self.sensory_states = observations

    def update_active_states(self, prescriptions: dict):
        """Apply combination logic and temporal smoothing to prescriptions."""
        for key, target in prescriptions.items():
            if key in self.active_states:
                # Gradual temporal smoothing (smoothing * prev + (1-smoothing) * target)
                self.active_states[key] = (self.smoothing * self.active_states[key]) + \
                                          ((1 - self.smoothing) * target)
        
        # Safety bounds (only apply to L1-L2 interface states if present)
        if 'noise_reduction' in self.active_states:
            self.active_states['noise_reduction'] = np.clip(self.active_states['noise_reduction'], 0.2, 1.2)
        if 'dwell_modifier' in self.active_states:
            self.active_states['dwell_modifier'] = np.clip(self.active_states['dwell_modifier'], 0.1, 1.0)
        if 'fatigue_buffer' in self.active_states:
            self.active_states['fatigue_buffer'] = np.clip(self.active_states['fatigue_buffer'], 0.1, 1.0)
        
        # Safety bounds for L2-L3 interface states if present
        if 'precision_modulation' in self.active_states:
            self.active_states['precision_modulation'] = np.clip(self.active_states['precision_modulation'], 0.5, 2.0)
        if 'theta_modulation' in self.active_states:
            self.active_states['theta_modulation'] = np.clip(self.active_states['theta_modulation'], 0.5, 2.0)

    def reset(self):
        """Clean initialization for new simulation runs."""
        for key in self.active_states:
            self.active_states[key] = 1.0
        self.sensory_states = {}