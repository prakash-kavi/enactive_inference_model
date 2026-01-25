import numpy as np

class MarkovBlanket:
    """
    The formal Active Interface (a) mediating between the Generative Model and the Generative Process.
    Acts as the 'Russian Doll' shell for the ActInfAgent.
    """
    def __init__(self, smoothing: float = 0.9):
        # Active States (a): Prescriptive modulations
        self.active_states = {
            'noise_reduction': 1.0,  # Attentional Gain
            'dwell_modifier': 1.0,   # Volitional Control
            'fatigue_buffer': 1.0    # Equanimity Buffer
        }
        self.smoothing = smoothing
        # Sensory States (s): The blanket's observation of Layer 1
        self.sensory_states = {}

    def update_sensory_states(self, observations: dict):
        """Update the sensory boundary with new data from Layer 1."""
        self.sensory_states = observations

    def update_active_states(self, prescriptions: dict):
        """Apply combination logic and temporal smoothing to prescriptions."""
        for key, target in prescriptions.items():
            if key in self.active_states:
                # Gradual temporal smoothing (0.9 prev + 0.1 target)
                self.active_states[key] = (self.smoothing * self.active_states[key]) + \
                                          ((1 - self.smoothing) * target)
        
        # Safety bounds for biological stability
        self.active_states['noise_reduction'] = np.clip(self.active_states['noise_reduction'], 0.2, 1.2)
        self.active_states['dwell_modifier'] = np.clip(self.active_states['dwell_modifier'], 0.1, 1.0)
        self.active_states['fatigue_buffer'] = np.clip(self.active_states['fatigue_buffer'], 0.1, 1.0)

    def reset(self):
        """Clean initialization for new simulation runs."""
        for key in self.active_states:
            self.active_states[key] = 1.0
        self.sensory_states = {}