"""Markov blankets: sensory/active state interfaces between hierarchical levels.

Each blanket enforces clean separation between levels:
- Sensory states: Read-only observations (bottom-up)
- Active states: Control signals (top-down)
- Optional EMA smoothing on sensory states
"""

import torch
from typing import Dict, Any

class MarkovBlanket:
    """Base class for Markov blanket interfaces."""
    
    def __init__(self, smoothing: float = 0.0):
        """
        Args:
            smoothing: EMA coefficient (0=no memory, 1=no update)
        """
        self.smoothing = float(max(0.0, min(1.0, smoothing)))
        self.sensory_states: Dict[str, Any] = {}
        self.active_states: Dict[str, Any] = {}
        
    def update_sensory_states(self, new_states: Dict[str, Any]) -> None:
        """Update sensory states. Numeric/Tensor: EMA smoothing. Other (str, dict): overwrite."""
        for key, new_val in new_states.items():
            if isinstance(new_val, (str, dict)):
                # Non-numeric: overwrite (e.g. current_state, thoughtseed_activations)
                self.sensory_states[key] = new_val
            elif key not in self.sensory_states:
                self.sensory_states[key] = new_val
            else:
                old_val = self.sensory_states[key]
                if isinstance(new_val, torch.Tensor):
                    self.sensory_states[key] = (
                        self.smoothing * old_val + (1 - self.smoothing) * new_val
                    )
                else:
                    self.sensory_states[key] = (
                        self.smoothing * float(old_val) + (1 - self.smoothing) * float(new_val)
                    )
    
    def update_active_states(self, new_states: Dict[str, Any]) -> None:
        """Update active states (top-down control)."""
        self.active_states.update(new_states)
    
    def reset(self) -> None:
        """Clear all states."""
        self.sensory_states.clear()
        self.active_states.clear()


class MarkovBlanketL1L2(MarkovBlanket):
    """Markov blanket between L1 (generative process) and L2 (attentional agent).
    
    Sensory (L1 -> L2):
        - Network activations: {DMN, VAN, DAN, FPN}
    
    Active (L2 -> L1):
        - mu_x: Target network activations (policy)
        - policy_drive: L2 transition urge
        - policy_confidence: L2 policy confidence (posterior)
    """
    
    def __init__(self, smoothing: float = 0.0):
        super().__init__(smoothing=smoothing)
        

class MarkovBlanketL2L3(MarkovBlanket):
    """Markov blanket between L2 (attentional agent) and L3 (metacognitive monitor).
    
    Sensory (L2 -> L3):
        - current_state: str (meditation state label)
        - dwell_progress: float (0-1, how long in current state)
        - thoughtseed_activations: Dict[str, float] (for meta-awareness)
    
    Active (L3 -> L2):
        - precision_sensory: float (0-1, sensory precision; Eq. 4)
        - policy_precision: float (>0, softmax inverse temperature)
        - policy_prior: list of 4 floats (log prior adjustment per candidate; L3 writes, L2 reads)
    """
    
    def __init__(self, smoothing: float = 0.0):
        super().__init__(smoothing=smoothing)
        self.active_states['precision_sensory'] = 0.5
        self.active_states['policy_precision'] = 1.0
        self.active_states['policy_prior'] = None  # neutral until L3 writes

    def reset(self) -> None:
        """Reset state and restore defaults."""
        super().reset()
        self.active_states['precision_sensory'] = 0.5
        self.active_states['policy_precision'] = 1.0
        self.active_states['policy_prior'] = None
