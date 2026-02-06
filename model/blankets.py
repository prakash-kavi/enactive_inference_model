"""Markov blankets: sensory/active state interfaces between hierarchical levels.

Each blanket enforces clean separation between levels:
- Sensory states: Read-only observations (bottom-up)
- Active states: Control signals (top-down)
- EMA smoothing: Prevents discontinuities across hierarchy
"""

import torch
from typing import Dict, Any

class MarkovBlanket:
    """Base class for Markov blanket interfaces."""
    
    def __init__(self, smoothing: float = 0.7):
        """
        Args:
            smoothing: EMA coefficient (0=no memory, 1=no update)
        """
        self.smoothing = float(max(0.0, min(1.0, smoothing)))
        self.sensory_states: Dict[str, Any] = {}
        self.active_states: Dict[str, Any] = {}
        
    def update_sensory_states(self, new_states: Dict[str, Any]) -> None:
        """Update sensory states with EMA smoothing."""
        for key, new_val in new_states.items():
            if key not in self.sensory_states:
                # First observation: initialize directly
                self.sensory_states[key] = new_val
            else:
                # EMA update
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
        - State burdens/costs (optional for lean model)
    
    Active (L2 -> L1):
        - action_mu: Target network activations (policy)
        - transition_drive: L2 policy confidence / drive
    """
    
    def __init__(self, smoothing: float = 0.7):
        super().__init__(smoothing=smoothing)
        # Initialize empty - will be populated by L1 process
        

class MarkovBlanketL2L3(MarkovBlanket):
    """Markov blanket between L2 (attentional agent) and L3 (metacognitive monitor).
    
    Sensory (L2 -> L3):
        - current_state: str (meditation state label)
        - dwell_progress: float (0-1, how long in current state)
        - thoughtseed_activations: Dict[str, float] (for meta-awareness)
    
    Active (L3 -> L2):
        - sensory_precision: float (0-1, precision surrogate)
        - transition_drive: float (0-1, policy-driven state change pressure)
    """
    
    def __init__(self, smoothing: float = 0.7):
        super().__init__(smoothing=smoothing)
        # Initialize with defaults
        self.active_states['sensory_precision'] = 0.5
        self.active_states['transition_drive'] = 0.0

    def reset(self) -> None:
        """Reset state and restore defaults."""
        super().reset()
        self.active_states['sensory_precision'] = 0.5
        self.active_states['transition_drive'] = 0.0
