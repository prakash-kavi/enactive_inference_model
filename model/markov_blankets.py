"""Markov blankets: sensory/active state interfaces between hierarchical levels.

Minimal nested interpretation (Russian-doll semantics):
L1↔L2 and L2↔L3 are separate blankets, but all cross-layer exchange is
strictly adjacent (L3 never reads L1 directly; L1 never reads L3 directly).
This enforces bottom-up emergence and top-down causation via L2.
"""

import torch
from typing import Dict, Any

class MarkovBlanket:
    """Base class for Markov blanket interfaces with strict key contracts."""

    allowed_sensory: set[str] = set()
    allowed_active: set[str] = set()

    def __init__(self, smoothing: float = 0.0, strict: bool = True):
        """
        Args:
            smoothing: EMA coefficient (0=no memory, 1=no update)
            strict: If True, reject unknown keys on update
        """
        self.smoothing = float(max(0.0, min(1.0, smoothing)))
        self.strict = bool(strict)
        self.sensory_states: Dict[str, Any] = {}
        self.active_states: Dict[str, Any] = {}

    def _detach_value(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach()
        if isinstance(value, dict):
            return {k: self._detach_value(v) for k, v in value.items()}
        return value

    def _validate_keys(self, new_states: Dict[str, Any], allowed: set[str], kind: str) -> None:
        if not self.strict:
            return
        extra = set(new_states.keys()) - allowed
        if extra:
            raise KeyError(f"Unknown {kind} keys for {self.__class__.__name__}: {sorted(extra)}")
        
    def update_sensory_states(self, new_states: Dict[str, Any]) -> None:
        """Update sensory states. Numeric/Tensor: EMA smoothing. Other (str, dict): overwrite."""
        self._validate_keys(new_states, self.allowed_sensory, "sensory")
        for key, new_val in new_states.items():
            new_val = self._detach_value(new_val)
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
        self._validate_keys(new_states, self.allowed_active, "active")
        detached = {k: self._detach_value(v) for k, v in new_states.items()}
        self.active_states.update(detached)
    
    def reset(self) -> None:
        """Clear all states."""
        self.sensory_states.clear()
        self.active_states.clear()


class MarkovBlanketL1L2(MarkovBlanket):
    """Markov blanket between L1 (generative process) and L2 (attentional agent).
    
    Sensory (L1 -> L2):
        - Network activations: {DMN, VAN, DAN, FPN}
        - dwell_progress: float (0-1, how long in current state)
    
    Active (L2 -> L1):
        - mu_x: Target network activations (policy)
        - policy_drive: L2 transition urge
    """
    
    def __init__(self, smoothing: float = 0.0):
        self.allowed_sensory = {"DMN", "VAN", "DAN", "FPN", "dwell_progress"}
        self.allowed_active = {"mu_x", "policy_drive"}
        super().__init__(smoothing=smoothing, strict=True)
        

class MarkovBlanketL2L3(MarkovBlanket):
    """Markov blanket between L2 (attentional agent) and L3 (metacognitive monitor).
    
    Sensory (L2 -> L3):
        - current_state: str (meditation state label)
        - dwell_progress: float (0-1, how long in current state)
        - thoughtseed_activations: Dict[str, float] (for meta-awareness)
    
    Active (L3 -> L2):
        - precision_sensory: float (0-1, sensory precision; Eq. 4)
        - policy_prior: list of 4 floats (log prior adjustment per candidate; L3 writes, L2 reads)
    """
    
    def __init__(self, smoothing: float = 0.0):
        self.allowed_sensory = {"current_state", "dwell_progress", "thoughtseed_activations"}
        self.allowed_active = {"precision_sensory", "policy_prior"}
        super().__init__(smoothing=smoothing, strict=True)
        self.active_states['precision_sensory'] = 0.5
        self.active_states['policy_prior'] = None  # neutral until L3 writes

    def reset(self) -> None:
        """Reset state and restore defaults."""
        super().reset()
        self.active_states['precision_sensory'] = 0.5
        self.active_states['policy_prior'] = None