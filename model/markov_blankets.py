"""Markov blankets: sensory/active state interfaces between hierarchical levels.

Minimal nested interpretation (Russian-doll semantics):
L1<->L2 and L2<->L3 are separate blankets, but all cross-layer exchange is
strictly adjacent (L3 never reads L1 directly; L1 never reads L3 directly).
This enforces bottom-up emergence and top-down causation via L2.
"""

import torch
from typing import Dict, Any

class MarkovBlanket:
    """Base class for Markov blanket interfaces with strict key contracts."""

    allowed_sensory: set[str] = set()
    allowed_active: set[str] = set()

    def __init__(self, strict: bool = True):
        """
        Args:
            strict: If True, reject unknown keys on update
        """
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
        """Update sensory states. Numeric/Tensor: overwrite. Other (str, dict): overwrite."""
        self._validate_keys(new_states, self.allowed_sensory, "sensory")
        for key, new_val in new_states.items():
            new_val = self._detach_value(new_val)
            if isinstance(new_val, (str, dict, list, tuple)):
                # Non-numeric: overwrite (e.g. labels, belief dicts, candidate lists)
                self.sensory_states[key] = new_val
            elif key not in self.sensory_states:
                self.sensory_states[key] = new_val
            else:
                self.sensory_states[key] = new_val
    
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
        - mu_x: Descending predictions over network activations (policy)
        - policy_state_probs: Optional[Dict[str,float]] policy posterior over candidate states
    """
    
    def __init__(self):
        self.allowed_sensory = {"DMN", "VAN", "DAN", "FPN", "dwell_progress"}
        self.allowed_active = {"mu_x", "policy_state_probs"}
        super().__init__(strict=True)
        

class MarkovBlanketL2L3(MarkovBlanket):
    """Markov blanket between L2 (attentional agent) and L3 (metacognitive monitor).
    
    Sensory (L2 -> L3):
        - state_belief: Dict[str, float] inferred state posterior from L2
        - policy_candidates: list[str] policy candidate labels
        - policy_priors: list[float] dwell-aware prior weights E(pi)
        - policy_costs: list[float] policy evidence G(pi)
        - thoughtseed_activations: list[float] inferred latent causes Z
    
    Active (L3 -> L2):
        - precision_sensory: float (0-1, sensory precision)
    """
    
    def __init__(self):
        self.allowed_sensory = {"state_belief", "policy_candidates", "policy_priors", "policy_costs", "thoughtseed_activations"}
        self.allowed_active = {"precision_sensory"}
        super().__init__(strict=True)
        self.active_states['precision_sensory'] = 0.5

    def reset(self) -> None:
        """Reset state and restore defaults."""
        super().reset()
        self.active_states['precision_sensory'] = 0.5
