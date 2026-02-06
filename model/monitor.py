"""Layer 3: Metacognitive monitor for precision inference.

Tracks meta-awareness and infers policy precision via a Gamma posterior.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from utils.config import THOUGHTSEEDS, compute_meta_awareness, EPS
from .blankets import MarkovBlanketL2L3
from utils.math_utils import clip_probability, to_float, precision_to_weight

class Layer3Monitor(nn.Module):
    """Metacognitive monitor: precision inference for policy and perception."""
    GAMMA_MIN = 0.1
    GAMMA_MAX = 3.0
    GAMMA_DECAY = 0.98
    
    def __init__(self, experience_level: str = 'expert',
                 blanket_l2l3: Optional[MarkovBlanketL2L3] = None):
        super().__init__()
        
        self.level = experience_level
        self.blanket_l2l3 = blanket_l2l3 or MarkovBlanketL2L3(smoothing=0.7)

        # Gamma posterior parameters for policy precision (inverse variance)
        self.alpha = 1.0
        self.beta = 1.0
        self.policy_precision = self._compute_precision()

        # Meta-awareness EMA
        self.meta_awareness_ema = None

        # Initialize active states
        self._update_precision_states()

    def _compute_precision(self) -> float:
        gamma = float(self.alpha / max(self.beta, EPS))
        return float(min(self.GAMMA_MAX, max(self.GAMMA_MIN, gamma)))

    def _update_precision_states(self) -> tuple[float, float]:
        """Update blanket with current precision signals."""
        gamma = self._compute_precision()
        precision_sensory = precision_to_weight(gamma)
        self.policy_precision = gamma
        self.blanket_l2l3.update_active_states({
            'policy_precision': gamma,
            'precision_sensory': precision_sensory,
        })
        return gamma, precision_sensory

    def reset(self) -> None:
        """Reset monitor state for an isolated training run."""
        self.alpha = 1.0
        self.beta = 1.0
        self.policy_precision = self._compute_precision()
        self.meta_awareness_ema = None
        self.blanket_l2l3.reset()
        self._update_precision_states()
    
    def update_meta_awareness(self, current_state: str, z: torch.Tensor) -> float:
        """Compute meta-awareness (A) from L2 thoughtseed activations."""
        z_dict = {ts: z[i].item() for i, ts in enumerate(THOUGHTSEEDS)}
        raw = compute_meta_awareness(current_state, z_dict)
        
        if self.meta_awareness_ema is None:
            self.meta_awareness_ema = raw
        else:
            self.meta_awareness_ema = 0.9 * self.meta_awareness_ema + 0.1 * raw
        
        meta = clip_probability(self.meta_awareness_ema)
        opacity = 1.0 - meta
        self.blanket_l2l3.update_sensory_states({
            'meta_awareness': meta,
            'opacity': opacity
        })
        
        return meta

    def infer_meta_posterior(self, forward_prediction_error: Optional[float]) -> Dict:
        """Infer policy precision via Gamma posterior updates.

        Args:
            forward_prediction_error: Scalar prediction error (epsilon)
        """
        if forward_prediction_error is not None:
            err = max(EPS, to_float(forward_prediction_error))
            decay = self.GAMMA_DECAY
            self.alpha = decay * self.alpha + 0.5
            self.beta = decay * self.beta + 0.5 * err

        gamma, precision_sensory = self._update_precision_states()

        return {
            'policy_precision': gamma,
            'precision_sensory': precision_sensory,
            'alpha': float(self.alpha),
            'beta': float(self.beta),
        }
