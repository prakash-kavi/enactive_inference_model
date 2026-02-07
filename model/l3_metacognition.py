"""Layer 3: Metacognitive monitor for meta-awareness tracking."""

import torch
import torch.nn as nn
from typing import Optional

from utils.config import THOUGHTSEEDS, compute_meta_awareness
from .markov_blankets import MarkovBlanketL2L3
from utils.math_utils import clip_probability

class Layer3Monitor(nn.Module):
    """Metacognitive monitor: tracks meta-awareness from thoughtseeds."""
    
    def __init__(self, experience_level: str = 'expert',
                 blanket_l2l3: Optional[MarkovBlanketL2L3] = None):
        super().__init__()
        
        self.level = experience_level
        self.blanket_l2l3 = blanket_l2l3 or MarkovBlanketL2L3(smoothing=0.7)

        # Meta-awareness EMA
        self.meta_awareness_ema = None

    def reset(self) -> None:
        """Reset monitor state for an isolated training run."""
        self.meta_awareness_ema = None
        self.blanket_l2l3.reset()
    
    def update_meta_awareness(self, current_state: str, z: torch.Tensor) -> float:
        """Compute meta-awareness (A) from L2 thoughtseed activations."""
        z_dict = {ts: z[i].item() for i, ts in enumerate(THOUGHTSEEDS)}
        raw = compute_meta_awareness(current_state, z_dict)
        
        if self.meta_awareness_ema is None:
            self.meta_awareness_ema = raw
        else:
            self.meta_awareness_ema = 0.9 * self.meta_awareness_ema + 0.1 * raw
        
        meta = clip_probability(self.meta_awareness_ema)
        self.blanket_l2l3.update_sensory_states({
            'meta_awareness': meta
        })
        
        return meta

