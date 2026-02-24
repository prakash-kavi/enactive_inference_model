"""Layer 3: Metacognitive monitor for meta-awareness and learned policy prior.

L3 maintains a learned policy prior (paper: ell_pi(s)) as a habit-like
Dirichlet pseudo-count updated by experience, and writes it to the L2<->L3 blanket.
Meta-awareness m_t is computed from L2 thoughtseeds and modulates precision in L2.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Union

from utils.config import (
    THOUGHTSEEDS,
    compute_meta_awareness,
    L3_POLICY_LR,
    L3_POLICY_STRENGTH,
    L3_META_EMA_ALPHA,
)
from .markov_blankets import MarkovBlanketL2L3
from utils.math_utils import clip_probability

class Layer3Monitor(nn.Module):
    """Metacognitive monitor: meta-awareness and learned policy prior (L3 -> L2)."""

    def __init__(self, blanket_l2l3: Optional[MarkovBlanketL2L3] = None):
        super().__init__()
        self.blanket_l2l3 = blanket_l2l3 or MarkovBlanketL2L3(smoothing=0.0)
        self.meta_awareness_ema = None
        self._learned_alpha: dict = {}

    def reset(self) -> None:
        """Reset monitor state for an isolated training run."""
        self.meta_awareness_ema = None
        self._learned_alpha.clear()
        self.blanket_l2l3.reset()

    def _get_l2_sensory(self) -> tuple[str, np.ndarray]:
        """Read current_state and thoughtseed activations from L2->L3 blanket (paper's interface)."""
        state = self.blanket_l2l3.sensory_states.get('current_state', 'breath_focus')
        ts_dict = self.blanket_l2l3.sensory_states.get('thoughtseed_activations') or {}
        z = np.array([float(ts_dict.get(ts, 0.0)) for ts in THOUGHTSEEDS], dtype=np.float64)
        return state, z

    def _get_prior_for_state(self, current_state: str) -> np.ndarray:
        """Return log prior adjustment for current state (length 4, neutral if not learned)."""
        alpha = self._learned_alpha.get(current_state)
        if alpha is None:
            alpha = np.ones(4, dtype=np.float64)
        alpha = np.maximum(alpha, 1e-6)
        alpha = alpha / alpha.sum()
        return np.log(alpha)

    def write_policy_prior(self, current_state: str) -> None:
        """Write learned policy prior ell_pi(s) to L2<->L3 blanket for L2 to use in infer_pi."""
        log_adj = self._get_prior_for_state(current_state)
        scaled = L3_POLICY_STRENGTH * log_adj
        self.blanket_l2l3.update_active_states({
            'policy_prior': list(scaled),
        })

    def update_policy_state(self, current_state: str, q_pi: np.ndarray) -> None:
        """Update learned habit prior from experience (Dirichlet-like EMA on pseudo-counts).
        Uses L3_POLICY_LR from first update; prior is initialized to uniform counts.
        """
        q = np.array(q_pi, dtype=np.float64)
        if q.size != 4:
            return
        q = np.clip(q, 1e-6, 1.0)
        q = q / q.sum()
        prev = self._learned_alpha.get(
            current_state, np.ones(4, dtype=np.float64)
        )
        self._learned_alpha[current_state] = (
            (1.0 - L3_POLICY_LR) * prev + L3_POLICY_LR * q
        )

    def update_meta_awareness(
        self,
        current_state: Optional[str] = None,
        z: Optional[torch.Tensor] = None,
    ) -> float:
        """Compute meta-awareness from L2 thoughtseed activations.
        If current_state/z are None, reads from L2->L3 blanket sensory (paper's interface)."""
        if current_state is None or z is None:
            current_state, z_np = self._get_l2_sensory()
            z_dict = {ts: float(z_np[i]) for i, ts in enumerate(THOUGHTSEEDS)}
        else:
            z_dict = {ts: z[i].item() for i, ts in enumerate(THOUGHTSEEDS)}
        raw = compute_meta_awareness(current_state, z_dict)
        if self.meta_awareness_ema is None:
            self.meta_awareness_ema = raw
        else:
            alpha = float(L3_META_EMA_ALPHA)
            self.meta_awareness_ema = (1.0 - alpha) * self.meta_awareness_ema + alpha * raw
        meta = clip_probability(self.meta_awareness_ema)
        return meta

