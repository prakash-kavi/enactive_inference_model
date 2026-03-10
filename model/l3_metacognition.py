"""Layer 3: Metacognitive monitor for policy selection and meta-awareness.

L3 selects the policy posterior q(pi) from expected free energy under a dwell prior
and exposes meta-awareness as a conflict signal gated by thoughtseed dynamics.
"""

import numpy as np
import torch.nn as nn
from typing import Optional, List

from utils.config import EPS, DEFAULT_DT, PRECISION_TAU, CLIP_MIN
from .markov_blankets import MarkovBlanketL2L3
from utils.math_utils import (
    clip_probability,
    normalize_scores,
    softmax,
)


class Layer3Monitor(nn.Module):
    """Metacognitive monitor: policy selection + meta-awareness (L3 -> L2)."""

    def __init__(
        self,
        blanket_l2l3: Optional[MarkovBlanketL2L3] = None,
        bptt_steps: int = 25,
    ):
        super().__init__()
        self.blanket_l2l3 = blanket_l2l3 or MarkovBlanketL2L3()
        self.meta_awareness = None
        self.bptt_steps = max(1, int(bptt_steps))

    def reset(self, preserve_habit: bool = False) -> None:
        """Reset monitor state for a new run."""
        self.meta_awareness = None
        self.blanket_l2l3.reset()

    def update_policy_state(self, state_belief: Optional[dict], q_pi: np.ndarray) -> None:
        """No-op: habit prior learning removed in simplified L3."""
        return

    def select_policy(
        self,
        g_vals: List[float],
        priors: List[float],
        state_belief: Optional[dict] = None,
        meta_precision: Optional[float] = None,
    ) -> np.ndarray:
        """Select policy posterior q(pi) from dwell prior and expected free energy."""
        if meta_precision is None:
            raise ValueError("meta_precision must be provided for policy selection.")

        log_prior = np.log(np.clip(np.array(priors, dtype=float), EPS, 1.0))
        g_raw = np.array(g_vals, dtype=float)
        g_array = normalize_scores(g_raw, EPS)

        gamma_eff = clip_probability(meta_precision)
        gamma_eff = max(EPS, gamma_eff)

        logits = log_prior - gamma_eff * g_array
        return softmax(logits)

    def _ou_update_meta(self, target: float) -> float:
        """Deterministic OU-style update for meta-awareness."""
        dt = float(DEFAULT_DT)
        tau = max(float(PRECISION_TAU), dt)
        theta = 1.0 / tau
        target = clip_probability(target)
        if self.meta_awareness is None:
            self.meta_awareness = target
        else:
            drift = -theta * (self.meta_awareness - target)
            self.meta_awareness = self.meta_awareness + drift * dt

        # Enforce baseline ambient minimum to prevent artificial zeros
        self.meta_awareness = max(float(CLIP_MIN), clip_probability(self.meta_awareness))
        return self.meta_awareness

    def update_meta_awareness_from_conflict(
        self,
        g_vals: List[float],
        state_belief: Optional[dict] = None,
        gate_belief: Optional[dict] = None,
    ) -> float:
        """Second-order belief: conflict between evidence and dwell prior, gated by ignition."""
        gate = 0.0
        ts_acts = self.blanket_l2l3.sensory_states.get('thoughtseed_activations', [])

        # Ignition gate from orchestrator thoughtseeds vs distractors
        if len(ts_acts) >= 5:
            aha_moment = float(ts_acts[3])
            equanimity = float(ts_acts[4])
            pain = float(ts_acts[1])
            pending = float(ts_acts[2])
            distractor = max(pain, pending)
            gate = max(0.0, aha_moment + equanimity - distractor)

        gate = clip_probability(gate)

        if not g_vals:
            raw_conflict = 0.0
        else:
            g_array = normalize_scores(np.array(g_vals, dtype=float), EPS)
            q_evid = softmax(-g_array)
            q_evid = np.clip(q_evid, EPS, 1.0)
            q_evid = q_evid / q_evid.sum()

            priors = self.blanket_l2l3.sensory_states.get('policy_priors')
            if priors is None or len(priors) != len(q_evid):
                q_prior = np.full_like(q_evid, 1.0 / max(len(q_evid), 1))
            else:
                q_prior = np.array(priors, dtype=float)
                q_prior = np.clip(q_prior, EPS, 1.0)
                q_prior = q_prior / q_prior.sum()

            kl = float(np.sum(q_evid * np.log(q_evid / q_prior)))
            raw_conflict = float(1.0 - np.exp(-kl))

        ambient = float(CLIP_MIN)
        target = ambient + (1.0 - ambient) * (raw_conflict * gate)

        return self._ou_update_meta(target)
