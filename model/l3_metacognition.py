"""Layer 3: Metacognitive monitor for policy selection and meta-awareness.

L3 maintains a learned policy prior (ell_pi(s)) as a habit-like
Dirichlet pseudo-count updated by inferred state belief, selects the
policy posterior q(pi), and exposes meta-awareness as policy--prior divergence.
"""

import numpy as np
import torch.nn as nn
from typing import Optional, Dict, List

from utils.config import EPS, STATES, DEFAULT_DT, PRECISION_TAU
from .markov_blankets import MarkovBlanketL2L3
from utils.math_utils import (
    clip_probability,
    normalize_scores,
    normalize_belief,
    policy_posterior,
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
        self.clarity_baseline = None
        self._learned_alpha: dict = {}
        self.bptt_steps = max(1, int(bptt_steps))
        self.policy_lr = 1.0 / self.bptt_steps

    def reset(self, preserve_habit: bool = False) -> None:
        """Reset monitor state for a new run.

        Set preserve_habit=True to keep learned habit priors across runs.
        """
        self.meta_awareness = None
        if not preserve_habit:
            self._learned_alpha.clear()
        self.blanket_l2l3.reset()

    def _get_prior_for_state(self, current_state: str) -> np.ndarray:
        """Return log prior adjustment for current state (length 4, neutral if not learned)."""
        alpha = self._learned_alpha.get(current_state)
        if alpha is None:
            alpha = np.ones(4, dtype=np.float64)
        alpha = np.maximum(alpha, 1e-6)
        alpha = alpha / alpha.sum()
        return np.log(alpha)

    def _habit_log_prior(self, state_belief: Optional[dict], scale: float = 1.0) -> np.ndarray:
        """Return habit log-prior adjustment weighted by state belief."""
        weights = normalize_belief(state_belief, keys=STATES, eps=EPS)
        log_adj = np.zeros(4, dtype=np.float64)
        for weight, state in zip(weights, STATES):
            if weight <= 0.0:
                continue
            log_adj += weight * self._get_prior_for_state(state)
        scale = float(np.clip(scale, 0.0, 1.0))
        return scale * log_adj

    def set_clarity_baseline(self, baseline: Optional[float]) -> None:
        """Set a learned clarity baseline (e.g., mean state confidence)."""
        if baseline is None:
            self.clarity_baseline = None
        else:
            self.clarity_baseline = clip_probability(baseline)

    def update_policy_state(self, state_belief: Optional[dict], q_pi: np.ndarray) -> None:
        """Update learned habit prior from inferred state belief (Dirichlet-like EMA).

        This is treated as an M-step update using E-step sufficient statistics.
        """
        q = np.array(q_pi, dtype=np.float64)
        if q.size != 4:
            return
        q = np.clip(q, 1e-6, 1.0)
        q = q / q.sum()
        weights = normalize_belief(state_belief, keys=STATES, eps=EPS)
        for weight, state in zip(weights, STATES):
            if weight <= 0.0:
                continue
            prev = self._learned_alpha.get(state, np.ones(4, dtype=np.float64))
            kappa = float(self.policy_lr) * float(weight)
            self._learned_alpha[state] = (1.0 - kappa) * prev + kappa * q

    def select_policy(
        self,
        g_vals: List[float],
        priors: List[float],
        state_belief: Optional[dict] = None,
        meta_precision: Optional[float] = None,
    ) -> np.ndarray:
        """Select policy posterior q(pi) from evidence and habit priors."""
        if meta_precision is None:
            raise ValueError("meta_precision must be provided for policy selection.")
        log_prior = np.log(np.array(priors, dtype=float))
        habit_scale = float(np.clip(1.0 - float(meta_precision), 0.0, 1.0))
        log_prior = log_prior + self._habit_log_prior(state_belief, scale=habit_scale)
        g_raw = np.array(g_vals, dtype=float)
        g_array = normalize_scores(g_raw, EPS)
        # Evidence precision from a single policy-precision signal.
        gamma_eff = clip_probability(meta_precision)
        gamma_eff = max(EPS, gamma_eff)
        q_pi = policy_posterior(log_prior, g_array, gamma_eff)
        return q_pi

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
        return clip_probability(self.meta_awareness)

    def update_meta_awareness_from_conflict(
        self,
        g_vals: List[float],
        state_belief: Optional[dict] = None,
        gate_belief: Optional[dict] = None,
    ) -> float:
        """Second-order belief: policy--prior divergence gated by detection state belief."""
        base_gate = float(self.clarity_baseline) if self.clarity_baseline is not None else 0.0
        gate = 0.0
        belief_for_gate = gate_belief if gate_belief is not None else state_belief
        if belief_for_gate:
            q_ma = float(belief_for_gate.get('meta_awareness', 0.0))
            q_ra = float(belief_for_gate.get('redirect_attention', 0.0))
            gate = q_ma + q_ra
        gate = clip_probability(gate + base_gate)

        if not g_vals:
            target = base_gate
        else:
            g_array = normalize_scores(np.array(g_vals, dtype=float), EPS)
            log_habit = self._habit_log_prior(state_belief, scale=1.0)
            q_evid = policy_posterior(np.zeros_like(log_habit), g_array, 1.0)
            q_habit = softmax(log_habit)
            q_evid = np.clip(q_evid, EPS, 1.0)
            q_habit = np.clip(q_habit, EPS, 1.0)
            q_evid = q_evid / q_evid.sum()
            q_habit = q_habit / q_habit.sum()
            kl = float(np.sum(q_evid * np.log(q_evid / q_habit)))
            raw_conflict = float(1.0 - np.exp(-kl))
            target = raw_conflict * gate

        return self._ou_update_meta(target)

