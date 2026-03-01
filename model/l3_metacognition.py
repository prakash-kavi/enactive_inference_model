"""Layer 3: Metacognitive monitor for policy selection and meta-awareness.

L3 maintains a learned policy prior (ell_pi(s)) as a habit-like
Dirichlet pseudo-count updated by inferred state belief, selects the
policy posterior q(pi), and exposes meta-awareness as policy--prior divergence.
"""

import numpy as np
import torch.nn as nn
from typing import Optional, Dict, List

from utils.config import EPS, STATES
from .markov_blankets import MarkovBlanketL2L3
from utils.math_utils import clip_probability, normalize_scores, policy_posterior, softmax

class Layer3Monitor(nn.Module):
    """Metacognitive monitor: policy selection + meta-awareness (L3 -> L2)."""

    def __init__(
        self,
        blanket_l2l3: Optional[MarkovBlanketL2L3] = None,
        bptt_steps: int = 25,
    ):
        super().__init__()
        self.blanket_l2l3 = blanket_l2l3 or MarkovBlanketL2L3()
        self.meta_awareness_ema = None
        self.q_pi_ema = None
        self.state_belief_ema = None
        self._learned_alpha: dict = {}
        self.bptt_steps = max(1, int(bptt_steps))
        self.policy_lr = 1.0 / self.bptt_steps
        self._meta_alpha = 1.0 / self.bptt_steps

    def reset(self, preserve_habit: bool = False) -> None:
        """Reset monitor state for a new run.

        Set preserve_habit=True to keep learned habit priors across runs.
        """
        self.meta_awareness_ema = None
        self.q_pi_ema = None
        self.state_belief_ema = None
        if not preserve_habit:
            self._learned_alpha.clear()
        self.blanket_l2l3.reset()

    def smooth_state_belief(self, state_belief: Optional[dict]) -> Optional[dict]:
        """EMA-smooth state belief for stable GNW gating."""
        if not state_belief:
            return state_belief
        weights = self._normalize_belief(state_belief)
        if self.state_belief_ema is None:
            self.state_belief_ema = weights
        else:
            alpha = self._meta_alpha
            self.state_belief_ema = (1.0 - alpha) * self.state_belief_ema + alpha * weights
            self.state_belief_ema = np.clip(self.state_belief_ema, EPS, 1.0)
            self.state_belief_ema = self.state_belief_ema / self.state_belief_ema.sum()
        return {s: float(self.state_belief_ema[i]) for i, s in enumerate(STATES)}

    def _normalize_belief(self, state_belief: Optional[dict]) -> np.ndarray:
        """Return normalized belief weights over STATES (uniform if missing)."""
        if not state_belief:
            return np.full(len(STATES), 1.0 / max(len(STATES), 1), dtype=np.float64)
        weights = np.array([float(state_belief.get(s, 0.0)) for s in STATES], dtype=np.float64)
        total = float(np.sum(weights))
        if total <= EPS:
            return np.full(len(STATES), 1.0 / max(len(STATES), 1), dtype=np.float64)
        return weights / total

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
        weights = self._normalize_belief(state_belief)
        log_adj = np.zeros(4, dtype=np.float64)
        for weight, state in zip(weights, STATES):
            if weight <= 0.0:
                continue
            log_adj += weight * self._get_prior_for_state(state)
        scale = float(np.clip(scale, 0.0, 1.0))
        return scale * log_adj

    def update_policy_state(self, state_belief: Optional[dict], q_pi: np.ndarray) -> None:
        """Update learned habit prior from inferred state belief (Dirichlet-like EMA).

        This is treated as an M-step update using E-step sufficient statistics.
        """
        q = np.array(q_pi, dtype=np.float64)
        if q.size != 4:
            return
        q = np.clip(q, 1e-6, 1.0)
        q = q / q.sum()
        weights = self._normalize_belief(state_belief)
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
        # Evidence reliability: coefficient of variation (scale-free)
        mean_abs = float(np.mean(np.abs(g_raw))) if g_raw.size else 0.0
        std_raw = float(np.std(g_raw)) if g_raw.size else 0.0
        cv = std_raw / (mean_abs + EPS)
        evidence_gain = cv / (cv + 1.0)  # in (0, 1), lower when evidence is flat
        gamma_eff = max(EPS, float(meta_precision) * evidence_gain)
        q_pi = policy_posterior(log_prior, g_array, gamma_eff)
        return q_pi

    def smooth_policy(self, q_pi: np.ndarray) -> np.ndarray:
        """EMA-smooth policy posterior (GNW-style broadcast stability)."""
        q = np.array(q_pi, dtype=np.float64)
        if q.size == 0:
            return q
        q = np.clip(q, EPS, 1.0)
        q = q / q.sum()
        if self.q_pi_ema is None:
            self.q_pi_ema = q
        else:
            alpha = self._meta_alpha
            self.q_pi_ema = (1.0 - alpha) * self.q_pi_ema + alpha * q
            self.q_pi_ema = np.clip(self.q_pi_ema, EPS, 1.0)
            self.q_pi_ema = self.q_pi_ema / self.q_pi_ema.sum()
        return self.q_pi_ema

    def update_meta_awareness_from_conflict(
        self,
        g_vals: List[float],
        state_belief: Optional[dict] = None,
        gate_belief: Optional[dict] = None,
    ) -> float:
        """Second-order belief: policy--prior divergence gated by detection state belief."""
        gate = 0.0
        belief_for_gate = gate_belief if gate_belief is not None else state_belief
        if belief_for_gate:
            q_ma = float(belief_for_gate.get('meta_awareness', 0.0))
            q_ra = float(belief_for_gate.get('redirect_attention', 0.0))
            gate = clip_probability(q_ma + q_ra)

        if not g_vals:
            raw = 0.0
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
            raw = max(raw_conflict * gate, 0.05)

        raw = clip_probability(raw)
        if self.meta_awareness_ema is None:
            self.meta_awareness_ema = raw
        else:
            alpha = self._meta_alpha
            self.meta_awareness_ema = (1.0 - alpha) * self.meta_awareness_ema + alpha * raw
        meta = clip_probability(self.meta_awareness_ema)
        return meta

