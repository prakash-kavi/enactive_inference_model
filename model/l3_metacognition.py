"""Layer 3: Metacognitive monitor for policy selection and meta-awareness.

L3 maintains a learned policy prior (ell_pi(s)) as a habit-like
Dirichlet pseudo-count updated by inferred state belief, selects the
policy posterior q(pi), and exposes meta-awareness as policy--prior divergence.
"""

import numpy as np
import torch.nn as nn
from typing import Optional, Dict, List

from utils.config import (
    DEFAULT_DT,
    EPS,
    STATES,
    L3_META_TAU,
    get_l3_policy_lr,
    get_l3_policy_strength,
)
from .markov_blankets import MarkovBlanketL2L3
from utils.math_utils import clip_probability, normalize_scores, policy_posterior, softmax

class Layer3Monitor(nn.Module):
    """Metacognitive monitor: policy selection + meta-awareness (L3 -> L2)."""

    def __init__(self, blanket_l2l3: Optional[MarkovBlanketL2L3] = None,
                 experience_level: str = "expert"):
        super().__init__()
        self.blanket_l2l3 = blanket_l2l3 or MarkovBlanketL2L3()
        self.meta_awareness_ema = None
        self._learned_alpha: dict = {}
        self.policy_lr = get_l3_policy_lr(experience_level)
        self.policy_strength = get_l3_policy_strength(experience_level)
        if L3_META_TAU > 0:
            self._meta_alpha = float(np.clip(DEFAULT_DT / L3_META_TAU, 0.0, 1.0))
        else:
            self._meta_alpha = 1.0

    def reset(self) -> None:
        """Reset monitor state for an isolated training run."""
        self.meta_awareness_ema = None
        self._learned_alpha.clear()
        self.blanket_l2l3.reset()

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

    def _habit_log_prior(self, state_belief: Optional[dict]) -> np.ndarray:
        """Return habit log-prior adjustment weighted by state belief."""
        weights = self._normalize_belief(state_belief)
        log_adj = np.zeros(4, dtype=np.float64)
        for weight, state in zip(weights, STATES):
            if weight <= 0.0:
                continue
            log_adj += weight * self._get_prior_for_state(state)
        return self.policy_strength * log_adj

    def update_policy_state(self, state_belief: Optional[dict], q_pi: np.ndarray) -> None:
        """Update learned habit prior from inferred state belief (Dirichlet-like EMA)."""
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
        log_prior = log_prior + self._habit_log_prior(state_belief)
        g_array = normalize_scores(np.array(g_vals, dtype=float), EPS)
        gamma_eff = max(EPS, float(meta_precision))
        q_pi = policy_posterior(log_prior, g_array, gamma_eff)
        return q_pi

    def update_meta_awareness_from_conflict(
        self,
        g_vals: List[float],
        state_belief: Optional[dict] = None,
    ) -> float:
        """Second-order belief: policy--prior divergence between evidence and habit."""
        if not g_vals:
            raw = 0.0
        else:
            g_array = normalize_scores(np.array(g_vals, dtype=float), EPS)
            log_habit = self._habit_log_prior(state_belief)
            q_evid = policy_posterior(np.zeros_like(log_habit), g_array, 1.0)
            q_habit = softmax(log_habit)
            q_evid = np.clip(q_evid, EPS, 1.0)
            q_habit = np.clip(q_habit, EPS, 1.0)
            q_evid = q_evid / q_evid.sum()
            q_habit = q_habit / q_habit.sum()
            kl = float(np.sum(q_evid * np.log(q_evid / q_habit)))
            raw = float(1.0 - np.exp(-kl))

        raw = clip_probability(raw)
        if self.meta_awareness_ema is None:
            self.meta_awareness_ema = raw
        else:
            alpha = self._meta_alpha
            self.meta_awareness_ema = (1.0 - alpha) * self.meta_awareness_ema + alpha * raw
        meta = clip_probability(self.meta_awareness_ema)
        return meta

