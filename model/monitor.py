"""Layer 3: Metacognitive monitor with EFE-based policy evaluation.

Implements planning/deliberation via Expected Free Energy (G):
- Meta-awareness: tracks attentional quality from L2 thoughtseeds
- Policy evaluation: risk + ambiguity decomposition
- Sensory precision: precision from VFE (F) fluctuations
- Policy confidence: transition drive signal
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional

from utils.config import THOUGHTSEEDS, compute_meta_awareness, get_exit_transition_probs, EPS
from .blankets import MarkovBlanketL2L3
from utils.math_utils import clip_probability, ema_update, to_float

class Layer3Monitor(nn.Module):
    """Metacognitive monitor: EFE-based policy inference (G)."""
    
    def __init__(self, experience_level: str = 'expert',
                 blanket_l2l3: Optional[MarkovBlanketL2L3] = None):
        super().__init__()
        
        self.level = experience_level
        self.blanket_l2l3 = blanket_l2l3 or MarkovBlanketL2L3(smoothing=0.7)

        # EMA tracking
        self.vfe_ema = 0.0
        self.vfe_mean = 0.0
        self.vfe_var = 1.0
        self.efe_ema = 0.0
        self.efe_risk_mean = 0.0
        self.efe_risk_var = 1.0
        self.efe_ambiguity_mean = 0.0
        self.efe_ambiguity_var = 1.0
        self.meta_awareness_ema = None
        self.prev_policy_confidence = 0.0

    def reset(self) -> None:
        """Reset monitor state for an isolated training run."""
        self.vfe_ema = 0.0
        self.vfe_mean = 0.0
        self.vfe_var = 1.0
        self.efe_ema = 0.0
        self.efe_risk_mean = 0.0
        self.efe_risk_var = 1.0
        self.efe_ambiguity_mean = 0.0
        self.efe_ambiguity_var = 1.0
        self.meta_awareness_ema = None
        self.prev_policy_confidence = 0.0
        self.blanket_l2l3.reset()
    
    def update_meta_awareness(self, current_state: str, z: torch.Tensor) -> float:
        """Compute meta-awareness (A) from L2 thoughtseed activations."""
        # Convert thoughtseed activations to dict
        z_dict = {ts: z[i].item() for i, ts in enumerate(THOUGHTSEEDS)}
        
        # Compute via weighted sum (state-dependent)
        raw = compute_meta_awareness(current_state, z_dict)
        
        # EMA smoothing
        if self.meta_awareness_ema is None:
            self.meta_awareness_ema = raw
        else:
            self.meta_awareness_ema = 0.9 * self.meta_awareness_ema + 0.1 * raw
        
        meta = clip_probability(self.meta_awareness_ema)
        
        # Send to L2 via Markov blanket
        opacity = 1.0 - meta  # Low meta-awareness = high opacity
        self.blanket_l2l3.update_sensory_states({
            'meta_awareness': meta,
            'opacity': opacity
        })
        
        return meta
    
    def infer_policy_posterior(self, current_state: str, vfe: float) -> Dict:
        """Infer policy posterior via expected free energy (G).
        
        Args:
            current_state: Current meditation state
            vfe: Variational free energy F from L2
        
        Returns:
            dict with precision_sensory (precision surrogate), policy_confidence, policy info
        """
        # Update VFE EMA (sigmoid-transformed for transition drive)
        vfe_sig = 1.0 / (1.0 + np.exp(-vfe))
        self.vfe_ema = 0.9 * self.vfe_ema + 0.1 * vfe_sig
        self.vfe_mean, self.vfe_var = ema_update(vfe, self.vfe_mean, self.vfe_var)

        # Meta-awareness from L2 (via Markov blanket)
        meta = self.blanket_l2l3.sensory_states.get('meta_awareness', 0.5)
        meta = to_float(meta)

        # Precision modulation: inverse of prediction error (self-organizing)
        # Low VFE -> high precision, high VFE -> low precision
        vfe_norm = self._zscore(vfe, self.vfe_mean, self.vfe_var)
        precision_sensory = clip_probability(1.0 / (1.0 + np.exp(vfe_norm)))
        transparency = meta  # (1 - opacity)

        # Policy evaluation via EFE (G)
        efe_risk, efe_ambiguity = self._compute_efe(current_state)
        self.efe_risk_mean, self.efe_risk_var = ema_update(
            efe_risk, self.efe_risk_mean, self.efe_risk_var
        )
        self.efe_ambiguity_mean, self.efe_ambiguity_var = ema_update(
            efe_ambiguity, self.efe_ambiguity_mean, self.efe_ambiguity_var
        )
        risk_norm = self._zscore(efe_risk, self.efe_risk_mean, self.efe_risk_var)
        ambiguity_norm = self._zscore(
            efe_ambiguity, self.efe_ambiguity_mean, self.efe_ambiguity_var
        )
        efe_total = 0.5 * (risk_norm + ambiguity_norm)

        # Update EFE EMA
        self.efe_ema = 0.9 * self.efe_ema + 0.1 * efe_total

        # Policy confidence: combine meta-awareness + VFE + EFE (heuristic)
        base_drive = 0.5 * (transparency + self.vfe_ema)
        efe_drive = 1.0 / (1.0 + np.exp(-self.efe_ema))
        policy_confidence = clip_probability(0.5 * (base_drive + efe_drive))
        
        # Store for next iteration
        self.prev_policy_confidence = policy_confidence
        
        # Send active states to L2 via Markov blanket
        self.blanket_l2l3.update_active_states({
            'precision_sensory': precision_sensory,
            'policy_confidence': policy_confidence
        })
        
        return {
            'precision_sensory': precision_sensory,
            'policy_confidence': policy_confidence,
            'efe_risk': efe_risk,
            'efe_ambiguity': efe_ambiguity,
            'efe_total': efe_total,
        }

    def _zscore(self, value: float, mean: float, var: float) -> float:
        """Normalize using EMA mean/variance."""
        return (value - mean) / np.sqrt(var + EPS)

    def _compute_efe(self, current_state: str) -> tuple[float, float]:
        """Compute Expected Free Energy: risk + ambiguity."""
        # E: policy prior
        policy_prior_E = get_exit_transition_probs(self.level, current_state)

        # C: preferences
        preferences_C = {
            'breath_focus': 0.8,
            'mind_wandering': 0.1,
            'meta_awareness': 0.5,
            'redirect_attention': 0.6
        }

        drive = clip_probability(self.prev_policy_confidence)
        blend = drive

        predicted_probs = {}
        for state in policy_prior_E:
            base_p = policy_prior_E[state]
            pref_p = preferences_C.get(state, 0.5)
            predicted_probs[state] = (1.0 - blend) * base_p + blend * pref_p

        total = sum(predicted_probs.values())
        if total > 0:
            predicted_probs = {s: p / total for s, p in predicted_probs.items()}

        risk = 0.0
        for state in predicted_probs:
            p_pred = max(EPS, predicted_probs[state])
            p_pref = max(EPS, preferences_C.get(state, 0.5))
            risk += p_pred * np.log(p_pred / p_pref)

        ambiguity = 0.0
        for state in predicted_probs:
            p = max(EPS, predicted_probs[state])
            ambiguity -= p * np.log(p)

        return float(risk), float(ambiguity)
    
