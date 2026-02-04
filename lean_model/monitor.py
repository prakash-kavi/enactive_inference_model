"""Layer 3: Metacognitive monitor with EFE-based policy evaluation.

Implements planning/deliberation via Expected Free Energy:
- Meta-awareness: Tracks attentional quality from L2 thoughtseeds
- Policy evaluation: Risk + ambiguity decomposition
- Precision modulation: Top-down attentional control signal
- Transition pressure: Policy-driven state change signal
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional

from .config import THOUGHTSEEDS, compute_meta_awareness, get_exit_transition_probs
from .blankets import MarkovBlanketL2L3
from .utils import clip_probability, to_float

class Layer3Monitor(nn.Module):
    """Metacognitive monitor: EFE-based policy selection."""
    
    def __init__(self, experience_level: str = 'expert',
                 blanket_l2l3: Optional[MarkovBlanketL2L3] = None,
                 params: Optional[Dict] = None):
        super().__init__()
        
        self.level = experience_level
        self.blanket_l2l3 = blanket_l2l3 or MarkovBlanketL2L3(smoothing=0.7)
        
        # Parameters (from config)
        if params is None:
            from .config import get_params
            params = get_params(experience_level)
        
        self.efe_ambiguity_weight = params['efe_ambiguity_weight']
        self.efe_cycle_strength = params['efe_cycle_strength']
        self.policy_temp_by_state = params['policy_temperature_by_state']
        self.precision_range = params['l3tol2_precision_range']
        
        # EMA tracking
        self.vfe_ema = 0.0
        self.efe_ema = 0.0
        self.meta_awareness_ema = None
        self.prev_transition_drive = 0.0
    
    def update_meta_awareness(self, current_state: str, z: torch.Tensor) -> float:
        """Compute meta-awareness from L2 thoughtseed activations."""
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
    
    def evaluate_policies(self, current_state: str, vfe: float) -> Dict:
        """Evaluate policies via EFE and output control signals.
        
        Args:
            current_state: Current meditation state
            vfe: Variational free energy from L2
        
        Returns:
            dict with precision_modulation, transition_pressure, policy info
        """
        # Update VFE EMA (sigmoid-transformed)
        vfe_sig = 1.0 / (1.0 + np.exp(-vfe))
        self.vfe_ema = 0.9 * self.vfe_ema + 0.1 * vfe_sig
        
        # Meta-awareness from L2 (via Markov blanket)
        meta = self.blanket_l2l3.sensory_states.get('meta_awareness', 0.5)
        meta = to_float(meta)
        
        # Precision modulation: Maps meta-awareness → attentional precision
        # High meta-awareness → high precision (tight prior adherence)
        transparency = meta  # (1 - opacity)
        prec_min, prec_max = self.precision_range
        precision = prec_min + (prec_max - prec_min) * transparency
        precision = clip_probability(precision)
        
        # Policy evaluation via EFE
        efe_risk, efe_ambiguity = self._compute_efe(current_state)
        efe_total = efe_risk + self.efe_ambiguity_weight * efe_ambiguity
        
        # Update EFE EMA
        self.efe_ema = 0.9 * self.efe_ema + 0.1 * efe_total
        
        # Transition pressure: Combine meta-awareness + VFE + EFE
        base_drive = 0.5 * (transparency + self.vfe_ema)
        efe_drive = 0.35 * self.efe_ema if self.efe_ema > 0.0 else 0.0
        
        # Apply state-specific temperature to transition pressure
        raw_pressure = base_drive + efe_drive
        temp = self.policy_temp_by_state.get(current_state, 1.0)
        transition_pressure = clip_probability(raw_pressure / temp)
        
        # Store for next iteration
        self.prev_transition_drive = transition_pressure
        
        # Send active states to L2 via Markov blanket
        self.blanket_l2l3.update_active_states({
            'precision_modulation': precision,
            'transition_drive': transition_pressure
        })
        
        return {
            'precision_modulation': precision,
            'transition_pressure': transition_pressure,
            'efe_risk': efe_risk,
            'efe_ambiguity': efe_ambiguity,
            'efe_total': efe_total,
        }
    
    def _compute_efe(self, current_state: str) -> tuple[float, float]:
        """Compute Expected Free Energy: risk + ambiguity.
        
        Args:
            current_state: Current meditation state
        
        Returns:
            (risk, ambiguity) floats
        """
        # Get base transition probabilities
        base_probs = get_exit_transition_probs(self.level, current_state)
        
        # Cycle preferences (meditation goal: breath_focus)
        cycle_prefs = {
            'breath_focus': 0.8,
            'mind_wandering': 0.1,
            'meta_awareness': 0.5,
            'redirect_attention': 0.6
        }
        
        # Predicted distribution: blend base + preferences
        drive = self.prev_transition_drive
        blend = drive * self.efe_cycle_strength
        
        predicted_probs = {}
        for state in base_probs:
            base_p = base_probs[state]
            pref_p = cycle_prefs.get(state, 0.5)
            predicted_probs[state] = (1.0 - blend) * base_p + blend * pref_p
        
        # Normalize
        total = sum(predicted_probs.values())
        if total > 0:
            predicted_probs = {s: p / total for s, p in predicted_probs.items()}
        
        # Risk: KL divergence between predicted and preferred
        eps = 1e-8
        risk = 0.0
        for state in predicted_probs:
            p_pred = max(eps, predicted_probs[state])
            p_pref = max(eps, cycle_prefs.get(state, 0.5))
            risk += p_pred * np.log(p_pred / p_pref)
        
        # Ambiguity: Entropy of predicted distribution
        ambiguity = 0.0
        for state in predicted_probs:
            p = max(eps, predicted_probs[state])
            ambiguity -= p * np.log(p)
        
        return float(risk), float(ambiguity)
