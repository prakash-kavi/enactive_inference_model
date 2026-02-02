"""Layer 3: Monitoring and policy layer.

Observes Layer 2 summary signals and emits lightweight control policies.
"""

import torch.nn as nn
import numpy as np

from ..generative_model import PolicyEnergyModel

class Layer3Monitor(nn.Module):
    """Layer 3 monitoring + policy module."""
    
    def __init__(self, experience_level: str = 'novice',
                 efe_ambiguity_weight: float = 0.4,
                 efe_cycle_strength: float = 0.35,
                 efe_gain: float = 0.35,
                 l3tol2_precision_range: tuple = (0.4, 0.6),
                 get_meta_awareness_fn=None, blanket_l2l3=None,
                 vfe_ema_alpha: float = 0.9):
        super().__init__()
        
        self.experience_level = experience_level
        self.efe_ambiguity_weight = efe_ambiguity_weight
        self.efe_cycle_strength = efe_cycle_strength
        self.efe_gain = efe_gain
        self.l3tol2_precision_range = l3tol2_precision_range
        self.get_meta_awareness_fn = get_meta_awareness_fn
        self.blanket_l2l3 = blanket_l2l3
        self.policy_energy_model = PolicyEnergyModel(
            experience_level=self.experience_level,
            ambiguity_weight=self.efe_ambiguity_weight,
            cycle_strength=self.efe_cycle_strength,
        )
        self.vfe_ema_alpha = vfe_ema_alpha
        self.vfe_ema = 0.0
        self.efe_ema = 0.0
        self.efe_value = 0.0
        self.efe_risk = 0.0
        self.efe_ambiguity = 0.0
        self.meta_awareness_ema = None
        self.prev_transition_drive = 0.0

    def update_meta_awareness(self, current_state: str, z) -> float:
        """Update meta-awareness from Layer 2 thoughtseed dynamics."""
        raw = 0.0
        if self.get_meta_awareness_fn:
            raw = float(self.get_meta_awareness_fn(current_state, z))
        raw = float(np.clip(raw, 0.0, 1.0))

        if self.meta_awareness_ema is None:
            self.meta_awareness_ema = raw
        else:
            alpha = float(np.clip(self.vfe_ema_alpha, 0.0, 1.0))
            self.meta_awareness_ema = (alpha * self.meta_awareness_ema) + ((1.0 - alpha) * raw)

        meta = float(np.clip(self.meta_awareness_ema, 0.0, 1.0))
        if self.blanket_l2l3:
            opacity = float(np.clip(1.0 - meta, 0.0, 1.0))
            self.blanket_l2l3.update_sensory_states({
                'meta_awareness': meta,
                'opacity': opacity
            })
        return meta

    def evaluate_policies(self) -> dict:
        """Evaluate L3 policies and return Layer-2-facing prescriptions."""
        sensory = self.blanket_l2l3.sensory_states
        current_state = sensory['current_state']

        vfe_val = sensory.get('vfe', None)
        if vfe_val is not None:
            try:
                vfe_val = float(vfe_val)
            except Exception:
                vfe_val = None
        if vfe_val is not None:
            vfe_sig = 1.0 / (1.0 + np.exp(-vfe_val))
            self.vfe_ema = (self.vfe_ema_alpha * self.vfe_ema) + ((1 - self.vfe_ema_alpha) * vfe_sig)
        
        prescription_l2l3 = {}

        opacity = float(np.clip(sensory.get('opacity', 1.0), 0.0, 1.0))
        prec_min, prec_max = self.l3tol2_precision_range
        transparency = 1.0 - opacity
        precision_drive = transparency
        precision = prec_min + (prec_max - prec_min) * precision_drive
        precision = float(np.clip(precision, 0.0, 1.0))
        prescription_l2l3['precision_modulation'] = precision
        transition_drive = float(np.clip(0.5 * (transparency + self.vfe_ema), 0.0, 1.0))

        # Expected Free Energy uses previous-step drive to avoid same-step feedback.
        efe_terms = self.policy_energy_model.compute_efe_terms(current_state, self.prev_transition_drive)
        self.efe_risk = float(efe_terms.risk)
        self.efe_ambiguity = float(efe_terms.ambiguity)
        self.efe_value = float(efe_terms.total)
        self.efe_ema = (self.vfe_ema_alpha * self.efe_ema) + ((1.0 - self.vfe_ema_alpha) * self.efe_value)

        if self.efe_ema > 0.0:
            transition_drive = float(np.clip(
                transition_drive + (self.efe_gain * self.efe_ema), 0.0, 1.0
            ))

        # Store final drive for next-step EFE
        self.prev_transition_drive = transition_drive
            
        if self.blanket_l2l3:
            self.blanket_l2l3.update_active_states(prescription_l2l3)

        return {
            'precision_modulation': precision,
            'transition_pressure': transition_drive,
        }
