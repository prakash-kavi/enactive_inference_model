"""Layer 3: Monitoring and policy layer.

Observes Layer 2 summary signals and emits lightweight control policies.
"""

import torch
import torch.nn as nn
import numpy as np

from utils.meditation_config import STATES, NETWORK_PROFILES
from utils.meditation_utils import get_exit_transition_probs, get_preferred_transition_probs

class Layer3Monitor(nn.Module):
    """Layer 3 monitoring + policy module."""
    
    def __init__(self, thoughtseeds: list,
                 experience_level: str = 'novice',
                 efe_ambiguity_weight: float = 0.4,
                 l3tol2_precision_range: tuple = (0.4, 0.6),
                 get_meta_awareness_fn=None, blanket_l2l3=None,
                 vfe_ema_alpha: float = 0.9):
        super().__init__()
        
        self.thoughtseeds = thoughtseeds
        self.experience_level = experience_level
        self.efe_ambiguity_weight = efe_ambiguity_weight
        self.l3tol2_precision_range = l3tol2_precision_range
        self.get_meta_awareness_fn = get_meta_awareness_fn
        self.blanket_l2l3 = blanket_l2l3
        self.vfe_ema_alpha = vfe_ema_alpha
        self.vfe_ema = 0.0
        self.efe_ema = 0.0
        self.efe_value = 0.0
        self.meta_awareness_ema = None
        self.prev_transition_drive = 0.0
    
    def _compute_efe(self, current_state: str, transition_drive: float) -> float:
        base = get_exit_transition_probs(self.experience_level, current_state)
        if not base:
            return 0.0
        drive = float(np.clip(transition_drive, 0.0, 1.0))
        pref = get_preferred_transition_probs(self.experience_level, current_state)
        if not pref:
            return 0.0
        pred = dict(base)
        if drive > 0.0:
            for s in pred.keys():
                pred[s] = (1.0 - drive) * base.get(s, 0.0) + drive * pref.get(s, 0.0)

        eps = 1e-8
        risk = 0.0
        for s in STATES:
            p_s = max(eps, float(pred.get(s, 0.0)))
            q_s = max(eps, float(pref.get(s, 0.0)))
            risk += p_s * (np.log(p_s) - np.log(q_s))

        ambiguity = 0.0
        for s, p_s in pred.items():
            profile = NETWORK_PROFILES.get(s, {}).get(self.experience_level, {})
            if not profile:
                continue
            for val in profile.values():
                p = float(np.clip(val, eps, 1.0 - eps))
                ambiguity += float(p_s) * (-(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)))

        return risk + (self.efe_ambiguity_weight * ambiguity)

    def compute_meta_awareness(self, current_state: str, z) -> float:
        """Compute meta-awareness using L3 signals with z as a calibrator."""
        base = 0.0
        if self.get_meta_awareness_fn:
            base = float(self.get_meta_awareness_fn(current_state, z))

        sensory = self.blanket_l2l3.sensory_states if self.blanket_l2l3 else {}
        recognition = float(np.clip(sensory.get('recognition_signal', 0.0), 0.0, 1.0))
        vfe_drive = float(np.clip(self.vfe_ema, 0.0, 1.0))
        raw = (0.7 * base) + (0.2 * recognition) + (0.1 * vfe_drive)
        raw = float(np.clip(raw, 0.0, 1.0))

        if self.meta_awareness_ema is None:
            self.meta_awareness_ema = raw
        else:
            alpha = float(np.clip(self.vfe_ema_alpha, 0.0, 1.0))
            self.meta_awareness_ema = (alpha * self.meta_awareness_ema) + ((1.0 - alpha) * raw)

        meta = float(np.clip(self.meta_awareness_ema, 0.0, 1.0))
        if self.blanket_l2l3:
            self.blanket_l2l3.update_sensory_states({'meta_awareness': meta})
        return meta

    def evaluate_policies(self) -> dict:
        """Evaluate policies and return prescriptions (non-differentiable)."""
        sensory = self.blanket_l2l3.sensory_states
        z = sensory['thoughtseed_activations'] # Tensor
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
        
        z_vals = z.detach().cpu().numpy() if isinstance(z, torch.Tensor) else np.array(z)
        
        prescription_l1l2 = {
            'noise_reduction': 1.0,
        }
        
        prescription_l2l3 = {
            'precision_modulation': 0.5
        }

        aha_idx = self.thoughtseeds.index('aha_moment')
        aha_accum = sensory.get('aha_accumulator_value', 0.0)
        recognition = sensory.get('recognition_signal', None)
        if recognition is None:
            recognition = max(z_vals[aha_idx], 0.0) * 0.7 + max(float(aha_accum), 0.0) * 0.3
        recognition_drive = float(np.clip(recognition, 0.0, 1.0))

        ma = float(np.clip(sensory.get('meta_awareness', 0.0), 0.0, 1.0))
        prec_min, prec_max = self.l3tol2_precision_range
        precision_drive = 0.5 * (ma + recognition_drive)
        precision = prec_min + (prec_max - prec_min) * precision_drive
        precision = float(np.clip(precision, 0.0, 1.0))
        if current_state in ('meta_awareness', 'redirect_breath'):
            precision = float(np.clip(precision + 0.1, 0.0, 1.0))
        prescription_l2l3['precision_modulation'] = precision

        noise_reduction = 1.0 - (0.6 * precision)
        prescription_l1l2['noise_reduction'] = float(np.clip(noise_reduction, 0.4, 1.0))

        transition_drive = (0.4 * ma) + (0.2 * self.vfe_ema) + (0.3 * recognition_drive)
        if current_state == 'mind_wandering' and recognition_drive > 0.6:
            transition_drive = float(np.clip(transition_drive + 0.2, 0.0, 1.0))
        if current_state in ('meta_awareness', 'redirect_breath'):
            progress = float(np.clip(sensory.get('dwell_progress', 0.0), 0.0, 1.0))
            transition_drive = float(np.clip(transition_drive + (0.2 * progress), 0.0, 1.0))
        prescription_l1l2['transition_drive'] = float(np.clip(transition_drive, 0.0, 1.0))

        # L2 -> L1 enactive bias tied to precision
        prescription_l1l2['l2tol1_enactive_bias'] = float(np.clip(precision, 0.0, 1.0))

        # Expected Free Energy uses previous-step drive to avoid same-step feedback.
        self.efe_value = float(self._compute_efe(current_state, self.prev_transition_drive))
        self.efe_ema = (self.vfe_ema_alpha * self.efe_ema) + ((1.0 - self.vfe_ema_alpha) * self.efe_value)

        if self.efe_ema > 0.0:
            efe_gain = 0.35
            transition_drive = float(np.clip(
                transition_drive + (efe_gain * self.efe_ema), 0.0, 1.0
            ))
            prescription_l1l2['transition_drive'] = transition_drive

        # Store final drive for next-step EFE
        self.prev_transition_drive = transition_drive
            
        if self.blanket_l2l3:
            self.blanket_l2l3.update_active_states(prescription_l2l3)

        return prescription_l1l2
