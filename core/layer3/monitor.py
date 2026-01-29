"""Layer 3: Monitoring and policy layer.

Observes Layer 2 summary signals and emits lightweight control policies.
"""

import torch
import torch.nn as nn
import numpy as np

class Layer3Monitor(nn.Module):
    """Layer 3 monitoring + policy module."""
    
    def __init__(self, thoughtseeds: list, 
                 sensory_precision_base: float, prior_precision_base: float,
                 precision_weight: float, complexity_penalty: float,
                 get_meta_awareness_fn=None, blanket_l2l3=None,
                 vfe_ema_alpha: float = 0.9):
        super().__init__()
        
        self.thoughtseeds = thoughtseeds
        self.sensory_precision_base = sensory_precision_base
        self.prior_precision_base = prior_precision_base
        self.precision_weight = precision_weight
        self.complexity_penalty = complexity_penalty
        self.get_meta_awareness_fn = get_meta_awareness_fn
        self.blanket_l2l3 = blanket_l2l3
        self.vfe_ema_alpha = vfe_ema_alpha
        self.vfe_ema = 0.0
    
    def compute_meta_metrics(self) -> dict:
        """Monitor internal state metrics (logging only)."""
        if not self.blanket_l2l3 or not self.blanket_l2l3.sensory_states:
            return {}
            
        return {
            "meta_awareness": self.blanket_l2l3.sensory_states.get('meta_awareness', 0.0),
            "dominant_thoughtseed": self.blanket_l2l3.sensory_states.get('dominant_thoughtseed')
        }

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
            'fatigue_buffer': 1.0
        }
        
        prescription_l2l3 = {
            'precision_modulation': 1.0
        }
        
        aha_idx = self.thoughtseeds.index('aha_moment')
        aha_accum = sensory.get('aha_accumulator_value', 0.0)
        dom_ts = sensory.get('dominant_thoughtseed', None)
        dom_act = sensory.get('dominant_activation', 0.0)
        try:
            dom_act = float(dom_act)
        except Exception:
            dom_act = 0.0
        
        recognition = sensory.get('recognition_signal', None)
        if recognition is None:
            recognition = max(z_vals[aha_idx], 0.0) * 0.7 + max(float(aha_accum), 0.0) * 0.3
        if dom_ts == 'aha_moment':
            recognition = max(recognition, float(dom_act))

        if recognition > 0.7:
            prescription_l1l2['noise_reduction'] = 0.5
            prescription_l2l3['precision_modulation'] = 1.5

        if self.vfe_ema > 0.6:
            prescription_l1l2['noise_reduction'] = min(prescription_l1l2['noise_reduction'], 0.8)
            prescription_l2l3['precision_modulation'] = max(prescription_l2l3['precision_modulation'], 1.1)
             
        # Policy 2: Attentional sharpening
        ma = sensory.get('meta_awareness', None)
        if ma is None and self.get_meta_awareness_fn:
            ma = self.get_meta_awareness_fn(current_state, z) # Returns float
        if ma is not None:
            ma = float(ma)
            ma = float(np.clip(ma, 0.0, 1.0))
            ma_precision = 0.8 + 1.0 * ma
            prescription_l2l3['precision_modulation'] = max(
                prescription_l2l3['precision_modulation'],
                ma_precision
            )
            if ma > 0.6:
                prescription_l1l2['noise_reduction'] = min(prescription_l1l2['noise_reduction'], 0.6)

        eq_idx = self.thoughtseeds.index('equanimity')
        if z_vals[eq_idx] > 0.6:
            prescription_l1l2['fatigue_buffer'] = 0.5
            
        if current_state == 'meta_awareness':
            prescription_l1l2['noise_reduction'] = min(prescription_l1l2['noise_reduction'], 0.4)
            prescription_l2l3['precision_modulation'] = max(prescription_l2l3['precision_modulation'], 1.3)

        if dom_ts in ('aha_moment', 'equanimity'):
            precision_boost = 1.1 + 0.2 * float(np.clip(dom_act, 0.0, 1.0))
            prescription_l2l3['precision_modulation'] = max(prescription_l2l3['precision_modulation'], precision_boost)

        recognition_drive = float(np.clip(recognition, 0.0, 1.0))
        ma_drive = float(np.clip(ma, 0.0, 1.0)) if ma is not None else 0.0
        transition_drive = (0.4 * ma_drive) + (0.3 * self.vfe_ema) + (0.3 * recognition_drive)
        if dom_ts in ('pending_tasks', 'pain_discomfort'):
            transition_drive = min(1.0, transition_drive + 0.15 * float(np.clip(dom_act, 0.0, 1.0)))
        elif dom_ts == 'attend_breath':
            transition_drive = max(0.0, transition_drive - 0.15 * float(np.clip(dom_act, 0.0, 1.0)))
        prescription_l1l2['transition_drive'] = float(np.clip(transition_drive, 0.0, 1.0))
            
        if self.blanket_l2l3:
            self.blanket_l2l3.update_active_states(prescription_l2l3)

        return prescription_l1l2
