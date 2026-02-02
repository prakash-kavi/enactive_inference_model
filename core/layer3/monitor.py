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
                 policy_horizon: int = 1,
                 policy_temperature: float = 1.0,
                 policy_temperature_by_state: dict = None,
                 policy_horizon_discount: float = 0.6,
                 l3tol2_precision_range: tuple = (0.4, 0.6),
                 get_meta_awareness_fn=None, blanket_l2l3=None,
                 vfe_ema_alpha: float = 0.9):
        super().__init__()
        
        self.experience_level = experience_level
        self.efe_ambiguity_weight = efe_ambiguity_weight
        self.efe_cycle_strength = efe_cycle_strength
        self.efe_gain = efe_gain
        self.policy_horizon = max(1, int(policy_horizon))
        self.policy_temperature = max(1e-6, float(policy_temperature))
        
        # Validate temperature config
        if policy_temperature_by_state is None:
            raise ValueError("policy_temperature_by_state required")
        from utils.meditation_config import STATES
        missing = [s for s in STATES if s not in policy_temperature_by_state]
        if missing:
            raise ValueError(f"policy_temperature_by_state missing states: {missing}")
        self.policy_temperature_by_state = dict(policy_temperature_by_state)
        
        self.policy_horizon_discount = float(np.clip(policy_horizon_discount, 0.0, 1.0))
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
        self.policy_posterior = {}
        self.policy_efe = {}
        self.selected_policy = ""
        self.policy_confidence = 0.0
        self.policy_entropy = 0.0
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
        opacity = float(np.clip(sensory['opacity'], 0.0, 1.0))

        vfe_val = sensory.get('vfe', None)
        if vfe_val is not None:
            try:
                vfe_val = float(vfe_val)
                vfe_sig = 1.0 / (1.0 + np.exp(-vfe_val))
                self.vfe_ema = (self.vfe_ema_alpha * self.vfe_ema) + ((1 - self.vfe_ema_alpha) * vfe_sig)
            except (TypeError, ValueError) as e:
                raise ValueError(f"Invalid VFE: {e}")
        
        prescription_l2l3 = {}

        opacity = float(np.clip(sensory['opacity'], 0.0, 1.0))
        prec_min, prec_max = self.l3tol2_precision_range
        transparency = 1.0 - opacity
        precision_drive = transparency
        precision = prec_min + (prec_max - prec_min) * precision_drive
        precision = float(np.clip(precision, 0.0, 1.0))
        prescription_l2l3['precision_modulation'] = precision
        transition_drive = float(np.clip(0.5 * (transparency + self.vfe_ema), 0.0, 1.0))

        # Policy inference with state-specific temperature
        state_temp = self.policy_temperature_by_state[current_state]
        policy_result = self.policy_energy_model.infer_policies(
            current_state=current_state,
            transition_pressure=self.prev_transition_drive,
            policy_horizon=self.policy_horizon,
            policy_temperature=float(max(1e-6, state_temp)),
            policy_horizon_discount=self.policy_horizon_discount,
        )
        self.policy_posterior = dict(policy_result.posterior)
        self.policy_efe = dict(policy_result.efe_by_policy)
        self.selected_policy = str(policy_result.selected_policy)
        self.policy_confidence = float(policy_result.selected_confidence)
        self.policy_entropy = float(policy_result.posterior_entropy)

        self.efe_risk = float(policy_result.expected_risk)
        self.efe_ambiguity = float(policy_result.expected_ambiguity)
        self.efe_value = float(policy_result.expected_total)
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
            'selected_policy': self.selected_policy,
            'policy_confidence': self.policy_confidence,
            'policy_entropy': self.policy_entropy,
        }
