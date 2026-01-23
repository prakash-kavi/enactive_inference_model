"""Configuration for the Vipassana Entropy meditation simulation."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import json
import os

# Helper: attempt to load a JSON config from the same `config/` directory (package)
def _load_config_json(name: str) -> Optional[Dict]:
    """Load optional JSON config from `config/` and return a dict or None."""
    base_dir = os.path.dirname(__file__)
    # First check current config directory, then `config/profiles/` for overrides.
    candidates = [
        os.path.join(base_dir, name),
        os.path.join(base_dir, 'profiles', name)
    ]
    for cfg_path in candidates:
        if not os.path.exists(cfg_path):
            continue
        try:
            with open(cfg_path, 'r', encoding='utf-8') as fh:
                return json.load(fh)
        except Exception:
            continue
    return None

# Core thoughtseed and state definitions
THOUGHTSEEDS = ['breath_focus', 'pain_discomfort', 'pending_tasks', 'self_reflection', 'equanimity']
STATES = ['breath_control', 'mind_wandering', 'meta_awareness', 'redirect_breath']

@dataclass
class DwellTimeConfig:
    breath_control: Tuple[int, int]
    mind_wandering: Tuple[int, int]
    meta_awareness: Tuple[int, int]
    redirect_breath: Tuple[int, int]
    
    @classmethod
    def novice(cls) -> 'DwellTimeConfig':
        return cls(
            breath_control=(5, 15),
            mind_wandering=(22, 42),
            meta_awareness=(2, 7),
            redirect_breath=(2, 5)
        )
    
    @classmethod
    def expert(cls) -> 'DwellTimeConfig':
        return cls(
            breath_control=(15, 25),
            mind_wandering=(8, 12),
            meta_awareness=(2, 4),
            redirect_breath=(2, 4)
        )

@dataclass
class NetworkProfile:
    DMN: float
    VAN: float
    DAN: float
    FPN: float
    
@dataclass
class TransitionThresholds:
    mind_wandering: float  # Distraction level threshold
    dmn_dan_ratio: float   # DMN/DAN ratio threshold
    meta_awareness: float  # Self-reflection threshold for meta-awareness
    return_focus: float    # Threshold to return to focused states
    
    @classmethod
    def novice(cls) -> 'TransitionThresholds':
        return cls(
            mind_wandering=0.6,
            dmn_dan_ratio=0.5,
            meta_awareness=0.4,
            return_focus=0.3
        )
    
    @classmethod
    def expert(cls) -> 'TransitionThresholds':
        return cls(
            mind_wandering=0.65,
            dmn_dan_ratio=0.6,
            meta_awareness=0.3,
            return_focus=0.35
        )

@dataclass
class StateTargetActivations:
    breath_focus: float
    equanimity: float
    pain_discomfort: float
    pending_tasks: float
    self_reflection: float
    
    def as_dict(self) -> Dict[str, float]:
        """Convert to dictionary for compatibility with existing code"""
        return {
            "breath_focus": self.breath_focus,
            "equanimity": self.equanimity,
            "pain_discomfort": self.pain_discomfort,
            "pending_tasks": self.pending_tasks,
            "self_reflection": self.self_reflection
        }




def load_actinf_params_from_json(path: Optional[str], experience_level: str = 'novice') -> 'ActInfParams':
    """Load `ActInfParams` from JSON; fall back to defaults if missing/invalid."""
    if not path:
        return ActInfParams.expert() if experience_level == 'expert' else ActInfParams.novice()

    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return ActInfParams.expert() if experience_level == 'expert' else ActInfParams.novice()

    if isinstance(data, dict) and experience_level in data:
        cfg = data[experience_level]
    else:
        cfg = data

    # Basic validation: ensure keys exist and types are plausible before constructing dataclass.
    def _validate_cfg(cfg_dict: Dict) -> bool:
        if not isinstance(cfg_dict, dict):
            return False
        # numeric keys we expect to be floats/ints
        numeric_keys = [
            'precision_weight', 'learning_rate', 'noise_level', 'memory_factor',
            'fpn_enhancement', 'distraction_pressure', 'fatigue_rate',
            'vfe_accum_decay', 'vfe_accum_alpha', 'fpn_accum_decay', 'fpn_accum_inc'
        ]
        for k in numeric_keys:
            if k in cfg_dict and not isinstance(cfg_dict[k], (int, float)):
                return False
        # transition thresholds must be dict with numeric entries if present
        tt = cfg_dict.get('transition_thresholds')
        if tt is not None:
            if not isinstance(tt, dict):
                return False
            for kk in ['mind_wandering', 'dmn_dan_ratio', 'meta_awareness', 'return_focus']:
                if kk in tt and not isinstance(tt[kk], (int, float)):
                    return False
        return True

    if not _validate_cfg(cfg):
        # fallback to defaults if validation fails
        return ActInfParams.expert() if experience_level == 'expert' else ActInfParams.novice()

    try:
        return ActInfParams.from_dict(cfg, experience_level=experience_level)
    except Exception:
        return ActInfParams.expert() if experience_level == 'expert' else ActInfParams.novice()

@dataclass
class ActInfParams:
    precision_weight: float
    complexity_penalty: float
    learning_rate: float
    noise_level: float
    memory_factor: float
    fpn_enhancement: float
    fpn_reflection_value: float
    fpn_equanimity_value: float
    distraction_pressure: float
    fatigue_rate: float
    fpn_accum_decay: float
    fpn_accum_inc: float
    fatigue_reset: float
    # FPN collapse / base demand tunables
    fpn_collapse_dan_mult: float
    fpn_collapse_dmn_inc: float
    fpn_base_demand: float
    fpn_focus_mult: float
    # Network/dynamics tunables (migrated from DEFAULTS)
    network_base: float
    fpn_to_dan_gain: float
    hysteresis_strength: float
    anticorrelation_force: float
    van_spike: float
    # Surface / modulation defaults (DMN/VAN/DAN effects)
    dmn_pending_value: float
    dmn_reflection_value: float
    dmn_breath_value: float
    van_pain_value: float
    van_reflection_value: float
    dan_breath_value: float
    dan_pending_value: float
    dan_pain_value: float
    # Efficiency weight (expert vs novice differences)
    efficiency_weight: float
    # Per-agent smoothing/blending and transition noise
    smoothing: float
    blend_factor_transition: float
    blend_factor_state: float
    blend_variation: float
    transition_perturb_std: float
    transition_variation_low: float
    transition_variation_high: float
    # VFE accumulator dynamics
    vfe_accum_decay: float
    vfe_accum_alpha: float
    base_theta: float
    base_sigma: float
    softmax_temperature: float
    fatigue_threshold: float
    transition_thresholds: TransitionThresholds
    
    @classmethod
    def novice(cls) -> 'ActInfParams':
        return cls(
            precision_weight=0.4,
            complexity_penalty=0.4,
            learning_rate=0.01,
            noise_level=0.04,
            memory_factor=0.85, 
            fpn_enhancement=1.0,
            fpn_reflection_value=0.15,
            fpn_equanimity_value=0.2,
            distraction_pressure=1.30,
            fatigue_rate=0.30,
            smoothing=0.6,
            blend_factor_transition=0.3,
            blend_factor_state=0.4,
            blend_variation=0.1,
            transition_perturb_std=0.02,
            transition_variation_low=-0.05,
            transition_variation_high=0.1,
            vfe_accum_decay=0.9,
            vfe_accum_alpha=0.1,
            base_theta=0.2,
            base_sigma=0.05,
            fpn_accum_decay=0.98,
            fpn_accum_inc=0.02,
            fatigue_reset=0.4,
            fpn_collapse_dan_mult=0.6,
            fpn_collapse_dmn_inc=0.2,
            fpn_base_demand=0.2,
            fpn_focus_mult=2.0,
            # network/dynamics defaults (may be overridden by config/*.json)
            network_base=0.1,
            fpn_to_dan_gain=0.4,
            hysteresis_strength=0.1,
            anticorrelation_force=0.25,
            van_spike=0.5,
            # surface/modulation defaults
            dmn_pending_value=0.15,
            dmn_reflection_value=0.05,
            dmn_breath_value=0.2,
            van_pain_value=0.15,
            van_reflection_value=0.2,
            dan_breath_value=0.2,
            dan_pending_value=0.15,
            dan_pain_value=0.1,
            softmax_temperature=2.5,
            efficiency_weight=0.3,
            fatigue_threshold=0.50,
            transition_thresholds=TransitionThresholds.novice()
        )
    
    @classmethod
    def expert(cls) -> 'ActInfParams':
        return cls(
            precision_weight=0.5,
            complexity_penalty=0.2,
            learning_rate=0.02,
            noise_level=0.03,  
            memory_factor=0.75,  
            fpn_enhancement=1.1, 
            fpn_reflection_value=0.2,
            fpn_equanimity_value=0.25,
            distraction_pressure=0.62,
            fatigue_rate=0.15,         
            smoothing=0.8,
            blend_factor_transition=0.3,
            blend_factor_state=0.4,
            blend_variation=0.1,
            transition_perturb_std=0.02,
            transition_variation_low=-0.05,
            transition_variation_high=0.1,
            vfe_accum_decay=0.9,
            vfe_accum_alpha=0.1,
            base_theta=0.25,
            base_sigma=0.035,
            fpn_accum_decay=0.98,
            fpn_accum_inc=0.02,
            fatigue_reset=0.4,
            fpn_collapse_dan_mult=0.6,
            fpn_collapse_dmn_inc=0.2,
            fpn_base_demand=0.2,
            fpn_focus_mult=2.0,
            # network/dynamics defaults (may be overridden by config/*.json)
            network_base=0.1,
            fpn_to_dan_gain=0.4,
            hysteresis_strength=0.2,
            anticorrelation_force=0.25,
            van_spike=0.5,
            # surface/modulation defaults
            dmn_pending_value=0.15,
            dmn_reflection_value=0.05,
            dmn_breath_value=0.2,
            van_pain_value=0.15,
            van_reflection_value=0.2,
            dan_breath_value=0.2,
            dan_pending_value=0.15,
            dan_pain_value=0.1,
            softmax_temperature=2.0,   
            efficiency_weight=0.7,
            fatigue_threshold=0.75,
            transition_thresholds=TransitionThresholds.expert()
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Union[float, Dict]], experience_level: str = 'novice') -> 'ActInfParams':
        """Create `ActInfParams` from a dict; fill missing keys from defaults."""
        base = cls.expert() if experience_level == 'expert' else cls.novice()
        base_dict = base.as_dict()

        # Merge provided values
        merged = base_dict.copy()
        merged.update(data or {})

        # Build TransitionThresholds object
        tt = merged.get('transition_thresholds', None)
        if isinstance(tt, dict):
            tt_obj = TransitionThresholds(
                mind_wandering=float(tt.get('mind_wandering', base.transition_thresholds.mind_wandering)),
                dmn_dan_ratio=float(tt.get('dmn_dan_ratio', base.transition_thresholds.dmn_dan_ratio)),
                meta_awareness=float(tt.get('meta_awareness', base.transition_thresholds.meta_awareness)),
                return_focus=float(tt.get('return_focus', base.transition_thresholds.return_focus)),
            )
        else:
            tt_obj = base.transition_thresholds

        merged['transition_thresholds'] = tt_obj

        return cls(**merged)
    
    def as_dict(self) -> Dict[str, Union[float, Dict]]:
        """Convert to dictionary for compatibility with existing code"""
        return {
            'precision_weight': self.precision_weight,
            'complexity_penalty': self.complexity_penalty,
            'learning_rate': self.learning_rate,
            'noise_level': self.noise_level,
            'memory_factor': self.memory_factor,
            'fpn_enhancement': self.fpn_enhancement,
            'base_theta': self.base_theta,
            'base_sigma': self.base_sigma,
            'fpn_reflection_value': self.fpn_reflection_value,
            'fpn_equanimity_value': self.fpn_equanimity_value,
            'distraction_pressure': self.distraction_pressure,
            'fatigue_rate': self.fatigue_rate,
            'smoothing': self.smoothing,
            'blend_factor_transition': self.blend_factor_transition,
            'blend_factor_state': self.blend_factor_state,
            'blend_variation': self.blend_variation,
            'transition_perturb_std': self.transition_perturb_std,
            'transition_variation_low': self.transition_variation_low,
            'transition_variation_high': self.transition_variation_high,
            'vfe_accum_decay': self.vfe_accum_decay,
            'vfe_accum_alpha': self.vfe_accum_alpha,
            'fpn_accum_decay': self.fpn_accum_decay,
            'fpn_accum_inc': self.fpn_accum_inc,
            'fatigue_reset': self.fatigue_reset,
            'fpn_collapse_dan_mult': self.fpn_collapse_dan_mult,
            'fpn_collapse_dmn_inc': self.fpn_collapse_dmn_inc,
            'fpn_base_demand': self.fpn_base_demand,
            'fpn_focus_mult': self.fpn_focus_mult,
            'network_base': self.network_base,
            'fpn_to_dan_gain': self.fpn_to_dan_gain,
            'hysteresis_strength': self.hysteresis_strength,
            'anticorrelation_force': self.anticorrelation_force,
            'van_spike': self.van_spike,
            'softmax_temperature': self.softmax_temperature,
            'dmn_pending_value': self.dmn_pending_value,
            'dmn_reflection_value': self.dmn_reflection_value,
            'dmn_breath_value': self.dmn_breath_value,
            'van_pain_value': self.van_pain_value,
            'van_reflection_value': self.van_reflection_value,
            'dan_breath_value': self.dan_breath_value,
            'dan_pending_value': self.dan_pending_value,
            'dan_pain_value': self.dan_pain_value,
            'efficiency_weight': self.efficiency_weight,
            'fatigue_threshold': self.fatigue_threshold,
            'transition_thresholds': {
                'mind_wandering': self.transition_thresholds.mind_wandering,
                'dmn_dan_ratio': self.transition_thresholds.dmn_dan_ratio,
                'meta_awareness': self.transition_thresholds.meta_awareness,
                'return_focus': self.transition_thresholds.return_focus
            }
        }


# Create the base configurations as module-level constants
STATE_DWELL_TIMES = {
    # Per-experience-level dwell time ranges (min, max) for each state
    'novice': DwellTimeConfig.novice().__dict__,
    'expert': DwellTimeConfig.expert().__dict__
}

# Project-wide numeric defaults to avoid magic numbers
DEFAULTS = {
    # Numeric clamps and thresholds used across the simulation
    'TARGET_CLIP_MIN': 0.05,  # lower bound for thoughtseed target activations
    'TARGET_CLIP_MAX': 1.0,   # upper bound for thoughtseed target activations
    'ACTIVATION_CLIP_MIN': 0.01,
    'ACTIVATION_CLIP_MAX': 0.99,
    'NETWORK_CLIP_MIN': 0.05,  # network activation lower bound
    'NETWORK_CLIP_MAX': 0.9,   # network activation upper bound
    'VAN_TRIGGER': 0.7,        # VAN accumulator threshold for salience spike
    'VAN_MAX': 0.85,           # physiological cap for VAN
    'DEFAULT_DT': 1.0,
    'MIN_HISTORY_FOR_LEARNING': 10
}

# Additional tunable defaults for dynamics and transitions
DEFAULTS.update({
    'TRANSITION_COUNTER_BASE': 3,
    'TRANSITION_COUNTER_RAND': 2,
})

# Additional surface/modulation defaults (moved from hard-coded locations)
# Surface/modulation defaults intentionally left out — per-agent values live in ActInfParams
# Default surface/modulation values are now provided by `ActInfParams` and optional JSON profiles.
 
# Network profiles for thoughtseeds and states
NETWORK_PROFILES = {
    "thoughtseed_contributions": {
        "breath_focus": NetworkProfile(DMN=0.2, VAN=0.3, DAN=0.65, FPN=0.6).__dict__,
        "pain_discomfort": NetworkProfile(DMN=0.5, VAN=0.7, DAN=0.3, FPN=0.4).__dict__,
        "pending_tasks": NetworkProfile(DMN=0.8, VAN=0.5, DAN=0.2, FPN=0.4).__dict__,
        "self_reflection": NetworkProfile(DMN=0.6, VAN=0.4, DAN=0.3, FPN=0.8).__dict__,
        "equanimity": NetworkProfile(DMN=0.3, VAN=0.3, DAN=0.5, FPN=0.9).__dict__
    },
    
    # State profiles differentiated by experience level
    # Expected network activations per high-level state and experience level
    "state_expected_profiles": {
        # BREATH CONTROL: Experts have lower DMN, higher DAN/FPN
        "breath_control": {
            "novice": NetworkProfile(DMN=0.35, VAN=0.4, DAN=0.7, FPN=0.5).__dict__,
            "expert": NetworkProfile(DMN=0.24, VAN=0.42, DAN=0.68, FPN=0.65).__dict__
        },
        
        # MIND WANDERING: Experts have much lower DMN, higher FPN control
        "mind_wandering": {
            "novice": NetworkProfile(DMN=0.85, VAN=0.45, DAN=0.2, FPN=0.35).__dict__,
            "expert": NetworkProfile(DMN=0.55, VAN=0.55, DAN=0.35, FPN=0.50).__dict__
        },
        
        # META-AWARENESS: Experts have higher VAN (detection) and FPN (control)
        "meta_awareness": {
            "novice": NetworkProfile(DMN=0.35, VAN=0.7, DAN=0.5, FPN=0.45).__dict__,
            "expert": NetworkProfile(DMN=0.32, VAN=0.85, DAN=0.48, FPN=0.55).__dict__
        },
        
        # REDIRECT BREATH: Experts have lower DMN, higher DAN/FPN (control)
        "redirect_breath": {
            "novice": NetworkProfile(DMN=0.3, VAN=0.45, DAN=0.65, FPN=0.55).__dict__,
            "expert": NetworkProfile(DMN=0.18, VAN=0.55, DAN=0.68, FPN=0.65).__dict__
        }
    }
}

# Attempt to override hard-coded constants with files in `config/` when available.
# This makes runtime behavior configurable without editing Python sources.
_cfg = _load_config_json('state_dwell_times.json')
if isinstance(_cfg, dict):
    try:
        STATE_DWELL_TIMES = _cfg
    except Exception:
        pass

_cfg = _load_config_json('network_profiles.json')
if isinstance(_cfg, dict):
    try:
        NETWORK_PROFILES = _cfg
    except Exception:
        pass

_cfg = _load_config_json('defaults_global.json')
if isinstance(_cfg, dict):
    try:
        DEFAULTS.update(_cfg)
    except Exception:
        pass

@dataclass
class ThoughtseedParams:
    
    # Base target activation patterns for each thoughtseed in each state
    BASE_ACTIVATIONS = {
        "breath_control": StateTargetActivations(
            breath_focus=0.7,
            equanimity=0.3,
            pain_discomfort=0.15,
            pending_tasks=0.1,
            self_reflection=0.2
        ).as_dict(),
        "mind_wandering": StateTargetActivations(
            breath_focus=0.1,
            equanimity=0.1,
            pain_discomfort=0.6,
            pending_tasks=0.7,
            self_reflection=0.1
        ).as_dict(),
        "meta_awareness": StateTargetActivations(
            breath_focus=0.2,
            equanimity=0.3,
            pain_discomfort=0.15,
            pending_tasks=0.15,
            self_reflection=0.8
        ).as_dict(),
        "redirect_breath": StateTargetActivations(
            breath_focus=0.6,
            equanimity=0.7,
            pain_discomfort=0.2,
            pending_tasks=0.1,
            self_reflection=0.4
        ).as_dict()
    }
    
    # How meta-awareness modulates each thoughtseed in each state
    META_AWARENESS_MODULATORS = {
        "breath_control": {
            "breath_focus": 0.1,
            "equanimity": 0.25,
            "pain_discomfort": 0.0,
            "pending_tasks": 0.0,
            "self_reflection": 0.1
        },
        "mind_wandering": {
            "breath_focus": 0.0,
            "equanimity": -0.05,
            "pain_discomfort": -0.1,
            "pending_tasks": -0.1,
            "self_reflection": 0.3
        },
        "meta_awareness": {
            "breath_focus": 0.1,
            "equanimity": 0.1,
            "pain_discomfort": 0.0,
            "pending_tasks": 0.0,
            "self_reflection": 0.1
        },
        "redirect_breath": {
            "breath_focus": 0.2,
            "equanimity": 0.25,
            "pain_discomfort": -0.1,
            "pending_tasks": 0.0,
            "self_reflection": 0.1
        }
    }
    
    # Experience-specific adjustments (values to add for experts)
    EXPERT_ADJUSTMENTS = {
        "breath_control": {
            "breath_focus": 0.1,
            "equanimity": 0.2,
            "pain_discomfort": 0.0,
            "pending_tasks": 0.0,
            "self_reflection": 0.0
        },
        "mind_wandering": {
            "breath_focus": 0.0,
            "equanimity": 0.05,
            "pain_discomfort": 0.4,
            "pending_tasks": 0.4,
            "self_reflection": 0.0
        },
        "meta_awareness": {
            "breath_focus": 0.0,
            "equanimity": 0.1,
            "pain_discomfort": 0.0,
            "pending_tasks": 0.0,
            "self_reflection": 0.1
        },
        "redirect_breath": {
            "breath_focus": 0.0,
            "equanimity": 0.2,
            "pain_discomfort": 0.0,
            "pending_tasks": 0.0,
            "self_reflection": 0.0
        }
    }
    
    @staticmethod
    def get_target_activations(state, meta_awareness, experience_level='novice'):
        """Get target activation values for each thoughtseed in the specified state."""
        # Start with base activations for this state
        activations = ThoughtseedParams.BASE_ACTIVATIONS[state].copy()
        
        # Apply meta-awareness modulation
        for ts in activations:
            modulator = ThoughtseedParams.META_AWARENESS_MODULATORS[state][ts]
            activations[ts] += modulator * meta_awareness
        
        # Apply expert adjustments if applicable
        if experience_level == 'expert':
            for ts in activations:
                activations[ts] += ThoughtseedParams.EXPERT_ADJUSTMENTS[state].get(ts, 0)
        
        return activations

@dataclass
class MetacognitionParams:
    
    # Base meta-awareness levels for each state
    BASE_AWARENESS = {
        "breath_control": 0.4,
        "mind_wandering": 0.2,
        "meta_awareness": 0.6,
        "redirect_breath": 0.5
    }
    
    # How thoughtseeds influence meta-awareness
    THOUGHTSEED_INFLUENCES = {
        "self_reflection": 0.1,  # Self-reflection strongly enhances meta-awareness
        "equanimity": 0.1        # Equanimity provides a stronger regulation boost for experts
    }
    
    @staticmethod
    def calculate_meta_awareness(state, thoughtseed_activations, experience_level='novice'):
        """Compute meta-awareness from state and thoughtseed activations."""
        # Get base awareness for this state
        base_awareness = MetacognitionParams.BASE_AWARENESS[state]
        
        # Calculate thoughtseed influence
        awareness_boost = 0
        for ts, influence in MetacognitionParams.THOUGHTSEED_INFLUENCES.items():
            if ts in thoughtseed_activations:
                awareness_boost += thoughtseed_activations[ts] * influence
        
        # Calculate total (without noise)
        meta_awareness = base_awareness + awareness_boost
        
        return meta_awareness

def get_actinf_params_dict(experience_level: str) -> Dict[str, Union[float, Dict]]:
    """Return a params dict for the requested experience level using `ActInfParams` dataclass."""
    return ActInfParams.expert().as_dict() if experience_level == 'expert' else ActInfParams.novice().as_dict()
