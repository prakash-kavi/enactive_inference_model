"""Configuration for the Vipassana Entropy meditation simulation.
Defines thoughtseed/network profiles, mediative states, and tunable parameters.
"""

from __future__ import annotations
from dataclasses import dataclass

# Core thoughtseed and mediative state definitions
THOUGHTSEEDS = ['attend_breath', 'pain_discomfort', 'pending_tasks', 'aha_moment', 'equanimity']
STATES = ['breath_focus', 'mind_wandering', 'meta_awareness', 'redirect_breath']

STATE_DWELL_TIMES = {
    # Per-experience-level dwell time ranges (min, max) for each mediative state
    'novice': {
        'breath_focus': (5, 15),
        'mind_wandering': (22, 42),
        'meta_awareness': (2, 7),
        'redirect_breath': (2, 5)
    },
    'expert': {
        'breath_focus': (15, 25),
        'mind_wandering': (8, 12),
        'meta_awareness': (2, 4),
        'redirect_breath': (2, 4)
    }
}

# Network profiles for thoughtseeds and mediative states
NETWORK_PROFILES = {
    "thoughtseed_contributions": {
        "attend_breath": {"DMN": 0.2, "VAN": 0.3, "DAN": 0.65, "FPN": 0.6},
        "pain_discomfort": {"DMN": 0.5, "VAN": 0.7, "DAN": 0.3, "FPN": 0.4},
        "pending_tasks": {"DMN": 0.8, "VAN": 0.5, "DAN": 0.2, "FPN": 0.4},
        "aha_moment": {"DMN": 0.6, "VAN": 0.4, "DAN": 0.3, "FPN": 0.8},
        "equanimity": {"DMN": 0.3, "VAN": 0.3, "DAN": 0.5, "FPN": 0.9}
    },

    # Expected network activations per high-level mediative state and experience level
    "state_expected_profiles": {
        # BREATH CONTROL: Experts have lower DMN, higher DAN/FPN
        "breath_focus": {
            "novice": {"DMN": 0.35, "VAN": 0.4, "DAN": 0.7, "FPN": 0.5},
            "expert": {"DMN": 0.24, "VAN": 0.42, "DAN": 0.68, "FPN": 0.65}
        },

        # MIND WANDERING: Experts have much lower DMN, higher FPN control
        "mind_wandering": {
            "novice": {"DMN": 0.85, "VAN": 0.45, "DAN": 0.2, "FPN": 0.35},
            "expert": {"DMN": 0.55, "VAN": 0.55, "DAN": 0.35, "FPN": 0.50}
        },

        # META-AWARENESS: Experts have higher VAN (detection) and FPN (control)
        "meta_awareness": {
            "novice": {"DMN": 0.35, "VAN": 0.7, "DAN": 0.5, "FPN": 0.45},
            "expert": {"DMN": 0.32, "VAN": 0.85, "DAN": 0.48, "FPN": 0.55}
        },

        # REDIRECT BREATH: Experts have lower DMN, higher DAN/FPN (control)
        "redirect_breath": {
            "novice": {"DMN": 0.3, "VAN": 0.45, "DAN": 0.65, "FPN": 0.55},
            "expert": {"DMN": 0.18, "VAN": 0.55, "DAN": 0.68, "FPN": 0.65}
        }
    }
}

NETWORK_MODULATION = {
    "DMN": {
        "pending_tasks": 0.15,
        "aha_moment": 0.05,
        "attend_breath": -0.2
    },
    "VAN": {
        "pain_discomfort": 0.15,
        "aha_moment_meta_awareness": 0.2
    },
    "DAN": {
        "attend_breath": 0.2,
        "pending_tasks": -0.15,
        "pain_discomfort": -0.1
    },
    "FPN": {
        "aha_moment": 0.15,
        "equanimity": 0.2
    }
}

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
    'MIN_HISTORY_FOR_LEARNING': 10,
    'TRANSITION_COUNTER_BASE': 3,
    'TRANSITION_COUNTER_RAND': 2
}

@dataclass
class ThoughtseedParams:

    # Base target activation patterns for each thoughtseed in each mediative state
    BASE_ACTIVATIONS = {
        "breath_focus": {
            "attend_breath": 0.7,
            "equanimity": 0.3,
            "pain_discomfort": 0.15,
            "pending_tasks": 0.1,
            "aha_moment": 0.2
        },
        "mind_wandering": {
            "attend_breath": 0.1,
            "equanimity": 0.1,
            "pain_discomfort": 0.6,
            "pending_tasks": 0.7,
            "aha_moment": 0.1
        },
        "meta_awareness": {
            "attend_breath": 0.2,
            "equanimity": 0.3,
            "pain_discomfort": 0.15,
            "pending_tasks": 0.15,
            "aha_moment": 0.8
        },
        "redirect_breath": {
            "attend_breath": 0.6,
            "equanimity": 0.7,
            "pain_discomfort": 0.2,
            "pending_tasks": 0.1,
            "aha_moment": 0.4
        }
    }

    # Target adjustments:  how meta-awareness and experience level modify base activations
    TARGET_ADJUSTMENTS = {
        "breath_focus": {
            "attend_breath": (0.1, 0.1),
            "equanimity": (0.25, 0.2),
            "pain_discomfort": (0.0, 0.0),
            "pending_tasks": (0.0, 0.0),
            "aha_moment": (0.1, 0.0)
        },
        "mind_wandering": {
            "attend_breath": (0.0, 0.0),
            "equanimity": (-0.05, 0.05),
            "pain_discomfort": (-0.1, 0.4),
            "pending_tasks": (-0.1, 0.4),
            "aha_moment": (0.3, 0.0)
        },
        "meta_awareness": {
            "attend_breath": (0.1, 0.0),
            "equanimity": (0.1, 0.1),
            "pain_discomfort": (0.0, 0.0),
            "pending_tasks": (0.0, 0.0),
            "aha_moment": (0.1, 0.1)
        },
        "redirect_breath": {
            "attend_breath": (0.2, 0.0),
            "equanimity": (0.25, 0.2),
            "pain_discomfort": (-0.1, 0.0),
            "pending_tasks": (0.0, 0.0),
            "aha_moment": (0.1, 0.0)
        }
    }

    @staticmethod
    def get_target_activations(state, meta_awareness, experience_level='novice'):
        """Get target activation values for each thoughtseed in the specified mediative state."""
        # Start with base activations for this mediative state
        activations = ThoughtseedParams.BASE_ACTIVATIONS[state].copy()

        # Apply unified adjustments
        for ts in activations:
            meta_mod, expert_offset = ThoughtseedParams.TARGET_ADJUSTMENTS[state][ts]
            activations[ts] += meta_mod * meta_awareness
            if experience_level == 'expert':
                activations[ts] += expert_offset

        return activations

@dataclass
class MetacognitionParams:

    # Base meta-awareness levels for each mediative state
    BASE_AWARENESS = {
        "breath_focus": 0.4,
        "mind_wandering": 0.2,
        "meta_awareness": 0.6,
        "redirect_breath": 0.5
    }

    # How thoughtseeds influence meta-awareness
    THOUGHTSEED_INFLUENCES = {
        "aha_moment": 0.1,  # Self-reflection strongly enhances meta-awareness
        "equanimity": 0.1   # Equanimity provides a stronger regulation boost for experts
    }

    @staticmethod
    def calculate_meta_awareness(state, thoughtseed_activations, experience_level='novice'):
        """Compute meta-awareness from mediative state and thoughtseed activations."""
        # Get base awareness for this mediative state
        base_awareness = MetacognitionParams.BASE_AWARENESS[state]

        # Calculate thoughtseed influence
        awareness_boost = 0
        for ts, influence in MetacognitionParams.THOUGHTSEED_INFLUENCES.items():
            if ts in thoughtseed_activations:
                awareness_boost += thoughtseed_activations[ts] * influence

        # Calculate total (without noise)
        meta_awareness = base_awareness + awareness_boost

        return meta_awareness

@dataclass
class ActInfParams:
    precision_weight: float
    complexity_penalty: float
    learning_rate: float
    noise_level: float
    memory_factor: float
    fpn_enhancement: float
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
    transition_weight_network: float
    transition_weight_activation: float
    fatigue_threshold: float
    # VFE Precision Parameters
    sensory_precision_base: float
    sensory_precision_van_scalar: float
    prior_precision_base: float
    prior_precision_meta_scalar: float

    # Learning Rate Precision
    learning_precision_base: float
    learning_precision_scalar: float

    # Network Targets & modulation
    dan_focus_target: float
    expert_meta_scalar: float

    _BASE_DEFAULTS = {
        "precision_weight": 0.4,
        "complexity_penalty": 0.4,
        "learning_rate": 0.01,
        "noise_level": 0.04,
        "memory_factor": 0.85,
        "fpn_enhancement": 1.0,
        "distraction_pressure": 1.30,
        "fatigue_rate": 0.30,
        "smoothing": 0.6,
        "blend_factor_transition": 0.3,
        "blend_factor_state": 0.4,
        "blend_variation": 0.1,
        "transition_perturb_std": 0.02,
        "transition_variation_low": -0.05,
        "transition_variation_high": 0.1,
        "vfe_accum_decay": 0.9,
        "vfe_accum_alpha": 0.1,
        "base_theta": 0.2,
        "base_sigma": 0.05,
        "transition_weight_network": 1.0,
        "transition_weight_activation": 1.0,
        "fpn_accum_decay": 0.98,
        "fpn_accum_inc": 0.02,
        "fatigue_reset": 0.4,
        "fpn_collapse_dan_mult": 0.6,
        "fpn_collapse_dmn_inc": 0.2,
        "fpn_base_demand": 0.2,
        "fpn_focus_mult": 2.0,
        # network/dynamics defaults (may be overridden by config/*.json)
        "network_base": 0.1,
        "fpn_to_dan_gain": 0.4,
        "hysteresis_strength": 0.1,
        "anticorrelation_force": 0.25,
        "van_spike": 0.5,
        "softmax_temperature": 2.5,
        "efficiency_weight": 0.3,
        "fatigue_threshold": 0.50,
        # Newly extracted magic numbers
        "sensory_precision_base": 0.1,
        "sensory_precision_van_scalar": 5.0,
        "prior_precision_base": 1.0,
        "prior_precision_meta_scalar": 3.0,
        "dan_focus_target": 0.9,
        "expert_meta_scalar": 1.0,  # Novice default
        "learning_precision_base": 1.0,
        "learning_precision_scalar": 2.0
    }

    @classmethod
    def novice(cls) -> 'ActInfParams':
        return cls(**cls._BASE_DEFAULTS)

    @classmethod
    def expert(cls) -> 'ActInfParams':
        base = dict(cls._BASE_DEFAULTS)
        base.update(
            precision_weight=0.5,
            complexity_penalty=0.2,
            learning_rate=0.02,
            noise_level=0.03,
            memory_factor=0.75,
            fpn_enhancement=1.1,
            distraction_pressure=0.62,
            fatigue_rate=0.15,
            smoothing=0.8,
            base_theta=0.25,
            base_sigma=0.035,
            hysteresis_strength=0.2,
            softmax_temperature=2.0,
            efficiency_weight=0.7,
            fatigue_threshold=0.75,
            expert_meta_scalar=1.05,
            learning_precision_scalar=5.0
        )
        return cls(**base)
