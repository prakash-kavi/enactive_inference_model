"""Configuration for the Vipassana meditation simulation.
Defines thoughtseed/network profiles, mediative states, and tunable parameters.
"""

from __future__ import annotations
from dataclasses import dataclass

# Core mediative states, networks and thoughtseed definitions
STATES = ['breath_focus', 'mind_wandering', 'meta_awareness', 'redirect_breath']
NETWORKS = ['DMN', 'VAN', 'DAN', 'FPN']
THOUGHTSEEDS = ['attend_breath', 'pain_discomfort', 'pending_tasks', 'aha_moment', 'equanimity']

# Network profiles for mediative states (used for agent learning initialization)
NETWORK_PROFILES = {
    "breath_focus": {
        "novice": {"DMN": 0.55, "VAN": 0.50, "DAN": 0.50, "FPN": 0.55},
        "expert": {"DMN": 0.40, "VAN": 0.45, "DAN": 0.60, "FPN": 0.65}
    },
    "mind_wandering": {
        "novice": {"DMN": 0.75, "VAN": 0.40, "DAN": 0.35, "FPN": 0.40},
        "expert": {"DMN": 0.65, "VAN": 0.50, "DAN": 0.40, "FPN": 0.50}
    },
    "meta_awareness": {
        "novice": {"DMN": 0.50, "VAN": 0.70, "DAN": 0.40, "FPN": 0.45},
        "expert": {"DMN": 0.40, "VAN": 0.70, "DAN": 0.45, "FPN": 0.50}
    },
    "redirect_breath": {
        "novice": {"DMN": 0.45, "VAN": 0.50, "DAN": 0.65, "FPN": 0.70},
        "expert": {"DMN": 0.35, "VAN": 0.55, "DAN": 0.70, "FPN": 0.75}
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
    'ACTIVATION_CLIP_MIN': 0.05,
    'ACTIVATION_CLIP_MAX': 0.9,
    'DEFAULT_DT': 0.2,
}

# Dwell Times (Seconds) for State Machine
DWELL_TIMES = {
    'expert': {
        'breath_focus': (15, 30),
        'mind_wandering': (10, 20),
        'meta_awareness': (1, 4),
        'redirect_breath': (1, 4)
    },
    'novice': {
        'breath_focus': (5, 15),
        'mind_wandering': (20, 40),
        'meta_awareness': (2, 6),
        'redirect_breath': (2, 5)
    }
}

@dataclass
class ThoughtseedParams:
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
        activations = ThoughtseedParams.BASE_ACTIVATIONS[state].copy()
        for ts in activations:
            meta_mod, expert_offset = ThoughtseedParams.TARGET_ADJUSTMENTS[state][ts]
            activations[ts] += meta_mod * meta_awareness
            if experience_level == 'expert':
                activations[ts] += expert_offset
        return activations

@dataclass
class MetacognitionParams:
    BASE_AWARENESS = {
        "breath_focus": 0.4,
        "mind_wandering": 0.2,
        "meta_awareness": 0.6,
        "redirect_breath": 0.5
    }

    THOUGHTSEED_INFLUENCES = {
        "aha_moment": 0.1,
        "equanimity": 0.1
    }

    @staticmethod
    def calculate_meta_awareness(state, thoughtseed_activations, experience_level='novice'):
        """Compute meta-awareness from mediative state and thoughtseed activations."""
        base_awareness = MetacognitionParams.BASE_AWARENESS[state]
        awareness_boost = 0
        for ts, influence in MetacognitionParams.THOUGHTSEED_INFLUENCES.items():
            if ts in thoughtseed_activations:
                awareness_boost += thoughtseed_activations[ts] * influence
        return base_awareness + awareness_boost

@dataclass
class ActInfParams:
    # VFE and learning
    precision_weight: float
    complexity_penalty: float
    learning_rate: float
    noise_level: float
    memory_factor: float
    
    # Thoughtseed dynamics
    distraction_pressure: float
    fatigue_rate: float
    smoothing: float
    base_theta: float
    base_sigma: float
    
    # Aha moment dynamics
    aha_target_gain: float
    aha_accum_decay: float
    aha_accum_inc: float
    
    # VFE accumulator
    vfe_accum_decay: float
    
    # Precision parameters
    sensory_precision_base: float
    prior_precision_base: float
    
    
    # State expectations and modulation
    fpn_enhancement: float
    expert_meta_scalar: float

    _BASE_DEFAULTS = {
        "precision_weight": 0.4,
        "complexity_penalty": 0.4,
        "learning_rate": 0.01,
        "noise_level": 0.04,
        "memory_factor": 0.85,
        "distraction_pressure": 1.30,
        "fatigue_rate": 0.30,
        "smoothing": 0.6,
        "base_theta": 0.3, # Increased from 0.2 to reach attractors faster
        "base_sigma": 0.05,
        "vfe_accum_decay": 0.9,
        "fpn_enhancement": 1.0,
        "aha_target_gain": 0.2,
        "aha_accum_decay": 0.95,
        "aha_accum_inc": 0.05,
        "sensory_precision_base": 0.1,
        "prior_precision_base": 1.0,
        "expert_meta_scalar": 1.0,
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
            base_theta=0.5, # Increased from 0.25 to 0.5 (Stiff Attractor)
            base_sigma=0.035,
            expert_meta_scalar=1.05
        )
        return cls(**base)
