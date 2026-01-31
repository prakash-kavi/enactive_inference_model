"""Configuration for the Vipassana meditation simulation.
Defines thoughtseed/network profiles, mediative states, and tunable parameters.
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# Common definitions
# -----------------------------------------------------------------------------
STATES = ['breath_focus', 'mind_wandering', 'meta_awareness', 'redirect_breath']
NETWORKS = ['DMN', 'VAN', 'DAN', 'FPN']
THOUGHTSEEDS = ['attend_breath', 'pain_discomfort', 'pending_tasks', 'aha_moment', 'equanimity']

DEFAULTS = {
    'ACTIVATION_CLIP_MIN': 0.05,
    'ACTIVATION_CLIP_MAX': 0.9,
    'DEFAULT_DT': 0.2,
    'REFRACTORY_SEC': 0.4,
}

# -----------------------------------------------------------------------------
# Layer 1: Generative process (MVOU) and state dynamics
# -----------------------------------------------------------------------------
# Base Multivariate Ornstein-Uhlenbeck (OU) coupling terms per state (dX = -Theta * dt).
# Positive = Inhibition/Restoring force (keeps activation stable).
# Negative = Synergy/Divergent force (drives co-activation).
THETA_BASE = {
    'breath_focus': {
        ('DMN', 'DAN'): 0.50, ('DAN', 'DMN'): 0.50,
        ('DAN', 'FPN'): 0.15, ('FPN', 'DAN'): 0.15
    },
    'mind_wandering': {
        ('DMN', 'VAN'): -0.30, ('VAN', 'DMN'): -0.30,
        ('DMN', 'FPN'): -0.15, ('FPN', 'DMN'): -0.15
    },
    'meta_awareness': {
        ('VAN', 'FPN'): -0.75, ('FPN', 'VAN'): -0.75,
        ('DMN', 'DAN'): -0.25, ('DAN', 'DMN'): -0.25,
        ('DMN', 'FPN'): -0.25, ('FPN', 'DMN'): -0.25
    },
    'redirect_breath': {
        ('DMN', 'DAN'): -0.40, ('DAN', 'DMN'): -0.40,
        ('DMN', 'FPN'): -0.30, ('FPN', 'DMN'): -0.30,
        ('DAN', 'FPN'): 0.40, ('FPN', 'DAN'): 0.40
    }
}

# Network profiles for mediative states (used for agent learning initialization)
NETWORK_PROFILES = {
    "breath_focus": {
        "novice": {"DMN": 0.50, "VAN": 0.45, "DAN": 0.58, "FPN": 0.60},
        "expert": {"DMN": 0.40, "VAN": 0.45, "DAN": 0.60, "FPN": 0.65}
    },
    "mind_wandering": {
        "novice": {"DMN": 0.82, "VAN": 0.35, "DAN": 0.30, "FPN": 0.33},
        "expert": {"DMN": 0.65, "VAN": 0.50, "DAN": 0.40, "FPN": 0.50}
    },
    "meta_awareness": {
        "novice": {"DMN": 0.45, "VAN": 0.78, "DAN": 0.45, "FPN": 0.52},
        "expert": {"DMN": 0.40, "VAN": 0.70, "DAN": 0.45, "FPN": 0.50}
    },
    "redirect_breath": {
        "novice": {"DMN": 0.40, "VAN": 0.55, "DAN": 0.72, "FPN": 0.78},
        "expert": {"DMN": 0.35, "VAN": 0.55, "DAN": 0.70, "FPN": 0.75}
    }
}

# Dwell Times (Seconds) for State Transitions (level-specific)
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

# Exit transition probabilities (self-transitions handled by dwell)
STATE_TRANSITION_PROBS = {
    'expert': {
        'breath_focus': {'mind_wandering': 0.40, 'meta_awareness': 0.40, 'redirect_breath': 0.20},
        'mind_wandering': {'breath_focus': 0.20, 'meta_awareness': 0.55, 'redirect_breath': 0.25},
        'meta_awareness': {'redirect_breath': 0.80, 'breath_focus': 0.15, 'mind_wandering': 0.05},
        'redirect_breath': {'breath_focus': 0.60, 'meta_awareness': 0.30, 'mind_wandering': 0.10},
    },
    'novice': {
        'breath_focus': {'mind_wandering': 0.88, 'meta_awareness': 0.08, 'redirect_breath': 0.04},
        'mind_wandering': {'meta_awareness': 0.65, 'redirect_breath': 0.20, 'breath_focus': 0.15},
        'meta_awareness': {'redirect_breath': 0.75, 'breath_focus': 0.15, 'mind_wandering': 0.10},
        'redirect_breath': {'breath_focus': 0.85, 'mind_wandering': 0.10, 'meta_awareness': 0.05},
    }
}

# Preferred transition bias (state-conditional preferences).
# Values are small deltas added to base transition probs and renormalized.
PREFERRED_TRANSITION_BIAS = {
    'novice': {
        'breath_focus': {'mind_wandering': 0.04, 'redirect_breath': -0.02},
        'mind_wandering': {'meta_awareness': 0.06, 'redirect_breath': -0.02},
        'meta_awareness': {'redirect_breath': 0.06, 'mind_wandering': -0.03},
        'redirect_breath': {'breath_focus': 0.06, 'mind_wandering': -0.03},
    },
    'expert': {
        'breath_focus': {'meta_awareness': 0.06, 'mind_wandering': -0.04},
        'mind_wandering': {'meta_awareness': 0.07, 'redirect_breath': 0.03},
        'meta_awareness': {'redirect_breath': 0.08, 'mind_wandering': -0.04},
        'redirect_breath': {'meta_awareness': 0.05, 'mind_wandering': -0.03},
    }
}

# -----------------------------------------------------------------------------
# Layer 2: Thoughtseed priors and meta-awareness mapping
# -----------------------------------------------------------------------------
THOUGHTSEED_BASE_ACTIVATIONS = {
    "breath_focus": {
        "attend_breath": 0.75,
        "equanimity": 0.35,
        "pain_discomfort": 0.1,
        "pending_tasks": 0.05,
        "aha_moment": 0.15
    },
    "mind_wandering": {
        "attend_breath": 0.05,
        "equanimity": 0.05,
        "pain_discomfort": 0.75,
        "pending_tasks": 0.8,
        "aha_moment": 0.05
    },
    "meta_awareness": {
        "attend_breath": 0.25,
        "equanimity": 0.4,
        "pain_discomfort": 0.1,
        "pending_tasks": 0.1,
        "aha_moment": 0.8
    },
    "redirect_breath": {
        "attend_breath": 0.7,
        "equanimity": 0.75,
        "pain_discomfort": 0.12,
        "pending_tasks": 0.05,
        "aha_moment": 0.5
    }
}

THOUGHTSEED_TARGET_ADJUSTMENTS = {
    "breath_focus": {
        "attend_breath": 0.1,
        "equanimity": 0.25,
        "pain_discomfort": 0.0,
        "pending_tasks": 0.0,
        "aha_moment": 0.1
    },
    "mind_wandering": {
        "attend_breath": 0.0,
        "equanimity": -0.05,
        "pain_discomfort": -0.1,
        "pending_tasks": -0.1,
        "aha_moment": 0.3
    },
    "meta_awareness": {
        "attend_breath": 0.1,
        "equanimity": 0.1,
        "pain_discomfort": 0.0,
        "pending_tasks": 0.0,
        "aha_moment": 0.1
    },
    "redirect_breath": {
        "attend_breath": 0.2,
        "equanimity": 0.25,
        "pain_discomfort": -0.1,
        "pending_tasks": 0.0,
        "aha_moment": 0.1
    }
}

THOUGHTSEED_LEVEL_OFFSETS = {
    "expert": {
        "breath_focus": {
            "attend_breath": 0.1,
            "equanimity": 0.1,
            "pain_discomfort": 0.0,
            "pending_tasks": 0.0,
            "aha_moment": 0.0
        },
        "mind_wandering": {
            "attend_breath": 0.0,
            "equanimity": 0.05,
            "pain_discomfort": 0.3,
            "pending_tasks": 0.3,
            "aha_moment": 0.0
        },
        "meta_awareness": {
            "attend_breath": 0.0,
            "equanimity": 0.1,
            "pain_discomfort": 0.0,
            "pending_tasks": 0.0,
            "aha_moment": 0.05
        },
        "redirect_breath": {
            "attend_breath": 0.0,
            "equanimity": 0.0,
            "pain_discomfort": 0.0,
            "pending_tasks": 0.0,
            "aha_moment": 0.0
        }
    }
}

META_BASE_AWARENESS = {
    "breath_focus": 0.4,
    "mind_wandering": 0.2,
    "meta_awareness": 0.6,
    "redirect_breath": 0.5
}

META_THOUGHTSEED_INFLUENCES = {
    "aha_moment": 0.1,
    "equanimity": 0.1
}

# -----------------------------------------------------------------------------
# Layer 2 / 3: Active inference parameters
# -----------------------------------------------------------------------------
ACTINF_DEFAULTS = {
    "learning_rate": 0.01,
    "z_ema_alpha": 0.75,
    "z_noise_sigma": 0.05,
    "aha_accum_decay": 0.95,
    "aha_accum_inc": 0.05,
    "state_embed_dim": 2,
    "vfe_ema_alpha": 0.9,
    "kl_beta": 1.0,
    "efe_risk_weight": 1.0,
    "efe_ambiguity_weight": 0.4,
    "l3tol2_precision_range": (0.4, 0.6),
    "network_target_reg": 0.05,
}

ACTINF_EXPERT_OVERRIDES = {
    "learning_rate": 0.02,
    "efe_ambiguity_weight": 0.35,
    "kl_beta": 0.55,
    "z_noise_sigma": 0.02
}
