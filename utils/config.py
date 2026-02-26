"""Consolidated configuration for lean meditation model.

Merges meditation_config.py + layer1_config.py + fine-tuned params.
All constants, no dynamic parameter loading.
"""

# =============================================================================
# Core Architecture Constants
# =============================================================================

STATES = ['breath_focus', 'mind_wandering', 'meta_awareness', 'redirect_attention']
NETWORKS = ['DMN', 'VAN', 'DAN', 'FPN']
THOUGHTSEEDS = ['attend_breath', 'pain_discomfort', 'pending_tasks', 'aha_moment', 'equanimity']

DEFAULTS = {
    'CLIP_MIN': 0.05,
    'CLIP_MAX': 0.9,
    'DEFAULT_DT': 0.2,
}

EPS = 1e-6

# =============================================================================
# Layer 1: MVOU Generative Process
# =============================================================================

# Global process noise variance (used in L1 MVOU)
NOISE_LEVEL = 0.002

# Multivariate Ornstein-Uhlenbeck coupling (dX = -Theta * dt)
# Positive = inhibition, Negative = synergy
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
    'redirect_attention': {
        ('DMN', 'DAN'): -0.40, ('DAN', 'DMN'): -0.40,
        ('DMN', 'FPN'): -0.30, ('FPN', 'DMN'): -0.30,
        ('DAN', 'FPN'): 0.40, ('FPN', 'DAN'): 0.40
    }
}

# Network activation attractors (state-dependent means)
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
        "novice": {"DMN": 0.45, "VAN": 0.85, "DAN": 0.42, "FPN": 0.56},
        "expert": {"DMN": 0.40, "VAN": 0.80, "DAN": 0.42, "FPN": 0.55}
    },
    "redirect_attention": {
        "novice": {"DMN": 0.40, "VAN": 0.50, "DAN": 0.78, "FPN": 0.74},
        "expert": {"DMN": 0.35, "VAN": 0.50, "DAN": 0.78, "FPN": 0.70}
    }
}

# State dwell times (seconds) - min/max ranges
DWELL_TIMES = {
    'expert': {
        'breath_focus': (15, 25),
        'mind_wandering': (10, 20),
        'meta_awareness': (3, 6),
        'redirect_attention': (3, 6)
    },
    'novice': {
        'breath_focus': (8, 15),
        'mind_wandering': (15, 30),
        'meta_awareness': (5, 10),
        'redirect_attention': (5, 10)
    }
}

# L1 state transition: once dwell has elapsed, hazard = L1_BASE_HAZARD + drive_boost.
# Set to 0 for strictly dwell- and policy-driven transitions; >0 adds baseline hazard.
L1_BASE_HAZARD = 0.3

# Exit transition probabilities priors (exclude self-transitions)
STATE_TRANSITION_PROBS = {
    'expert': {
        'breath_focus': {'mind_wandering': 0.60, 'meta_awareness': 0.20, 'redirect_attention': 0.20},
        'mind_wandering': {'meta_awareness': 0.75, 'redirect_attention': 0.15, 'breath_focus': 0.10},
        'meta_awareness': {'redirect_attention': 0.85, 'breath_focus': 0.10, 'mind_wandering': 0.05},
        'redirect_attention': {'breath_focus': 0.80, 'meta_awareness': 0.15, 'mind_wandering': 0.05}
    },
    'novice': {
        'breath_focus': {'mind_wandering': 0.80, 'meta_awareness': 0.10, 'redirect_attention': 0.10},
        'mind_wandering': {'meta_awareness': 0.60, 'redirect_attention': 0.20, 'breath_focus': 0.20},
        'meta_awareness': {'redirect_attention': 0.60, 'breath_focus': 0.15, 'mind_wandering': 0.25},
        'redirect_attention': {'breath_focus': 0.60, 'mind_wandering': 0.35, 'meta_awareness': 0.05}
    }
}

# =============================================================================
# Layer 2: Thoughtseeds + recognition/decoder
# =============================================================================

# Fixed-step VI hyperparameters (L2)
VI_STEPS = 2
VI_LR = 0.2
# Apply VI refinement only in these states (System 2 sharpening)
VI_REFINEMENT_STATES = {'meta_awareness', 'redirect_attention'}

# Thoughtseed priors (state-dependent activation baselines)
THOUGHTSEED_STATE_PRIORS = {
    "breath_focus": {
        "attend_breath": 0.85,
        "equanimity": 0.45,
        "pain_discomfort": 0.2,
        "pending_tasks": 0.05,
        "aha_moment": 0.15
    },
    "mind_wandering": {
        "attend_breath": 0.15,
        "equanimity": 0.1,
        "pain_discomfort": 0.65,  
        "pending_tasks": 0.6,      
        "aha_moment": 0.15
    },
    "meta_awareness": {
        "attend_breath": 0.25,
        "equanimity": 0.35,
        "pain_discomfort": 0.2,
        "pending_tasks": 0.2,
        "aha_moment": 0.85
    },
    "redirect_attention": {
        "attend_breath": 0.7,
        "equanimity": 0.85,
        "pain_discomfort": 0.15,
        "pending_tasks": 0.15,
        "aha_moment": 0.25
    }

}

# Meta-awareness computation: state-aware thoughtseed weighting
META_THOUGHTSEED_WEIGHTS = {
    "breath_focus": {
        "attend_breath": 0.25,
        "equanimity": 0.35,
        "aha_moment": 0.25
    },
    "mind_wandering": {
        "attend_breath": 0.15,
        "equanimity": 0.25,
        "aha_moment": 0.20
    },
    "meta_awareness": {
        "attend_breath": 0.15,
        "equanimity": 0.35,
        "aha_moment": 0.60
    },
    "redirect_attention": {
        "attend_breath": 0.30,
        "equanimity": 0.40,
        "aha_moment": 0.25
    }
}

# =============================================================================
# Active Inference Parameters (Fine-Tuned)
# =============================================================================
LEARNING_RATES = {
    "novice": 0.01,
    "expert": 0.02,
}

# Layer 3: learned policy tendencies
L3_POLICY_LR = 0.05       # EMA learning rate for L3 policy prior update
L3_POLICY_STRENGTH = 0.5  # Scale for L3 prior influence on L2 (0 = neutral, 1 = full)
L3_META_EMA_ALPHA = 0.1   # EMA rate for meta-awareness smoothing


# =============================================================================
# Utility Functions
# =============================================================================
def compute_meta_awareness(state, thoughtseed_activations):
    """Compute meta-awareness from weighted thoughtseeds."""
    weights = META_THOUGHTSEED_WEIGHTS.get(state, {})
    weighted_sum = sum(float(thoughtseed_activations.get(ts, 0.0)) * float(w) for ts, w in weights.items())
    weight_total = sum(float(w) for w in weights.values())
    meta = weighted_sum / weight_total if weight_total > 0.0 else 0.0
    return float(max(0.0, min(1.0, meta)))

def get_policy_candidate_order(current_state: str):
    """Return policy candidate order: [current_state, ...others in STATES order].
    """
    return [current_state] + [s for s in STATES if s != current_state]


def get_exit_transition_probs(experience_level, current_state):
    """Return exit transition probabilities for the given state (rows sum to 1.0)."""
    return dict(STATE_TRANSITION_PROBS[experience_level][current_state])
