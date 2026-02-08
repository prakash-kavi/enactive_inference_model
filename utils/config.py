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
    'ACTIVATION_CLIP_MIN': 0.05,
    'ACTIVATION_CLIP_MAX': 0.9,
    'DEFAULT_DT': 0.2,
}

EPS = 1e-6

# =============================================================================
# Layer 1: MVOU Generative Process
# =============================================================================

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
        'breath_focus': (15, 30),
        'mind_wandering': (10, 20),
        'meta_awareness': (3, 6),
        'redirect_attention': (3, 6)
    },
    'novice': {
        'breath_focus': (8, 15),
        'mind_wandering': (15, 35),
        'meta_awareness': (5, 10),
        'redirect_attention': (5, 10)
    }
}

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
# Layer 2: Thoughtseeds + VAE
# =============================================================================

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
        "pain_discomfort": 0.8,  
        "pending_tasks": 0.75,      
        "aha_moment": 0.15
    },
    "meta_awareness": {
        "attend_breath": 0.25,
        "equanimity": 0.35,
        "pain_discomfort": 0.2,
        "pending_tasks": 0.2,
        "aha_moment": 0.95
    },
    "redirect_attention": {
        "attend_breath": 0.7,
        "equanimity": 0.85,
        "pain_discomfort": 0.15,
        "pending_tasks": 0.15,
        "aha_moment": 0.35
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
        "attend_breath": 0.10,
        "equanimity": 0.20,
        "aha_moment": 0.45
    },
    "meta_awareness": {
        "attend_breath": 0.15,
        "equanimity": 0.35,
        "aha_moment": 0.60
    },
    "redirect_attention": {
        "attend_breath": 0.30,
        "equanimity": 0.30,
        "aha_moment": 0.50
    }
}

# =============================================================================
# Active Inference Parameters (Fine-Tuned)
# =============================================================================
LEARNING_RATES = {
    "novice": 0.01,
    "expert": 0.02,
}

FORWARD_LOSS_WEIGHT = 0.5

# =============================================================================
# Utility Functions
# =============================================================================

def get_params(experience_level):
    """Get all parameters for experience level."""
    params = {"learning_rate": LEARNING_RATES[experience_level]}
    return params

def get_thoughtseed_priors(state): 
    """Get thoughtseed prior activations for state (Universal)."""
    return THOUGHTSEED_STATE_PRIORS[state].copy()

def compute_meta_awareness(state, thoughtseed_activations):
    """Compute meta-awareness from weighted thoughtseeds."""
    weights = META_THOUGHTSEED_WEIGHTS.get(state, {})
    weighted_sum = sum(float(thoughtseed_activations.get(ts, 0.0)) * float(w) for ts, w in weights.items())
    weight_total = sum(float(w) for w in weights.values())
    meta = weighted_sum / weight_total if weight_total > 0.0 else 0.0
    return float(max(0.0, min(1.0, meta)))

def get_exit_transition_probs(experience_level, current_state):
    """Get normalized exit probabilities (exclude self-transition)."""
    probs = STATE_TRANSITION_PROBS[experience_level][current_state]
    total = sum(probs.values())
    return {s: p / total for s, p in probs.items()} if total else {s: 1.0 / len(probs) for s in probs}
