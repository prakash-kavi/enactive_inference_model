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
        "expert": {"DMN": 0.40, "VAN": 0.78, "DAN": 0.42, "FPN": 0.55}
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
        'meta_awareness': (1, 4),
        'redirect_attention': (1, 4)
    },
    'novice': {
        'breath_focus': (5, 15),
        'mind_wandering': (20, 40),
        'meta_awareness': (2, 6),
        'redirect_attention': (3, 8)
    }
}

# Exit transition probabilities (exclude self-transitions)
STATE_TRANSITION_PROBS = {
    'expert': {
        'breath_focus': {'mind_wandering': 0.35, 'meta_awareness': 0.45, 'redirect_attention': 0.20},
        'mind_wandering': {'breath_focus': 0.18, 'meta_awareness': 0.56, 'redirect_attention': 0.26},
        'meta_awareness': {'redirect_attention': 0.85, 'breath_focus': 0.14, 'mind_wandering': 0.01},
        'redirect_attention': {'breath_focus': 0.59, 'meta_awareness': 0.34, 'mind_wandering': 0.07},
    },
    'novice': {
        'breath_focus': {'mind_wandering': 0.90, 'meta_awareness': 0.08, 'redirect_attention': 0.02},
        'mind_wandering': {'meta_awareness': 0.68, 'redirect_attention': 0.17, 'breath_focus': 0.15},
        'meta_awareness': {'redirect_attention': 0.79, 'breath_focus': 0.15, 'mind_wandering': 0.06},
        'redirect_attention': {'breath_focus': 0.88, 'mind_wandering': 0.07, 'meta_awareness': 0.05},
    }
}

# =============================================================================
# Layer 2: Thoughtseeds + VAE
# =============================================================================

# Thoughtseed priors (state-dependent activation baselines)
THOUGHTSEED_BASE_ACTIVATIONS = {
    "novice": {
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
            "equanimity": 0.3,
            "pain_discomfort": 0.1,
            "pending_tasks": 0.1,
            "aha_moment": 0.9
        },
        "redirect_attention": {
            "attend_breath": 0.7,
            "equanimity": 0.85,
            "pain_discomfort": 0.12,
            "pending_tasks": 0.05,
            "aha_moment": 0.35
        }
    },
    "expert": {
        "breath_focus": {
            "attend_breath": 0.85,
            "equanimity": 0.45,
            "pain_discomfort": 0.1,
            "pending_tasks": 0.05,
            "aha_moment": 0.15
        },
        "mind_wandering": {
            "attend_breath": 0.05,
            "equanimity": 0.1,
            "pain_discomfort": 0.95,  # High attractor (was 1.05, clipped to preserve distinction)
            "pending_tasks": 1.0,      # Strongest attractor (was 1.1, clipped)
            "aha_moment": 0.05
        },
        "meta_awareness": {
            "attend_breath": 0.25,
            "equanimity": 0.35,
            "pain_discomfort": 0.1,
            "pending_tasks": 0.1,
            "aha_moment": 0.95
        },
        "redirect_attention": {
            "attend_breath": 0.7,
            "equanimity": 0.85,
            "pain_discomfort": 0.12,
            "pending_tasks": 0.05,
            "aha_moment": 0.35
        }
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

ACTINF_PARAMS = {
    "novice": {
        # Training
        "learning_rate": 0.01,
        "kl_beta": 1.0,
        "init_noise_sigma": 0.05,
        
        # L2 inference
        "l2_vi_steps": 2,
        "l2_vi_lr": 0.24,
        "l2_vi_obs_weight": 1.0,
        "l2_vi_prior_weight": 1.0,
        "l2_vi_sensory_weight": 0.35,
        "l2_vi_temporal_weight": 0.12,
        
        # L3 policy
        "efe_ambiguity_weight": 0.4,
        "efe_cycle_strength": 0.35,
        "policy_temperature_by_state": {
            "breath_focus": 1.8,
            "mind_wandering": 1.15,
            "meta_awareness": 0.95,
            "redirect_attention": 1.0,
        },
        "l3tol2_precision_range": (0.4, 0.6),
    },
    "expert": {
        # Training
        "learning_rate": 0.02,
        "kl_beta": 0.55,
        "init_noise_sigma": 0.02,
        
        # L2 inference
        "l2_vi_steps": 2,
        "l2_vi_lr": 0.24,
        "l2_vi_obs_weight": 1.0,
        "l2_vi_prior_weight": 1.0,
        "l2_vi_sensory_weight": 0.35,
        "l2_vi_temporal_weight": 0.12,
        
        # L3 policy
        "efe_ambiguity_weight": 0.35,
        "efe_cycle_strength": 0.35,
        "policy_temperature_by_state": {
            "breath_focus": 1.05,
            "mind_wandering": 0.95,
            "meta_awareness": 0.85,
            "redirect_attention": 0.9,
        },
        "l3tol2_precision_range": (0.4, 0.6),
    }
}

# Phase 4: Forward model action loss weighting
FORWARD_LOSS_BASE_WEIGHT = 0.05
FORWARD_LOSS_PRECISION_SCALE = 0.1  # Modulated by L3 precision

# =============================================================================
# Utility Functions
# =============================================================================

def get_params(experience_level):
    """Get all parameters for experience level."""
    return ACTINF_PARAMS[experience_level].copy()

def get_thoughtseed_priors(state, experience_level):
    """Get thoughtseed prior activations for state."""
    return THOUGHTSEED_BASE_ACTIVATIONS[experience_level][state].copy()

def compute_meta_awareness(state, thoughtseed_activations):
    """Compute meta-awareness from weighted thoughtseeds."""
    weights = META_THOUGHTSEED_WEIGHTS.get(state, {})
    if not weights:
        return 0.0
    
    weighted_sum = 0.0
    weight_total = 0.0
    for ts, weight in weights.items():
        weight_total += float(weight)
        weighted_sum += float(thoughtseed_activations.get(ts, 0.0)) * float(weight)
    
    if weight_total <= 0.0:
        return 0.0
    return float(max(0.0, min(1.0, weighted_sum / weight_total)))

def get_exit_transition_probs(experience_level, current_state):
    """Get normalized exit probabilities (exclude self-transition)."""
    probs = STATE_TRANSITION_PROBS[experience_level][current_state].copy()
    total = sum(probs.values())
    if total <= 0.0:
        # Uniform fallback
        return {s: 1.0 / len(probs) for s in probs}
    return {s: p / total for s, p in probs.items()}
