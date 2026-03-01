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

DEFAULT_DT = 0.2  # Step size
BPTT_STEPS = 25   # BPTT window length (steps)
CLIP_MIN = 0.05
CLIP_MAX = 0.9

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
        "novice": {"DMN": 0.40, "VAN": 0.45, "DAN": 0.78, "FPN": 0.72},
        "expert": {"DMN": 0.35, "VAN": 0.40, "DAN": 0.78, "FPN": 0.68}
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
        'breath_focus': (10, 18),
        'mind_wandering': (15, 25),
        'meta_awareness': (5, 10),
        'redirect_attention': (5, 10)
    }
}

# L1 state transition: once dwell has elapsed, hazard is set by transition_drive only.

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
# Layer 2: Thoughtseeds + encoder/decoder
# =============================================================================

# Fixed-step VI hyperparameters (L2)
VI_STEPS = 2
VI_LR = 0.2
# Trigger VI refinement when latent mismatch exceeds this threshold (MSE in z-space)
VI_MISMATCH_THRESHOLD = 0.02

# State-dependent latent noise for stochastic inference (posterior variance proxy)
Z_NOISE_STD_BY_STATE = {
    "breath_focus": 0.02,
    "mind_wandering": 0.08,
    "meta_awareness": 0.03,
    "redirect_attention": 0.03,
}

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

# =============================================================================
# Precision and Active Inference Parameters
# =============================================================================
# Time constant (seconds) for EMA scale of forward surprisal (sigma_fwd^2).
# Meta-awareness smoothing and habit-prior learning are derived from the
# BPTT window; we align precision smoothing to the same timescale.
PRECISION_TAU = BPTT_STEPS * DEFAULT_DT

# =============================================================================
# Optimization Parameters
# =============================================================================
# Optimizer learning rate for model parameters (phi, theta, psi).
LEARNING_RATES = {
    "novice": 0.01,
    "expert": 0.02,
}

def get_policy_candidate_order(current_state: str):
    """Return policy candidate order: [current_state, ...others in STATES order].
    """
    return [current_state] + [s for s in STATES if s != current_state]

def get_exit_transition_probs(experience_level, current_state):
    """Return exit transition probabilities for the given state (rows sum to 1.0)."""
    return dict(STATE_TRANSITION_PROBS[experience_level][current_state])

