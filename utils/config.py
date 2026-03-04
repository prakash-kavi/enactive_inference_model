"""Consolidated configuration for the lean meditation model.

Single source of truth for constants used across training, simulation, and plotting.

We distinguish between \"phenomenological\" parameters (which define the expert/novice
profiles) and \"technical\" hyperparameters (which define numerical behaviour). This
file is the only place where either should be changed.
"""

# ---------------------------------------------------------------------------
# Phenomenological parameters (states, networks, priors, dwell/transition)
# ---------------------------------------------------------------------------
STATES = ['breath_focus', 'mind_wandering', 'meta_awareness', 'redirect_attention']
NETWORKS = ['DMN', 'VAN', 'DAN', 'FPN']
THOUGHTSEEDS = ['attend_breath', 'pain_discomfort', 'pending_tasks', 'aha_moment', 'equanimity']

THETA_BASE = {
    'breath_focus': {
        ('DMN', 'DAN'): 0.40, ('DAN', 'DMN'): 0.60,
        ('DAN', 'FPN'): 0.15, ('FPN', 'DAN'): 0.15
    },
    'mind_wandering': {
        ('DMN', 'VAN'): -0.30, ('VAN', 'DMN'): -0.30,
        ('DMN', 'FPN'): -0.15, ('FPN', 'DMN'): -0.15
    },
    'meta_awareness': {
        ('VAN', 'FPN'): 0.50, ('FPN', 'VAN'): 0.50,
        ('DMN', 'DAN'): -0.25, ('DAN', 'DMN'): -0.25,
        ('DMN', 'FPN'): -0.25, ('FPN', 'DMN'): -0.25
    },
    'redirect_attention': {
        ('DMN', 'DAN'): -0.40, ('DAN', 'DMN'): -0.40,
        ('DMN', 'FPN'): -0.20, ('FPN', 'DMN'): -0.40,
        ('DAN', 'FPN'): 0.40, ('FPN', 'DAN'): 0.40
    }
}

NETWORK_PROFILES = {
    "breath_focus": {
        "novice": {"DMN": 0.50, "VAN": 0.45, "DAN": 0.58, "FPN": 0.60},
        "expert": {"DMN": 0.35, "VAN": 0.45, "DAN": 0.65, "FPN": 0.70}
    },
    "mind_wandering": {
        "novice": {"DMN": 0.82, "VAN": 0.35, "DAN": 0.30, "FPN": 0.33},
        "expert": {"DMN": 0.70, "VAN": 0.40, "DAN": 0.28, "FPN": 0.38}
    },
    "meta_awareness": {
        "novice": {"DMN": 0.45, "VAN": 0.85, "DAN": 0.42, "FPN": 0.56},
        "expert": {"DMN": 0.38, "VAN": 0.85, "DAN": 0.42, "FPN": 0.60}
    },
    "redirect_attention": {
        "novice": {"DMN": 0.40, "VAN": 0.45, "DAN": 0.78, "FPN": 0.72},
        "expert": {"DMN": 0.30, "VAN": 0.40, "DAN": 0.82, "FPN": 0.72}
    }
}

DWELL_TIMES = {
    'expert': {
        'breath_focus': (15, 25),
        'mind_wandering': (10, 18),
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
        'meta_awareness': {'redirect_attention': 0.60, 'breath_focus': 0.10, 'mind_wandering': 0.30},
        'redirect_attention': {'breath_focus': 0.40, 'mind_wandering': 0.40, 'meta_awareness': 0.20}
    }
}

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

# ---------------------------------------------------------------------------
# Technical hyperparameters (timescales, learning rates, numerics)
# ---------------------------------------------------------------------------
DEFAULT_DT = 0.2
BPTT_STEPS = 25

TRAIN_STEPS = 8000
EVAL_STEPS = 2000
PLOT_STEPS = 2000
TOTAL_STEPS = TRAIN_STEPS + EVAL_STEPS + PLOT_STEPS

CLIP_MIN = 0.05
CLIP_MAX = 0.9

EPS = 1e-6

NOISE_LEVEL = 0.002

VI_STEPS = 2
VI_LR = 0.2

STATE_BELIEF_VAR = 0.1

PRECISION_TAU = BPTT_STEPS * DEFAULT_DT

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
