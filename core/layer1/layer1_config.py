"""Layer-1 specific configuration: MVOU dynamics, state machine, and generative process costs.
These are internal to Layer-1 and not exposed through the Markov blanket.
"""

# -----------------------------------------------------------------------------
# MVOU Coupling Structure (Theta matrices)
# -----------------------------------------------------------------------------
# Base Multivariate Ornstein-Uhlenbeck coupling terms per state (dX = -Theta * dt).
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
    'redirect_attention': {
        ('DMN', 'DAN'): -0.40, ('DAN', 'DMN'): -0.40,
        ('DMN', 'FPN'): -0.30, ('FPN', 'DMN'): -0.30,
        ('DAN', 'FPN'): 0.40, ('FPN', 'DAN'): 0.40
    }
}

# -----------------------------------------------------------------------------
# Attractor Means (Network activation profiles per state)
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# State Machine Configuration
# -----------------------------------------------------------------------------
# Dwell Times (Seconds) for State Transitions
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

# Exit Transition Probabilities (self-transitions handled by dwell)
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

# -----------------------------------------------------------------------------
# Generative Costs 
# -----------------------------------------------------------------------------
# Layer-1 state-aware generative costs: MW detection and RA reorienting dynamics
L1_GENERATIVE_COSTS = {
    'novice': {
        # MW Detection Parameters
        'mw_detection_alpha': 0.02,          # Burden accumulation rate
        'mw_detection_threshold': 0.11,      # Detection trigger threshold
        'mw_detection_gain': 1.20,           # Hazard sensitivity to burden
        'mw_detection_decay': 0.08,          # Burden decay rate outside MW
        
        # Burden Computation Weights
        'cost_activation_weight': 1.25,      # Weight for activation cost
        'cost_coupling_weight': 0.75,        # Weight for coupling cost
        'cost_activation_scale': 0.02,       # Normalization scale for activation
        'cost_coupling_scale': 0.25,         # Normalization scale for coupling
        
        # RA Reorienting Dynamics
        'ra_bf_pull_strength': 0.30,         # BF attractor pull during RA
        'ra_diffusion_scale': 3.5,           # Noise scaling during RA
        'ra_to_bf_transition_bias': 0.0,     # Explicit transition bias to BF
    },
    'expert': {
        # MW Detection Parameters
        'mw_detection_alpha': 0.05,
        'mw_detection_threshold': 0.09,
        'mw_detection_gain': 0.95,
        'mw_detection_decay': 0.15,
        
        # Burden Computation Weights
        'cost_activation_weight': 0.80,
        'cost_coupling_weight': 0.45,
        'cost_activation_scale': 0.02,
        'cost_coupling_scale': 0.25,
        
        # RA Reorienting Dynamics
        'ra_bf_pull_strength': 2.0,
        'ra_diffusion_scale': 0.5,
        'ra_to_bf_transition_bias': 0.22,
    },
}
