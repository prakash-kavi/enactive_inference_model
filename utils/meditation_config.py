"""Configuration for the Vipassana meditation simulation.
Defines thoughtseed/network profiles, mediative states, and tunable parameters.
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# Common definitions
# -----------------------------------------------------------------------------
EXPERIENCE_LEVELS = ['novice', 'expert']
STATES = ['breath_focus', 'mind_wandering', 'meta_awareness', 'redirect_attention']
NETWORKS = ['DMN', 'VAN', 'DAN', 'FPN']
THOUGHTSEEDS = ['attend_breath', 'pain_discomfort', 'pending_tasks', 'aha_moment', 'equanimity']

DEFAULTS = {
    'ACTIVATION_CLIP_MIN': 0.05,
    'ACTIVATION_CLIP_MAX': 0.9,
    'DEFAULT_DT': 0.2,
    'REFRACTORY_SEC': 0.4,
}

# Numerical stability epsilon
EPS = 1e-6

# -----------------------------------------------------------------------------
# Layer 2: Thoughtseed priors and meta-awareness mapping
# -----------------------------------------------------------------------------
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
            "pain_discomfort": 1.05,
            "pending_tasks": 1.1,
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

# Meta-awareness: state-aware weighting over thoughtseed activations.
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

# -----------------------------------------------------------------------------
# Layer 2 / 3: Active inference parameters
# -----------------------------------------------------------------------------
ACTINF_DEFAULTS = {
    "learning_rate": 0.01,
    "z_noise_sigma": 0.05,
    "vfe_ema_alpha": 0.9,
    "kl_beta": 1.0,
    "l2_vi_steps": 2,
    "l2_vi_lr": 0.24,
    "l2_vi_obs_weight": 1.0,
    "l2_vi_prior_weight": 1.0,
    "l2_vi_sensory_weight": 0.35,
    "l2_vi_temporal_weight": 0.12,
    "l2_vi_grad_clip": 5.0,
    "efe_ambiguity_weight": 0.4,
    "efe_cycle_strength": 0.35,
    "efe_gain": 0.35,
    "policy_horizon": 1,
    "policy_temperature": 1.0,
    "policy_temperature_by_state": {
        "breath_focus": 1.8,
        "mind_wandering": 1.15,
        "meta_awareness": 0.95,
        "redirect_attention": 1.0,
    },
    "policy_horizon_discount": 0.6,
    "l3tol2_precision_range": (0.4, 0.6),
    "network_target_reg": 0.05,
}

ACTINF_EXPERT_OVERRIDES = {
    "learning_rate": 0.02,
    "efe_ambiguity_weight": 0.35,
    "kl_beta": 0.55,
    "z_noise_sigma": 0.02,
    "policy_temperature_by_state": {
        "breath_focus": 1.05,
        "mind_wandering": 0.95,
        "meta_awareness": 0.85,
        "redirect_attention": 0.9,
    },
}
