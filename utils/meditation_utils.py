"""Small utility helpers used across the simulation."""

import os
import logging
import numpy as np
from utils.meditation_config import (
    STATE_TRANSITION_PROBS,
    PREFERRED_TRANSITION_BIAS,
    STATES,
    THOUGHTSEED_BASE_ACTIVATIONS,
    META_THOUGHTSEED_WEIGHTS,
    ACTINF_DEFAULTS,
    ACTINF_EXPERT_OVERRIDES,
)

def ensure_directories(base_dir=None):
    """Create `data/` and `plots/` under `base_dir` (or package root)."""
    if not base_dir:
        base_dir = os.getcwd()
    base_dir = os.fspath(base_dir)
    os.makedirs(os.path.join(base_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "plots"), exist_ok=True)
    logging.info("Directories created/verified: data/, plots/")

def _normalize_probs(probs: dict, states: list) -> dict:
    vals = {s: max(0.0, float(probs.get(s, 0.0))) for s in states}
    total = sum(vals.values())
    if total <= 0.0:
        uniform = 1.0 / max(1, len(states))
        return {s: uniform for s in states}
    return {k: v / total for k, v in vals.items()}

def get_exit_transition_probs(experience_level: str, current_state: str) -> dict:
    """Return exit-only transition distribution for current_state."""
    base = STATE_TRANSITION_PROBS.get(experience_level, {}).get(current_state, {})
    return _normalize_probs(base, [s for s in STATES if s != current_state])

def get_preferred_transition_probs(experience_level: str, current_state: str) -> dict:
    """Return preferred next-state distribution P*(s'|s) as normalized base + bias."""
    base = get_exit_transition_probs(experience_level, current_state)
    bias = PREFERRED_TRANSITION_BIAS.get(experience_level, {}).get(current_state, {})
    if not base:
        return {}
    adjusted = {}
    for state, prob in base.items():
        adjusted[state] = max(0.0, float(prob) + float(bias.get(state, 0.0)))
    return _normalize_probs(adjusted, list(adjusted.keys()))

def get_thoughtseed_targets(state, experience_level='novice'):
    """Get target activation values for each thoughtseed in the specified state."""
    level_map = THOUGHTSEED_BASE_ACTIVATIONS.get(experience_level, {})
    return level_map[state].copy()

def compute_meta_awareness(state, thoughtseed_activations):
    """Compute meta-awareness from state-weighted thoughtseed activations."""
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
    return float(np.clip(weighted_sum / weight_total, 0.0, 1.0))

def get_actinf_params(experience_level='novice'):
    params = dict(ACTINF_DEFAULTS)
    if experience_level == 'expert':
        params.update(ACTINF_EXPERT_OVERRIDES)
    return params
