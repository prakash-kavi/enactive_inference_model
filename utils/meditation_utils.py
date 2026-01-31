"""Small utility helpers used across the simulation."""

import os
import logging
import numpy as np
import json
import torch

from utils.meditation_config import (
    STATE_TRANSITION_PROBS,
    PREFERRED_TRANSITION_BIAS,
    STATES,
    THOUGHTSEEDS,
    NETWORKS,
    THOUGHTSEED_BASE_ACTIVATIONS,
    THOUGHTSEED_TARGET_ADJUSTMENTS,
    THOUGHTSEED_LEVEL_OFFSETS,
    META_BASE_AWARENESS,
    META_THOUGHTSEED_INFLUENCES,
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

def get_thoughtseed_targets(state, meta_awareness, experience_level='novice'):
    """Get target activation values for each thoughtseed in the specified state."""
    activations = THOUGHTSEED_BASE_ACTIVATIONS[state].copy()
    for ts in activations:
        meta_mod = THOUGHTSEED_TARGET_ADJUSTMENTS[state][ts]
        activations[ts] += meta_mod * meta_awareness
        level_offset = THOUGHTSEED_LEVEL_OFFSETS.get(experience_level, {}).get(state, {}).get(ts, 0.0)
        activations[ts] += level_offset
    return activations

def compute_meta_awareness(state, thoughtseed_activations):
    """Compute meta-awareness from mediative state and thoughtseed activations."""
    base_awareness = META_BASE_AWARENESS[state]
    awareness_boost = 0
    for ts, influence in META_THOUGHTSEED_INFLUENCES.items():
        if ts in thoughtseed_activations:
            awareness_boost += thoughtseed_activations[ts] * influence
    return base_awareness + awareness_boost

def get_actinf_params(experience_level='novice'):
    params = dict(ACTINF_DEFAULTS)
    if experience_level == 'expert':
        params.update(ACTINF_EXPERT_OVERRIDES)
    return params
