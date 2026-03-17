"""Core invariant tests for the enactive inference model.

Covers: probability normalization, VFE finiteness, transition matrix row sums,
state belief normalization, and numerical stability of softmax/entropy/OU.
"""

import numpy as np
import torch
import pytest

from utils.config import (
    STATES, NETWORKS, THOUGHTSEEDS, CLIP_MIN, CLIP_MAX, EPS,
    VI_STEPS, GRAD_CLIP, THETA_MW_DIAG, THETA_DEFAULT_DIAG,
)
from utils.math_utils import (
    softmax, policy_posterior, normalize_belief, belief_entropy, policy_entropy,
    ou_step_scalar, clip_probability,
)
from model.phenotype import EXPERT_PHENOTYPE, NOVICE_PHENOTYPE
from model.l2_recognition import Layer2Agent
from model.l3_metacognition import Layer3Monitor
from model.markov_blankets import MarkovBlanketL1L2, MarkovBlanketL2L3


# ---------------------------------------------------------------------------
# Probability / normalization invariants
# ---------------------------------------------------------------------------

class TestProbabilityNormalization:

    def test_policy_posterior_sums_to_one(self):
        log_prior = np.log(np.array([0.4, 0.3, 0.2, 0.1]))
        g_vals = np.array([0.5, 1.2, 0.3, 0.8])
        q = policy_posterior(log_prior, g_vals, gamma=0.7)
        assert abs(q.sum() - 1.0) < 1e-6
        assert np.all(q >= 0.0)

    def test_softmax_sums_to_one(self):
        for logits in [np.zeros(4), np.array([1e10, -1e10, 0, 0]), np.array([-100.0]*4)]:
            result = softmax(logits)
            assert abs(result.sum() - 1.0) < 1e-6

    def test_normalize_belief_sums_to_one(self):
        belief = {'a': 2.0, 'b': 3.0, 'c': 1.0}
        result = normalize_belief(belief)
        assert abs(result.sum() - 1.0) < 1e-6

    def test_normalize_belief_degenerate_returns_uniform(self):
        result = normalize_belief(None, keys=STATES)
        assert abs(result.sum() - 1.0) < 1e-6
        assert np.allclose(result, 1.0 / len(STATES))

    def test_state_belief_from_l2_sums_to_one(self):
        agent = Layer2Agent(phenotype=EXPERT_PHENOTYPE)
        z = torch.full((len(THOUGHTSEEDS),), 0.5)
        belief = agent.infer_state_belief(z)
        total = sum(belief.values())
        assert abs(total - 1.0) < 1e-5
        assert all(v >= 0.0 for v in belief.values())


# ---------------------------------------------------------------------------
# VFE / free energy finiteness
# ---------------------------------------------------------------------------

class TestVFEFiniteness:

    def test_vfe_is_finite_for_normal_inputs(self):
        agent = Layer2Agent(phenotype=EXPERT_PHENOTYPE)
        for state in STATES:
            z = torch.rand(len(THOUGHTSEEDS)) * (CLIP_MAX - CLIP_MIN) + CLIP_MIN
            x = torch.rand(len(NETWORKS)) * (CLIP_MAX - CLIP_MIN) + CLIP_MIN
            vfe = agent.compute_vfe(state=state, z=z, observed_x=x)
            assert torch.isfinite(vfe), f"VFE non-finite for state={state}"
            assert vfe.item() >= 0.0

    def test_vfe_is_finite_for_boundary_inputs(self):
        agent = Layer2Agent(phenotype=EXPERT_PHENOTYPE)
        z = torch.full((len(THOUGHTSEEDS),), CLIP_MIN)
        x = torch.full((len(NETWORKS),), CLIP_MAX)
        vfe = agent.compute_vfe(state=STATES[0], z=z, observed_x=x)
        assert torch.isfinite(vfe)


# ---------------------------------------------------------------------------
# Transition matrix row sums
# ---------------------------------------------------------------------------

class TestTransitionMatrix:

    def test_transition_probs_row_sums(self):
        from utils.config import STATE_TRANSITION_PROBS
        for level in ['expert', 'novice']:
            for state, row in STATE_TRANSITION_PROBS[level].items():
                total = sum(row.values())
                assert abs(total - 1.0) < 1e-6, \
                    f"{level}/{state} transition probs sum to {total}, not 1.0"


# ---------------------------------------------------------------------------
# Numerical stability: softmax, entropy, OU
# ---------------------------------------------------------------------------

class TestNumericalStability:

    @pytest.mark.parametrize("logits", [
        np.array([1e15, -1e15, 0.0, 0.0]),
        np.array([-1e15] * 4),
        np.zeros(4),
    ])
    def test_softmax_extreme_inputs(self, logits):
        result = softmax(logits)
        assert np.all(np.isfinite(result))
        assert abs(result.sum() - 1.0) < 1e-6

    @pytest.mark.parametrize("belief", [
        {'a': 0.0, 'b': 0.0},       # all-zero → uniform
        {'a': 1e-15, 'b': 1e-15},   # near-zero
        {'a': 1.0, 'b': 0.0},       # degenerate
    ])
    def test_belief_entropy_finite(self, belief):
        result = belief_entropy(belief)
        assert np.isfinite(result)
        assert result >= 0.0

    def test_policy_entropy_finite(self):
        for probs in [None, [0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.25]*4]:
            result = policy_entropy(probs)
            assert np.isfinite(result)
            assert result >= 0.0

    def test_ou_step_stays_in_bounds(self):
        value = torch.tensor([CLIP_MIN, CLIP_MAX, 0.5, 0.5, 0.5])
        target = torch.tensor([0.5] * 5)
        for _ in range(100):
            value = ou_step_scalar(value, target, dt=0.2, tau=5.0,
                                   noise_level=0.002, clip_min=CLIP_MIN, clip_max=CLIP_MAX)
        assert torch.all(value >= CLIP_MIN)
        assert torch.all(value <= CLIP_MAX)
        assert torch.all(torch.isfinite(value))


# ---------------------------------------------------------------------------
# Config constants sanity
# ---------------------------------------------------------------------------

class TestConfigSanity:

    def test_thoughtseeds_order_matches_expected(self):
        expected = ['attend_breath', 'pain_discomfort', 'pending_tasks', 'aha_moment', 'equanimity']
        assert THOUGHTSEEDS == expected, \
            "THOUGHTSEEDS order changed — update l3_metacognition.py named lookups if intentional"

    def test_states_count_consistent(self):
        assert len(STATES) == 4

    def test_grad_clip_positive(self):
        assert GRAD_CLIP > 0.0

    def test_theta_diag_constants(self):
        assert 0 < THETA_DEFAULT_DIAG < THETA_MW_DIAG <= 1.0
