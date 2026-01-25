"""Core meditation models: AgentConfig and ActInfAgent."""

import numpy as np
import os
import json
import logging
from collections import defaultdict
import copy
from typing import Optional, Dict, List, Tuple, Any

from config.meditation_config import (
    THOUGHTSEEDS, STATES,
    ActInfParams, ThoughtseedParams, MetacognitionParams,
    NETWORK_PROFILES, DEFAULTS, NETWORK_MODULATION
)
from meditation_utils import ou_update, clip_array, LeakyAccumulator

class AgentConfig:
    """Base class for thoughtseed dynamics and mediative state handling.
    Handles initialization of history tracking, parameter loading, and 
    stochastic generation of dwell times and target activations.
    """
    
    def __init__(self, experience_level: str = 'novice', timesteps_per_cycle: int = 1000, seed: Optional[int] = None):
        self.experience_level = experience_level
        self.timesteps = timesteps_per_cycle
        self.thoughtseeds = THOUGHTSEEDS
        self.states = STATES
        self.num_thoughtseeds = len(self.thoughtseeds)
        
        # State tracking
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        self.natural_transition_count = 0
        self.forced_transition_count = 0
        
        # History tracking
        self.activations_history = []
        self.state_history = []
        self.meta_awareness_history = []
        self.dominant_ts_history = []
        
        # Load per-agent params from dataclass defaults.
        self.params = ActInfParams.expert() if experience_level == 'expert' else ActInfParams.novice()
        # Copy params into explicit agent attributes (preserve `self.params`).
        for k, v in vars(self.params).items():
            if not hasattr(self, k):
                setattr(self, k, v)

        # Agent RNG (RandomState)
        self.rng = np.random.RandomState(seed)
        # Track activation patterns at transition points
        self.transition_activations = {state: [] for state in self.states}
        
        # Track distraction buildup patterns
        self.distraction_buildup_rates = []
        
        # Initialize accumulators (Leaky Integrators) - AFTER params are loaded
        # Note: dmn_accumulator and fpn_accumulator removed (dynamics now in generative_process)
        self.aha_accumulator = LeakyAccumulator(decay=self.aha_accum_decay, gain=self.aha_accum_inc, activation='linear')  # Linear to avoid double-sigmoid
        self.vfe_accumulator = LeakyAccumulator(decay=self.vfe_accum_decay, gain=1.0 - self.vfe_accum_decay, activation='linear')
        # Track last observed networks (from generative process) for modulation
        self.prev_network_acts: Optional[Dict[str, float]] = None


    def get_target_activations(self, state: str, meta_awareness: float) -> np.ndarray:
        """Calculate target thoughtseed activations for a given mediative state.
        Delegates to `ThoughtseedParams` for base values and applies agent-specific noise.
        """
        targets_dict = ThoughtseedParams.get_target_activations(
            state, meta_awareness, self.experience_level)
        
        target_activations = np.zeros(self.num_thoughtseeds)
        for i, ts in enumerate(self.thoughtseeds):
            target_activations[i] = targets_dict[ts]
        
        # Uses agent RNG
        target_activations += self.rng.normal(0, self.noise_level, size=self.num_thoughtseeds)
        return np.clip(target_activations, 0.05, 1.0)


    def get_meta_awareness(self, current_state: str, activations: np.ndarray) -> float:
        """Compute meta-awareness from thoughtseed activations and current mediative state.
        Delegates to `MetacognitionParams`.
        """
        activations_dict = {ts: activations[i] for i, ts in enumerate(self.thoughtseeds)}
        
        return MetacognitionParams.calculate_meta_awareness(
            state=current_state,
            thoughtseed_activations=activations_dict,
            experience_level=self.experience_level
        )

class ActInfAgent(AgentConfig):
    """Active Inference agent implementing Layers 2 & 3 (Generative Model + Recognition/Inference).
    
    This class models the agent's internal model:
    1. Generative Model (Layer 2): Predicts network activations from thoughtseeds and state
    2. Recognition/Inference (Layer 3): Computes prediction errors, VFE, and learns
    
    NOTE: Network generation is now handled by generative_process.py (Layer 1).
    The agent receives observations from the process and predicts/infers.
    """
    
    def __init__(self, experience_level: str = 'novice', timesteps_per_cycle: int = 200):
        super().__init__(experience_level, timesteps_per_cycle)
        
        self.networks = ['DMN', 'VAN', 'DAN', 'FPN']
        # Track observed networks (from generative process)
        self.network_activations_history = []
        self.free_energy_history = []
        self.prediction_error_history = []
        self.precision_history = []
        
        # SCIENTIFIC MATRIX FORM: The Weight Matrix (W)
        # Rows: Thoughtseeds [attend_breath, pain_discomfort, pending_tasks, aha_moment, equanimity]
        # Cols: Networks [DMN, VAN, DAN, FPN]
        # Aligned with "Neural Efficiency" and empirical targets
        self.W = np.array([
            [0.20, 0.30, 0.85, 0.50],  # attend_breath: Anchor focus (high DAN, moderate FPN)
            [0.40, 0.75, 0.30, 0.40],  # pain_discomfort: Sensory distraction (high VAN)
            [0.85, 0.40, 0.20, 0.30],  # pending_tasks: Internal narrative (high DMN)
            [0.30, 0.80, 0.40, 0.70],  # aha_moment: State recognition (high VAN, high FPN)
            [0.25, 0.30, 0.50, 0.90]   # equanimity: Executive regulation (very high FPN)
        ], dtype=np.float64)
        
        # Track learned network profiles (generative model parameters)
        # NOTE: W matrix is now fixed (scientific form), but state expectations are still learned
        self.learned_network_profiles = {
            "state_network_expectations": {state: {} for state in self.states}
        }
        
        self.prev_meta_awareness = 0.0
        self.aha_drive = 0.0  # Standardized access for Aha signal
        
        # Metrics tracking for algorithmic framework
        self.neural_efficiency_history = []
        self.expert_mind_wandering_detections = 0
        self.stability_indicators = []
        self.van_spike_detections = 0
        self.van_history = []  # Track VAN for spike detection
        
        # Initialize state expectations with default profiles
        for state in self.states:
            self.learned_network_profiles["state_network_expectations"][state] = NETWORK_PROFILES["state_expected_profiles"][state][self.experience_level].copy()
        self._state_expect_vectors = {
            state: self._build_state_expect_vector(state) for state in self.states
        }
        
        # State abbreviation mapping (generative_process uses BF/MW/MA/RA)
        self._state_abbrev_map = {
            'BF': 'breath_focus', 'MW': 'mind_wandering',
            'MA': 'meta_awareness', 'RA': 'redirect_breath'
        }
        self._state_full_to_abbrev = {v: k for k, v in self._state_abbrev_map.items()}

    def _build_state_expect_vector(self, state: str) -> np.ndarray:
        """Build a dense vector of network expectations for a given state."""
        expect = self.learned_network_profiles["state_network_expectations"][state]
        return np.array([expect[net] for net in self.networks])

    def map_state_abbrev_to_full(self, abbrev_state: str) -> str:
        """Map state abbreviation (BF/MW/MA/RA) to full state name."""
        return self._state_abbrev_map.get(abbrev_state, abbrev_state)
    
    def map_state_full_to_abbrev(self, full_state: str) -> str:
        """Map full state name to abbreviation (BF/MW/MA/RA)."""
        return self._state_full_to_abbrev.get(full_state, full_state)

    def compute_generative_predictions(self, thoughtseed_activations: np.ndarray, current_state: str, meta_awareness: float) -> Dict[str, float]:
        """
        Layer 2 (Generative Model): μ_pred = z · W
        
        The agent's 'world model' predicting network activity from latent thoughtseeds.
        Uses the Scientific Matrix Form: single dot product replaces dictionary lookups.
        
        Args:
            thoughtseed_activations: z vector (5×1) of thoughtseed strengths
            current_state: Mediative state (full name or abbreviation)
            meta_awareness: Meta-awareness level (modulates state expectation bias)
        
        Returns:
            Dictionary of predicted network activations {DMN, VAN, DAN, FPN}
        """
        # Normalize state name (handle both full and abbrev)
        if current_state in self._state_abbrev_map:
            current_state = self.map_state_abbrev_to_full(current_state)
        
        # 1. Prediction based strictly on latent thoughtseed strength
        # μ_pred is a 1×4 vector of [DMN, VAN, DAN, FPN]
        mu_pred = thoughtseed_activations @ self.W
        
        # 2. Apply Top-Down State Expectation Bias
        # In Active Inference, the 'State' is a prior that modulates the likelihood
        state_expect_vec = self._state_expect_vectors[current_state]
        
        # Meta-awareness determines how much the 'Internal Map' weights expectations vs. seeds
        meta_factor = meta_awareness * self.expert_meta_scalar
        
        # Balanced prediction: 50% Thoughtseed profile + 50% State-contextual bias
        # This prevents the agent from 'hallucinating' focus when distraction seeds are high
        final_prediction = (0.5 * mu_pred) + (0.5 * state_expect_vec * meta_factor)
            
        return dict(zip(self.networks, final_prediction))
    
    def compute_prediction_errors(self, predicted: Dict[str, float], observed: Dict[str, float]) -> Dict[str, float]:
        """
        Prediction Error: δ = observed - predicted
        Explicit error signal for Active Inference learning.
        """
        return {net: observed[net] - predicted[net] for net in self.networks}
    
    
    def get_sensory_inference(self, network_acts: Dict[str, float]) -> np.ndarray:
        """
        Layer 3: Recognition/Inference.
        
        Infers thoughtseed beliefs (z) by inverting the generative mapping.
        Uses matrix-transpose projection: z_inferred = (network_obs · W^T) / row_sums(W)
        
        This allows Layer 3 to recognize patterns:
        - VAN spike (0.80) → high aha_moment activation (row has 0.80 in VAN)
        - High FPN distinguishes aha_moment (0.70) from pain_discomfort (0.40)
        
        Args:
            network_acts: Observed network activations {DMN, VAN, DAN, FPN}
        
        Returns:
            Inferred thoughtseed activations (5×1 vector)
        """
        # Convert dictionary observation to vector
        net_vec = np.array([network_acts.get(net, 0.0) for net in self.networks])
        
        # Sensory Inference (A-Matrix inversion): 
        # Project network data back onto the Thoughtseed space
        # Normalized by the row-sums of W to ensure activation scales are preserved
        match_scores = net_vec @ self.W.T  # (1×4) @ (4×5) = (1×5)
        row_sums = np.sum(self.W, axis=1)  # Sum across networks for each thoughtseed
        inferred = np.where(row_sums > 0, match_scores / row_sums, 0.1)
                
        return clip_array(inferred, DEFAULTS['ACTIVATION_CLIP_MIN'], DEFAULTS['ACTIVATION_CLIP_MAX'])

    def calculate_vfe(self, current_seeds: np.ndarray, prior_seeds: np.ndarray, sensory_inference: np.ndarray, meta_awareness: float, vfe_trend: float = 0.0) -> Tuple[float, float, float]:
        """Compute Variational Free Energy (VFE).
        VFE = (Sensory NLL * Sensory Precision) + (Prior NLL * Prior Precision)
        Minimizing VFE maximizes the evidence for the agent's internal model.
        """
        # Sensory NLL (Accuracy)
        sensory_nll = np.sum(
            sensory_inference * np.log(sensory_inference / (current_seeds + 1e-9)) + 
            (1 - sensory_inference) * np.log((1 - sensory_inference) / (1 - current_seeds + 1e-9))
        )
        
        # Prior NLL (Complexity)
        prior_nll = np.sum(
            current_seeds * np.log(current_seeds / (prior_seeds + 1e-9)) + 
            (1 - current_seeds) * np.log((1 - current_seeds) / (1 - prior_seeds + 1e-9))
        )
        
        # Precision modulation based on VFE history (Attention)
        precision_mod = np.clip(-1.0 * vfe_trend, -0.3, 0.3)

        # Sensory precision scales with VAN proxy (Salience) AND Aha Drive
        van_proxy = sensory_inference[self.thoughtseeds.index('aha_moment')]
        
        # Option C: Aha Drive strongly boosts precision
        aha_precision_boost = 1.0 + (self.aha_vfe_gain * self.aha_drive)
        
        pi_sensory = (self.sensory_precision_base + (self.sensory_precision_van_scalar * van_proxy)) * (1.0 + precision_mod) * self.precision_weight * aha_precision_boost
        
        # Prior precision scales with meta-awareness (Top-down control)
        pi_prior = (self.prior_precision_base + (self.prior_precision_meta_scalar * meta_awareness)) * (1.0 + precision_mod) * self.complexity_penalty
        
        # Total VFE
        vfe = (sensory_nll * pi_sensory) + (prior_nll * pi_prior)
        
        return vfe, sensory_nll, prior_nll
   
    def update_network_profiles(self, thoughtseed_activations: np.ndarray, network_activations: Dict[str, float], current_state: str, prediction_errors: Dict[str, float]):
        """Update learned mappings (generative model) based on prediction errors.
        
        NOTE: W matrix is now fixed (Scientific Matrix Form). Only state expectations are learned.
        This preserves the mathematically precise form while allowing state-specific learning.
        
        Implements a Hebbian-like associative learning rule modulated by precision.
        """
        if len(self.network_activations_history) < 10:
            return
        
        # W matrix is fixed - no updates to thoughtseed contributions
        # Learning now focuses only on state expectations (which are state-specific)
        
        # Update mediative state expectations (slower rate)
        # Use empirical targets as anchors to prevent coupling effects from biasing learning
        slow_rate = self.learning_rate * 0.3
        
        # Empirical targets from ACTIVE_INFERENCE_FRAMEWORK.md (user's table)
        empirical_targets = {
            "breath_focus": {
                "expert": {"DMN": 0.30, "VAN": 0.45, "DAN": 0.60, "FPN": 0.45},
                "novice": {"DMN": 0.55, "VAN": 0.50, "DAN": 0.60, "FPN": 0.65}
            },
            "mind_wandering": {
                "expert": {"DMN": 0.60, "VAN": 0.65, "DAN": 0.40, "FPN": 0.65},
                "novice": {"DMN": 0.75, "VAN": 0.40, "DAN": 0.35, "FPN": 0.40}
            },
            "meta_awareness": {
                "expert": {"DMN": 0.40, "VAN": 0.80, "DAN": 0.50, "FPN": 0.70},
                "novice": {"DMN": 0.50, "VAN": 0.60, "DAN": 0.50, "FPN": 0.55}
            },
            "redirect_breath": {
                "expert": {"DMN": 0.35, "VAN": 0.55, "DAN": 0.65, "FPN": 0.55},
                "novice": {"DMN": 0.45, "VAN": 0.50, "DAN": 0.65, "FPN": 0.70}
            }
        }
        
        for net in self.networks:
            current_expect = self.learned_network_profiles["state_network_expectations"][current_state][net]
            observed = network_activations[net]
            empirical_target = empirical_targets.get(current_state, {}).get(self.experience_level, {}).get(net, None)
            
            if empirical_target is not None:
                # Use empirical target as anchor - blend between current expectation, observed, and target
                # Weight: 70% current, 15% observed, 15% empirical target
                # This prevents runaway while allowing learning from observations
                new_value = (0.70 * current_expect + 
                            0.15 * observed + 
                            0.15 * empirical_target)
                
                # Allow some flexibility but keep close to empirical target
                # Clip to ±0.15 of empirical target
                new_value = np.clip(new_value, 
                                  max(0.05, empirical_target - 0.15),
                                  min(0.95, empirical_target + 0.15))
            else:
                # No empirical target - use standard update (shouldn't happen)
                pred_error = prediction_errors[net]
                update_factor = 0.1 if abs(pred_error) > 0.2 else 1.0
                new_value = (1 - slow_rate * update_factor) * current_expect + (slow_rate * update_factor) * observed
            
            self.learned_network_profiles["state_network_expectations"][current_state][net] = new_value
        self._state_expect_vectors[current_state] = self._build_state_expect_vector(current_state)
    
    def get_network_modulation(self, network_acts: Dict[str, float], current_state: str) -> Dict[str, float]:
        """Calculate how current network activity modulates thoughtseed targets.
        Example: High DMN activity increases 'pending_tasks' and 'aha_moment' (mind-wandering).
        """
        modulations = {ts: 0.0 for ts in self.thoughtseeds}
        
        mods = NETWORK_MODULATION

        dmn_strength = network_acts.get('DMN', 0)
        modulations['pending_tasks'] += mods['DMN']['pending_tasks'] * dmn_strength
        modulations['aha_moment'] += mods['DMN']['aha_moment'] * dmn_strength
        modulations['attend_breath'] += mods['DMN']['attend_breath'] * dmn_strength

        van_strength = network_acts.get('VAN', 0)
        modulations['pain_discomfort'] += mods['VAN']['pain_discomfort'] * van_strength
        if current_state == "meta_awareness":
            modulations['aha_moment'] += mods['VAN']['aha_moment_meta_awareness'] * van_strength

        dan_strength = network_acts.get('DAN', 0)
        modulations['attend_breath'] += mods['DAN']['attend_breath'] * dan_strength
        modulations['pending_tasks'] += mods['DAN']['pending_tasks'] * dan_strength
        modulations['pain_discomfort'] += mods['DAN']['pain_discomfort'] * dan_strength
        
        fpn_strength = network_acts.get('FPN', 0)
        fpn_enhancement = self.fpn_enhancement
        modulations['aha_moment'] += mods['FPN']['aha_moment'] * fpn_strength * fpn_enhancement
        modulations['equanimity'] += mods['FPN']['equanimity'] * fpn_strength * fpn_enhancement
                
        return modulations

    def get_transition_probabilities(self, activations: np.ndarray, network_acts: Dict[str, float], targets_by_state: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate probability of transitioning to each mediative state.
        Combines two evidence sources:
        1. Network Similarity: How close current networks are to mediative state expectations.
        2. Activation Similarity: How close current thoughtseeds are to mediative state targets.
        """
        scores = {}
        temp = max(1e-6, self.softmax_temperature)

        w_net = self.transition_weight_network
        w_act = self.transition_weight_activation

        net_vec = np.array([network_acts.get(net, 0.0) for net in self.networks])
        for state in self.states:
            # Network expectation similarity (negative L2 distance)
            expect_vec = self._state_expect_vectors[state]
            net_dist = np.linalg.norm(net_vec - expect_vec)

            # Activation similarity: compare to target activations for that state
            target_ts = targets_by_state[state]
            act_dist = np.linalg.norm(activations - target_ts)

            # Combine scores (deterministic, weighted distances)
            scores[state] = float(np.exp(-(w_net * net_dist + w_act * act_dist) / (temp * 1.0)))

        # Normalize into probabilities (avoid division by zero)
        total = sum(scores.values())
        if total <= 0:
            # Uniform over states as a safe fallback
            n = len(self.states)
            return {s: 1.0 / n for s in self.states}

        return {s: v / total for s, v in scores.items()}

    def update_thoughtseed_dynamics(self, current_activations: np.ndarray, target_activations: np.ndarray, 
                                   current_state: str, current_dwell: int, dwell_limit: int,
                                   observed_networks: Optional[Dict[str, float]] = None,
                                   sensory_inference: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Evolve thoughtseed activations using Ornstein-Uhlenbeck dynamics with Bayesian blending.
        
        Implements Prior-Likelihood blending:
        - Prior (μ_prior): target_activations from ThoughtseedParams (where agent wants to be)
        - Likelihood (μ_likelihood): sensory_inference from get_sensory_inference (where brain actually is)
        - Final target: μ = α·μ_prior + (1-α)·μ_likelihood
        
        This allows thoughtseeds to be pulled toward both:
        1. State-specific targets (top-down control)
        2. Sensory evidence from networks (bottom-up recognition)
        
        Args:
            observed_networks: Current observed network activations from generative process.
            sensory_inference: Inferred thoughtseed activations from networks (Likelihood).
                              If None, uses Prior only (backward compatibility).
        """
        dt = DEFAULTS['DEFAULT_DT']
        updated_activations = current_activations.copy()
        
        # 1. PRIOR: State-specific targets (where agent wants to be)
        mu_prior = np.array(target_activations)
        
        # Apply Network Modulation to Prior (use observed networks if provided)
        networks_for_modulation = observed_networks if observed_networks is not None else self.prev_network_acts
        if networks_for_modulation:
            # Normalize state name for modulation lookup
            mod_state = current_state
            if current_state in self._state_abbrev_map:
                mod_state = self.map_state_abbrev_to_full(current_state)
            
            modulations = self.get_network_modulation(networks_for_modulation, mod_state)
            for i, ts in enumerate(self.thoughtseeds):
                mu_prior[i] += modulations[ts]
                
            # Option C: Aha Drive directly boosts aha_moment target
            if self.aha_drive > 0.01:
                idx = self.thoughtseeds.index('aha_moment')
                mu_prior[idx] += self.aha_drive * self.aha_target_gain
        
        # 2. LIKELIHOOD: Sensory inference from networks (where brain actually is)
        if sensory_inference is not None:
            mu_likelihood = np.array(sensory_inference)
        else:
            # Fallback: Use Prior only (backward compatibility)
            mu_likelihood = mu_prior.copy()
        
        # 3. BAYESIAN BLENDING: Combine Prior and Likelihood
        # Weight: 60% Prior (state targets) + 40% Likelihood (sensory evidence)
        # This allows bottom-up recognition to influence thoughtseed dynamics
        prior_weight = 0.6
        mu = prior_weight * mu_prior + (1 - prior_weight) * mu_likelihood
        
        # Apply dynamic modifiers (Distraction Buildup & Fatigue)
        if current_state in ["breath_focus", "redirect_breath"]:
            # Distraction increases over time in focused states
            progress = min(1.5, current_dwell / max(5, dwell_limit))
            
            # Distraction Pressure: Accumulation of internal stimuli
            distraction_buildup = self.distraction_pressure * progress
            
            for ts in ["pain_discomfort", "pending_tasks"]:
                idx = self.thoughtseeds.index(ts)
                mu[idx] += distraction_buildup
                
            # Cognitive Fatigue: Decay of focus capability
            bf_idx = self.thoughtseeds.index("attend_breath")
            mu[bf_idx] = max(0.1, mu[bf_idx] - (self.fatigue_rate * progress))

        # 2. Set Stochastic Parameters (Ornstein-Uhlenbeck)
        # Theta (Reversion Speed)
        base_theta = self.base_theta
        # Sigma (Volatility)
        base_sigma = self.base_sigma
        
        # 3. Apply OU Update
        for i, ts in enumerate(self.thoughtseeds):
            x_prev = current_activations[i]
            target = mu[i]

            theta = base_theta
            sigma = base_sigma

            # Mind wandering is more volatile and "sticky"
            if current_state == "mind_wandering":
                sigma *= 1.5
                if ts in ["pending_tasks", "pain_discomfort"]:
                    theta *= 0.5

            # Focused states are more stable
            if current_state == "breath_focus" and ts == "attend_breath":
                sigma *= 0.5
                theta *= 1.5

            # OU update using agent RNG
            updated_activations[i] = float(ou_update(x_prev, target, theta, sigma, dt, rng=self.rng))
            
        from meditation_utils import clip_array
        return clip_array(updated_activations, DEFAULTS['ACTIVATION_CLIP_MIN'], DEFAULTS['ACTIVATION_CLIP_MAX'])
