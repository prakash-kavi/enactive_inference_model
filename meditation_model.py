"""Core meditation models: AgentConfig and ActInfAgent."""

import numpy as np
import os
import json
import logging
from collections import defaultdict
import copy
from typing import Optional, Dict, List, Tuple, Any

from config.meditation_config import (
    THOUGHTSEEDS, STATES, STATE_DWELL_TIMES,
    ActInfParams, ThoughtseedParams, MetacognitionParams,
    NETWORK_PROFILES, DEFAULTS
)
from meditation_utils import ou_update, clip_array

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
        self.state_indices = {state: i for i, state in enumerate(self.states)}
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        self.natural_transition_count = 0
        self.forced_transition_count = 0
        
        # History tracking
        self.activations_history = []
        self.state_history = []
        self.meta_awareness_history = []
        self.dominant_ts_history = []
        self.state_history_over_time = []
        
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
        
        # Initialize accumulators (Leaky Integrators)
        self.dmn_accumulator = 0.0
        self.fpn_accumulator = 0.0
        # Placeholder for previous network activations (initialized in ActInfAgent)
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

    def get_dwell_time(self, state: str) -> int:
        """Determine how long the agent stays in a mediative state.
        Draws a random integer from the configuration range defined in `STATE_DWELL_TIMES`.
        Enforces minimal biological plausibility constraints.
        """
        config_min, config_max = STATE_DWELL_TIMES[self.experience_level][state]
        
        # Ensure minimal biological plausibility
        if state in ['meta_awareness', 'redirect_breath']:
            min_biological = 1
            max_biological = config_max
        else:
            min_biological = 3
            max_biological = config_max
        
        # Uses agent RNG
        return max(min_biological, min(max_biological, int(self.rng.randint(config_min, config_max + 1))))

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
    """Active Inference agent implementing the Free Energy Principle (FEP).
    This class models the agent's 'brain' including:
    1. Network Dynamics (DMN, VAN, DAN, FPN)
    2. Active Inference (Minimizing Variational Free Energy)
    3. Learning (Updating internal models of world statistics)
    """
    
    def __init__(self, experience_level: str = 'novice', timesteps_per_cycle: int = 200):
        super().__init__(experience_level, timesteps_per_cycle)
        
        self.networks = ['DMN', 'VAN', 'DAN', 'FPN']
        # previous network activations
        self.prev_network_acts = {net: 0.0 for net in self.networks}
        self.network_activations_history = []
        self.free_energy_history = []
        self.prediction_error_history = []
        self.precision_history = []
        
        # Track learned network profiles
        self.learned_network_profiles = {
            "thoughtseed_contributions": {ts: {} for ts in self.thoughtseeds},
            "state_network_expectations": {state: {} for state in self.states}
        }
        
        self.in_transition = False
        self.transition_counter = 0
        self.transition_target = None
        self.prev_meta_awareness = 0.0
        
        # Initialize with default profiles
        for ts in self.thoughtseeds:
            self.learned_network_profiles["thoughtseed_contributions"][ts] = NETWORK_PROFILES["thoughtseed_contributions"][ts].copy()

        for state in self.states:
            self.learned_network_profiles["state_network_expectations"][state] = NETWORK_PROFILES["state_expected_profiles"][state][self.experience_level].copy()
            
    def compute_network_activations(self, thoughtseed_activations: np.ndarray, current_state: str, meta_awareness: float, dt: float = 1.0) -> Dict[str, float]:
        """Compute network activations for the current timestep.
        Integrates top-down expectations, lateral coupling, and fatigue dynamics.
        """
        # 1. Top-down: Thoughtseeds drive networks
        target_acts = self._apply_top_down_influence(thoughtseed_activations, current_state, meta_awareness)

        # 2. Coupled Dynamics: Interactions (if history exists)
        if self.prev_network_acts:
            target_acts = self._apply_lateral_coupling(target_acts, current_state)
            target_acts = self._apply_accumulator_dynamics(target_acts)

        # 3. Stochastic Update: Ornstein-Uhlenbeck process
        current_acts = self._ou_update_networks(target_acts, dt)

        # 4. Global Limits
        max_van = DEFAULTS['VAN_MAX']
        if current_acts['VAN'] > max_van:
            current_acts['VAN'] = max_van
            
        return current_acts

    def _apply_top_down_influence(self, thoughtseed_activations: np.ndarray, current_state: str, meta_awareness: float) -> Dict[str, float]:
        """Calculate base network targets from thoughtseeds and mediative state expectations."""
        ts_to_network = self.learned_network_profiles["thoughtseed_contributions"]
        target_acts = {net: self.network_base for net in self.networks}
        
        # Contribution 1: Thoughtseed activation -> Network
        for i, ts in enumerate(self.thoughtseeds):
            ts_act = thoughtseed_activations[i]
            for net in self.networks:
                target_acts[net] += ts_act * ts_to_network[ts][net]
    
        # Contribution 2: Mediative State Expectation (Modulated by Meta-awareness)
        state_expect = self.learned_network_profiles["state_network_expectations"][current_state]
        for net in self.networks:
            # Meta-awareness scales state influence (experts have stronger top-down control)
            meta_factor = meta_awareness * self.expert_meta_scalar
            state_influence = state_expect[net] * meta_factor
            target_acts[net] = 0.5 * target_acts[net] + 0.5 * state_influence
            
        return target_acts

    def _apply_lateral_coupling(self, target_acts: Dict[str, float], current_state: str) -> Dict[str, float]:
        """Apply lateral interactions: FPN regulation, DAN hysteresis, and DMN/DAN inhibition."""
        current_dan = self.prev_network_acts['DAN']
        current_fpn = self.prev_network_acts['FPN']
        current_dmn = self.prev_network_acts.get('DMN', 0)
        
        # A. FPN Regulation (Neural Efficiency)
        if current_state in ['breath_focus', 'redirect_breath']:
            focus_error = max(0, self.dan_focus_target - current_dan)
            fpn_demand = self.fpn_base_demand + (self.fpn_focus_mult * focus_error)

            # Experts relax control faster when stable
            efficiency_weight = self.efficiency_weight
            target_acts['FPN'] = (1 - efficiency_weight) * target_acts['FPN'] + efficiency_weight * fpn_demand

        # B. FPN drives DAN; apply hysteresis
        target_acts['DAN'] += self.fpn_to_dan_gain * current_fpn * (1.0 - current_dan)
        
        # DAN Hysteresis: Momentum of sustained attention; disrupted by DMN
        hysteresis_strength = self.hysteresis_strength
        target_acts['DAN'] += hysteresis_strength * current_dan * (1.0 - current_dmn)

        # C. Mutual Inhibition (DMN vs DAN)
        anticorrelation_force = self.anticorrelation_force
        target_acts['DAN'] -= anticorrelation_force * current_dmn * target_acts['DAN']
        target_acts['DMN'] -= anticorrelation_force * current_dan * target_acts['DMN']
        
        return target_acts

    def _apply_accumulator_dynamics(self, target_acts: Dict[str, float]) -> Dict[str, float]:
        """Apply leaky integrators for VAN spikes and FPN fatigue."""
        # VAN Accumulator (Salience Spike)
        current_dmn = self.prev_network_acts['DMN']
        self.dmn_accumulator = 0.9 * self.dmn_accumulator + 0.1 * current_dmn
        
        if self.dmn_accumulator > DEFAULTS.get('VAN_TRIGGER', 0.7):
            target_acts['VAN'] += self.van_spike
            self.dmn_accumulator = 0.0 # Reset

        # FPN Accumulator (Cognitive Fatigue)
        current_fpn = self.prev_network_acts['FPN']
        self.fpn_accumulator = self.fpn_accum_decay * self.fpn_accumulator + self.fpn_accum_inc * current_fpn
        
        # Collapse when effort exceeds capacity
        if self.fpn_accumulator > self.fatigue_threshold:
            target_acts['DAN'] *= self.fpn_collapse_dan_mult
            target_acts['DMN'] += self.fpn_collapse_dmn_inc
            self.fpn_accumulator = self.fatigue_reset
            
        return target_acts

    def _ou_update_networks(self, target_acts: Dict[str, float], dt: float) -> Dict[str, float]:
        """Update network state using Ornstein-Uhlenbeck stochastic process."""
        current_acts = {}
        
        if self.prev_network_acts:
            theta = 1.0 - self.memory_factor
            sigma = self.noise_level
            
            for net in self.networks:
                x_prev = self.prev_network_acts[net]
                mu = target_acts[net]
                current_acts[net] = float(ou_update(x_prev, mu, theta, sigma, dt, rng=self.rng))
        else:
            current_acts = target_acts.copy()
            
        # Clip to valid network range
        for net in self.networks:
            current_acts[net] = float(clip_array(current_acts[net], DEFAULTS['NETWORK_CLIP_MIN'], DEFAULTS['NETWORK_CLIP_MAX']))
            
        return current_acts
    
    def get_sensory_inference(self, network_acts: Dict[str, float]) -> np.ndarray:
        """Infer thoughtseed beliefs (states) from network activations (observations)."""
        ts_contribs = self.learned_network_profiles["thoughtseed_contributions"]
        inferred = np.zeros(self.num_thoughtseeds)
        
        for i, ts in enumerate(self.thoughtseeds):
            match_score = 0.0
            total_weight = 0.0
            
            for net in self.networks:
                weight = ts_contribs[ts][net]
                match_score += network_acts[net] * weight
                total_weight += weight
            
            if total_weight > 0:
                inferred[i] = match_score / total_weight
            else: # pragma: no cover
                inferred[i] = 0.1
                
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

        # Sensory precision scales with VAN proxy (Salience)
        van_proxy = sensory_inference[self.thoughtseeds.index('self_reflection')]
        pi_sensory = (self.sensory_precision_base + (self.sensory_precision_van_scalar * van_proxy)) * (1.0 + precision_mod) * self.precision_weight
        
        # Prior precision scales with meta-awareness (Top-down control)
        pi_prior = (self.prior_precision_base + (self.prior_precision_meta_scalar * meta_awareness)) * (1.0 + precision_mod) * self.complexity_penalty
        
        # Total VFE
        vfe = (sensory_nll * pi_sensory) + (prior_nll * pi_prior)
        
        return vfe, sensory_nll, prior_nll
   
    def update_network_profiles(self, thoughtseed_activations: np.ndarray, network_activations: Dict[str, float], current_state: str, prediction_errors: Dict[str, float]):
        """Update learned mappings (generative model) based on prediction errors.
        Implements a Hebbian-like associative learning rule modulated by precision.
        """
        if len(self.network_activations_history) < 10:
            return
        
        for i, ts in enumerate(self.thoughtseeds):
            ts_act = thoughtseed_activations[i]  # z_i(t)
            
            if ts_act > 0.2:
                for net in self.networks:
                    current_error = prediction_errors[net] # δ_k(t)
                    
                    # Precision (confidence) and signed Bayesian-like update
                    precision = self.learning_precision_base + self.learning_precision_scalar * len(self.network_activations_history)/self.timesteps
                    
                    # Signed update scaled by learning_rate and ts activation
                    error_sign = 1 if network_activations[net] < self.learned_network_profiles["state_network_expectations"][current_state][net] else -1
                    update = self.learning_rate * (error_sign * current_error) * ts_act / precision
                    
                    # Update contribution
                    self.learned_network_profiles["thoughtseed_contributions"][ts][net] += update
                    
                    # Bound weights [W_min, W_max]
                    self.learned_network_profiles["thoughtseed_contributions"][ts][net] = np.clip(
                        self.learned_network_profiles["thoughtseed_contributions"][ts][net], 0.1, 0.9)
        
        # Update mediative state expectations (slower rate)
        slow_rate = self.learning_rate * 0.3
        for net in self.networks:
            current_expect = self.learned_network_profiles["state_network_expectations"][current_state][net]
            new_value = (1 - slow_rate) * current_expect + slow_rate * network_activations[net]
            self.learned_network_profiles["state_network_expectations"][current_state][net] = new_value
    
    def get_network_modulation(self, network_acts: Dict[str, float], current_state: str) -> Dict[str, float]:
        """Calculate how current network activity modulates thoughtseed targets.
        Example: High DMN activity increases 'pending_tasks' and 'self_reflection' (mind-wandering).
        """
        modulations = {ts: 0.0 for ts in self.thoughtseeds}
        
        # DMN enhances pending_tasks and self_reflection, suppresses breath_focus
        dmn_strength = network_acts.get('DMN', 0)

        modulations['pending_tasks'] += self.dmn_pending_value * dmn_strength
        modulations['self_reflection'] += self.dmn_reflection_value * dmn_strength
        modulations['attend_breath'] -= self.dmn_breath_value * dmn_strength

        # VAN enhances pain_discomfort (salience) and self_reflection during meta_awareness
        van_strength = network_acts.get('VAN', 0)

        modulations['pain_discomfort'] += self.van_pain_value * van_strength

        if current_state == "meta_awareness":
            modulations['self_reflection'] += self.van_reflection_value * van_strength

        # DAN enhances breath_focus, suppresses distractions
        dan_strength = network_acts.get('DAN', 0)

        modulations['attend_breath'] += self.dan_breath_value * dan_strength
        modulations['pending_tasks'] -= self.dan_pending_value * dan_strength
        modulations['pain_discomfort'] -= self.dan_pain_value * dan_strength
        
        # FPN enhances self_reflection and equanimity (metacognition and regulation)
        fpn_strength = network_acts.get('FPN', 0)
        fpn_enhancement = self.fpn_enhancement

        modulations['self_reflection'] += self.fpn_reflection_value * fpn_strength * fpn_enhancement
        modulations['equanimity'] += self.fpn_equanimity_value * fpn_strength * fpn_enhancement
                
        return modulations

    def get_transition_probabilities(self, activations: np.ndarray, network_acts: Dict[str, float]) -> Dict[str, float]:
        """Calculate probability of transitioning to each mediative state.
        Combines two evidence sources:
        1. Network Similarity: How close current networks are to mediative state expectations.
        2. Activation Similarity: How close current thoughtseeds are to mediative state targets.
        """
        scores = {}
        temp = max(1e-6, self.softmax_temperature)

        # Use previous meta-awareness as a proxy (fallback to 0.5)
        meta = getattr(self, 'prev_meta_awareness', 0.5) if getattr(self, 'prev_meta_awareness', None) is not None else 0.5

        for state in self.states:
            # Network expectation similarity (negative L2 distance)
            expect = self.learned_network_profiles["state_network_expectations"][state]
            expect_vec = np.array([expect[net] for net in self.networks])
            net_vec = np.array([network_acts.get(net, 0.0) for net in self.networks])
            net_dist = np.linalg.norm(net_vec - expect_vec)
            net_score = np.exp(-net_dist / (temp * 1.0))

            # Activation similarity: compare to target activations for that state
            try:
                target_ts = self.get_target_activations(state, meta)
                act_dist = np.linalg.norm(activations - target_ts)
                act_score = np.exp(-act_dist / (temp * 1.0))
            except Exception:
                act_score = 1.0

            # Combine scores (multiplicative fusion)
            scores[state] = float(net_score * act_score)

        # Normalize into probabilities (avoid division by zero)
        total = sum(scores.values())
        if total <= 0:
            # Uniform over states as a safe fallback
            n = len(self.states)
            return {s: 1.0 / n for s in self.states}

        return {s: v / total for s, v in scores.items()}

    def update_thoughtseed_dynamics(self, current_activations: np.ndarray, target_activations: np.ndarray, current_state: str, current_dwell: int, dwell_limit: int) -> np.ndarray:
        """Evolve thoughtseed activations using Ornstein-Uhlenbeck dynamics.
        Includes distraction buildup, cognitive fatigue, and state-dependent volatility.
        """
        dt = DEFAULTS['DEFAULT_DT']
        updated_activations = current_activations.copy()
        
        # 1. Calculate Dynamic Targets (mu)
        mu = np.array(target_activations)
        
        # Apply Network Modulation to Targets
        if self.prev_network_acts:
            modulations = self.get_network_modulation(self.prev_network_acts, current_state)
            for i, ts in enumerate(self.thoughtseeds):
                mu[i] += modulations[ts]
        
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