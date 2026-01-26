"""Core meditation models: AgentConfig and GNWBottleneck (Layer 2) with PhenomenologicalMonitor (Layer 3)."""

import numpy as np
from typing import Optional, Dict, Tuple

from config.meditation_config import (
    THOUGHTSEEDS, STATES,
    ActInfParams, ThoughtseedParams, MetacognitionParams,
    NETWORK_PROFILES, DEFAULTS, NETWORK_MODULATION
)
from meditation_utils import ou_update, clip_array, LeakyAccumulator
from markov_blanket import MarkovBlanket


class PhenomenologicalMonitor:
    """
    Layer 3: The Meta-Cognitive Observer (Phenomenological Monitor).
    
    This is the innermost kernel of the Russian Doll architecture. It monitors
    the alignment between Layer 2's intentional state and the meditative goal,
    computing Variational Free Energy (VFE) and selecting policies for precision gating.
    """
    
    def __init__(self, networks: list, thoughtseeds: list, 
                 sensory_precision_base: float, prior_precision_base: float,
                 precision_weight: float, complexity_penalty: float,
                 get_meta_awareness_fn=None, blanket_l2l3=None):
        """
        Initialize the Phenomenological Monitor.
        
        Args:
            networks: List of network names ['DMN', 'VAN', 'DAN', 'FPN']
            thoughtseeds: List of thoughtseed names
            sensory_precision_base: Base precision for sensory (accuracy) term
            prior_precision_base: Base precision for prior (complexity) term
            precision_weight: Weight for meta-awareness modulation of sensory precision
            complexity_penalty: Weight for meta-awareness modulation of prior precision
            get_meta_awareness_fn: Function to compute meta-awareness (for policy evaluation)
            blanket_l2l3: Reference to Markov Blanket L2-L3 (optional, for full integration)
        """
        self.networks = networks
        self.thoughtseeds = thoughtseeds
        self.sensory_precision_base = sensory_precision_base
        self.prior_precision_base = prior_precision_base
        self.precision_weight = precision_weight
        self.complexity_penalty = complexity_penalty
        self.get_meta_awareness_fn = get_meta_awareness_fn
        self.blanket_l2l3 = blanket_l2l3
    
    def compute_vfe(self, observed_networks: Dict[str, float] = None, predicted_networks: Dict[str, float] = None, 
                    current_seeds: np.ndarray = None, prior_seeds: np.ndarray = None, meta_awareness: float = None) -> Tuple[float, float, float]:
        """
        Layer 3: Compute Variational Free Energy (VFE) using standard Active Inference formulation.
        
        VFE = Accuracy (NLL of prediction errors) + Complexity (KL divergence from prior)
        
        Can read from blanket_l2l3.sensory_states if blanket is provided, or use direct parameters.
        
        Args:
            observed_networks: Observed network activations from generative process (Layer 1) - optional if using blanket
            predicted_networks: Predicted network activations from generative model (Layer 2) - optional if using blanket
            current_seeds: Current thoughtseed activations q(z) - optional if using blanket
            prior_seeds: Prior thoughtseed targets p(z) - optional if using blanket
            meta_awareness: Meta-awareness level for precision modulation - optional if using blanket
        
        Returns:
            Tuple of (vfe, accuracy_nll, complexity_kl)
        """
        # Read from blanket if available, otherwise use direct parameters
        if self.blanket_l2l3 and self.blanket_l2l3.sensory_states:
            sensory = self.blanket_l2l3.sensory_states
            observed_networks = sensory.get('observed_networks', observed_networks)
            predicted_networks = sensory.get('predicted_networks', predicted_networks)
            current_seeds = sensory.get('thoughtseed_activations', current_seeds)
            prior_seeds = sensory.get('prior_seeds', prior_seeds)
            meta_awareness = sensory.get('meta_awareness', meta_awareness)
        
        # Validate required parameters
        if observed_networks is None or predicted_networks is None or current_seeds is None or prior_seeds is None or meta_awareness is None:
            raise ValueError("compute_vfe requires all parameters or blanket_l2l3 with sensory_states populated")
        # 1. ACCURACY: Negative Log-Likelihood of prediction errors
        # Use Beta/Bernoulli likelihood for activations in [0,1] range
        # NLL = -sum(observed * log(predicted) + (1-observed) * log(1-predicted))
        eps = 1e-9  # Small epsilon to avoid log(0)
        accuracy_nll = 0.0
        for net in self.networks:
            observed = observed_networks.get(net, 0.0)
            predicted = predicted_networks.get(net, 0.0)
            # Clip predicted to valid range [eps, 1-eps]
            predicted = np.clip(predicted, eps, 1.0 - eps)
            # Bernoulli/Beta NLL: -log(p(observed | predicted))
            accuracy_nll += -(observed * np.log(predicted) + (1.0 - observed) * np.log(1.0 - predicted))
        
        # 2. COMPLEXITY: KL divergence from prior
        # KL(q(z) || p(z)) = sum(q(z) * log(q(z) / p(z)))
        complexity_kl = np.sum(
            current_seeds * np.log((current_seeds + eps) / (prior_seeds + eps)) + 
            (1 - current_seeds) * np.log((1 - current_seeds + eps) / (1 - prior_seeds + eps))
        )
        
        # 3. PRECISION (Simplified: base precision with meta-awareness modulation)
        # Sensory precision: base + meta-awareness boost
        pi_sensory = self.sensory_precision_base * (1.0 + meta_awareness * self.precision_weight)
        
        # Prior precision: base + meta-awareness boost  
        pi_prior = self.prior_precision_base * (1.0 + meta_awareness * self.complexity_penalty)
        
        # 4. TOTAL VFE
        vfe = (accuracy_nll * pi_sensory) + (complexity_kl * pi_prior)
        
        return vfe, accuracy_nll, complexity_kl
    
    def evaluate_policies(self, z: np.ndarray = None, vfe: float = None, current_state: str = None) -> dict:
        """
        Layer 3: Evaluate policies and return prescriptions for precision gating.
        
        Selects Active State prescriptions based on internal beliefs and VFE.
        Writes precision modulations to blanket_l2l3.active_states (L3 → L2).
        Also returns L1-L2 prescriptions for backward compatibility.
        
        Policies:
        1. Aha! Moment Short-Circuit: If aha_moment > 0.75, truncate MW dwell
        2. Attentional Sharpening: If meta_awareness > 0.6, reduce noise
        3. Equanimity Buffer: If equanimity > 0.6, reduce fatigue
        4. Precision Reset: If in meta_awareness state, clear signal
        
        Args:
            z: Current thoughtseed activations (from perceptual_inference) - optional if using blanket
            vfe: Current Variational Free Energy - optional if using blanket
            current_state: Current meditative state (full name) - optional if using blanket
            
        Returns:
            Prescription dictionary for Markov Blanket L1-L2 (for backward compatibility)
        """
        # Read from blanket if available, otherwise use direct parameters
        if self.blanket_l2l3 and self.blanket_l2l3.sensory_states:
            sensory = self.blanket_l2l3.sensory_states
            z = sensory.get('thoughtseed_activations', z)
            vfe = sensory.get('vfe', vfe)
            current_state = sensory.get('current_state', current_state)
        
        # Validate required parameters
        if z is None or vfe is None or current_state is None:
            raise ValueError("evaluate_policies requires all parameters or blanket_l2l3 with sensory_states populated")
        
        # L1-L2 prescriptions (for biological modulations)
        prescription_l1l2 = {
            'noise_reduction': 1.0,  # 1.0 = no change; < 1.0 = clearer signal
            'dwell_modifier': 1.0,   # 1.0 = no change; < 1.0 = shorter duration
            'fatigue_buffer': 1.0    # 1.0 = no change; < 1.0 = less fatigue
        }
        
        # L2-L3 prescriptions (for precision gating)
        prescription_l2l3 = {
            'precision_modulation': 1.0,  # 1.0 = no change; > 1.0 = higher precision
            'theta_modulation': 1.0,       # 1.0 = no change; > 1.0 = faster reversion
            'target_adjustment': 1.0       # 1.0 = no change; > 1.0 = stronger prior pull
        }
        
        # Policy 1: Aha! Moment Short-Circuit (Immediate MW termination)
        aha_idx = self.thoughtseeds.index('aha_moment')
        if z[aha_idx] > 0.75:
            prescription_l1l2['dwell_modifier'] = 0.2  # Truncate to 20% of max dwell
            prescription_l1l2['noise_reduction'] = 0.5  # Strong attentional gain
            prescription_l2l3['precision_modulation'] = 1.5  # Increase precision for focus
            prescription_l2l3['theta_modulation'] = 1.3  # Faster reversion to BF
        
        # Policy 2: Attentional Sharpening (Meta-Awareness → Noise Reduction)
        if self.get_meta_awareness_fn:
            meta_awareness = self.get_meta_awareness_fn(current_state, z)
            if meta_awareness > 0.6:
                # Use minimum (strongest reduction wins) for noise
                prescription_l1l2['noise_reduction'] = min(prescription_l1l2['noise_reduction'], 0.6)
                prescription_l2l3['precision_modulation'] = 1.2  # Moderate precision boost
        
        # Policy 3: Equanimity Buffer (Reduces fatigue/distraction)
        equanimity_idx = self.thoughtseeds.index('equanimity')
        if z[equanimity_idx] > 0.6:
            prescription_l1l2['fatigue_buffer'] = 0.5
            prescription_l2l3['target_adjustment'] = 1.1  # Slight boost to equanimity target
        
        # Policy 4: Precision Reset (Meta-Awareness state → Signal clearing)
        if current_state == 'meta_awareness':
            prescription_l1l2['noise_reduction'] = min(prescription_l1l2['noise_reduction'], 0.4)
            prescription_l2l3['precision_modulation'] = 1.3  # Strong precision boost
        
        # Write L2-L3 prescriptions to blanket if available
        if self.blanket_l2l3:
            self.blanket_l2l3.update_active_states(prescription_l2l3)
        
        # Return L1-L2 prescriptions (for backward compatibility)
        return prescription_l1l2


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

class GNWBottleneck(AgentConfig):
    """
    Layer 2: GNWBottleneck (The Intentional Bridge).
    
    This class implements the Generative Model (GM) that actively reconciles:
    - Bottom-Up: High-dimensional neural signals from Layer 1 (biological reality)
    - Top-Down: Meta-cognitive context from Layer 3 (meditative goals)
    
    The bottleneck performs competitive alignment between biological emergence and
    intentional priorities, using a 5-dimensional thoughtseed space.
    
    Architecture:
    - Contains nested PhenomenologicalMonitor (Layer 3) for VFE computation and policy evaluation
    - Owns MarkovBlanket L1-L2 (biological interface) and L2-L3 (mind interface)
    - Uses Scientific Matrix Form (W) for generative predictions
    
    NOTE: Network generation is handled by generative_process.py (Layer 1).
    The bottleneck receives observations from the process and predicts/infers.
    """
    
    def __init__(self, experience_level: str = 'novice', timesteps_per_cycle: int = 200):
        super().__init__(experience_level, timesteps_per_cycle)
        
        self.networks = ['DMN', 'VAN', 'DAN', 'FPN']
        # Track observed networks (from generative process)
        self.network_activations_history = []
        self.free_energy_history = []
        self.prediction_error_history = []
        self.precision_history = []
        
        # The Weight Matrix (W)
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
        # W matrix is now fixed, but state expectations are still learned
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
        
        # Markov Blanket L1-L2: The agent's statistical boundary with Layer 1 (biological interface)
        self.blanket = MarkovBlanket(smoothing=0.9)
        
        # Markov Blanket L2-L3: The interface between Layer 2 (GNWBottleneck) and Layer 3 (Monitor)
        # Sensory States (L2 → L3): thoughtseed_activations, predicted_networks, current_state, meta_awareness
        # Active States (L3 → L2): precision_modulation, theta_modulation, target_adjustment
        blanket_l2l3_template = {
            'precision_modulation': 1.0,  # Precision weight adjustment (1.0 = no change)
            'theta_modulation': 1.0,       # Reversion speed adjustment for MVOU (1.0 = no change)
            'target_adjustment': 1.0       # Prior target adjustment (1.0 = no change)
        }
        self.blanket_l2l3 = MarkovBlanket(smoothing=0.9, active_states_template=blanket_l2l3_template)
        
        # Layer 3: Phenomenological Monitor (nested within Layer 2)
        self.monitor = PhenomenologicalMonitor(
            networks=self.networks,
            thoughtseeds=self.thoughtseeds,
            sensory_precision_base=self.sensory_precision_base,
            prior_precision_base=self.prior_precision_base,
            precision_weight=self.precision_weight,
            complexity_penalty=self.complexity_penalty,
            get_meta_awareness_fn=self.get_meta_awareness,
            blanket_l2l3=self.blanket_l2l3
        )

    # ========================================================================
    # LAYER 2: GENERATIVE MODEL
    # ========================================================================
    # The agent's internal "world model" that predicts network activations
    # from latent thoughtseeds and state context.
    # ========================================================================

    def _build_state_expect_vector(self, state: str) -> np.ndarray:
        """Build a dense vector of network expectations for a given state."""
        expect = self.learned_network_profiles["state_network_expectations"][state]
        return np.array([expect[net] for net in self.networks])

    def compute_generative_predictions(self, thoughtseed_activations: np.ndarray, current_state: str, meta_awareness: float) -> Dict[str, float]:
        """
        Layer 2 (Generative Model): μ_pred = z · W
        
        The agent's 'world model' predicting network activity from latent thoughtseeds.
        Uses the Scientific Matrix Form: single dot product replaces dictionary lookups.
        
        Args:
            thoughtseed_activations: z vector (5×1) of thoughtseed strengths
            current_state: Mediative state (full name from STATES)
            meta_awareness: Meta-awareness level (modulates state expectation bias)
        
        Returns:
            Dictionary of predicted network activations {DMN, VAN, DAN, FPN}
        """
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
    
    def update_thoughtseed_dynamics(self, current_activations: np.ndarray, target_activations: np.ndarray, 
                                   current_state: str, current_dwell: int, dwell_limit: int,
                                   observed_networks: Optional[Dict[str, float]] = None,
                                   sensory_inference: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Layer 2: Evolve thoughtseed activations using Ornstein-Uhlenbeck dynamics with Bayesian blending.
        
        Implements Prior-Likelihood blending:
        - Prior (μ_prior): target_activations from ThoughtseedParams (where agent wants to be)
        - Likelihood (μ_likelihood): sensory_inference from perceptual_inference (where brain actually is)
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
            modulations = self.get_network_modulation(networks_for_modulation, current_state)
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
        # Apply Equanimity Buffer from Markov Blanket if available
        fatigue_buffer = self.blanket.active_states.get('fatigue_buffer', 1.0)
        effective_fatigue_rate = self.fatigue_rate * fatigue_buffer
        effective_distraction_pressure = self.distraction_pressure * fatigue_buffer
        
        if current_state in ["breath_focus", "redirect_breath"]:
            # Distraction increases over time in focused states
            progress = min(1.5, current_dwell / max(5, dwell_limit))
            
            # Distraction Pressure: Accumulation of internal stimuli (modulated by equanimity)
            distraction_buildup = effective_distraction_pressure * progress
            
            for ts in ["pain_discomfort", "pending_tasks"]:
                idx = self.thoughtseeds.index(ts)
                mu[idx] += distraction_buildup
                
            # Cognitive Fatigue: Decay of focus capability (modulated by equanimity)
            bf_idx = self.thoughtseeds.index("attend_breath")
            mu[bf_idx] = max(0.1, mu[bf_idx] - (effective_fatigue_rate * progress))

        # Set Stochastic Parameters (Ornstein-Uhlenbeck)
        # Theta (Reversion Speed)
        base_theta = self.base_theta
        # Sigma (Volatility)
        base_sigma = self.base_sigma
        
        # Apply OU Update
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

    def get_network_modulation(self, network_acts: Dict[str, float], current_state: str) -> Dict[str, float]:
        """
        Layer 2: Calculate how current network activity modulates thoughtseed targets.
        Helper method for update_thoughtseed_dynamics().
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

    # ========================================================================
    # LAYER 3: RECOGNITION/INFERENCE
    # ========================================================================
    # The agent's inference and learning mechanisms that minimize Variational
    # Free Energy through perceptual inference, VFE calculation, and learning.
    # ========================================================================

    def perceptual_inference(self) -> np.ndarray:
        """
        Layer 3: Perceptual Inference (Bottom-Up Recognition).
        Infers thoughtseed beliefs (z) from sensory states in the Markov Blanket.
        Uses amortized inference via matrix-transpose projection.
        """
        # Get observations from blanket's sensory states
        network_acts = self.blanket.sensory_states
        if not network_acts:
            # Fallback: return neutral inference if no sensory data
            return np.array([0.2] * self.num_thoughtseeds)
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
    
    def calculate_vfe(self, observed_networks: Dict[str, float], predicted_networks: Dict[str, float], 
                     current_seeds: np.ndarray, prior_seeds: np.ndarray, meta_awareness: float) -> Tuple[float, float, float]:
        """
        Layer 2: Delegate VFE calculation to Layer 3 (Phenomenological Monitor).
        
        Writes sensory data to Blanket_2 and delegates computation to Monitor.
        
        Args:
            observed_networks: Observed network activations from generative process (Layer 1)
            predicted_networks: Predicted network activations from generative model (Layer 2)
            current_seeds: Current thoughtseed activations q(z)
            prior_seeds: Prior thoughtseed targets p(z)
            meta_awareness: Meta-awareness level for precision modulation
        
        Returns:
            Tuple of (vfe, accuracy_nll, complexity_kl)
        """
        # Write sensory data to Blanket_2 (L2 → L3)
        self.blanket_l2l3.update_sensory_states({
            'observed_networks': observed_networks,
            'predicted_networks': predicted_networks,
            'thoughtseed_activations': current_seeds,
            'prior_seeds': prior_seeds,
            'meta_awareness': meta_awareness
        })
        
        # Delegate to Layer 3 Monitor
        return self.monitor.compute_vfe()

    def prescriptive_action(self, z: np.ndarray, vfe: float, current_state: str) -> dict:
        """
        Layer 2: Delegate policy evaluation to Layer 3 (Phenomenological Monitor).
        
        Writes sensory data to Blanket_2 and delegates policy evaluation to Monitor.
        Monitor writes L2-L3 prescriptions to Blanket_2 and returns L1-L2 prescriptions.
        
        Args:
            z: Current thoughtseed activations (from perceptual_inference)
            vfe: Current Variational Free Energy
            current_state: Current meditative state
            
        Returns:
            Prescription dictionary for Markov Blanket L1-L2 (biological modulations)
        """
        # Update Blanket_2 sensory states with current context
        self.blanket_l2l3.update_sensory_states({
            'thoughtseed_activations': z,
            'vfe': vfe,
            'current_state': current_state
        })
        
        # Delegate to Layer 3 Monitor (writes to Blanket_2, returns L1-L2 prescriptions)
        prescription_l1l2 = self.monitor.evaluate_policies()
        
        # Update Blanket L1-L2 with L1-L2 prescriptions
        self.blanket.update_active_states(prescription_l1l2)
        
        return prescription_l1l2

    def compute_prediction_errors(self, predicted: Dict[str, float], observed: Dict[str, float]) -> Dict[str, float]:
        """
        Layer 3: Prediction Error: δ = observed - predicted
        Explicit error signal for Active Inference learning.
        """
        return {net: observed[net] - predicted[net] for net in self.networks}
   
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

