"""
meditation_trainer.py

Trainer that orchestrates Active Inference training for an `ActInfAgent`.
This module pulls the training loop out of the agent to keep the agent focused
on dynamics, inference and small learning steps.

Architecture:
- Layer 1 (Generative Process): generative_process.py - generates observations
- Layers 2 & 3 (Agent): meditation_model.py - predicts and infers
"""
import os
import json
import logging
import numpy as np
from typing import Tuple, List, Dict, Any, Optional

from meditation_utils import ensure_directories, _save_json_outputs, compute_state_aggregates, build_transition_stats
from config.meditation_config import DEFAULTS
from meditation_diagnostics import (
    compute_neural_efficiency_ratio, detect_expert_mind_wandering,
    compute_dmn_dan_anticorrelation, detect_van_spike_transition,
    detect_meta_awareness_transition, detect_redirect_transition,
    validate_state_signature
)
from generative_process import MeditationGenerativeProcess

class Trainer:
    """Orchestrates the simulation loop for an Active Inference Agent.
    
    Coordinates Layer 1 (Generative Process) and Layers 2 & 3 (Agent).
    """
    
    def __init__(self, agent, generative_process: Optional[MeditationGenerativeProcess] = None):
        self.agent = agent
        # Initialize generative process if not provided
        if generative_process is None:
            self.process = MeditationGenerativeProcess(
                experience_level=agent.experience_level,
                seed=agent.rng.randint(0, 2**31) if hasattr(agent.rng, 'randint') else None
            )
        else:
            self.process = generative_process

    def train(self, save_outputs: bool = True, output_dir: str = None, seed: int = None, enable_learning: bool = True):
        """Run training using the provided `agent`.

        - `output_dir` (optional): directory to save JSON outputs.
        - `seed` (optional): reproducibility seed (sets numpy RNG).
        - `enable_learning` (optional): if False, generative model weights are fixed (Fixed Attractor mode).
        """

        if seed is not None:
            self.agent.rng = np.random.RandomState(seed)
            # Also set seed for generative process
            self.process.rng = np.random.RandomState(seed)

        # 1. Initialization
        current_state, current_dwell, dwell_limit, activations, network_acts, meta_awareness = self._initialize_simulation()
        
        state_transition_patterns = []
        transition_timestamps = []
        old_state_abbrev = self.agent.map_state_full_to_abbrev(current_state)  # Track for transitions

        # 2. Training Loop (Russian Doll Enactive Inference)
        for t in range(self.agent.timesteps):
            # 2.1 Biology (Layer 1): Generate network signals with downward causation
            network_acts, process_state_abbrev = self.process.update(self.agent.blanket.active_states)
            process_state = self.agent.map_state_abbrev_to_full(process_state_abbrev)
            
            # Check if state transition occurred in process
            state_changed = (process_state != current_state)
            if state_changed:
                old_state_abbrev = self.agent.map_state_full_to_abbrev(current_state)
                current_state = process_state
                current_dwell = 0
                dwell_limit = self.process.current_max_dwell
            else:
                current_dwell += 1
            
            # 2.2 Sensory Step: Update blanket's sensory states
            self.agent.blanket.update_sensory_states(network_acts)
            
            # 2.3 Perception (Layer 3): Perceptual inference from blanket
            sensory_inference = self.agent.perceptual_inference()
            
            meta_awareness, activations, targets_by_state = self._pass_top_down(
                current_state, current_dwell, dwell_limit, activations, meta_awareness, 
                network_acts, sensory_inference
            )
            free_energy, sensory_nll = self._pass_bottom_up(
                current_state, activations, meta_awareness, targets_by_state[current_state], 
                network_acts, sensory_inference, enable_learning
            )
            
            # 2.4 Action (Layer 3): Prescriptive action based on beliefs and VFE
            prescription = self.agent.prescriptive_action(activations, free_energy, current_state)
            # Note: prescription is already applied to blanket in prescriptive_action()

            # 2.5 Record History & Transitions (Now correct: VFE is ready)
            self._record_history(current_state, activations, meta_awareness, network_acts, free_energy, sensory_nll)
            
            if state_changed:
                # Record transition with accurate VFE (computed above)
                pattern = (
                    old_state_abbrev,  # from
                    process_state_abbrev,  # to
                    {ts: activations[i] for i, ts in enumerate(self.agent.thoughtseeds)},
                    {net: val for net, val in network_acts.items()},
                    free_energy,  # Now contains the actual VFE for this transition
                )
                state_transition_patterns.append(pattern)
                transition_timestamps.append(t)
                old_state_abbrev = process_state_abbrev  # Update for next potential transition
            
            # Update previous network activations for next step
            self.agent.prev_network_acts = network_acts.copy()

        # 3. Save Results
        if save_outputs:
            self._save_results(output_dir, state_transition_patterns, transition_timestamps)
            
        return self.agent

    def _initialize_simulation(self) -> Tuple[str, int, int, np.ndarray, Dict[str, float], float]:
        """Initialize simulation state, dwell times, and activations."""
        # Reset generative process
        self.process.reset(state='BF')
        
        # Get initial state from process (abbreviation)
        process_state_abbrev = self.process.current_state
        current_state = self.agent.map_state_abbrev_to_full(process_state_abbrev)
        current_dwell = 0
        # Dwell times are now handled by generative process, but we track for compatibility
        dwell_limit = self.process.current_max_dwell

        # Legacy RNG alignment: Match baseline which had a redundant uniform call here
        _ = self.agent.rng.uniform(0.05, 0.15)

        # Initialize activations
        activations = self.agent.get_target_activations(current_state, 0.6)
        
        # Get initial network observations from generative process
        network_acts, _ = self.process.update(self.agent.blanket.active_states)
        self.agent.prev_network_acts = network_acts.copy()
        
        # Initialize blanket's sensory states
        self.agent.blanket.update_sensory_states(network_acts)

        # Initialize meta-awareness and VFE accumulator
        meta_awareness = self.agent.get_meta_awareness(current_state, activations)
        self.agent.prev_meta_awareness = meta_awareness
        self.agent.vfe_accumulator.reset(0.0)
        
        return current_state, current_dwell, dwell_limit, activations, network_acts, meta_awareness

    def _pass_top_down(self, current_state: str, current_dwell: int, dwell_limit: int, 
                      activations: np.ndarray, prev_meta_awareness: float,
                      observed_networks: Dict[str, float], sensory_inference: np.ndarray) -> Tuple[float, np.ndarray, Dict[str, np.ndarray]]:
        """Execute Top-Down dynamics: Meta-awareness update, transition blending, and thoughtseed evolution.
        
        Args:
            sensory_inference: Likelihood from Layer 3 (inferred thoughtseed activations from networks)
        """
        
        # A. Calculate and Smooth Meta-awareness
        raw_meta = self.agent.get_meta_awareness(current_state, activations)
        smoothing = self.agent.smoothing
        meta_awareness = smoothing * prev_meta_awareness + (1 - smoothing) * raw_meta
        self.agent.prev_meta_awareness = meta_awareness  # Update agent state

        # B. Handle Transition Blending (if needed)
        activations = self._apply_transition_blending(activations)

        # C. Get Target Activations (L3 -> L2 Prior) once per timestep
        targets_by_state = {
            state: self.agent.get_target_activations(state, meta_awareness)
            for state in self.agent.states
        }
        target_activations = targets_by_state[current_state]

        # D. Update Thoughtseed Dynamics (L2 Belief Update with Bayesian Blending)
        # Pass observed networks AND sensory_inference for Prior-Likelihood blending
        activations = self.agent.update_thoughtseed_dynamics(
            activations, target_activations, current_state, current_dwell, dwell_limit,
            observed_networks=observed_networks, sensory_inference=sensory_inference
        )

        # E. Track Distraction Buildup
        if current_state in ["breath_focus", "redirect_breath"]:
            progress = min(1.5, current_dwell / max(10, dwell_limit))
            self.agent.distraction_buildup_rates.append(self.agent.distraction_pressure * progress)
        else:
            self.agent.distraction_buildup_rates.append(0)
            
        return meta_awareness, activations, targets_by_state

    def _apply_transition_blending(self, activations: np.ndarray) -> np.ndarray:
        """Apply smoothing to activations during state transitions.
        
        NOTE: State transitions are now handled by generative_process.py,
        so this method is kept for backward compatibility but does minimal work.
        """
        # Transitions are handled by the process, so no blending needed here
        return activations

    def _pass_bottom_up(self, current_state: str, activations: np.ndarray, meta_awareness: float, 
                       target_activations: np.ndarray, observed_networks: Dict[str, float],
                       sensory_inference: np.ndarray, enable_learning: bool = True) -> Tuple[float, float]:
        """Execute Bottom-Up dynamics: VFE calculation and Learning.
        
        NOTE: Sensory inference is now computed before _pass_top_down for Bayesian blending.
        Network activations are provided from generative_process (Layer 1).
        """
        
        # A. Calculate VFE (L2 Belief Revision & L3 Monitoring)
        vfe_trend = 0.0
        if len(self.agent.free_energy_history) > 5:
            vfe_trend = np.mean(np.diff(self.agent.free_energy_history[-5:]))

        free_energy, sensory_nll, _ = self.agent.calculate_vfe(
            activations, target_activations, sensory_inference, meta_awareness, vfe_trend
        )
        
        # B. Update network profiles (Learning)
        if enable_learning:
            # Compute proper prediction errors: δ = observed - predicted
            # Generative model predicts networks from thoughtseeds and state
            predicted_networks = self.agent.compute_generative_predictions(activations, current_state, meta_awareness)
            prediction_errors = self.agent.compute_prediction_errors(predicted_networks, observed_networks)
            self.agent.update_network_profiles(activations, observed_networks, current_state, prediction_errors)
        
        return free_energy, sensory_nll

    def _record_history(self, current_state: str, activations: np.ndarray, meta_awareness: float, 
                       network_acts: Dict[str, float], free_energy: float, sensory_nll: float):
        """Record simulation data to agent history."""
        self.agent.network_activations_history.append(network_acts.copy())
        self.agent.free_energy_history.append(free_energy)
        self.agent.prediction_error_history.append(sensory_nll)
        self.agent.precision_history.append(0.5 + self.agent.precision_weight * meta_awareness)

        dominant_ts = self.agent.thoughtseeds[np.argmax(activations)]
        
        self.agent.state_history.append(current_state)
        self.agent.activations_history.append(activations.copy())
        self.agent.meta_awareness_history.append(meta_awareness)
        self.agent.dominant_ts_history.append(dominant_ts)
        
        # Track diagnostic metrics
        # Neural efficiency ratio
        efficiency_ratio = compute_neural_efficiency_ratio(network_acts, current_state)
        if efficiency_ratio is not None:
            self.agent.neural_efficiency_history.append(efficiency_ratio)
        
        # Expert mind wandering detection
        expert_mw = detect_expert_mind_wandering(network_acts)
        if expert_mw is True:
            self.agent.expert_mind_wandering_detections += 1
        
        # Stability indicator (DMN-DAN anti-correlation)
        is_stable = compute_dmn_dan_anticorrelation(network_acts)
        self.agent.stability_indicators.append(is_stable)

    # NOTE: State transitions are now handled by generative_process.py
    # The _handle_state_transitions and _execute_transition methods are removed
    # as the process manages its own state transitions based on dwell times.

    def _save_results(self, output_dir: str, state_transition_patterns: List[Dict], transition_timestamps: List[int]):
        """Save simulation results to JSON files."""
        out_dir = output_dir or os.path.join(os.path.dirname(__file__), "data")
        os.makedirs(out_dir, exist_ok=True)

        aggregates = compute_state_aggregates(self.agent)

        transition_stats = build_transition_stats(
            self.agent,
            state_transition_patterns,
            transition_timestamps,
            aggregates
        )

        # Debug logging
        logging.info("%s NETWORK VALUES BY STATE:", self.agent.experience_level.upper())
        state_network_means = aggregates.get('average_network_activations_by_state', {})
        for state in self.agent.states:
            logging.info("  %s:", state)
            state_networks = state_network_means.get(state, {})
            for net in self.agent.networks:
                if net in state_networks:
                    logging.info("    %s: %.2f", net, state_networks[net])

        # Save to file
        out_path = os.path.join(out_dir, f"transition_stats_{self.agent.experience_level}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(transition_stats, f, indent=2)
        logging.info("Saved transition stats -> %s", os.path.relpath(out_path))

        logging.info("Active Inference training complete for %s.", self.agent.experience_level)
        logging.info("  - Natural transitions: %d, Forced transitions: %d", self.agent.natural_transition_count, self.agent.forced_transition_count)

        # Generate consumer-ready JSONs
        _save_json_outputs(self.agent, output_dir=out_dir, aggregates=aggregates)

