"""
meditation_trainer.py

Trainer that orchestrates Active Inference training for an `ActInfAgent`.
This module pulls the training loop out of the agent to keep the agent focused
on dynamics, inference and small learning steps.
"""
import os
import json
import logging
import numpy as np
from typing import Tuple, List, Dict, Any, Optional

from meditation_utils import ensure_directories, _save_json_outputs, compute_state_aggregates
from config.meditation_config import STATE_DWELL_TIMES, DEFAULTS

class Trainer:
    """Orchestrates the simulation loop for an Active Inference Agent."""
    
    def __init__(self, agent):
        self.agent = agent

    def train(self, save_outputs: bool = True, output_dir: str = None, seed: int = None):
        """Run training using the provided `agent`.

        - `output_dir` (optional): directory to save JSON outputs.
        - `seed` (optional): reproducibility seed (sets numpy RNG).
        """
        if seed is not None:
            self.agent.rng = np.random.RandomState(seed)

        # 1. Initialization
        current_state, current_dwell, dwell_limit, activations, network_acts, meta_awareness = self._initialize_simulation()
        
        # Local tracking variables
        time_in_focused_state = 0
        state_transition_patterns = []
        transition_timestamps = []

        # 2. Training Loop
        for t in range(self.agent.timesteps):
            # 2.1 Top-Down Pass (Dynamics)
            meta_awareness, activations, time_in_focused_state = self._pass_top_down(
                current_state, current_dwell, dwell_limit, activations, meta_awareness, time_in_focused_state
            )
            
            # 2.2 Bottom-Up Pass (Perception & Learning)
            network_acts, sensory_inference, free_energy, sensory_nll = self._pass_bottom_up(
                current_state, activations, meta_awareness
            )

            # 2.3 Record History
            self._record_history(current_state, activations, meta_awareness, network_acts, free_energy, sensory_nll)

            # 2.4 State Transitions (Policy)
            transition_result = self._handle_state_transitions(
                current_state, current_dwell, dwell_limit, activations, network_acts, free_energy, t
            )
            
            if transition_result:
                # Unpack transition result
                current_state, current_dwell, dwell_limit, activations, pattern = transition_result
                state_transition_patterns.append(pattern)
                transition_timestamps.append(t)
            else:
                current_dwell += 1
            
            # Update previous activations for next step memory
            self.agent.prev_network_acts = network_acts.copy()

        # 3. Save Results
        if save_outputs:
            self._save_results(output_dir, state_transition_patterns, transition_timestamps)
            
        return self.agent

    def _initialize_simulation(self) -> Tuple[str, int, int, np.ndarray, Dict[str, float], float]:
        """Initialize simulation state, dwell times, and activations."""
        # Initialize training sequence (use agent's canonical state list)
        state_sequence = self.agent.states
        current_state = state_sequence[0]
        current_dwell = 0
        dwell_limit = self.agent.get_dwell_time(current_state)

        # Legacy RNG alignment: Match baseline which had a redundant uniform call here
        _ = self.agent.rng.uniform(0.05, 0.15)

        # Initialize activations
        activations = self.agent.get_target_activations(current_state, 0.6)
        
        # Initialize network tracking
        network_acts = self.agent.compute_network_activations(activations, current_state, 0.6)
        # self.agent.prev_network_acts remains zeros for first step, matching baseline

        # Initialize meta-awareness and VFE accumulator
        meta_awareness = self.agent.get_meta_awareness(current_state, activations)
        self.agent.prev_meta_awareness = meta_awareness
        self.agent.vfe_accumulator = 0.0
        
        return current_state, current_dwell, dwell_limit, activations, network_acts, meta_awareness

    def _pass_top_down(self, current_state: str, current_dwell: int, dwell_limit: int, 
                      activations: np.ndarray, prev_meta_awareness: float, time_in_focused_state: int) -> Tuple[float, np.ndarray, int]:
        """Execute Top-Down dynamics: Meta-awareness update, transition blending, and thoughtseed evolution."""
        
        # A. Calculate and Smooth Meta-awareness
        raw_meta = self.agent.get_meta_awareness(current_state, activations)
        smoothing = self.agent.smoothing
        meta_awareness = smoothing * prev_meta_awareness + (1 - smoothing) * raw_meta
        self.agent.prev_meta_awareness = meta_awareness  # Update agent state

        # B. Handle Transition Blending (if needed)
        activations = self._apply_transition_blending(activations)

        # C. Get Target Activations (L3 -> L2 Prior)
        target_activations = self.agent.get_target_activations(current_state, meta_awareness)

        # D. Update Thoughtseed Dynamics (L2 Belief Update)
        activations = self.agent.update_thoughtseed_dynamics(
            activations, target_activations, current_state, current_dwell, dwell_limit
        )

        # E. Track Distraction Buildup
        if current_state in ["breath_focus", "redirect_breath"]:
            time_in_focused_state += 1
            progress = min(1.5, current_dwell / max(10, dwell_limit))
            self.agent.distraction_buildup_rates.append(self.agent.distraction_pressure * progress)
        else:
            time_in_focused_state = 0
            self.agent.distraction_buildup_rates.append(0)
            
        return meta_awareness, activations, time_in_focused_state

    def _apply_transition_blending(self, activations: np.ndarray) -> np.ndarray:
        """Apply smoothing to activations during state transitions."""
        if hasattr(self.agent, 'in_transition') and self.agent.in_transition:
            base_blend = self.agent.blend_factor_transition
            blend_factor = base_blend * (1.0 + self.agent.rng.uniform(-self.agent.blend_variation, self.agent.blend_variation))

            # Small perturbations
            perturbed_target = self.agent.transition_target.copy()
            perturbed_target += self.agent.rng.normal(0, self.agent.transition_perturb_std, size=len(perturbed_target))
            perturbed_target = np.clip(perturbed_target, DEFAULTS['TARGET_CLIP_MIN'], DEFAULTS['TARGET_CLIP_MAX'])

            # Apply blending
            activations = (1 - blend_factor) * activations + blend_factor * perturbed_target

            self.agent.transition_counter -= 1
            if self.agent.transition_counter <= 0:
                self.agent.in_transition = False
                
        return activations

    def _pass_bottom_up(self, current_state: str, activations: np.ndarray, meta_awareness: float) -> Tuple[Dict[str, float], np.ndarray, float, float]:
        """Execute Bottom-Up dynamics: Network computation, Sensory Inference, and VFE calculation."""
        
        # A. Compute network activations (L1 Generative Process)
        network_acts = self.agent.compute_network_activations(activations, current_state, meta_awareness)

        # B. Infer sensory state from networks (L1 -> L2 Sensory Evidence)
        sensory_inference = self.agent.get_sensory_inference(network_acts)

        # C. Calculate VFE (L2 Belief Revision & L3 Monitoring)
        vfe_trend = 0.0
        if len(self.agent.free_energy_history) > 5:
            vfe_trend = np.mean(np.diff(self.agent.free_energy_history[-5:]))

        target_activations = self.agent.get_target_activations(current_state, meta_awareness)
        free_energy, sensory_nll, _ = self.agent.calculate_vfe(
            activations, target_activations, sensory_inference, meta_awareness, vfe_trend
        )

        # D. Update network profiles (Learning)
        dummy_errors = {net: sensory_nll * 0.1 for net in self.agent.networks}
        self.agent.update_network_profiles(activations, network_acts, current_state, dummy_errors)
        
        return network_acts, sensory_inference, free_energy, sensory_nll

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
        self.agent.state_history_over_time.append(self.agent.state_indices[current_state])

    def _handle_state_transitions(self, current_state: str, current_dwell: int, dwell_limit: int,
                                activations: np.ndarray, network_acts: Dict[str, float], free_energy: float, t: int) -> Optional[Tuple]:
        """Check for and execute state transitions based on VFE and dwell time."""
        
        # Accumulate VFE (Evidence of Policy Failure)
        decay = self.agent.vfe_accum_decay
        alpha = self.agent.vfe_accum_alpha
        self.agent.vfe_accumulator = decay * self.agent.vfe_accumulator + alpha * free_energy

        # Dynamic params
        base_threshold = 2.5 if self.agent.experience_level == 'expert' else 3.5
        min_dwell = STATE_DWELL_TIMES[self.agent.experience_level][current_state][0]

        # Transition Condition: Dwell expired OR Critical VFE (after refractory period)
        critical_vfe = self.agent.vfe_accumulator > (base_threshold * 1.5)
        dwell_expired = current_dwell >= dwell_limit
        
        if dwell_expired or (critical_vfe and current_dwell >= min_dwell):
            return self._execute_transition(current_state, activations, network_acts, free_energy, meta_awareness=self.agent.prev_meta_awareness)
            
        return None

    def _execute_transition(self, current_state: str, activations: np.ndarray, network_acts: Dict[str, float], 
                           free_energy: float, meta_awareness: float) -> Tuple:
        """Execute the transition logic: sample next state, blend activations, and reset counters."""
        
        # 1. Get Transition Probabilities
        probs = self.agent.get_transition_probabilities(activations, network_acts)

        # Force transition (exclude current)
        if current_state in probs:
            probs[current_state] = 0.0
        
        total_prob = sum(probs.values())
        if total_prob > 0:
            probs = {k: v / total_prob for k, v in probs.items()}
        else:
            probs = {k: 1.0 / (len(probs) - 1) for k in probs if k != current_state}

        # 2. Sample Next State
        states = list(probs.keys())
        probabilities = list(probs.values())
        next_state = self.agent.rng.choice(states, p=probabilities)

        # 3. Update Agent Counters
        self.agent.vfe_accumulator = 0.0
        self.agent.transition_activations[current_state].append(activations.copy())
        self.agent.natural_transition_count += 1
        
        # 4. Prepare return pattern
        pattern = (
            current_state,
            next_state,
            {ts: activations[i] for i, ts in enumerate(self.agent.thoughtseeds)},
            {net: val for net, val in network_acts.items()},
            free_energy,
        )

        # 5. Calculate new targets and blend
        new_target = self.agent.get_target_activations(next_state, meta_awareness)
        
        # Introduce biological variability
        low = self.agent.transition_variation_low
        high = self.agent.transition_variation_high
        for i in range(len(new_target)):
            variation = 1.0 + self.agent.rng.uniform(low, high)
            new_target[i] = max(DEFAULTS['TARGET_CLIP_MIN'], new_target[i] * variation)

        # Blend
        base_blend = self.agent.blend_factor_state
        blend_factor = base_blend * (1.0 + self.agent.rng.uniform(-self.agent.blend_variation, self.agent.blend_variation))
        new_activations = (1 - blend_factor) * activations + blend_factor * new_target

        # Set transition flags
        self.agent.in_transition = True
        self.agent.transition_counter = DEFAULTS['TRANSITION_COUNTER_BASE'] + int(self.agent.rng.randint(0, DEFAULTS['TRANSITION_COUNTER_RAND']))
        self.agent.transition_target = new_target.copy()

        # Return updated state
        new_dwell_limit = self.agent.get_dwell_time(next_state)
        
        return next_state, 0, new_dwell_limit, new_activations, pattern

    def _save_results(self, output_dir: str, state_transition_patterns: List[Dict], transition_timestamps: List[int]):
        """Save simulation results to JSON files."""
        out_dir = output_dir or os.path.join(os.path.dirname(__file__), "data")
        os.makedirs(out_dir, exist_ok=True)

        aggregates = compute_state_aggregates(self.agent)

        # Serialize transition counts
        serial_transition_counts = {k: {kk: int(vv) for kk, vv in inner.items()} for k, inner in self.agent.transition_counts.items()}

        tt = self.agent.params.transition_thresholds # Access via params in new structure if needed, or agent itself if updated
        # Note: In previous code, it accessed self.agent.transition_thresholds. 
        # But ActInfParams put it in self.params.transition_thresholds? 
        # Let's check agent init. `for k, v in vars(self.params).items(): setattr(self, k, v)`. 
        # So `agent.transition_thresholds` exists.
        
        serial_transition_thresholds = {
            'mind_wandering': float(self.agent.transition_thresholds.mind_wandering),
            'dmn_dan_ratio': float(self.agent.transition_thresholds.dmn_dan_ratio),
            'meta_awareness': float(self.agent.transition_thresholds.meta_awareness),
            'return_focus': float(self.agent.transition_thresholds.return_focus)
        }

        # Serialize patterns
        serial_patterns = []
        for (frm, to, ts_dict, net_dict, fe) in state_transition_patterns:
            serial_patterns.append({
                'from': frm,
                'to': to,
                'thoughtseed_activations': {k: float(v) for k, v in ts_dict.items()},
                'network_acts': {k: float(v) for k, v in net_dict.items()},
                'free_energy': float(fe)
            })

        transition_stats = {
            'transition_counts': serial_transition_counts,
            'transition_thresholds': serial_transition_thresholds,
            'natural_transitions': int(self.agent.natural_transition_count),
            'forced_transitions': int(self.agent.forced_transition_count),
            'transition_timestamps': [int(x) for x in transition_timestamps],
            'state_transition_patterns': serial_patterns,
            'distraction_buildup_rates': [float(x) for x in self.agent.distraction_buildup_rates],
            'average_activations_at_transition': aggregates.get('average_activations_at_transition', {}),
            'average_network_activations_by_state': aggregates.get('average_network_activations_by_state', {}),
            'average_free_energy_by_state': aggregates.get('average_free_energy_by_state', {}),
        }

        # Debug logging
        logging.info("%s NETWORK VALUES BY STATE:", self.agent.experience_level.upper())
        for state in self.agent.states:
            logging.info("  %s:", state)
            indices = [j for j, s in enumerate(self.agent.state_history) if s == state]
            if indices:
                state_networks = {
                    net: float(np.mean([self.agent.network_activations_history[j][net] for j in indices])) 
                    for net in self.agent.networks
                }
                for net in self.agent.networks:
                    logging.info("    %s: %.2f", net, state_networks[net])

        # Save to file
        out_path = os.path.join(out_dir, f"transition_stats_{self.agent.experience_level}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(transition_stats, f, indent=2)
        logging.info("Saved transition stats -> %s", os.path.relpath(out_path))

        logging.info("Active Inference training complete for %s.", self.agent.experience_level)
        logging.info("  - Natural transitions: %d, Forced transitions: %d", self.agent.natural_transition_count, self.agent.forced_transition_count)

        # Generate consumer-ready JSONs
        _save_json_outputs(self.agent, output_dir=out_dir)
