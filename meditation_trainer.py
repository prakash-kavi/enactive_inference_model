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

from meditation_utils import ensure_directories, _save_json_outputs, compute_state_aggregates
from config.meditation_config import STATE_DWELL_TIMES, DEFAULTS, ActInfParams, get_actinf_params_dict

class Trainer:
    def __init__(self, agent):
        self.agent = agent

    def train(self, save_outputs: bool = True, output_dir: str = None, seed: int = None):
        """Run training using the provided `agent`.

        - `output_dir` (optional): directory to save JSON outputs.
        - `seed` (optional): reproducibility seed (sets numpy RNG).
        """
        agent = self.agent

        if seed is not None:
            np.random.seed(seed)

        # Initialize training sequence
        state_sequence = ["breath_control", "mind_wandering", "meta_awareness", "redirect_breath"]
        current_state_index = 0
        current_state = state_sequence[current_state_index]
        current_dwell = 0
        dwell_limit = agent.get_dwell_time(current_state)

        # Initialize activations
        activations = np.full(agent.num_thoughtseeds, np.random.uniform(0.05, 0.15))
        activations = agent.get_target_activations(current_state, 0.6)
        prev_activations = activations.copy()

        # Track focused state timing
        time_in_focused_state = 0
        state_transition_patterns = []
        transition_timestamps = []

        # Initialize network tracking
        network_acts = agent.compute_network_activations(activations, current_state, 0.6)
        prev_network_acts = network_acts.copy()

        # Initialize meta-awareness for smoothing
        agent.prev_meta_awareness = agent.get_meta_awareness(current_state, activations)

        # Initialize VFE accumulator for policy switching
        agent.vfe_accumulator = 0.0

        # Training loop
        for t in range(agent.timesteps):
            # --- PASS 1: TOP-DOWN (Action & Prediction) ---

            # Calculate instantaneous meta-awareness
            raw_meta_awareness = agent.get_meta_awareness(current_state, activations)

            # Apply smoothing (EMA) using agent-level smoothing
            smoothing = agent.smoothing

            meta_awareness = smoothing * agent.prev_meta_awareness + (1 - smoothing) * raw_meta_awareness
            agent.prev_meta_awareness = meta_awareness

            if hasattr(agent, 'in_transition') and agent.in_transition:
                # Continue smoothing the transition over multiple timesteps
                base_blend = agent.blend_factor_transition
                blend_factor = base_blend * (1.0 + np.random.uniform(-agent.blend_variation, agent.blend_variation))

                # Add small random perturbations to transition target
                perturbed_target = agent.transition_target.copy()
                perturbed_target += np.random.normal(0, agent.transition_perturb_std, size=len(perturbed_target))
                perturbed_target = np.clip(perturbed_target, DEFAULTS['TARGET_CLIP_MIN'], DEFAULTS['TARGET_CLIP_MAX'])

                # Apply blending
                activations = (1 - blend_factor) * activations + blend_factor * perturbed_target

                agent.transition_counter -= 1
                if agent.transition_counter <= 0:
                    agent.in_transition = False

            # Get target activations (L3 -> L2 Prior)
            target_activations = agent.get_target_activations(current_state, meta_awareness)

            # Update thoughtseed dynamics (L2 Belief Update)
            activations = agent.update_thoughtseed_dynamics(
                activations,
                target_activations,
                current_state,
                current_dwell,
                dwell_limit
            )

            # Handle distraction tracking
            if current_state in ["breath_control", "redirect_breath"]:
                time_in_focused_state += 1
                progress = min(1.5, current_dwell / max(10, dwell_limit))
                # Use agent-provided distraction_pressure
                distraction_pressure = agent.distraction_pressure
                agent.distraction_buildup_rates.append(distraction_pressure * progress)
            else:
                time_in_focused_state = 0
                agent.distraction_buildup_rates.append(0)

            # --- PASS 2: BOTTOM-UP (Perception & Learning) ---

            # Compute network activations (L1 Generative Process)
            network_acts = agent.compute_network_activations(activations, current_state, meta_awareness)

            # Infer sensory state from networks (L1 -> L2 Sensory Evidence)
            sensory_inference = agent.get_sensory_inference(network_acts)

            # Calculate VFE (L2 Belief Revision & L3 Monitoring)
            vfe_trend = 0.0
            if len(agent.free_energy_history) > 5:
                vfe_trend = np.mean(np.diff(agent.free_energy_history[-5:]))

            free_energy, sensory_nll, prior_nll = agent.calculate_vfe(
                activations, target_activations, sensory_inference, meta_awareness, vfe_trend
            )

            # Update network profiles based on prediction errors
            dummy_errors = {net: sensory_nll * 0.1 for net in agent.networks}
            agent.update_network_profiles(activations, network_acts, current_state, dummy_errors)

            # Record network state and free energy
            agent.network_activations_history.append(network_acts.copy())
            agent.free_energy_history.append(free_energy)
            agent.prediction_error_history.append(sensory_nll)
            agent.precision_history.append(0.5 + agent.precision_weight * meta_awareness)

            # Identify dominant thoughtseed
            dominant_ts = agent.thoughtseeds[np.argmax(activations)]

            # Track histories
            agent.state_history.append(current_state)
            agent.activations_history.append(activations.copy())
            agent.meta_awareness_history.append(meta_awareness)
            agent.dominant_ts_history.append(dominant_ts)
            agent.state_history_over_time.append(agent.state_indices[current_state])

            # --- L3 POLICY SWITCHING (VFE-Based) ---
            transition_happened = False

            # Accumulate VFE (Evidence of Policy Failure) using agent-level dynamics
            decay = agent.vfe_accum_decay
            alpha = agent.vfe_accum_alpha
            agent.vfe_accumulator = decay * agent.vfe_accumulator + alpha * free_energy

            # Dynamic Threshold based on Experience
            base_threshold = 2.5 if agent.experience_level == 'expert' else 3.5

            # Check for transition if we've been in the state long enough OR VFE is critical
            critical_vfe = agent.vfe_accumulator > (base_threshold * 1.5)
            dwell_expired = current_dwell >= dwell_limit

            # Refractory period: Don't force transition via VFE if we just arrived
            min_dwell = STATE_DWELL_TIMES[agent.experience_level][current_state][0]
            if dwell_expired or (critical_vfe and current_dwell >= min_dwell):
                # 1. Get Transition Probabilities
                probs = agent.get_transition_probabilities(activations, network_acts)

                # 2. Force Transition
                if current_state in probs:
                    probs[current_state] = 0.0

                # Renormalize
                total_prob = sum(probs.values())
                if total_prob > 0:
                    probs = {k: v / total_prob for k, v in probs.items()}
                else:  # pragma: no cover
                    probs = {k: 1.0 / (len(probs) - 1) for k in probs if k != current_state}

                # 3. Sample Next State
                states = list(probs.keys())
                probabilities = list(probs.values())
                next_state = np.random.choice(states, p=probabilities)

                # 4. Execute Transition
                transition_happened = True
                agent.vfe_accumulator = 0.0
                agent.transition_activations[current_state].append(activations.copy())
                agent.natural_transition_count += 1
                transition_timestamps.append(t)
                state_transition_patterns.append((
                    current_state,
                    next_state,
                    {ts: activations[i] for i, ts in enumerate(agent.thoughtseeds)},
                    {net: val for net, val in network_acts.items()},
                    free_energy,
                ))

                agent.transition_counts[current_state][next_state] += 1

                # Update state
                if next_state in state_sequence:
                    current_state_index = state_sequence.index(next_state)
                current_state = next_state
                current_dwell = 0
                dwell_limit = agent.get_dwell_time(current_state)

                # Calculate new state targets
                new_target = agent.get_target_activations(current_state, meta_awareness)

                # Introduce biological variability
                low = agent.transition_variation_low
                high = agent.transition_variation_high
                for i in range(len(new_target)):
                    variation = 1.0 + np.random.uniform(low, high)
                    new_target[i] *= variation
                    new_target[i] = max(DEFAULTS['TARGET_CLIP_MIN'], new_target[i])

                # Blend current state into new state (Smooth Transition)
                base_blend = agent.blend_factor_state
                blend_factor = base_blend * (1.0 + np.random.uniform(-agent.blend_variation, agent.blend_variation))
                activations = (1 - blend_factor) * activations + blend_factor * new_target

                # Add transition markers
                agent.in_transition = True
                agent.transition_counter = DEFAULTS['TRANSITION_COUNTER_BASE'] + np.random.randint(0, DEFAULTS['TRANSITION_COUNTER_RAND'])
                agent.transition_target = new_target.copy()

            if not transition_happened:
                current_dwell += 1

            # Store for next iteration
            prev_activations = activations.copy()
            prev_network_acts = network_acts.copy()
            agent.prev_network_acts = prev_network_acts.copy()

        # Save learned weights and network profiles
        if save_outputs:
            # Ensure output dir exists (configurable)
            out_dir = output_dir or os.path.join(os.path.dirname(__file__), "data")
            os.makedirs(out_dir, exist_ok=True)

            aggregates = compute_state_aggregates(agent)

            # Convert transition-related objects to JSON-serializable primitives
            serial_transition_counts = {k: {kk: int(vv) for kk, vv in inner.items()} for k, inner in agent.transition_counts.items()}

            tt = agent.transition_thresholds
            serial_transition_thresholds = {
                'mind_wandering': float(tt.mind_wandering),
                'dmn_dan_ratio': float(tt.dmn_dan_ratio),
                'meta_awareness': float(tt.meta_awareness),
                'return_focus': float(tt.return_focus)
            }

            serial_state_transition_patterns = []
            for (frm, to, ts_dict, net_dict, fe) in state_transition_patterns:
                serial_state_transition_patterns.append({
                    'from': frm,
                    'to': to,
                    'thoughtseed_activations': {k: float(v) for k, v in ts_dict.items()},
                    'network_acts': {k: float(v) for k, v in net_dict.items()},
                    'free_energy': float(fe)
                })

            # Create named local variables for saved fields so they are easy to inspect
            transition_counts = serial_transition_counts
            transition_thresholds = serial_transition_thresholds
            natural_transitions = int(agent.natural_transition_count)
            forced_transitions = int(agent.forced_transition_count)
            transition_timestamps_serial = [int(x) for x in transition_timestamps]
            state_transition_patterns_serial = serial_state_transition_patterns
            distraction_buildup_rates_serial = [float(x) for x in agent.distraction_buildup_rates]

            average_activations_at_transition = aggregates.get('average_activations_at_transition', {})
            average_network_activations_by_state = aggregates.get('average_network_activations_by_state', {})
            average_free_energy_by_state = aggregates.get('average_free_energy_by_state', {})

            transition_stats = {
                'transition_counts': transition_counts,  # matrix of observed transitions between states
                'transition_thresholds': transition_thresholds,  # per-agent threshold params used during training
                'natural_transitions': natural_transitions,  # count of natural (policy-driven) transitions
                'forced_transitions': forced_transitions,  # count of forced transitions
                'transition_timestamps': transition_timestamps_serial,  # list of timesteps where transitions occurred
                'state_transition_patterns': state_transition_patterns_serial,  # activations & network states captured at transitions
                'distraction_buildup_rates': distraction_buildup_rates_serial,  # distraction pressure timecourse during run
                'average_activations_at_transition': average_activations_at_transition,  # mean thoughtseed activations at transitions
                'average_network_activations_by_state': average_network_activations_by_state,  # per-state mean network activations
                'average_free_energy_by_state': average_free_energy_by_state,  # per-state mean free energy
            }

            # Debug: report network values by state
            logging.info("%s NETWORK VALUES BY STATE:", agent.experience_level.upper())
            for state in agent.states:
                logging.info("  %s:", state)
                indices = [j for j, s in enumerate(agent.state_history) if s == state]

                if not indices:  # pragma: no cover
                    logging.info("    (No visits to this state)")
                    continue

                state_networks = {
                    net: float(np.mean([
                        agent.network_activations_history[j][net]
                        for j in indices
                    ])) for net in agent.networks
                }
                for net in agent.networks:
                    logging.info("    %s: %.2f", net, state_networks[net])

            # Save the transition statistics
            out_path = os.path.join(out_dir, f"transition_stats_{agent.experience_level}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(transition_stats, f, indent=2)
            logging.info("Saved transition stats -> %s", os.path.relpath(out_path))

            logging.info("Active Inference training complete for %s.", agent.experience_level)
            logging.info("  - Natural transitions: %d, Forced transitions: %d", agent.natural_transition_count, agent.forced_transition_count)

            # Generate JSON outputs (time series data and parameters)
            _save_json_outputs(agent, output_dir=out_dir)

        return agent
