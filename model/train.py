"""Training orchestrator: BPTT loop with hierarchical free energy minimization.

Implements the "awareness as hyperparameter optimization" framework:
- L2 minimizes VFE (F): reconstruction + KL divergence + forward prediction error
- L3 modulates L2 precision via meta-awareness
- Forward model trains on action-conditioned prediction
"""

import numpy as np
import torch
import torch.optim as optim
from typing import Dict, Optional
import json
from pathlib import Path

from utils.config import (
    STATES, NETWORKS, THOUGHTSEEDS, DEFAULTS,
    FORWARD_LOSS_BASE_WEIGHT, FORWARD_LOSS_PRECISION_SCALE,
    get_params, get_thoughtseed_priors
)
from .process import Layer1Process
from .agent import Layer2Agent
from .monitor import Layer3Monitor
from .blankets import MarkovBlanketL1L2, MarkovBlanketL2L3
from utils.math_utils import networks_to_tensor, clip_probability

class MeditationTrainer:
    """BPTT training for hierarchical meditation model."""
    
    def __init__(self, experience_level: str = 'expert', seed: int = 42, debug_anomaly: bool = False):
        self.level = experience_level
        self.seed = seed
        self.params = get_params(experience_level)
        self.debug_anomaly = debug_anomaly
        
        if self.debug_anomaly:
            torch.autograd.set_detect_anomaly(True)
        
        # Set seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Markov blankets
        self.blanket_l1l2 = MarkovBlanketL1L2(smoothing=0.7)
        self.blanket_l2l3 = MarkovBlanketL2L3(smoothing=0.7)
        
        # Hierarchical layers
        self.process = Layer1Process(experience_level=experience_level, seed=seed)
        self.agent = Layer2Agent(
            experience_level=experience_level,
            blanket_l1l2=self.blanket_l1l2,
            blanket_l2l3=self.blanket_l2l3
        )
        self.monitor = Layer3Monitor(
            experience_level=experience_level,
            blanket_l2l3=self.blanket_l2l3
        )
        
        # Optimizer
        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.params['learning_rate'])
        
        # History tracking (Initialized in reset)
        self.history = {}
        self._last_x_actual = None
    
    def step(self, t: int, current_state: str, activations: torch.Tensor, enable_learning: bool = True) -> tuple[torch.Tensor, str, torch.Tensor, Dict]:
        """Execute a single simulation step.
        
        Args:
            t: Current timestep
            current_state: State at start of step
            activations: Current thoughtseed activations (L2 hidden state)
            enable_learning: Whether to track gradients
            
        Returns:
            step_loss: Loss for this step (F + forward prediction error)
            new_state: Updated state
            activations: Updated thoughtseed activations
            metrics: Dict of step metrics for history
        """
        # ===== Layer 1: Generative Process =====
        # Update L1 state based on previous active states (action)
        network_acts, new_state = self.process.update(self.blanket_l1l2.active_states)
        
        # Update L1->L2 sensory states (observations)
        self.blanket_l1l2.update_sensory_states(network_acts)
        
        # ===== Layer 2: Attentional Agent =====
        # Perception: Encode networks -> thoughtseeds
        z_recognition = self.agent.infer_z_from_x()
        
        # Variational inference: Update thoughtseed dynamics
        activations = self.agent.update_posterior_z(
            current_state=new_state,
            activations=activations,
            observed_networks=network_acts,
            z_recognition=z_recognition
        ).detach()  # VI not part of BPTT gradient flow
        
        # Compute VFE (L2's free energy)
        free_energy = self.agent.compute_vfe(
            new_state, activations, network_acts
        )
        
        # ===== Layer 3: Metacognitive Monitor =====
        # Update meta-awareness using new state
        meta_awareness = self.monitor.update_meta_awareness(new_state, activations)
        
        # ===== Policy Inference =====
        prescription = self.agent.infer_pi(activations, new_state)
        
        # Update L2->L1 active states (action policy)
        self.blanket_l1l2.update_active_states(prescription)
        
        # Forward model loss calculation
        step_loss = free_energy
        forward_prediction_error_val = 0.0
        meta_result = None
        
        # Current observation (for forward model)
        x_current = networks_to_tensor(network_acts, NETWORKS)
        selected_action_mu = prescription['selected_action_mu']
        
        if self._last_x_actual is not None:
            # Predict current observation from previous (x, action)
            x_pred = self.agent.vae.predict_next(
                self._last_x_actual['x'],
                self._last_x_actual['action']
            )
            forward_prediction_error = torch.nn.functional.binary_cross_entropy(
                x_pred,
                x_current.detach(),
            )
            forward_prediction_error_val = forward_prediction_error.item()
            
            # L3 precision posterior (Gamma update from prediction error)
            meta_result = self.monitor.infer_meta_posterior(forward_prediction_error_val)
            precision = meta_result['precision_sensory']
            precision_gain = clip_probability(precision)
            forward_weight = (FORWARD_LOSS_BASE_WEIGHT + FORWARD_LOSS_PRECISION_SCALE * precision_gain)
            forward_weight = max(forward_weight, 0.5)
            
            # Combined loss
            step_loss = free_energy + forward_weight * forward_prediction_error
            
            # Recognition Loss (Amortized Inference)
            # Train encoder to predict the optimized thoughtseeds (z) from observations (x)
            # This enables "Expert Intuition" mechanism
            if enable_learning and self.level == 'expert':
                # Re-run encoder on current observation
                z_recognition = self.agent.vae.encode(x_current.detach())
                
                # Target is the optimized z from VFE loop (which is detached)
                # Recognition loss = MSE(Encoder(x), VI_Optimized_z)
                rec_loss = torch.nn.functional.mse_loss(z_recognition, activations.detach())
                
                # Add to total loss (weighted)
                step_loss = step_loss + rec_loss
            
        if meta_result is not None and t % 500 == 0:
            print(
                f"[diag t={t}] gamma={meta_result['policy_precision']:.4f} "
                f"precision_sensory={meta_result['precision_sensory']:.4f} "
                f"policy_confidence={prescription['policy_confidence']:.4f} "
                f"policy_drive={prescription.get('policy_drive', 0.0):.4f} "
                f"forward_pe={forward_prediction_error_val:.6f}"
            )

        # Store for next iteration
        self._last_x_actual = {
            'x': x_current.detach(),
            'action': selected_action_mu.detach()
        }
        
        # Collect metrics
        # Convert network activations tensors to floats for JSON serialization
        network_acts_serializable = {
            net: float(val.detach().item()) if isinstance(val, torch.Tensor) else float(val)
            for net, val in network_acts.items()
        }
        
        ts_acts = activations.detach().cpu().numpy().tolist()
        dominant_idx = np.argmax(ts_acts)
        
        metrics = {
            'timestamp': t,
            'current_state': current_state,
            'new_state': new_state,
            'free_energy': free_energy.detach().item(),
            'meta_awareness': meta_awareness,
            'action_error': forward_prediction_error_val,
            'network_activations': network_acts_serializable,
            'thoughtseed_activations': ts_acts,
            'dominant_thoughtseed': THOUGHTSEEDS[dominant_idx]
        }
        
        return step_loss, new_state, activations, metrics

    def train(
        self,
        timesteps: int = 10000,
        enable_learning: bool = True,
        reseed_rng: bool = False,
        run_seed: Optional[int] = None,
    ) -> Dict:
        """Run BPTT training.
        
        Args:
            timesteps: Total simulation steps
            enable_learning: Whether to update weights
            reseed_rng: Reinitialize random generators for an isolated/reproducible run
            run_seed: Optional per-run seed override (used only when reseed_rng=True)
        
        Returns:
            Training results dict
        """
        # Initialize run state
        self._reset_run_state(reseed_rng=reseed_rng, run_seed=run_seed)
        
        # Log Phenotype status
        if self.level == 'novice':
            print(f"PHENOTYPE: NOVICE (Encoder Frozen - No Amortized Inference Learning)")
        else:
            print(f"PHENOTYPE: EXPERT (Encoder Unfrozen - Amortized Inference Learning Active)")

        self.process.reset(state='breath_focus')
        current_state = self.process.current_state
        
        # Initial thoughtseed activations
        device = next(self.agent.parameters()).device
        priors = get_thoughtseed_priors(current_state)
        activations_np = np.array([priors[ts] for ts in THOUGHTSEEDS], dtype=np.float32)
        activations_np += np.random.normal(0, self.params['noise_sigma'], size=len(activations_np))
        activations = torch.tensor(activations_np, dtype=torch.float32, device=device)
        activations = torch.clamp(activations, DEFAULTS['ACTIVATION_CLIP_MIN'], DEFAULTS['ACTIVATION_CLIP_MAX'])
        
        # Initialize blankets
        self.blanket_l1l2.update_active_states({
            'mu_x': None,
            'precision_gain': 0.0,
            'noise_reduction': 1.0,
            'policy_confidence': 0.0,
            'policy_drive': 0.0
        })
        self.blanket_l2l3.reset()
        
        # Update meta-awareness
        self.monitor.update_meta_awareness(current_state, activations)
        
        # BPTT windowing
        bptt_steps = 50
        
        for t_start in range(0, timesteps, bptt_steps):
            # Clear blanket history (BPTT window boundary)
            self.blanket_l1l2.sensory_states.clear()
            self.blanket_l2l3.sensory_states.clear()
            
            # Zero gradients at window start
            if enable_learning:
                self.optimizer.zero_grad()
                window_loss = None
            
            steps_to_run = min(bptt_steps, timesteps - t_start)
            
            # Control autograd context (disable for inference-only runs to save memory/compute)
            with torch.set_grad_enabled(enable_learning):
                for t_sub in range(steps_to_run):
                    t = t_start + t_sub
                    
                    # Execute step
                    step_loss, new_state, activations, metrics = self.step(
                        t=t,
                        current_state=current_state,
                        activations=activations,
                        enable_learning=enable_learning
                    )
                    
                    # Record state transitions
                    if new_state != current_state:
                        self.history['transitions'].append({
                            'timestamp': t,
                            'from': current_state,
                            'to': new_state,
                            'free_energy': metrics['free_energy']
                        })
                        current_state = new_state
                    
                    # Accumulate loss
                    if enable_learning and step_loss.requires_grad:
                        window_loss = step_loss if window_loss is None else (window_loss + step_loss)
                    
                    # Record history
                    self._append_history(current_state, metrics)
            
            # Gradient step (after accumulating gradients from all steps in window)
            if enable_learning:
                if window_loss is not None and window_loss.requires_grad:
                    window_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=1.0)
                    self.optimizer.step()
            
            # Detach states (BPTT boundary)
            with torch.no_grad():
                self.process.x = self.process.x.detach()
                self.process.smoothed_x = self.process.smoothed_x.detach()
                if hasattr(self.agent, 'z_ema'):
                    self.agent.z_ema = self.agent.z_ema.detach()
                activations = activations.detach()
        
        return self._package_results()
    
    def _append_history(self, current_state: str, metrics: Dict) -> None:
        """Append per-timestep metrics to training history."""
        self.history['states'].append(current_state)
        self.history['free_energy'].append(metrics['free_energy'])
        self.history['meta_awareness'].append(metrics['meta_awareness'])
        self.history['action_errors'].append(metrics['action_error'])
        self.history['network_activations'].append(metrics['network_activations'])
        self.history['thoughtseed_activations'].append(metrics['thoughtseed_activations'])
        self.history['dominant_thoughtseed'].append(metrics['dominant_thoughtseed'])

    
    def _package_results(self) -> Dict:
        """Package training results for analysis."""
        # Compute dwell times directly from contiguous state runs.
        dwell_times = {state: [] for state in STATES}
        state_sequence = self.history['states']
        if state_sequence:
            run_state = state_sequence[0]
            run_len = 1
            for state in state_sequence[1:]:
                if state == run_state:
                    run_len += 1
                else:
                    dwell_times[run_state].append(run_len)
                    run_state = state
                    run_len = 1
            dwell_times[run_state].append(run_len)
        
        avg_dwell = {state: np.mean(dwells) if dwells else 0.0 
                     for state, dwells in dwell_times.items()}
        
        # Compute transition matrix
        trans_matrix = {s: {t: 0 for t in STATES} for s in STATES}
        for trans in self.history['transitions']:
            trans_matrix[trans['from']][trans['to']] += 1
        
        # Normalize to probabilities
        for from_state in trans_matrix:
            total = sum(trans_matrix[from_state].values())
            if total > 0:
                trans_matrix[from_state] = {
                    to_state: count / total 
                    for to_state, count in trans_matrix[from_state].items()
                }
        
        # Action prediction errors by state
        action_errors_by_state = {state: [] for state in STATES}
        for i, state in enumerate(self.history['states']):
            if i < len(self.history['action_errors']):
                action_errors_by_state[state].append(self.history['action_errors'][i])
        
        avg_action_errors = {
            state: np.mean(errors) if errors else 0.0
            for state, errors in action_errors_by_state.items()
        }
        
        return {
            'experience_level': self.level,
            'seed': self.seed,
            'timesteps': len(self.history['states']),
            'free_energy_history': self.history['free_energy'],
            'meta_awareness_history': self.history['meta_awareness'],
            'state_history': self.history['states'],  # Renamed for viz compatibility
            'state_sequence': self.history['states'],  # Keep for backward compat
            'transitions': self.history['transitions'],
            'avg_dwell_times': avg_dwell,
            'transition_matrix': trans_matrix,
            'avg_action_errors': avg_action_errors,
            'final_free_energy': self.history['free_energy'][-1] if self.history['free_energy'] else 0.0,
            'network_activations_history': self.history['network_activations'],
            'thoughtseed_activations_history': self.history['thoughtseed_activations'],
            'dominant_ts_history': self.history['dominant_thoughtseed'],
            'action_errors_history': self.history['action_errors']
        }
    
    def _reset_run_state(self, reseed_rng: bool = False, run_seed: Optional[int] = None):
        """Reset internal state for a new run."""
        self.history = {
            'states': [],
            'free_energy': [],
            'meta_awareness': [],
            'transitions': [],
            'action_errors': [],
            'network_activations': [],
            'thoughtseed_activations': [],
            'dominant_thoughtseed': []
        }
        self._last_x_actual = None
        self.blanket_l1l2.reset()
        self.monitor.reset()
        self.agent.z_ema.zero_()
        self.agent.z_ema_initialized = False

        if reseed_rng:
            seed = self.seed if run_seed is None else int(run_seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
            self.process.rng = np.random.RandomState(seed)

    def save_results(self, output_dir: str = 'data/lean_results') -> None:
        """Save training results to JSON."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = self._package_results()
        
        # Save compact JSON
        filepath = output_path / f'training_results_{self.level}_seed{self.seed}.json'
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {filepath}")

def train_meditation(
    experience_level: str = 'expert',
    timesteps: int = 10000,
    seed: int = 42,
    debug_anomaly: bool = False,
    reseed_rng: bool = False,
    run_seed: Optional[int] = None,
    save_results: bool = True,
    output_dir: str = 'data/lean_results',
) -> Dict:
    """Convenience wrapper for MeditationTrainer."""
    trainer = MeditationTrainer(
        experience_level=experience_level,
        seed=seed,
        debug_anomaly=debug_anomaly,
    )
    results = trainer.train(
        timesteps=timesteps,
        enable_learning=True,
        reseed_rng=reseed_rng,
        run_seed=run_seed,
    )

    if save_results:
        trainer.save_results(output_dir=output_dir)

    return results
