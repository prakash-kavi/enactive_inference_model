"""Training orchestrator: BPTT loop with hierarchical free energy minimization.

Implements the "awareness as hyperparameter optimization" framework:
- L2 minimizes VFE: reconstruction + KL divergence + forward prediction error
- L3 modulates L2's precision (hyperparameter) via meta-awareness
- Phase 4: Forward model trains on action-conditioned prediction
"""

import numpy as np
import torch
import torch.optim as optim
from typing import Dict
import json
from pathlib import Path

from config import (
    STATES, NETWORKS, THOUGHTSEEDS, DEFAULTS, EPS,
    FORWARD_LOSS_BASE_WEIGHT, FORWARD_LOSS_PRECISION_SCALE,
    get_params, get_thoughtseed_priors
)
from process import Layer1Process
from agent import Layer2Agent
from monitor import Layer3Monitor
from blankets import MarkovBlanketL1L2, MarkovBlanketL2L3
from utils import bernoulli_kl, networks_to_tensor

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
            blanket_l2l3=self.blanket_l2l3,
            params=self.params
        )
        
        # Optimizer
        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.params['learning_rate'])
        
        # History tracking
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
        
        # Phase 4: Track previous observation for forward loss
        self._last_x_actual = None
    
    def train(self, timesteps: int = 10000, enable_learning: bool = True) -> Dict:
        """Run BPTT training.
        
        Args:
            timesteps: Total simulation steps
            enable_learning: Whether to update weights
        
        Returns:
            Training results dict
        """
        # Initialize
        self.process.reset(state='breath_focus')
        current_state = self.process.current_state
        
        # Initial thoughtseed activations
        device = next(self.agent.parameters()).device
        priors = get_thoughtseed_priors(current_state, self.level)
        activations_np = np.array([priors[ts] for ts in THOUGHTSEEDS], dtype=np.float32)
        activations_np += np.random.normal(0, self.params['init_noise_sigma'], size=len(activations_np))
        activations = torch.tensor(activations_np, dtype=torch.float32, device=device)
        activations = torch.clamp(activations, DEFAULTS['ACTIVATION_CLIP_MIN'], DEFAULTS['ACTIVATION_CLIP_MAX'])
        
        # Initialize blankets
        self.blanket_l1l2.update_active_states({
            'agent_bias': None,
            'l2tol1_enactive_bias': 0.0,
            'noise_reduction': 1.0,
            'transition_drive': 0.0
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
            
            for t_sub in range(steps_to_run):
                t = t_start + t_sub
                
                # ===== Layer 1: Generative Process =====
                network_acts, new_state = self.process.update(self.blanket_l1l2.active_states)
                
                # Record state transitions
                if new_state != current_state:
                    self.history['transitions'].append({
                        'timestamp': t,
                        'from': current_state,
                        'to': new_state,
                        'free_energy': self.history['free_energy'][-1] if self.history['free_energy'] else 0.0
                    })
                    current_state = new_state
                
                # Update L1→L2 sensory states
                self.blanket_l1l2.update_sensory_states(network_acts)
                
                # ===== Layer 2: Attentional Agent =====
                # Perception: Encode networks → thoughtseeds
                sensory_inference = self.agent.perceptual_inference()
                
                # Variational inference: Update thoughtseed dynamics
                activations = self.agent.update_thoughtseeds(
                    current_state=current_state,
                    activations=activations,
                    observed_networks=network_acts,
                    sensory_inference=sensory_inference
                ).detach()  # VI not part of BPTT gradient flow
                
                # Compute VFE (L2's free energy)
                free_energy = self._compute_vfe(
                    current_state, activations, network_acts, device
                )
                
                # ===== Layer 3: Metacognitive Monitor =====
                # Update meta-awareness from thoughtseeds
                meta_awareness = self.monitor.update_meta_awareness(current_state, activations)
                
                # Policy evaluation (EFE-based)
                policy_result = self.monitor.evaluate_policies(current_state, free_energy.item())
                
                # ===== Phase 4: Forward-Informed Action Selection =====
                prescription = self.agent.prescriptive_action(activations, current_state)
                
                # Update L2→L1 active states (action)
                self.blanket_l1l2.update_active_states(prescription)
                
                # Current observation (for forward model)
                x_current = networks_to_tensor(network_acts, NETWORKS)
                selected_action_mu = prescription['selected_action_mu']
                
                # Forward model loss: Predict current from previous state+action
                action_pred_error_val = 0.0
                if self._last_x_actual is not None:
                    # Predict current observation from previous (x, action)
                    x_pred = self.agent.vae.predict_next(
                        self._last_x_actual['x'],
                        self._last_x_actual['action']
                    )
                    action_pred_error = torch.mean((x_current.detach() - x_pred)**2)
                    action_pred_error_val = action_pred_error.item()
                    
                    # L3 precision weighting
                    precision = policy_result['precision_modulation']
                    action_loss_weight = FORWARD_LOSS_BASE_WEIGHT + FORWARD_LOSS_PRECISION_SCALE * precision
                    
                    # Combined loss
                    step_loss = free_energy + action_loss_weight * action_pred_error
                else:
                    # First step: only VFE
                    step_loss = free_energy
                
                # Accumulate loss across BPTT window, then run one backward pass.
                # This avoids retaining old graphs across repeated backward() calls.
                if enable_learning and step_loss.requires_grad:
                    window_loss = step_loss if window_loss is None else (window_loss + step_loss)
                
                # Store for next iteration
                self._last_x_actual = {
                    'x': x_current.detach(),
                    'action': selected_action_mu.detach()
                }
                
                # Record history
                self.history['states'].append(current_state)
                self.history['free_energy'].append(free_energy.detach().item())
                self.history['meta_awareness'].append(meta_awareness)
                self.history['action_errors'].append(action_pred_error_val)
                
                # Convert network activations tensors to floats for JSON serialization
                network_acts_serializable = {
                    net: float(val.detach().item()) if isinstance(val, torch.Tensor) else float(val)
                    for net, val in network_acts.items()
                }
                self.history['network_activations'].append(network_acts_serializable)
                
                # Record thoughtseed activations and dominant thoughtseed
                ts_acts = activations.detach().cpu().numpy().tolist()
                self.history['thoughtseed_activations'].append(ts_acts)
                dominant_idx = np.argmax(ts_acts)
                self.history['dominant_thoughtseed'].append(THOUGHTSEEDS[dominant_idx])
            
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
    
    def _compute_vfe(self, state: str, z: torch.Tensor, 
                     network_acts: Dict, device: torch.device) -> torch.Tensor:
        """Compute variational free energy: reconstruction + KL divergence."""
        # Decode thoughtseeds → predicted networks
        recon_x = self.agent.decode_with_state(z)
        
        # Observed networks
        observed_x = networks_to_tensor(network_acts, NETWORKS, device=device, detach=True)
        
        # Reconstruction loss (observation model)
        recon_loss = torch.nn.functional.mse_loss(recon_x, observed_x)
        
        # KL divergence (prior)
        prior = self.agent.mu_params[state].detach()
        kl_div = bernoulli_kl(z, prior, EPS)
        
        # Total VFE
        beta = self.params['kl_beta']
        vfe = recon_loss + beta * kl_div
        
        return vfe
    
    def _package_results(self) -> Dict:
        """Package training results for analysis."""
        # Compute dwell times directly from contiguous state runs.
        # This is deterministic and does not depend on transition bookkeeping.
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
            'dominant_ts_history': self.history['dominant_thoughtseed']
        }
    
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


def train_meditation(experience_level: str = 'expert', 
                     timesteps: int = 10000, 
                     seed: int = 42,
                     debug_anomaly: bool = False,
                     save_results: bool = True,
                     output_dir: str = 'data/lean_results') -> Dict:
    """Convenience function for training meditation model.
    
    Args:
        experience_level: 'expert' or 'novice'
        timesteps: Number of simulation steps
        seed: Random seed
        debug_anomaly: Enable PyTorch anomaly detection for debugging
        save_results: Whether to save to disk
        output_dir: Output directory for results
    
    Returns:
        Training results dict
    """
    trainer = MeditationTrainer(
        experience_level=experience_level,
        seed=seed,
        debug_anomaly=debug_anomaly
    )
    results = trainer.train(timesteps=timesteps, enable_learning=True)
    
    if save_results:
        trainer.save_results(output_dir=output_dir)
    
    return results
