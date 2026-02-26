"""Training orchestrator: variational EM loop for hierarchical free energy minimisation.

Timescale separation (Dempster et al., 1977; Friston et al., 2019):
- E-step (every t): perceptual inference via VI + policy selection. No parameter gradients.
- M-step (every T=50 steps): parameter update via BPTT over the accumulated buffer.
"""

import numpy as np
import torch
import torch.optim as optim
from typing import Dict, List, Optional
import json
from pathlib import Path

from utils.config import (
    STATES, NETWORKS, THOUGHTSEEDS, CLIP_MIN, CLIP_MAX, EPS,
    THOUGHTSEED_STATE_PRIORS,
    PRECISION_CLIP_MIN, PRECISION_CLIP_MAX,
)
from .l1_generative_process import Layer1Process
from .l2_recognition import Layer2Agent
from .l3_metacognition import Layer3Monitor
from .markov_blankets import MarkovBlanketL1L2, MarkovBlanketL2L3
from .phenotype import PhenotypeConfig, EXPERT_PHENOTYPE, RECOGNITION_LOSS_ALPHA
from utils.math_utils import (
    networks_to_tensor,
    precision_from_surprisal,
    mse_error,
    recon_error,
    prior_error,
    forward_error,
    compute_precision_sensory,
)

class MeditationTrainer:
    """BPTT training for hierarchical meditation model."""
    
    def __init__(self, phenotype: PhenotypeConfig = None, seed: int = 42):
        self.phenotype = phenotype if phenotype is not None else EXPERT_PHENOTYPE
        self.level = self.phenotype.level
        self.seed = seed

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.blanket_l1l2 = MarkovBlanketL1L2(smoothing=0.0)
        self.blanket_l2l3 = MarkovBlanketL2L3(smoothing=0.0)

        self.process = Layer1Process(phenotype=self.phenotype, seed=seed)
        self.agent = Layer2Agent(
            phenotype=self.phenotype,
            blanket_l1l2=self.blanket_l1l2,
            blanket_l2l3=self.blanket_l2l3
        )
        self.monitor = Layer3Monitor(blanket_l2l3=self.blanket_l2l3)

        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.phenotype.learning_rate)
        
        self.history = {}
        self._last_x_actual = None
    
    # ------------------------------------------------------------------
    # E-step: perceptual inference and action (no parameter gradients)
    # ------------------------------------------------------------------
    def _e_step(
        self, t: int, activations: torch.Tensor
    ) -> tuple[Dict, str, torch.Tensor, Dict]:
        """E-step: forward inference over one timestep.

        Runs entirely under torch.no_grad(). Infers z* via VI,
        selects action via EFE, and steps L1. Returns a buffer
        entry containing the detached tensors the M-step will re-run
        differentiable passes over.
        """
        with torch.no_grad():
            # Cache (x_{t-1}, a_{t-1}) for M-step forward surprisal
            x_prev     = self._last_x_actual['x'].clone()      if self._last_x_actual is not None else None
            action_prev = self._last_x_actual['action'].clone() if self._last_x_actual is not None else None

            # ===== Layer 1: Generative Process =====
            network_acts, new_state = self.process.update(self.blanket_l1l2.active_states)

            dwell_progress = 0.0
            if self.process.current_max_dwell > 0:
                dwell_progress = min(1.0, self.process.current_dwell / self.process.current_max_dwell)
            sensory_payload = dict(network_acts)
            sensory_payload['dwell_progress'] = float(dwell_progress)
            self.blanket_l1l2.update_sensory_states(sensory_payload)

            x_current = networks_to_tensor(network_acts, NETWORKS)

            # Forward prediction error - used for precision + diagnostics
            # (gradient-tracked version is recomputed in M-step)
            forward_error_val = 0.0
            base_precision = float(self.blanket_l2l3.active_states.get('precision_sensory', 0.5))
            if x_prev is not None:
                x_pred = self.agent.thoughtseed_model.predict_next(x_prev, action_prev)
                fwd_err_raw = forward_error(x_pred, x_current)
                forward_error_val = float(fwd_err_raw.item())
                base_precision = precision_from_surprisal(forward_error_val, EPS)
                self.blanket_l2l3.update_active_states({'precision_sensory': base_precision})

            # ===== Layer 3: Metacognitive Monitor (precision update) =====
            meta_awareness = self.monitor.update_meta_awareness()

            clip_min = PRECISION_CLIP_MIN
            clip_max = PRECISION_CLIP_MAX
            precision_sensory = compute_precision_sensory(
                base_precision, meta_awareness, EPS, clip_min, clip_max
            )
            self.blanket_l2l3.update_active_states({'precision_sensory': precision_sensory})

            # ===== Layer 2: Attentional Agent =====
            activations, _ = self.agent.infer_z_step(
                current_state=new_state,
                activations=activations,
            )
            activations = activations.detach()  # z* fixed; VI not part of BPTT

            # VFE as scalar metric (M-step will recompute with grad)
            free_energy_val = float(
                self.agent.compute_vfe(
                    new_state, activations, x_current, precision_sensory=precision_sensory
                ).item()
            )

            # Update L2->L3 sensory interface for next step
            thoughtseed_dict = {ts: float(activations[i].item()) for i, ts in enumerate(THOUGHTSEEDS)}
            self.blanket_l2l3.update_sensory_states({
                'current_state':          new_state,
                'dwell_progress':         dwell_progress,
                'thoughtseed_activations': thoughtseed_dict,
            })
            self.monitor.write_policy_prior(new_state)

            # ===== Policy Inference =====
            prescription      = self.agent.infer_pi(new_state)
            selected_action_mu = prescription['selected_action_mu']
            self.monitor.update_policy_state(new_state, prescription['q_pi'])
            self.blanket_l1l2.update_active_states({
                'mu_x': prescription['mu_x'],
                'transition_drive': prescription['transition_drive'],
                'policy_state_probs': prescription.get('policy_state_probs'),
            })

            # Update rolling store for next E-step
            self._last_x_actual = {
                'x':      x_current.detach(),
                'action': selected_action_mu.detach(),
            }

            # Buffer entry - all tensors detached; M-step re-runs grad passes
            buf: Dict = {
                'x_prev':      x_prev,
                'action_prev': action_prev,
                'x_curr':      x_current.detach(),
                'z_star':      activations,          # already detached by VI
                'network_acts': network_acts,
                'state':       new_state,
                'precision_sensory': precision_sensory,
            }

            # Metrics (float approximation of per-step loss for diagnostics)
            network_acts_serializable = {
                net: float(val.detach().item()) if isinstance(val, torch.Tensor) else float(val)
                for net, val in network_acts.items()
            }
            ts_acts      = activations.cpu().numpy().tolist()
            dominant_idx = int(np.argmax(ts_acts))
            metrics = {
                'timestamp':               t,
                'free_energy':             free_energy_val,
                'loss':                    free_energy_val + forward_error_val,
                'meta_awareness':          meta_awareness,
                'action_error':            forward_error_val,
                'efe_prag_mean':           float(prescription.get('efe_prag_mean', 0.0)),
                'efe_epi_mean':            float(prescription.get('efe_epi_mean', 0.0)),
                'network_activations':     network_acts_serializable,
                'thoughtseed_activations': ts_acts,
                'dominant_thoughtseed':    THOUGHTSEEDS[dominant_idx],
            }

        return buf, new_state, activations, metrics

    # ------------------------------------------------------------------
    # M-step: parameter update over one BPTT window
    # ------------------------------------------------------------------
    def _m_step(self, buffer: List[Dict]) -> Optional[torch.Tensor]:
        """M-step: update (phi, theta, psi) from accumulated E-step buffer.

        Re-runs only the three differentiable forward passes over the stored
        detached (z*, x) pairs - no VI, no blanket I/O:
          - VFE   : decoder theta  (reconstruction + prior matching)
          - S_fwd : forward model psi  (next-step prediction error)
          - L_rec : encoder phi        (amortised inference alignment)

        This is the variational EM M-step; beliefs z* are fixed from the E-step.
        Friston et al. (2019); Dempster, Laird & Rubin (1977).
        """
        window_loss: Optional[torch.Tensor] = None
        for entry in buffer:
            # -- VFE: decoder gradient (theta) --------------------------------
            vfe = self.agent.compute_vfe(
                entry['state'], entry['z_star'], entry['x_curr'],
                precision_sensory=entry.get('precision_sensory')
            )
            step_loss = vfe

            # Forward surprisal: forward-model gradient (psi)
            if entry['x_prev'] is not None:
                x_pred = self.agent.thoughtseed_model.predict_next(entry['x_prev'], entry['action_prev'])
                step_loss = step_loss + forward_error(x_pred, entry['x_curr'])

            # Recognition loss: encoder gradient (phi)
            rec_scale = RECOGNITION_LOSS_ALPHA
            if rec_scale > 0.0:
                z_enc     = self.agent.thoughtseed_model.encode(entry['x_curr'])
                step_loss = step_loss + rec_scale * mse_error(z_enc, entry['z_star'])

            window_loss = step_loss if window_loss is None else (window_loss + step_loss)

        return window_loss

    def _run_e_step_window(
        self,
        t_start: int,
        steps_to_run: int,
        activations: torch.Tensor,
        current_state: str,
    ) -> tuple[List[Dict], torch.Tensor, str]:
        """Run the E-step loop for one window and record history."""
        e_step_buffer: List[Dict] = []
        for t_sub in range(steps_to_run):
            t = t_start + t_sub

            buf, new_state, activations, metrics = self._e_step(t, activations)
            e_step_buffer.append(buf)

            if new_state != current_state:
                transition_debug = getattr(self.process, 'last_transition_info', None)
                self.history['transitions'].append({
                    'timestamp':   t,
                    'from':        current_state,
                    'to':          new_state,
                    'free_energy': metrics['free_energy'],
                    'weighted_exit_probs': transition_debug.get('weighted_exit_probs') if transition_debug else None,
                    'base_exit_probs': transition_debug.get('base_exit_probs') if transition_debug else None,
                    'policy_state_probs': transition_debug.get('policy_state_probs') if transition_debug else None,
                })
                current_state = new_state

            self.history['states'].append(current_state)
            self.history['free_energy'].append(metrics['free_energy'])
            self.history['loss'].append(metrics['loss'])
            self.history['meta_awareness'].append(metrics['meta_awareness'])
            self.history['action_errors'].append(metrics['action_error'])
            self.history['efe_prag'].append(metrics['efe_prag_mean'])
            self.history['efe_epi'].append(metrics['efe_epi_mean'])
            self.history['network_activations'].append(metrics['network_activations'])
            self.history['thoughtseed_activations'].append(metrics['thoughtseed_activations'])
            self.history['dominant_thoughtseed'].append(metrics['dominant_thoughtseed'])

        return e_step_buffer, activations, current_state

    def _run_m_step_window(self, buffer: List[Dict], enable_learning: bool) -> None:
        """Run the M-step update for one window."""
        if not enable_learning:
            return
        self.optimizer.zero_grad()
        window_loss = self._m_step(buffer)
        if window_loss is not None and window_loss.requires_grad:
            window_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=1.0)
            self.optimizer.step()

    def train(
        self,
        timesteps: int = 10000,
        enable_learning: bool = True,
        reseed_rng: bool = False,
        run_seed: Optional[int] = None,
    ) -> Dict:
        """Run variational EM training.

        Each BPTT window T runs:
          1. E-step loop  - T calls to _e_step(), no parameter gradients.
          2. M-step once  - _m_step(buffer) re-runs differentiable passes and
                            backprops through (phi, theta, psi).

        Args:
            timesteps:     Total simulation steps
            enable_learning: Whether to update weights (M-step)
            reseed_rng:    Reinitialise random generators for reproducibility
            run_seed:      Per-run seed override (used only with reseed_rng)

        Returns:
            Training results dict
        """
        self._reset_run_state(reseed_rng=reseed_rng, run_seed=run_seed)

        print(f"PHENOTYPE: {self.phenotype.label} "
              f"(lr={self.phenotype.learning_rate:.4f}, "
              f"alpha_rec={RECOGNITION_LOSS_ALPHA:.3f})")

        self.process.reset(state='breath_focus')
        current_state = self.process.current_state

        # Initial thoughtseed activations
        priors = THOUGHTSEED_STATE_PRIORS[current_state].copy()
        activations_np = np.array([priors[ts] for ts in THOUGHTSEEDS], dtype=np.float32)
        activations = torch.tensor(activations_np, dtype=torch.float32)
        activations = torch.clamp(activations, CLIP_MIN, CLIP_MAX)

        # Initialise blankets
        self.blanket_l1l2.update_active_states({
            'mu_x': None,
            'transition_drive': 0.0
        })
        self.blanket_l2l3.reset()
        thoughtseed_dict = {ts: float(activations[i].item()) for i, ts in enumerate(THOUGHTSEEDS)}
        self.blanket_l2l3.update_sensory_states({
            'current_state': current_state,
            'dwell_progress': 0.0,
            'thoughtseed_activations': thoughtseed_dict,
        })

        bptt_steps = 50

        for t_start in range(0, timesteps, bptt_steps):
            # Window boundary: clear sensory states (hidden state carries across)
            self.blanket_l1l2.sensory_states.clear()
            self.blanket_l2l3.sensory_states.clear()

            steps_to_run = min(bptt_steps, timesteps - t_start)

            # E-step loop (no gradient accumulation)
            e_step_buffer, activations, current_state = self._run_e_step_window(
                t_start, steps_to_run, activations, current_state
            )

            # M-step: gradient update over window
            self._run_m_step_window(e_step_buffer, enable_learning)

            # BPTT boundary detach (activations already detached from E-step)
            with torch.no_grad():
                self.process.x = self.process.x.detach()
                activations = activations.detach()

        return self._package_results()

    
    def _package_results(self) -> Dict:
        """Package training results for analysis."""
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
            'loss_history': self.history['loss'],
            'meta_awareness_history': self.history['meta_awareness'],
            'state_history': self.history['states'],  # Renamed for viz compatibility
            'transitions': self.history['transitions'],
            'avg_action_errors': avg_action_errors,
            'final_free_energy': self.history['free_energy'][-1] if self.history['free_energy'] else 0.0,
            'final_loss': self.history['loss'][-1] if self.history['loss'] else 0.0,
            'network_activations_history': self.history['network_activations'],
            'thoughtseed_activations_history': self.history['thoughtseed_activations'],
            'dominant_ts_history': self.history['dominant_thoughtseed'],
            'action_errors_history': self.history['action_errors'],
            'efe_prag_history': self.history['efe_prag'],
            'efe_epi_history': self.history['efe_epi'],
        }
    
    def _reset_run_state(self, reseed_rng: bool = False, run_seed: Optional[int] = None):
        """Reset internal state for a new run."""
        self.history = {
            'states': [],
            'free_energy': [],
            'loss': [],
            'meta_awareness': [],
            'transitions': [],
            'action_errors': [],
            'efe_prag': [],
            'efe_epi': [],
            'network_activations': [],
            'thoughtseed_activations': [],
            'dominant_thoughtseed': []
        }
        self._last_x_actual = None
        self.blanket_l1l2.reset()
        self.monitor.reset()

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
    phenotype: PhenotypeConfig = None,
    timesteps: int = 10000,
    seed: int = 42,
    reseed_rng: bool = False,
    run_seed: Optional[int] = None,
    save_results: bool = True,
    output_dir: str = 'data/lean_results',
) -> Dict:
    """Convenience wrapper for MeditationTrainer.

    Pass ``phenotype=EXPERT_PHENOTYPE`` or ``phenotype=NOVICE_PHENOTYPE``.
    Defaults to EXPERT_PHENOTYPE if omitted.
    """
    trainer = MeditationTrainer(
        phenotype=phenotype,
        seed=seed,
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
