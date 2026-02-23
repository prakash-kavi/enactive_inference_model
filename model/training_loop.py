"""Training orchestrator: variational EM loop for hierarchical free energy minimisation.

Timescale separation (Dempster et al., 1977; Friston et al., 2019):
- E-step (every t): perceptual inference via VI + policy selection (Eqs. 5-8).  No parameter gradients.
- M-step (every T=50 steps): parameter update via BPTT over the accumulated buffer (Eq. 9).

Key equation references:
- Eq. (2): VFE = reconstruction surprisal + KL complexity
- Eq. (4): lambda_sens integrates base (forward surprisal) and meta-awareness m_t
- Eq. (9): L = VFE + S_fwd + alpha_rec * L_rec  (M-step loss)
- L3: meta-awareness m_t, learned policy prior
"""

import numpy as np
import torch
import torch.optim as optim
from typing import Dict, List, Optional
import json
from pathlib import Path

from utils.config import (
    STATES, NETWORKS, THOUGHTSEEDS, DEFAULTS, EPS,
    THOUGHTSEED_STATE_PRIORS,
)
from .l1_generative_process import Layer1Process
from .l2_recognition import Layer2Agent
from .l3_metacognition import Layer3Monitor
from .markov_blankets import MarkovBlanketL1L2, MarkovBlanketL2L3
from .phenotype import PhenotypeConfig, EXPERT_PHENOTYPE
from utils.math_utils import (
    networks_to_tensor,
    precision_from_surprisal,
    integrate_precision_logit,
    bernoulli_nll,
    mse_error,
)

class MeditationTrainer:
    """BPTT training for hierarchical meditation model."""
    
    def __init__(self, phenotype: PhenotypeConfig = None, seed: int = 42):
        self.phenotype = phenotype if phenotype is not None else EXPERT_PHENOTYPE
        self.level = self.phenotype.level
        self.seed = seed
        self.params = {'learning_rate': self.phenotype.learning_rate}

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
        self._loss_balance = {}
    
    # ------------------------------------------------------------------
    # E-step: perceptual inference and action (no parameter gradients)
    # ------------------------------------------------------------------
    def _e_step(
        self, t: int, activations: torch.Tensor
    ) -> tuple[Dict, str, torch.Tensor, Dict]:
        """E-step: forward inference over one timestep (Eqs. 1-8).

        Runs entirely under torch.no_grad(). Infers z* via VI (Eq. 3),
        selects action via EFE (Eqs. 5-8), and steps L1. Returns a buffer
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

            # Forward prediction error — float for precision update
            # (gradient-tracked version is recomputed in M-step)
            forward_prediction_error_val = 0.0
            base_precision = float(self.blanket_l2l3.active_states.get('precision_sensory', 0.5))
            if x_prev is not None:
                x_pred = self.agent.vae.predict_next(x_prev, action_prev)
                fwd_err_val = bernoulli_nll(x_pred, x_current, EPS)
                forward_prediction_error_val = float(fwd_err_val.item())
                base_precision = precision_from_surprisal(forward_prediction_error_val, EPS)
                self.blanket_l2l3.update_active_states({'precision_sensory': base_precision})

            # ===== Layer 2: Attentional Agent =====
            activations, _ = self.agent.infer_z_step(
                current_state=new_state,
                activations=activations,
                observed_networks=network_acts,
            )
            activations = activations.detach()  # z* fixed; VI not part of BPTT

            # VFE as scalar metric (M-step will recompute with grad)
            free_energy_val = float(
                self.agent.compute_vfe(new_state, activations, network_acts).item()
            )

            # ===== Layer 3: Metacognitive Monitor =====
            thoughtseed_dict = {ts: float(activations[i].item()) for i, ts in enumerate(THOUGHTSEEDS)}
            self.blanket_l2l3.update_sensory_states({
                'current_state':          new_state,
                'dwell_progress':         dwell_progress,
                'thoughtseed_activations': thoughtseed_dict,
            })
            meta_awareness = self.monitor.update_meta_awareness()

            clip_min = DEFAULTS['CLIP_MIN']
            clip_max = DEFAULTS['CLIP_MAX']
            base_precision = float(np.clip(base_precision, clip_min, clip_max))
            meta_awareness = float(np.clip(float(meta_awareness), clip_min, clip_max))
            precision_sensory = integrate_precision_logit(base_precision, meta_awareness, EPS)
            self.blanket_l2l3.update_active_states({'precision_sensory': precision_sensory})
            self.monitor.write_policy_prior(new_state)

            # ===== Policy Inference =====
            prescription      = self.agent.infer_pi(activations, new_state)
            selected_action_mu = prescription['selected_action_mu']
            self.monitor.update_policy_state(new_state, prescription['q_pi'])
            self.blanket_l1l2.update_active_states(prescription)

            # Update rolling store for next E-step
            self._last_x_actual = {
                'x':      x_current.detach(),
                'action': selected_action_mu.detach(),
            }

            # Buffer entry — all tensors detached; M-step re-runs grad passes
            buf: Dict = {
                'x_prev':      x_prev,
                'action_prev': action_prev,
                'x_curr':      x_current.detach(),
                'z_star':      activations,          # already detached by VI
                'network_acts': network_acts,
                'state':       new_state,
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
                'loss':                    free_energy_val + forward_prediction_error_val,
                'meta_awareness':          meta_awareness,
                'action_error':            forward_prediction_error_val,
                'network_activations':     network_acts_serializable,
                'thoughtseed_activations': ts_acts,
                'dominant_thoughtseed':    THOUGHTSEEDS[dominant_idx],
            }

        return buf, new_state, activations, metrics

    # ------------------------------------------------------------------
    # M-step: parameter update over one BPTT window (Eq. 9)
    # ------------------------------------------------------------------
    def _m_step(self, buffer: List[Dict]) -> Optional[torch.Tensor]:
        """M-step: update (phi, theta, psi) from accumulated E-step buffer (Eq. 9).

        Re-runs only the three differentiable forward passes over the stored
        detached (z*, x) pairs — no VI, no blanket I/O:
          - VFE   : decoder theta  (reconstruction surprisal + KL)
          - S_fwd : forward model psi  (next-step prediction error)
          - L_rec : encoder phi        (amortised inference alignment)

        This is the variational EM M-step; beliefs z* are fixed from the E-step.
        Friston et al. (2019); Dempster, Laird & Rubin (1977).
        """
        window_loss: Optional[torch.Tensor] = None
        for entry in buffer:
            # ── VFE: decoder gradient (theta) ────────────────────────────────
            vfe = self.agent.compute_vfe(
                entry['state'], entry['z_star'], entry['network_acts']
            )
            step_loss = vfe

            # ── Forward surprisal: forward-model gradient (psi) ──────────────
            if entry['x_prev'] is not None:
                x_pred = self.agent.vae.predict_next(entry['x_prev'], entry['action_prev'])
                step_loss = step_loss + bernoulli_nll(x_pred, entry['x_curr'], EPS)

            # ── Recognition loss: encoder gradient (phi) — Eq. 9 ─────────────
            rec_scale = self.phenotype.alpha_rec
            if rec_scale > 0.0:
                z_enc     = self.agent.vae.encode(entry['x_curr'])
                step_loss = step_loss + rec_scale * mse_error(z_enc, entry['z_star'])

            window_loss = step_loss if window_loss is None else (window_loss + step_loss)

        return window_loss

    def train(
        self,
        timesteps: int = 10000,
        enable_learning: bool = True,
        reseed_rng: bool = False,
        run_seed: Optional[int] = None,
    ) -> Dict:
        """Run variational EM training.

        Each BPTT window T runs:
          1. E-step loop  — T calls to _e_step(), no parameter gradients.
          2. M-step once  — _m_step(buffer) re-runs differentiable passes and
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
              f"α_rec={self.phenotype.alpha_rec:.3f})")

        self.process.reset(state='breath_focus')
        current_state = self.process.current_state

        # Initial thoughtseed activations
        priors = THOUGHTSEED_STATE_PRIORS[current_state].copy()
        activations_np = np.array([priors[ts] for ts in THOUGHTSEEDS], dtype=np.float32)
        activations = torch.tensor(activations_np, dtype=torch.float32)
        activations = torch.clamp(activations, DEFAULTS['CLIP_MIN'], DEFAULTS['CLIP_MAX'])

        # Initialise blankets
        self.blanket_l1l2.update_active_states({
            'mu_x': None,
            'policy_confidence': 0.0,
            'policy_drive': 0.0
        })
        self.blanket_l2l3.reset()
        thoughtseed_dict = {ts: float(activations[i].item()) for i, ts in enumerate(THOUGHTSEEDS)}
        self.blanket_l2l3.update_sensory_states({
            'current_state': current_state,
            'dwell_progress': 0.0,
            'thoughtseed_activations': thoughtseed_dict,
        })
        self.monitor.update_meta_awareness()

        bptt_steps = 50

        for t_start in range(0, timesteps, bptt_steps):
            # Window boundary: clear sensory states (hidden state carries across)
            self.blanket_l1l2.sensory_states.clear()
            self.blanket_l2l3.sensory_states.clear()

            steps_to_run = min(bptt_steps, timesteps - t_start)
            e_step_buffer: List[Dict] = []

            # ── E-step loop (no gradient accumulation) ───────────────────────
            for t_sub in range(steps_to_run):
                t = t_start + t_sub

                buf, new_state, activations, metrics = self._e_step(t, activations)
                e_step_buffer.append(buf)

                # Record state transitions
                if new_state != current_state:
                    self.history['transitions'].append({
                        'timestamp':   t,
                        'from':        current_state,
                        'to':          new_state,
                        'free_energy': metrics['free_energy'],
                    })
                    current_state = new_state

                # Record history
                self.history['states'].append(current_state)
                self.history['free_energy'].append(metrics['free_energy'])
                self.history['loss'].append(metrics['loss'])
                self.history['meta_awareness'].append(metrics['meta_awareness'])
                self.history['action_errors'].append(metrics['action_error'])
                self.history['network_activations'].append(metrics['network_activations'])
                self.history['thoughtseed_activations'].append(metrics['thoughtseed_activations'])
                self.history['dominant_thoughtseed'].append(metrics['dominant_thoughtseed'])

            # ── M-step: gradient update over window ──────────────────────────
            if enable_learning:
                self.optimizer.zero_grad()
                window_loss = self._m_step(e_step_buffer)
                if window_loss is not None and window_loss.requires_grad:
                    window_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=1.0)
                    self.optimizer.step()

            # BPTT boundary detach (activations already detached from E-step)
            with torch.no_grad():
                self.process.x = self.process.x.detach()
                activations = activations.detach()

        return self._package_results()

    
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
            'loss_history': self.history['loss'],
            'meta_awareness_history': self.history['meta_awareness'],
            'state_history': self.history['states'],  # Renamed for viz compatibility
            'transitions': self.history['transitions'],
            'avg_dwell_times': avg_dwell,
            'transition_matrix': trans_matrix,
            'avg_action_errors': avg_action_errors,
            'final_free_energy': self.history['free_energy'][-1] if self.history['free_energy'] else 0.0,
            'final_loss': self.history['loss'][-1] if self.history['loss'] else 0.0,
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
            'loss': [],
            'meta_awareness': [],
            'transitions': [],
            'action_errors': [],
            'network_activations': [],
            'thoughtseed_activations': [],
            'dominant_thoughtseed': []
        }
        self._last_x_actual = None
        self._loss_balance = {}
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