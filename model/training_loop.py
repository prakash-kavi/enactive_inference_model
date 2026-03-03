"""Training orchestrator: variational EM loop for hierarchical free energy minimisation.

Timescale separation (Dempster et al., 1977; Friston et al., 2019):
- E-step (every t): perceptual inference via VI + policy selection. No parameter gradients.
- M-step (every T steps): parameter update via BPTT over the accumulated buffer,
  plus habit-prior updates from E-step sufficient statistics.
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
    DEFAULT_DT, PRECISION_TAU, BPTT_STEPS, PLOT_STEPS,
)
from .l1_generative_process import Layer1Process
from .l2_recognition import Layer2Agent
from .l3_metacognition import Layer3Monitor
from .markov_blankets import MarkovBlanketL1L2, MarkovBlanketL2L3
from .phenotype import PhenotypeConfig, EXPERT_PHENOTYPE
from utils.math_utils import (
    networks_to_tensor,
    mse_error,
    forward_error,
    state_confidence,
    policy_entropy,
)

class MeditationTrainer:
    """BPTT training for hierarchical meditation model."""
    
    def __init__(self, phenotype: PhenotypeConfig = None, seed: int = 42):
        self.phenotype = phenotype if phenotype is not None else EXPERT_PHENOTYPE
        self.level = self.phenotype.level
        self.seed = seed
        self.bptt_steps = int(BPTT_STEPS)

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.blanket_l1l2 = MarkovBlanketL1L2()
        self.blanket_l2l3 = MarkovBlanketL2L3()

        self.process = Layer1Process(phenotype=self.phenotype, seed=seed)
        self.agent = Layer2Agent(
            phenotype=self.phenotype,
            blanket_l1l2=self.blanket_l1l2,
            blanket_l2l3=self.blanket_l2l3
        )
        self.monitor = Layer3Monitor(
            blanket_l2l3=self.blanket_l2l3,
            bptt_steps=self.bptt_steps,
        )

        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.phenotype.learning_rate)
        
        self.history = {}
        self._last_x_actual = None
        self._sigma_fwd2 = None
        self.clarity_baseline_train = None
        self._surprisal_alpha = float(
            np.clip(DEFAULT_DT / PRECISION_TAU, 0.0, 1.0)
        ) if PRECISION_TAU > 0 else 1.0
    
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
                # EMA forward-surprisal scale (sigma_fwd^2)
                if self._sigma_fwd2 is None:
                    self._sigma_fwd2 = forward_error_val
                else:
                    self._sigma_fwd2 = (
                        (1.0 - self._surprisal_alpha) * self._sigma_fwd2
                        + self._surprisal_alpha * forward_error_val
                    )
                sigma_fwd2 = max(float(self._sigma_fwd2), EPS)
                base_precision = float(np.exp(-forward_error_val / (sigma_fwd2 + EPS)))
                self.blanket_l2l3.update_active_states({'precision_sensory': base_precision})

            # ===== L3 precision signal (used by L2) =====
            base_precision = float(np.clip(base_precision, CLIP_MIN, CLIP_MAX))
            precision_sensory = base_precision
            self.blanket_l2l3.update_active_states({'precision_sensory': precision_sensory})

            # ===== Layer 2: Attentional Agent =====
            activations, _, z_prior = self.agent.infer_z_step(
                current_state=new_state,
                activations=activations,
            )
            activations = activations.detach()  # z* fixed; VI not part of BPTT

            # VFE as scalar metric (M-step will recompute with grad)
            free_energy_val = float(
                self.agent.compute_vfe(
                    new_state,
                    activations,
                    x_current,
                    precision_sensory=precision_sensory,
                    prior_target=None,
                ).item()
            )

            # Update L2->L3 sensory interface for policy selection
            state_belief_raw = self.agent.infer_state_belief(activations)
            state_belief = state_belief_raw
            state_conf = state_confidence(state_belief_raw, keys=STATES)
            policy_eval = self.agent.evaluate_policies(new_state, state_belief=state_belief)
            self.blanket_l2l3.update_sensory_states({
                'state_belief':      state_belief,
                'policy_candidates': policy_eval['candidates'],
                'policy_priors':     policy_eval['priors'],
                'policy_costs':      policy_eval['g_vals'],
            })

            # ===== Policy Selection (L3) =====
            meta_awareness = self.monitor.update_meta_awareness_from_conflict(
                g_vals=policy_eval['g_vals'],
                state_belief=state_belief,
                gate_belief=state_belief,
            )
            q_pi = self.monitor.select_policy(
                g_vals=policy_eval['g_vals'],
                priors=policy_eval['priors'],
                state_belief=state_belief,
                meta_precision=meta_awareness,
            )
            q_pi = np.array(q_pi, dtype=float)
            if not np.all(np.isfinite(q_pi)) or q_pi.sum() <= 0.0:
                raise ValueError(f"Invalid policy posterior at t={t}: {q_pi}")
            q_pi = q_pi / q_pi.sum()
            sb_vals = np.array(list(state_belief.values()), dtype=float)
            if sb_vals.size > 0 and (not np.all(np.isfinite(sb_vals)) or sb_vals.sum() <= 0.0):
                raise ValueError(f"Invalid state belief at t={t}: {state_belief}")

            prescription = self.agent.action_from_policy(
                q_pi=q_pi,
                candidates=policy_eval['candidates'],
                mu_candidates=policy_eval['mu_candidates'],
            )
            selected_action_mu = prescription['selected_action_mu']
            self.blanket_l1l2.update_active_states({
                'mu_x': prescription['mu_x'],
                'policy_state_probs': prescription.get('policy_state_probs'),
                'meta_precision': meta_awareness,
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
                'z_prior':     z_prior.detach(),
                'state':       new_state,
                'precision_sensory': precision_sensory,
                'state_belief': state_belief,
                'policy_posterior': q_pi.tolist() if hasattr(q_pi, "tolist") else list(q_pi),
            }

            # Metrics (float approximation of per-step loss for diagnostics)
            network_acts_serializable = {
                net: float(val.detach().item()) if isinstance(val, torch.Tensor) else float(val)
                for net, val in network_acts.items()
            }
            ts_acts      = activations.cpu().numpy().tolist()
            ts_prior_acts = z_prior.cpu().numpy().tolist() if z_prior is not None else ts_acts
            dominant_idx = int(np.argmax(ts_acts))
            policy_state_probs = prescription.get('policy_state_probs') or {}
            p_stay = float(policy_state_probs.get(new_state, q_pi[0] if len(q_pi) > 0 else 0.0))
            switch_prob = float(1.0 - p_stay)
            transition_floor = 1.0 / max(self.process.current_max_dwell, 1)
            transition_prob = max(switch_prob, transition_floor)
            policy_entropy_val = policy_entropy(q_pi, eps=EPS)
            metrics = {
                'timestamp':               t,
                'free_energy':             free_energy_val,
                'loss':                    free_energy_val + forward_error_val,
                'meta_awareness':          meta_awareness,
                'action_error':            forward_error_val,
                'efe_prag_mean':           float(policy_eval.get('efe_prag_mean', 0.0)),
                'efe_epi_mean':            float(policy_eval.get('efe_epi_mean', 0.0)),
                'state_belief':            state_belief,
                'policy_posterior':        q_pi.tolist(),
                'policy_entropy':          policy_entropy_val,
                'switch_prob':            switch_prob,
                'transition_prob':         transition_prob,
                'precision_sensory':       precision_sensory,
                'state_confidence':        state_conf,
                'state_belief_ma':         float(state_belief.get('meta_awareness', 0.0)),
                'state_belief_ra':         float(state_belief.get('redirect_attention', 0.0)),
                'network_activations':     network_acts_serializable,
                'thoughtseed_activations': ts_acts,
                'thoughtseed_prior_activations': ts_prior_acts,
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
        vfe_terms: List[torch.Tensor] = []
        fwd_terms: List[torch.Tensor] = []
        rec_terms: List[torch.Tensor] = []
        vfe_sum = 0.0
        rec_sum = 0.0

        for entry in buffer:
            # -- VFE: decoder gradient (theta) --------------------------------
            vfe = self.agent.compute_vfe(
                entry['state'], entry['z_star'], entry['x_curr'],
                precision_sensory=entry.get('precision_sensory'),
                prior_target=None,
            )
            vfe_terms.append(vfe)
            vfe_sum += float(vfe.detach().item())

            # Forward surprisal: forward-model gradient (psi)
            if entry['x_prev'] is not None:
                x_pred = self.agent.thoughtseed_model.predict_next(entry['x_prev'], entry['action_prev'])
                fwd_terms.append(forward_error(x_pred, entry['x_curr']))
            else:
                fwd_terms.append(torch.tensor(0.0, dtype=vfe.dtype, device=vfe.device))

            # Recognition loss: encoder gradient (phi)
            z_enc = self.agent.thoughtseed_model.encode(entry['x_curr'])
            rec_loss = mse_error(z_enc, entry['z_star'])
            rec_terms.append(rec_loss)
            rec_sum += float(rec_loss.detach().item())

        # Adaptive recognition scaling: match mean VFE and mean recognition loss
        if rec_sum > 0.0:
            rec_scale = vfe_sum / (rec_sum + EPS)
        else:
            rec_scale = 0.0

        for vfe, fwd, rec in zip(vfe_terms, fwd_terms, rec_terms):
            step_loss = vfe + fwd + rec_scale * rec
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
                self.history['transitions'].append({
                    'timestamp':   t,
                    'from':        current_state,
                    'to':          new_state,
                    'free_energy': metrics['free_energy'],
                })
                current_state = new_state

            self.history['states'].append(current_state)
            self.history['free_energy'].append(metrics['free_energy'])
            self.history['loss'].append(metrics['loss'])
            self.history['meta_awareness'].append(metrics['meta_awareness'])
            self.history['state_confidence'].append(metrics['state_confidence'])
            self.history['network_activations'].append(metrics['network_activations'])
            self.history['thoughtseed_activations'].append(metrics['thoughtseed_activations'])
            self.history['thoughtseed_prior_activations'].append(
                metrics.get('thoughtseed_prior_activations', metrics['thoughtseed_activations'])
            )

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
        self._update_policy_prior_from_buffer(buffer)

    def _update_policy_prior_from_buffer(self, buffer: List[Dict]) -> None:
        """M-step: update learned policy prior from E-step sufficient statistics."""
        for entry in buffer:
            state_belief = entry.get('state_belief')
            q_pi = entry.get('policy_posterior')
            if state_belief is None or q_pi is None:
                continue
            self.monitor.update_policy_state(state_belief, q_pi)

    def train(
        self,
        timesteps: int = 10000,
        enable_learning: bool = True,
        train_steps: Optional[int] = None,
        reseed_rng: bool = False,
        run_seed: Optional[int] = None,
        preserve_habit_prior: bool = False,
    ) -> Dict:
        """Run variational EM: train phase then eval+plot phases with frozen weights.

        When train_steps is set (e.g. TRAIN_STEPS), learning runs only for steps
        0..train_steps; steps train_steps..timesteps run with weights frozen (eval + plot).

        Each BPTT window T runs:
          1. E-step loop  - T calls to _e_step(), no parameter gradients.
          2. M-step once  - _m_step(buffer) only when effective_learning is True.

        Args:
            timesteps:     Total simulation steps (train + eval + plot)
            enable_learning: Whether to update weights during the train phase
            train_steps:   If set, learning only for t < train_steps; rest frozen
            reseed_rng:    Reinitialise random generators for reproducibility
            run_seed:      Per-run seed override (used only with reseed_rng)

        Returns:
            Training results dict (full run)
        """
        self._reset_run_state(
            reseed_rng=reseed_rng,
            run_seed=run_seed,
            preserve_habit=preserve_habit_prior,
        )
        if enable_learning:
            self.monitor.set_clarity_baseline(None)

        print(f"PHENOTYPE: {self.phenotype.label} "
              f"(lr={self.phenotype.learning_rate:.4f}, "
              f"alpha_rec=adaptive)")

        self.process.reset(state='breath_focus')
        current_state = self.process.current_state

        # Initial thoughtseed activations
        priors = THOUGHTSEED_STATE_PRIORS[current_state].copy()
        activations_np = np.array([priors[ts] for ts in THOUGHTSEEDS], dtype=np.float32)
        activations = torch.tensor(activations_np, dtype=torch.float32)
        activations = torch.clamp(activations, CLIP_MIN, CLIP_MAX)

        # Initialise blankets
        self.blanket_l1l2.update_active_states({
            'mu_x': None
        })
        self.blanket_l2l3.reset()
        self.blanket_l2l3.update_sensory_states({
            'state_belief': {state: 1.0 if state == current_state else 0.0 for state in STATES},
        })

        bptt_steps = self.bptt_steps
        clarity_baseline_set_for_frozen = False

        for t_start in range(0, timesteps, bptt_steps):
            # Window boundary: clear sensory states (hidden state carries across)
            self.blanket_l1l2.sensory_states.clear()
            self.blanket_l2l3.sensory_states.clear()

            steps_to_run = min(bptt_steps, timesteps - t_start)
            # Freeze weights after train phase: only run M-step when whole window is inside train phase
            effective_learning = enable_learning and (
                train_steps is None or (t_start + steps_to_run <= train_steps)
            )
            # Set clarity_baseline when entering frozen phase so eval/plot use it
            if (
                train_steps is not None
                and t_start >= train_steps
                and not clarity_baseline_set_for_frozen
                and enable_learning
            ):
                self.clarity_baseline_train = self._compute_clarity_baseline(train_steps=train_steps)
                if self.clarity_baseline_train is not None:
                    self.monitor.set_clarity_baseline(self.clarity_baseline_train)
                clarity_baseline_set_for_frozen = True

            # E-step loop (no gradient accumulation)
            e_step_buffer, activations, current_state = self._run_e_step_window(
                t_start,
                steps_to_run,
                activations,
                current_state,
            )

            # M-step: gradient update only during train phase
            self._run_m_step_window(e_step_buffer, effective_learning)

            # BPTT boundary detach (activations already detached from E-step)
            with torch.no_grad():
                self.process.x = self.process.x.detach()
                activations = activations.detach()

        if enable_learning and train_steps is not None:
            self.clarity_baseline_train = self._compute_clarity_baseline(train_steps=train_steps)
            if self.clarity_baseline_train is not None:
                self.monitor.set_clarity_baseline(self.clarity_baseline_train)
        elif enable_learning:
            self.clarity_baseline_train = self._compute_clarity_baseline()
            if self.clarity_baseline_train is not None:
                self.monitor.set_clarity_baseline(self.clarity_baseline_train)
        return self._package_results()

    def _compute_clarity_baseline(self, train_steps: Optional[int] = None) -> Optional[float]:
        """Compute mean state-confidence over the training tail (for clarity_baseline)."""
        values = self.history.get('state_confidence', [])
        if not values:
            return None
        if train_steps is not None:
            start = max(0, train_steps - PLOT_STEPS)
            tail = values[start:train_steps]
        else:
            tail = values[-PLOT_STEPS:] if len(values) > PLOT_STEPS else values
        if not tail:
            return None
        return float(np.mean(tail))

    
    def _package_results(self) -> Dict:
        """Package training results for analysis."""
        return {
            'experience_level': self.level,
            'seed': self.seed,
            'timesteps': len(self.history['states']),
            'free_energy_history': self.history['free_energy'],
            'loss_history': self.history['loss'],
            'meta_awareness_history': self.history['meta_awareness'],
            'state_confidence_history': self.history['state_confidence'],
            'state_history': self.history['states'],  # Renamed for viz compatibility
            'transitions': self.history['transitions'],
            'network_activations_history': self.history['network_activations'],
            'thoughtseed_activations_history': self.history['thoughtseed_activations'],
            'thoughtseed_prior_activations_history': self.history['thoughtseed_prior_activations'],
            'clarity_baseline_train': self.clarity_baseline_train,
        }
    
    def _reset_run_state(
        self,
        reseed_rng: bool = False,
        run_seed: Optional[int] = None,
        preserve_habit: bool = False,
    ):
        """Reset internal state for a new run.

        Set preserve_habit=True to keep learned habit priors across runs.
        """
        self.history = {
            'states': [],
            'free_energy': [],
            'loss': [],
            'meta_awareness': [],
            'state_confidence': [],
            'transitions': [],
            'network_activations': [],
            'thoughtseed_activations': [],
            'thoughtseed_prior_activations': [],
        }
        self._last_x_actual = None
        self._sigma_fwd2 = None
        self.blanket_l1l2.reset()
        self.monitor.reset(preserve_habit=preserve_habit)

        if reseed_rng:
            seed = self.seed if run_seed is None else int(run_seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
            self.process.rng = np.random.RandomState(seed)

    def save_results(
        self,
        output_dir: str = 'data/lean_results',
        prefix: str = 'training_results',
    ) -> None:
        """Save results to JSON."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = self._package_results()
        
        # Save compact JSON
        filepath = output_path / f'{prefix}_{self.level}_seed{self.seed}.json'
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {filepath}")

def train_meditation_model(
    phenotype: PhenotypeConfig = None,
    timesteps: int = 10000,
    train_steps: Optional[int] = None,
    seed: int = 42,
    reseed_rng: bool = False,
    run_seed: Optional[int] = None,
    save_results: bool = True,
    output_dir: str = 'data/lean_results',
) -> tuple[MeditationTrainer, Dict]:
    """Train a model (train phase, then eval+plot with frozen weights) and return trainer + results."""
    trainer = MeditationTrainer(
        phenotype=phenotype,
        seed=seed,
    )
    results = trainer.train(
        timesteps=timesteps,
        enable_learning=True,
        train_steps=train_steps,
        reseed_rng=reseed_rng,
        run_seed=run_seed,
        preserve_habit_prior=False,
    )
    if save_results:
        trainer.save_results(output_dir=output_dir, prefix='training_results')
    return trainer, results
