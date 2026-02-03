"""Trainer orchestrating BPTT for the Layer 2 attentional model."""
import numpy as np
import torch
import torch.optim as optim
from typing import Tuple, Dict, Optional

from utils.meditation_config import (
    DEFAULTS,
    THOUGHTSEED_BASE_ACTIVATIONS
)
from core.layer1.layer1_config import NETWORK_PROFILES
from ..generative_model import ObservationModel
from ..layer1.generative_process import Layer1Process

from .logger import SimulationLogger

class PracticeTrainer:
    """Orchestrates the BPTT simulation loop."""
    
    def __init__(self, agent, generative_process: Optional[Layer1Process] = None):
        self.agent = agent
        if generative_process is None:
            self.process = Layer1Process(
                experience_level=agent.experience_level,
                seed=agent.rng.randint(0, 2**31) if hasattr(agent.rng, 'randint') else None
            )
        else:
            self.process = generative_process
            
        self.optimizer = optim.Adam(self.agent.parameters(), lr=agent.learning_rate)
        self.obs_model = ObservationModel(eps=1e-6)
        self._init_prior_cache()
        self.logger = SimulationLogger(agent.experience_level)

    def _init_prior_cache(self):
        self.state_index = {s: i for i, s in enumerate(self.agent.states)}

        num_states = len(self.agent.states)
        num_ts = len(self.agent.thoughtseeds)
        base = np.zeros((num_states, num_ts), dtype=np.float32)
        for s in self.agent.states:
            s_idx = self.state_index[s]
            base_map = THOUGHTSEED_BASE_ACTIVATIONS.get(self.agent.experience_level, {}).get(s, {})
            for ts in self.agent.thoughtseeds:
                t_idx = self.agent.thoughtseeds.index(ts)
                raw_val = float(base_map.get(ts, 0.0))
                base[s_idx, t_idx] = float(np.clip(
                    raw_val,
                    DEFAULTS['ACTIVATION_CLIP_MIN'],
                    DEFAULTS['ACTIVATION_CLIP_MAX']
                ))
        self._prior_base_np = base
        self._prior_cache = {}

    @staticmethod
    def _detach_active_states(active_states: dict) -> None:
        for key, value in list(active_states.items()):
            if isinstance(value, torch.Tensor):
                active_states[key] = value.detach()

    def _get_prior_tensors(self, device: torch.device):
        cached = self._prior_cache.get(device)
        if cached is None:
            cached = torch.tensor(self._prior_base_np, device=device)
            self._prior_cache[device] = cached
        return cached

    def _get_prior_vector(self, state: str, device: torch.device) -> torch.Tensor:
        idx = self.state_index[state]
        base = self._get_prior_tensors(device)
        return base[idx]

    def train(self, save_outputs: bool = True, output_dir: str = None, seed: int = None, enable_learning: bool = True):
        """Run BPTT training."""

        if seed is not None:
             torch.manual_seed(seed)
             np.random.seed(seed)
             self.agent.rng = np.random.RandomState(seed)
             self.rng_np = np.random.RandomState(seed)
             self.process.set_rng(self.rng_np)

        self.optimizer.zero_grad()
        current_state, current_dwell, dwell_limit, activations = self._initialize_simulation()
        
        last_free_energy = 0.0
        
        # Phase 4: Track previous actual observation for forward model training
        self._last_x_actual = None
        
        bptt_steps = 50
        total_steps = self.agent.timesteps
        
        for t_start in range(0, total_steps, bptt_steps):
            self.agent.blanket.sensory_states.clear()
            self.agent.blanket_l2l3.sensory_states.clear()
            
            steps_to_run = min(bptt_steps, total_steps - t_start)
            
            loss = torch.tensor(0.0, device=activations.device)
            for t_sub in range(steps_to_run):
                t = t_start + t_sub

                network_acts, process_state = self.process.update(self.agent.blanket.active_states)
                
                state_changed = (process_state != current_state)
                if state_changed:
                    self._record_transition(
                        t,
                        current_state,
                        process_state,
                        activations,
                        network_acts,
                        last_free_energy
                    )
                    current_state = process_state
                    current_dwell = 0
                    dwell_limit = self.process.current_max_dwell
                else:
                    current_dwell += 1
                    
                self.agent.blanket.update_sensory_states(network_acts)
                
                sensory_inference = self.agent.perceptual_inference()
                
                meta_awareness, activations = self._update_latents(
                    current_state, current_dwell, dwell_limit, activations,
                    network_acts, sensory_inference
                )
                
                free_energy, recon_loss, kl_div = self._perception_bottom_up(
                     current_state, activations, network_acts
                )
                
                loss = loss + free_energy
                last_free_energy = free_energy.detach().item()
                
                # Forward-informed action selection: Choose action FIRST
                prescription = self.agent.prescriptive_action(activations, free_energy, current_state)
                
                # Current observation and selected action
                x_current = torch.stack([network_acts[net] for net in self.agent.networks])
                selected_action_mu = prescription['selected_action_mu']
                
                # Phase 4: Forward model loss (enactive inference)
                # Predict what current observation should be, given previous state+action
                action_pred_error = torch.tensor(0.0, device=activations.device)
                action_pred_error_val = 0.0
                if self._last_x_actual is not None:
                    # Predict current observation from previous state+action (gradients flow here!)
                    x_pred_from_last = self.agent.vae.predict_next(self._last_x_actual['x'], 
                                                                     self._last_x_actual['action'])
                    action_pred_error = torch.mean((x_current.detach() - x_pred_from_last)**2)
                    action_pred_error_val = action_pred_error.detach().item()
                    
                    # Weight by L3 precision modulation
                    precision = self.agent.blanket_l2l3.active_states.get('precision_modulation', 0.5)
                    if isinstance(precision, torch.Tensor):
                        precision = precision.item()
                    
                    # Scale precision to reasonable loss weight (0.05 to 0.15 range)
                    action_loss_weight = 0.05 + (0.1 * precision)
                    loss = loss + (action_loss_weight * action_pred_error)
                
                # Store current state+action for next iteration's forward loss
                self._last_x_actual = {
                    'x': x_current.detach(),
                    'action': selected_action_mu.detach()
                }
                
                transition_drive = float(self.agent.blanket.active_states.get('transition_drive', 0.0))
                self._record_history(
                    current_state, activations, meta_awareness, network_acts,
                    free_energy, recon_loss, transition_drive, kl_div, action_pred_error_val
                )

            if steps_to_run > 0:
                loss = loss / float(steps_to_run)

            if enable_learning and loss.requires_grad:
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            with torch.no_grad():
                self.process.x = self.process.x.detach()
                self.process.smoothed_x = self.process.smoothed_x.detach()
            
            with torch.no_grad():
                if hasattr(self.agent, "z_ema"):
                    self.agent.z_ema = self.agent.z_ema.detach()
            
            self._detach_active_states(self.agent.blanket.active_states)
            self._detach_active_states(self.agent.blanket_l2l3.active_states)
            
            activations = activations.detach()
            
        # Save results
        if save_outputs:
            self._save_results(output_dir)
            
        return self.agent

    def _initialize_simulation(self) -> Tuple[str, int, int, torch.Tensor]:
        """Initialize simulation."""
        self.process.reset(state='breath_focus')
        current_state = self.process.current_state
        current_dwell = 0
        dwell_limit = self.process.current_max_dwell
        
        device = self.agent.z_ema.device
        prior = self._get_prior_vector(current_state, device).detach().cpu().numpy()
        activations_np = np.array(prior, dtype=np.float32)
        rng = getattr(self, "rng_np", None)
        init_sigma = float(getattr(self.agent, "init_noise_sigma", 0.0))
        if rng is None:
            activations_np += np.random.normal(0, init_sigma, size=len(activations_np))
        else:
            activations_np += rng.normal(0, init_sigma, size=len(activations_np))
        activations = torch.tensor(activations_np, dtype=torch.float32, device=device)
        activations = torch.clamp(activations, DEFAULTS['ACTIVATION_CLIP_MIN'], DEFAULTS['ACTIVATION_CLIP_MAX'])
        
        network_acts, _ = self.process.update(self.agent.blanket.active_states)
        self.agent.blanket.update_sensory_states(network_acts)
        
        self.agent.blanket_l2l3.reset()
        
        self.agent.monitor.update_meta_awareness(current_state, activations)
        
        return current_state, current_dwell, dwell_limit, activations

    def _update_latents(self, current_state: str, current_dwell: int, dwell_limit: int,
                        activations: torch.Tensor, observed_networks: Dict[str, torch.Tensor],
                        sensory_inference: torch.Tensor) -> Tuple[float, torch.Tensor]:
        activations = self.agent.update_thoughtseed_dynamics(
            activations, current_state, current_dwell, dwell_limit, observed_networks, sensory_inference
        )
        self.agent.blanket_l2l3.update_sensory_states({
            'current_state': current_state,
            'dwell_progress': (current_dwell / max(1, dwell_limit)) if dwell_limit else 0.0
        })
        meta_awareness = self.agent.monitor.update_meta_awareness(current_state, activations)

        return meta_awareness, activations

    def _perception_bottom_up(self, current_state: str, activations: torch.Tensor,
                              observed_networks: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        recon_x = self.agent.decode_with_state(activations, current_state)
        
        device = activations.device
        
        x = torch.stack([observed_networks[net] for net in self.agent.networks]).to(device)
        recon_loss = self.obs_model.reconstruction_nll(recon_x, x)

        z = activations
        if z.dim() != 1:
            z = z.squeeze(0)
        prior_vec = self._get_prior_vector(current_state, device)
        kl_div = self.obs_model.latent_bernoulli_kl(z, prior_vec)

        reg_weight = self.agent.params.get("network_target_reg", 0.0)
        target_loss = torch.tensor(0.0, device=device)
        if reg_weight > 0.0 and current_state in NETWORK_PROFILES:
            prof = NETWORK_PROFILES[current_state][self.agent.experience_level]
            target_vec = torch.tensor([prof[n] for n in self.agent.networks], device=device, dtype=torch.float32)
            target_loss = torch.nn.functional.mse_loss(x, target_vec, reduction='mean')

        beta = float(self.agent.params.get("kl_beta", 1.0))
        loss = recon_loss + beta * kl_div + (reg_weight * target_loss)

        free_energy = loss
        return free_energy, recon_loss, kl_div

    def _record_transition(self, t, old, new, acts, nets, vfe):
        acts_dict = {ts: acts[i].detach().item() for i, ts in enumerate(self.agent.thoughtseeds)}
        nets_dict = {k: v.detach().item() for k, v in nets.items()}
        self.logger.record_transition(
            timestamp=t,
            from_state=old,
            to_state=new,
            thoughtseed_activations=acts_dict,
            network_acts=nets_dict,
            free_energy=float(vfe)
        )

    def _record_history(self, current_state, activations, meta_awareness, network_acts, 
                       free_energy, recon_loss, transition_drive, kl_div, action_pred_error=0.0):
        # Calculate derived metrics for logging
        net_acts_float = {k: v.detach().item() for k, v in network_acts.items()}
        
        prec_min, prec_max = self.agent.params.get('l3tol2_precision_range', (0.4, 0.6))
        prec = prec_min + (prec_max - prec_min) * meta_awareness
        prec = float(np.clip(prec, min(prec_min, prec_max), max(prec_min, prec_max)))
        
        efe_val = float(self.agent.monitor.efe_value)
        efe_risk = float(self.agent.monitor.efe_risk)
        efe_ambiguity = float(self.agent.monitor.efe_ambiguity)
        selected_policy = str(self.agent.monitor.selected_policy)
        policy_confidence = float(self.agent.monitor.policy_confidence)
        policy_entropy = float(self.agent.monitor.policy_entropy)
        policy_posterior = dict(self.agent.monitor.policy_posterior)
        latent_terms = self.agent.last_latent_vfe_terms
        mw_burden = float(getattr(self.process, 'last_network_burden', 0.0))
        transition_hazard = float(getattr(self.process, 'last_transition_hazard', 0.0))
        activation_burden_component = float(getattr(self.process, 'last_activation_burden_component', 0.0))
        coupling_burden_component = float(getattr(self.process, 'last_coupling_burden_component', 0.0))
        dom = self.agent.thoughtseeds[torch.argmax(activations).item()]
        
        self.logger.record_step(
            current_state=current_state,
            activations=activations.detach().cpu().numpy(),
            meta_awareness=float(meta_awareness),
            network_acts=net_acts_float,
            free_energy=free_energy.detach().item(),
            recon_loss=recon_loss.detach().item(),
            kl_div=kl_div.detach().item(),
            transition_drive=float(transition_drive),
            precision=prec,
            efe=efe_val,
            efe_risk=efe_risk,
            efe_ambiguity=efe_ambiguity,
            selected_policy=selected_policy,
            policy_confidence=policy_confidence,
            policy_entropy=policy_entropy,
            policy_posterior=policy_posterior,
            mw_burden=mw_burden,
            transition_hazard=transition_hazard,
            activation_burden_component=activation_burden_component,
            coupling_burden_component=coupling_burden_component,
            latent_reconstruction=latent_terms['reconstruction'],
            latent_prior_kl=latent_terms['prior_kl'],
            latent_sensory_consistency=latent_terms['sensory_consistency'],
            latent_temporal_consistency=latent_terms['temporal_consistency'],
            latent_vfe_total=latent_terms['total'],
            dominant_ts=str(dom),
            action_pred_error=float(action_pred_error)  # Phase 4
        )

    def _save_results(self, output_dir):
        self.logger.save_results(self.agent, output_dir)
