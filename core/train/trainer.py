"""Trainer orchestrating BPTT for the Layer 2 attentional model."""
import numpy as np
import torch
import torch.optim as optim
from typing import Tuple, Dict, Optional

from utils.meditation_config import (
    DEFAULTS, NETWORK_PROFILES,
    THOUGHTSEED_BASE_ACTIVATIONS
)
from ..layer1.process import Layer1Process

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
        self._init_prior_cache()
        self.logger = SimulationLogger(agent.experience_level)

    def _init_prior_cache(self):
        self.state_index = {s: i for i, s in enumerate(self.agent.states)}
        self.ts_index = {ts: i for i, ts in enumerate(self.agent.thoughtseeds)}

        num_states = len(self.agent.states)
        num_ts = len(self.agent.thoughtseeds)
        base = np.zeros((num_states, num_ts), dtype=np.float32)
        for s in self.agent.states:
            s_idx = self.state_index[s]
            base_map = THOUGHTSEED_BASE_ACTIVATIONS.get(self.agent.experience_level, {}).get(s, {})
            for ts in self.agent.thoughtseeds:
                t_idx = self.ts_index[ts]
                base[s_idx, t_idx] = float(base_map.get(ts, 0.0))
        self._prior_base_np = base
        self._prior_cache = {}

    def _get_prior_tensors(self, device: torch.device):
        cached = self._prior_cache.get(device)
        if cached is None:
            base = torch.tensor(self._prior_base_np, device=device)
            cached = base
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
        current_state, current_dwell, dwell_limit, activations, network_acts, meta_awareness = self._initialize_simulation()
        
        last_free_energy = 0.0
        
        bptt_steps = 50
        total_steps = self.agent.timesteps
        
        for t_start in range(0, total_steps, bptt_steps):
            self.agent.blanket.sensory_states.clear()
            self.agent.blanket_l2l3.sensory_states.clear()
            
            steps_to_run = min(bptt_steps, total_steps - t_start)
            
            loss = torch.tensor(0.0, device=self.agent.vae.encoder_net[0].weight.device)
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
                     current_state, activations, meta_awareness, network_acts
                )
                
                loss = loss + free_energy
                last_free_energy = free_energy.detach().item()
                
                self._action_top_down(activations, free_energy, current_state)
                
                transition_drive = float(self.agent.blanket.active_states.get('transition_drive', 0.0))
                self._record_history(
                    current_state, activations, meta_awareness, network_acts,
                    free_energy, recon_loss, transition_drive, kl_div
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
                self.agent.aha_accum_val.data = self.agent.aha_accum_val.detach()
                if hasattr(self.agent, "z_ema"):
                    self.agent.z_ema = self.agent.z_ema.detach()
            
            for k in list(self.agent.blanket.active_states.keys()):
                v = self.agent.blanket.active_states[k]
                if isinstance(v, torch.Tensor):
                    self.agent.blanket.active_states[k] = v.detach()
                    
            for k in list(self.agent.blanket_l2l3.active_states.keys()):
                v = self.agent.blanket_l2l3.active_states[k]
                if isinstance(v, torch.Tensor):
                    self.agent.blanket_l2l3.active_states[k] = v.detach()
            
            activations = activations.detach()
            
        # Save results
        if save_outputs:
            self._save_results(output_dir)
            
        return self.agent

    def _initialize_simulation(self) -> Tuple[str, int, int, torch.Tensor, Dict[str, torch.Tensor], float]:
        """Initialize simulation."""
        self.process.reset(state='breath_focus')
        current_state = self.process.current_state
        current_dwell = 0
        dwell_limit = self.process.current_max_dwell
        
        device = self.agent.vae.encoder_net[0].weight.device
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
        
        meta_awareness = self.agent.get_meta_awareness(current_state, activations)
        
        return current_state, current_dwell, dwell_limit, activations, network_acts, meta_awareness

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
        meta_awareness = self.agent.monitor.compute_meta_awareness(current_state, activations)

        return meta_awareness, activations

    def _perception_bottom_up(self, current_state: str, activations: torch.Tensor, meta_awareness: float,
                              observed_networks: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        recon_x = self.agent.decode_with_state(activations, current_state)
        
        device = self.agent.vae.encoder_net[0].weight.device
        
        x = torch.stack([observed_networks[net] for net in self.agent.networks]).to(device)
        recon_x = recon_x.to(device)

        recon_loss = torch.nn.functional.mse_loss(recon_x, x, reduction='mean')

        eps = 1e-6
        z = activations
        if z.dim() != 1:
            z = z.squeeze(0)
        z = torch.clamp(z, eps, 1.0 - eps)
        prior_vec = self._get_prior_vector(current_state, device)
        prior = torch.clamp(prior_vec, eps, 1.0 - eps)
        kl_div = torch.mean(
            z * torch.log(z / prior) +
            (1 - z) * torch.log((1 - z) / (1 - prior))
        )

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

    def _action_top_down(self, activations: torch.Tensor, free_energy: torch.Tensor, current_state: str) -> dict:
        return self.agent.prescriptive_action(activations, free_energy, current_state)

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

    def _record_history(self, current_state, activations, meta_awareness, network_acts, free_energy, recon_loss, transition_drive, kl_div):
        # Calculate derived metrics for logging
        net_acts_float = {k: v.detach().item() for k, v in network_acts.items()}
        
        prec_min, prec_max = self.agent.params.get('l3tol2_precision_range', (0.4, 0.6))
        prec = prec_min + (prec_max - prec_min) * meta_awareness
        prec = float(np.clip(prec, min(prec_min, prec_max), max(prec_min, prec_max)))
        
        efe_val = getattr(self.agent.monitor, "efe_value", 0.0)
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
            efe=float(efe_val),
            dominant_ts=str(dom)
        )

    def _save_results(self, output_dir):
        self.logger.save_results(self.agent, output_dir)
