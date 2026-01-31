"""Trainer orchestrating BPTT for the Layer 2 attentional model."""
import os
import json
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from typing import Tuple, Dict, Optional

from utils.meditation_utils import ensure_directories
from utils.meditation_config import (
    DEFAULTS, NETWORK_PROFILES,
    THOUGHTSEED_BASE_ACTIVATIONS,
    THOUGHTSEED_TARGET_ADJUSTMENTS,
    THOUGHTSEED_LEVEL_OFFSETS
)
from utils.meditation_diagnostics import (
    compute_neural_efficiency_ratio,
    compute_dmn_dan_anticorrelation
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
        adjust = np.zeros((num_states, num_ts), dtype=np.float32)
        offsets = np.zeros((num_states, num_ts), dtype=np.float32)

        level_offsets = THOUGHTSEED_LEVEL_OFFSETS.get(self.agent.experience_level, {})

        for s in self.agent.states:
            s_idx = self.state_index[s]
            base_map = THOUGHTSEED_BASE_ACTIVATIONS.get(s, {})
            adjust_map = THOUGHTSEED_TARGET_ADJUSTMENTS.get(s, {})
            offset_map = level_offsets.get(s, {})
            for ts in self.agent.thoughtseeds:
                t_idx = self.ts_index[ts]
                base[s_idx, t_idx] = float(base_map.get(ts, 0.0))
                adjust[s_idx, t_idx] = float(adjust_map.get(ts, 0.0))
                offsets[s_idx, t_idx] = float(offset_map.get(ts, 0.0))

        self._prior_base_np = base
        self._prior_adjust_np = adjust
        self._prior_offset_np = offsets
        self._prior_cache = {}

    def _get_prior_tensors(self, device: torch.device):
        cached = self._prior_cache.get(device)
        if cached is None:
            base = torch.tensor(self._prior_base_np, device=device)
            adjust = torch.tensor(self._prior_adjust_np, device=device)
            offsets = torch.tensor(self._prior_offset_np, device=device)
            cached = (base, adjust, offsets)
            self._prior_cache[device] = cached
        return cached

    def _get_prior_vector(self, state: str, meta_awareness: float, device: torch.device) -> torch.Tensor:
        idx = self.state_index[state]
        base, adjust, offsets = self._get_prior_tensors(device)
        return base[idx] + (meta_awareness * adjust[idx]) + offsets[idx]

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
        
        state_transition_patterns = []
        transition_timestamps = []
        old_state = current_state
        
        bptt_steps = 50
        total_steps = self.agent.timesteps
        
        for t_start in range(0, total_steps, bptt_steps):
            self.agent.blanket.sensory_states.clear()
            self.agent.blanket_l2l3.sensory_states.clear()
            
            steps_to_run = min(bptt_steps, total_steps - t_start)
            
            loss = torch.tensor(0.0, device=self.agent.vae.encoder_net[0].weight.device)
            state_net_accum = {s: [] for s in self.agent.states}
            
            for t_sub in range(steps_to_run):
                t = t_start + t_sub

                network_acts, process_state = self.process.update(self.agent.blanket.active_states)
                
                state_changed = (process_state != current_state)
                if state_changed:
                    self._record_transition(state_transition_patterns, transition_timestamps, t, old_state, process_state, activations, network_acts, 0.0)
                    old_state = current_state
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
                
                self._action_top_down(activations, free_energy, current_state)
                
                transition_drive = float(self.agent.blanket.active_states.get('transition_drive', 0.0))
                self._record_history(
                    current_state, activations, meta_awareness, network_acts,
                    free_energy, recon_loss, transition_drive, kl_div
                )

                net_vec = torch.stack([network_acts[net] for net in self.agent.networks])
                state_net_accum[current_state].append(net_vec)
                
            if enable_learning and loss.requires_grad:
                contrastive_weight = self.agent.params.get("state_contrastive_weight", 0.0)
                if contrastive_weight > 0.0:
                    structural_loss = self._compute_structural_loss(state_net_accum)
                    if structural_loss is not None:
                        loss = loss + (contrastive_weight * structural_loss)
                
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
            self._save_results(output_dir, state_transition_patterns, transition_timestamps)
            
        return self.agent

    def _initialize_simulation(self) -> Tuple[str, int, int, torch.Tensor, Dict[str, torch.Tensor], float]:
        """Initialize simulation."""
        self.process.reset(state='breath_focus')
        current_state = self.process.current_state
        current_dwell = 0
        dwell_limit = self.process.current_max_dwell
        
        device = self.agent.vae.encoder_net[0].weight.device
        prior = self._get_prior_vector(current_state, 0.6, device).detach().cpu().numpy()
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
        prior_vec = self._get_prior_vector(current_state, meta_awareness, device)
        prior = torch.clamp(prior_vec, eps, 1.0 - eps)
        kl_div = torch.mean(
            z * torch.log(z / prior) +
            (1 - z) * torch.log((1 - z) / (1 - prior))
        )

        reg_weight = self.agent.params.get("network_target_reg", 0.0)
        target_loss = torch.tensor(0.0, device=device)
        van_boost_loss = torch.tensor(0.0, device=device)
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

    def _compute_structural_loss(self, state_net_accum: Dict[str, list]) -> Optional[torch.Tensor]:
        state_means = {}
        for s, vecs in state_net_accum.items():
            if vecs:
                state_means[s] = torch.stack(vecs, dim=0).mean(dim=0)

        states = list(state_means.keys())
        if len(states) < 2:
            return None

        pair_weights = {
            ('breath_focus', 'mind_wandering'): 2.0,
            ('mind_wandering', 'meta_awareness'): 1.5,
            ('mind_wandering', 'redirect_breath'): 1.5,
            ('breath_focus', 'meta_awareness'): 1.2,
            ('breath_focus', 'redirect_breath'): 1.2,
            ('meta_awareness', 'redirect_breath'): 1.0
        }

        sims = []
        for i in range(len(states)):
            for j in range(i + 1, len(states)):
                a = state_means[states[i]]
                b = state_means[states[j]]
                key = tuple(sorted((states[i], states[j])))
                weight = pair_weights.get(key, 1.0)
                sims.append(weight * F.cosine_similarity(a, b, dim=0))

        if not sims:
            return None
        return torch.stack(sims).clamp(min=0.0).mean()

    def _record_transition(self, patterns, timestamps, t, old, new, acts, nets, vfe):
        acts_dict = {ts: acts[i].detach().item() for i, ts in enumerate(self.agent.thoughtseeds)}
        nets_dict = {k: v.detach().item() for k, v in nets.items()}
        patterns.append((old, new, acts_dict, nets_dict, float(vfe)))
        timestamps.append(t)

    def _record_history(self, current_state, activations, meta_awareness, network_acts, free_energy, recon_loss, transition_drive, kl_div):
        # Calculate derived metrics for logging
        net_acts_float = {k: v.detach().item() for k, v in network_acts.items()}
        
        prec_min, prec_max = self.agent.params.get('l3tol2_precision_range', (0.4, 0.6))
        prec = prec_min + (prec_max - prec_min) * meta_awareness
        
        efe_val = getattr(self.agent.monitor, "last_efe", 0.0)
        dom = self.agent.thoughtseeds[torch.argmax(activations).item()]
        
        eff = compute_neural_efficiency_ratio(net_acts_float, current_state)
        stability = compute_dmn_dan_anticorrelation(net_acts_float)
        
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
            dominant_ts=str(dom),
            neural_efficiency=eff,
            stability_indicator=stability
        )

    def _save_results(self, output_dir, state_transition_patterns, transition_timestamps):
        self.logger.save_results(self.agent, state_transition_patterns, transition_timestamps, output_dir)
