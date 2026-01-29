"""Trainer orchestrating BPTT for the Layer 2 attentional model."""
import os
import json
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional

from utils.meditation_utils import ensure_directories, _save_json_outputs, compute_state_aggregates, build_transition_stats
from config.meditation_config import DEFAULTS, NETWORK_PROFILES, get_thoughtseed_targets
from utils.meditation_diagnostics import (
    compute_neural_efficiency_ratio, detect_expert_mind_wandering,
    compute_dmn_dan_anticorrelation
)
from ..layer1.process import Layer1Process

class PracticeTrainer:
    """Orchestrates the BPTT simulation loop."""
    
    def __init__(self, agent, generative_process: Optional[Layer1Process] = None):
        self.agent = agent
        # Initialize generative process if not provided
        if generative_process is None:
            self.process = Layer1Process(
                experience_level=agent.experience_level,
                seed=agent.rng.randint(0, 2**31) if hasattr(agent.rng, 'randint') else None
            )
        else:
            self.process = generative_process
            
        # Optimizer
        # Optimize Agent's parameters (Theta, Mu, etc.)
        # Layer 1 parameters are usually fixed, but if we wanted to learn L1, we'd add them.
        # User goal: "Gradient-based parameter optimization to fix attractor collapse" -> Agent parameters.
        self.optimizer = optim.Adam(self.agent.parameters(), lr=agent.learning_rate)

    def train(self, save_outputs: bool = True, output_dir: str = None, seed: int = None, enable_learning: bool = True):
        """Run BPTT training."""

        if seed is not None:
             # Set torch seed
             torch.manual_seed(seed)
             np.random.seed(seed)
             # Agent RNG (kept for legacy random calls if any)
             self.agent.rng = np.random.RandomState(seed)
             # Process RNG (numpy part)
             self.process.rng_np = np.random.RandomState(seed)

        # 1. Initialization
        self.optimizer.zero_grad()
        current_state, current_dwell, dwell_limit, activations, network_acts, meta_awareness = self._initialize_simulation()
        
        state_transition_patterns = []
        transition_timestamps = []
        old_state = current_state
        
        # BPTT Configuration
        bptt_steps = 50 # Longer sequence to capture dwell transitions
        total_steps = self.agent.timesteps
        
        # 2. Training Loop (BPTT)
        # We process in chunks of `bptt_steps`
        
        for t_start in range(0, total_steps, bptt_steps):
            # Clear old sensory states to prevent graph leaks
            self.agent.blanket.sensory_states.clear()
            self.agent.blanket_l2l3.sensory_states.clear()
            
            steps_to_run = min(bptt_steps, total_steps - t_start)
            
            # Run Sequence
            loss = torch.tensor(0.0, device=self.agent.vae.encoder_net[0].weight.device)
            state_net_accum = {s: [] for s in self.agent.states}
            
            for t_sub in range(steps_to_run):
                t = t_start + t_sub
                
                # --- Step Logic ---
                # 2.1 Biology (Layer 1)
                # Reads actives from L1-L2 (which were set in previous step)
                # L1 returns Differentiable Tensors
                network_acts, process_state = self.process.update(self.agent.blanket.active_states)
                
                # State Check
                state_changed = (process_state != current_state)
                if state_changed:
                    # Record transition info (detached)
                    self._record_transition(state_transition_patterns, transition_timestamps, t, old_state, process_state, activations, network_acts, 0.0)
                    # Update pointers
                    old_state = current_state
                    current_state = process_state
                    current_dwell = 0
                    dwell_limit = self.process.current_max_dwell
                else:
                    current_dwell += 1
                    
                # 2.2 Blanket L1-L2 (Sensory)
                self.agent.blanket.update_sensory_states(network_acts)
                
                # 2.3 Perception (Layer 2)
                sensory_inference = self.agent.perceptual_inference()
                
                # 2.4 Top-Down (Layer 2) - Top-down priors
                # Need to handle variables passing.
                meta_awareness, activations, _ = self._pass_top_down(
                    current_state, current_dwell, dwell_limit, activations, meta_awareness,
                    network_acts, sensory_inference
                )
                
                # 2.5 Bottom-Up (L2 -> L3)
                # Pass None for target_activations as we use prior_seeds from blanket
                free_energy, accuracy_nll = self._pass_bottom_up(
                     current_state, activations, meta_awareness, 
                     None, 
                     network_acts, sensory_inference
                )
                
                # Accumulate Loss (Global VFE)
                # Simple accumulation - PyTorch handles the graph correctly
                loss = loss + free_energy
                
                # Note: Structural Contrastive Loss is now computed at the batch level
                # via _compute_structural_loss() to be efficient and holistic.
                # Old per-step logic removed.
                
                # 2.6 Action (L3 -> L2 -> L1)
                prescription = self.agent.prescriptive_action(activations, free_energy, current_state)
                # Prescriptions applied to blanket
                
                # 2.7 Record History (Detached)
                self._record_history(current_state, activations, meta_awareness, network_acts, free_energy, accuracy_nll)

                # Accumulate per-state network vectors for contrastive loss
                net_vec = torch.stack([network_acts[net] for net in self.agent.networks])
                state_net_accum[current_state].append(net_vec)
                
            # --- End of Sequence: Backprop ---
            if enable_learning and loss.requires_grad:
                # Contrastive loss over state means (within this BPTT window)
                contrastive_weight = self.agent.params.get("state_contrastive_weight", 0.0)
                if contrastive_weight > 0.0 and 'state_net_accum' in locals():
                    state_means = {}
                    for s, vecs in state_net_accum.items():
                        if vecs:
                            state_means[s] = torch.stack(vecs, dim=0).mean(dim=0)
                    states = list(state_means.keys())
                    if len(states) > 1:
                        sims = []
                        for i in range(len(states)):
                            for j in range(i + 1, len(states)):
                                a = state_means[states[i]]
                                b = state_means[states[j]]
                                sims.append(F.cosine_similarity(a, b, dim=0))
                        if sims:
                            contrastive_loss = torch.stack(sims).clamp(min=0.0).mean()
                            loss = loss + (contrastive_weight * contrastive_loss)
                
                loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # --- Detach States for TBPTT ---
            # Layer 1: Detach regular tensors (no longer registered buffers)
            with torch.no_grad():
                self.process.x = self.process.x.detach()
                self.process.smoothed_x = self.process.smoothed_x.detach()
            
            # Layer 2: Use .data to safely modify buffers in-place
            with torch.no_grad():
                self.agent.aha_accum_val.data = self.agent.aha_accum_val.detach()
                if hasattr(self.agent, "z_ema"):
                    self.agent.z_ema = self.agent.z_ema.detach()
            
            # Detach Blanket Active States (Critical for BPTT across batches)
            # Replace dictionary values with detached copies
            for k in list(self.agent.blanket.active_states.keys()):
                v = self.agent.blanket.active_states[k]
                if isinstance(v, torch.Tensor):
                    self.agent.blanket.active_states[k] = v.detach()
                    
            for k in list(self.agent.blanket_l2l3.active_states.keys()):
                v = self.agent.blanket_l2l3.active_states[k]
                if isinstance(v, torch.Tensor):
                    self.agent.blanket_l2l3.active_states[k] = v.detach()
            
            # Activations variable (curr latent state)
            activations = activations.detach()
            
            # Clear graph for next iteration implicitly by new tensors

        # 3. Save Results
        if save_outputs:
            self._save_results(output_dir, state_transition_patterns, transition_timestamps)
            
        return self.agent

    def _initialize_simulation(self) -> Tuple[str, int, int, torch.Tensor, Dict[str, torch.Tensor], float]:
        """Initialize simulation."""
        self.process.reset(state='breath_focus')
        current_state = self.process.current_state
        current_dwell = 0
        dwell_limit = self.process.current_max_dwell
        
        # Initial activations (Tensor)
        device = self.agent.vae.encoder_net[0].weight.device
        
        mu_dict = get_thoughtseed_targets(current_state, 0.6, self.agent.experience_level)
        activations_np = np.array([mu_dict[ts] for ts in self.agent.thoughtseeds])
        # Add noise
        activations_np += np.random.normal(0, self.agent.noise_level, size=len(activations_np))
        activations = torch.tensor(activations_np, dtype=torch.float32, device=device)
        activations = torch.clamp(activations, DEFAULTS['ACTIVATION_CLIP_MIN'], DEFAULTS['ACTIVATION_CLIP_MAX'])
        
        # Initial Network obs
        network_acts, _ = self.process.update(self.agent.blanket.active_states) # Tensor dict
        self.agent.blanket.update_sensory_states(network_acts)
        
        self.agent.blanket_l2l3.reset()
        
        meta_awareness = self.agent.get_meta_awareness(current_state, activations) # returns float
        
        return current_state, current_dwell, dwell_limit, activations, network_acts, meta_awareness

    def _pass_top_down(self, current_state: str, current_dwell: int, dwell_limit: int, 
                      activations: torch.Tensor, prev_meta_awareness: float,
                      observed_networks: Dict[str, torch.Tensor], sensory_inference: torch.Tensor) -> Tuple[float, torch.Tensor, Dict[str, torch.Tensor]]:
        
        raw_meta = self.agent.get_meta_awareness(current_state, activations)
        smoothing = self.agent.params['smoothing']
        meta_awareness = smoothing * prev_meta_awareness + (1 - smoothing) * raw_meta
        
        # Make meta-awareness available to L3/L2 policies and precision logic
        self.agent.blanket_l2l3.update_sensory_states({
            'meta_awareness': meta_awareness,
            'current_state': current_state
        })
        
        activations = self.agent.update_thoughtseed_dynamics(
            activations, current_state, current_dwell, dwell_limit, observed_networks, sensory_inference
        )
        # update_thoughtseed_dynamics returns updated z.
        # It also updates blanket with prior_seeds etc.
        
        return meta_awareness, activations, {} # Return empty targets map

    def _pass_bottom_up(self, current_state: str, activations: torch.Tensor, meta_awareness: float, 
                       target_activations: Any, observed_networks: Dict[str, torch.Tensor],
                       sensory_inference: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # A. Reconstruction from the same latent used for dynamics (state-aware)
        recon_x = self.agent.decode_with_state(activations, current_state)
        
        # B. VAE Loss (VFE)
        # Prepare Tensors
        device = self.agent.vae.encoder_net[0].weight.device
        
        # Stack Observed (Generic order check?)
        # self.agent.networks list order is fixed ['DMN', 'VAN', 'DAN', 'FPN']
        x = torch.stack([observed_networks[net] for net in self.agent.networks]).to(device)
        recon_x = recon_x.to(device)

        # Reconstruction loss (MSE)
        recon_loss = torch.nn.functional.mse_loss(recon_x, x, reduction='sum')

        # KL for independent strengths: Bernoulli q(z) vs uniform prior p=0.5
        eps = 1e-6
        z = activations
        if z.dim() != 1:
            z = z.squeeze(0)
        z = torch.clamp(z, eps, 1.0 - eps)
        prior = 0.5
        kl_div = torch.sum(
            z * torch.log(z / prior) +
            (1 - z) * torch.log((1 - z) / (1 - prior))
        )

        # C. Network target regularizer (lightweight)
        reg_weight = self.agent.params.get("network_target_reg", 0.0)
        target_loss = torch.tensor(0.0, device=device)
        if reg_weight > 0.0 and current_state in NETWORK_PROFILES:
            prof = NETWORK_PROFILES[current_state][self.agent.experience_level]
            target_vec = torch.tensor([prof[n] for n in self.agent.networks], device=device, dtype=torch.float32)
            target_loss = torch.nn.functional.mse_loss(x, target_vec, reduction='sum')

        # Total loss
        beta = 1.0
        loss = recon_loss + beta * kl_div + (reg_weight * target_loss)
        
        # We treat VAE loss as "Free Energy"
        free_energy = loss
        accuracy_nll = recon_loss # approximation
        
        return free_energy, accuracy_nll

    def _record_transition(self, patterns, timestamps, t, old, new, acts, nets, vfe):
        # Detach and convert
        acts_dict = {ts: acts[i].detach().item() for i, ts in enumerate(self.agent.thoughtseeds)}
        nets_dict = {k: v.detach().item() for k, v in nets.items()}
        patterns.append((old, new, acts_dict, nets_dict, float(vfe)))
        timestamps.append(t)

    def _record_history(self, current_state, activations, meta_awareness, network_acts, free_energy, accuracy_nll):
        # Detach and convert
        self.agent.state_history.append(current_state)
        self.agent.activations_history.append(activations.detach().cpu().numpy())
        self.agent.meta_awareness_history.append(meta_awareness)
        
        net_acts_float = {k: v.detach().item() for k, v in network_acts.items()}
        self.agent.network_activations_history.append(net_acts_float)
        
        self.agent.free_energy_history.append(free_energy.detach().item())
        self.agent.prediction_error_history.append(accuracy_nll.detach().item())
        
        prec = 0.5 + self.agent.params['precision_weight'] * meta_awareness
        self.agent.precision_history.append(prec)
        
        dom = self.agent.thoughtseeds[torch.argmax(activations).item()]
        self.agent.dominant_ts_history.append(dom)
        
        # Diagnostics
        eff = compute_neural_efficiency_ratio(net_acts_float, current_state)
        if eff is not None: self.agent.neural_efficiency_history.append(eff)
        
        if detect_expert_mind_wandering(net_acts_float) is True:
             self.agent.expert_mind_wandering_detections += 1
             
        self.agent.stability_indicators.append(compute_dmn_dan_anticorrelation(net_acts_float))

    def _save_results(self, output_dir, state_transition_patterns, transition_timestamps):
        """Save results (Helper)."""
        out_dir = output_dir or os.path.join(os.path.dirname(__file__), "data")
        os.makedirs(out_dir, exist_ok=True)
        # Need to ensure agent history lists are populated with floats/numpy, not tensors (Done in _record_history)
        
        aggregates = compute_state_aggregates(self.agent)
        transition_stats = build_transition_stats(self.agent, state_transition_patterns, transition_timestamps, aggregates)
        
        out_path = os.path.join(out_dir, f"transition_stats_{self.agent.experience_level}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(transition_stats, f, indent=2)

        _save_json_outputs(self.agent, output_dir=out_dir, aggregates=aggregates)
