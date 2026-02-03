"""Layer 2: attentional agent bottleneck (VAE + policies)."""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, List

from utils.meditation_config import THOUGHTSEEDS, STATES, NETWORKS, DEFAULTS
from utils.meditation_utils import get_actinf_params, get_thoughtseed_targets, compute_meta_awareness
from ..generative_model import ObservationModel
from ..blankets.l1_l2 import MarkovBlanketL1L2
from ..blankets.l2_l3 import MarkovBlanketL2L3
from ..layer3.monitor import Layer3Monitor

class Layer2AttentionalModel(nn.Module):
    """Layer 2: attentional agents + learnable dynamics."""
    
    def __init__(self, experience_level: str = 'novice', timesteps_per_cycle: int = 200, seed: Optional[int] = None,
                 enable_forward_model: bool = True, enable_forward_actions: bool = True):
        super().__init__()
        
        # Ablation flags
        self.enable_forward_model = enable_forward_model
        self.enable_forward_actions = enable_forward_actions
        
        self.experience_level = experience_level
        self.timesteps = timesteps_per_cycle
        self.thoughtseeds = THOUGHTSEEDS
        self.states = STATES
        self.num_thoughtseeds = len(self.thoughtseeds)
        self.networks = NETWORKS
        self.ts_index = {ts: i for i, ts in enumerate(self.thoughtseeds)}
        self.rng = np.random.RandomState(seed)
        
        self.params = get_actinf_params(experience_level)
        
        self.van_history: List[float] = []

        from .vae import MeditationVAE
        self.vae = MeditationVAE(
            input_dim=len(self.networks),
            latent_dim=self.num_thoughtseeds,
            enable_forward_model=self.enable_forward_model
        )
        
        self.mu_params = nn.ParameterDict()
        self.state_embed_dim = 2
        self.state_embeddings = nn.ParameterDict()
        self.state_embed_to_bias = nn.Linear(self.state_embed_dim, len(self.networks), bias=False)
        
        for state in self.states:
            mu_dict = get_thoughtseed_targets(state, self.experience_level)
            mu_vec = [mu_dict[ts] for ts in self.thoughtseeds]
            self.mu_params[state] = nn.Parameter(torch.tensor(mu_vec, dtype=torch.float32))
            embed = torch.randn(self.state_embed_dim, dtype=torch.float32) * 0.05
            self.state_embeddings[state] = nn.Parameter(embed)
            
        
        self.register_buffer('z_ema', torch.zeros(self.num_thoughtseeds))
        self.z_ema_initialized = False

        self.blanket = MarkovBlanketL1L2(smoothing=0.7)
        self.blanket_l2l3 = MarkovBlanketL2L3(smoothing=0.7)
        
        def _meta_fn(current_state, z):
            act_dict = {ts: z[i].item() for i, ts in enumerate(self.thoughtseeds)}
            return compute_meta_awareness(current_state, act_dict)

        self.monitor = Layer3Monitor(
            experience_level=self.experience_level,
            efe_ambiguity_weight=self.params['efe_ambiguity_weight'],
            efe_cycle_strength=self.params['efe_cycle_strength'],
            efe_gain=self.params['efe_gain'],
            policy_horizon=self.params['policy_horizon'],
            policy_temperature=self.params['policy_temperature'],
            policy_temperature_by_state=self.params['policy_temperature_by_state'],
            policy_horizon_discount=self.params['policy_horizon_discount'],
            l3tol2_precision_range=self.params['l3tol2_precision_range'],
            get_meta_awareness_fn=_meta_fn,
            blanket_l2l3=self.blanket_l2l3,
            vfe_ema_alpha=self.params['vfe_ema_alpha']
        )
        
        self.init_noise_sigma = float(self.params.get('z_noise_sigma', 0.0))
        self.learning_rate = self.params['learning_rate']
        self.z_ema_alpha = 0.75
        self.obs_model = ObservationModel(eps=1e-6)
        self.last_latent_vfe_terms = {
            'reconstruction': 0.0,
            'prior_kl': 0.0,
            'sensory_consistency': 0.0,
            'temporal_consistency': 0.0,
            'total': 0.0,
        }


    def detect_van_spike(self, current_van: float) -> bool:
        """Detect VAN spike."""
        self.van_history.append(current_van)
        if len(self.van_history) > 10:
            self.van_history.pop(0)
        
        if len(self.van_history) < 2:
            return False
            
        prev_van = self.van_history[-2]
        spike_threshold = 0.7
        spike_delta = 0.15
        
        is_spike = (current_van > spike_threshold) and ((current_van - prev_van) > spike_delta)
        return is_spike

    def perceptual_inference(self) -> torch.Tensor:
        """Perceptual inference (networks -> thoughtseed strengths)."""
        network_acts = self.blanket.sensory_states
        device = self.vae.encoder_net[0].weight.device
        
        # Convert inputs to tensor
        net_vec = torch.stack([network_acts.get(net, torch.tensor(0.0, device=device)) for net in self.networks])
        
        # Encoder (q(z|x))
        if net_vec.dim() == 1:
            x_in = net_vec.unsqueeze(0)
            logits = self.vae.encode(x_in)
            # Independent strengths: sigmoid per thoughtseed (no simplex constraint)
            inferred = torch.sigmoid(logits).squeeze(0)
        else:
            logits = self.vae.encode(net_vec)
            inferred = torch.sigmoid(logits)
        
        return torch.clamp(inferred, DEFAULTS['ACTIVATION_CLIP_MIN'], DEFAULTS['ACTIVATION_CLIP_MAX'])

    def decode_with_state(self, z: torch.Tensor, current_state: str) -> torch.Tensor:
        """Decode thoughtseeds into networks with a learned state bias."""
        if z.dim() == 1:
            z_in = z.unsqueeze(0)
            decoded = self.vae.decode(z_in).squeeze(0)
        else:
            decoded = self.vae.decode(z)
        embed = self.state_embeddings.get(current_state)
        if embed is not None:
            bias = self.state_embed_to_bias(embed.unsqueeze(0)).squeeze(0)
            decoded = decoded + bias
        return torch.clamp(decoded, 0.0, 1.0)

    def _latent_variational_inference(
        self,
        current_state: str,
        current_activations: torch.Tensor,
        observed_networks: Dict[str, torch.Tensor],
        sensory_inference: Optional[torch.Tensor],
        precision_modulation: float,
    ) -> torch.Tensor:
        """Run explicit variational updates over latent thoughtseed activations."""
        device = current_activations.device
        clip_min = DEFAULTS['ACTIVATION_CLIP_MIN']
        clip_max = DEFAULTS['ACTIVATION_CLIP_MAX']

        observed_vec = torch.stack([
            observed_networks.get(net, torch.tensor(0.0, device=device))
            for net in self.networks
        ]).to(device)

        z_prev = torch.clamp(current_activations.detach(), clip_min, clip_max)
        if sensory_inference is None:
            sensory_target = z_prev
        else:
            sensory_target = torch.clamp(sensory_inference.detach(), clip_min, clip_max)
        prior_target = torch.clamp(self.mu_params[current_state].detach(), clip_min, clip_max)

        z_init = 0.5 * z_prev + 0.5 * sensory_target
        z_init = ((1.0 - (0.5 * precision_modulation)) * z_init) + ((0.5 * precision_modulation) * prior_target)
        z_var = torch.clamp(z_init, clip_min, clip_max).requires_grad_(True)

        vi_steps = max(1, int(self.params['l2_vi_steps']))
        vi_lr = float(self.params['l2_vi_lr'])
        vi_grad_clip = float(self.params['l2_vi_grad_clip'])
        obs_w = float(self.params['l2_vi_obs_weight'])
        prior_w = float(self.params['l2_vi_prior_weight'])
        sensory_w = float(self.params['l2_vi_sensory_weight']) * (1.0 - precision_modulation)
        temporal_w = float(self.params['l2_vi_temporal_weight'])

        final_terms = None
        for _ in range(vi_steps):
            recon_x = self.decode_with_state(z_var, current_state)
            terms = self.obs_model.latent_variational_terms(
                z=z_var,
                prior=prior_target,
                recon_x=recon_x,
                observed_x=observed_vec,
                precision=precision_modulation,
                sensory_target=sensory_target,
                temporal_target=z_prev,
                obs_weight=obs_w,
                prior_weight=prior_w,
                sensory_weight=sensory_w,
                temporal_weight=temporal_w,
            )
            final_terms = terms
            grad = torch.autograd.grad(terms.total, z_var, retain_graph=False, create_graph=False)[0]
            if vi_grad_clip > 0.0:
                grad = torch.clamp(grad, -vi_grad_clip, vi_grad_clip)
            with torch.no_grad():
                z_var = torch.clamp(z_var - (vi_lr * grad), clip_min, clip_max)
            z_var.requires_grad_(True)

        if final_terms is None:
            raise RuntimeError('Latent VI did not produce terms')

        self.last_latent_vfe_terms = {
            'reconstruction': float(final_terms.reconstruction.detach().item()),
            'prior_kl': float(final_terms.prior_kl.detach().item()),
            'sensory_consistency': float(final_terms.sensory_consistency.detach().item()),
            'temporal_consistency': float(final_terms.temporal_consistency.detach().item()),
            'total': float(final_terms.total.detach().item()),
        }

        return z_var.detach()

    def update_thoughtseed_dynamics(self, current_activations: torch.Tensor, 
                                   current_state: str, current_dwell: int, dwell_limit: int,
                                   observed_networks: Dict[str, torch.Tensor],
                                   sensory_inference: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Update attentional-agent strengths."""
        van = observed_networks.get('VAN', torch.tensor(0.0)).item()
        van_spike_detected = self.detect_van_spike(van)

        progress = 0.0
        if dwell_limit > 0:
            progress = min(1.0, current_dwell / max(1, dwell_limit))
        prec_mod = self.blanket_l2l3.active_states.get('precision_modulation', 0.5)
        if isinstance(prec_mod, torch.Tensor):
            prec_mod = prec_mod.item()
        prec_mod = float(np.clip(prec_mod, 0.0, 1.0))

        z_inferred = self._latent_variational_inference(
            current_state=current_state,
            current_activations=current_activations,
            observed_networks=observed_networks,
            sensory_inference=sensory_inference,
            precision_modulation=prec_mod,
        )

        # Dwell-scheduled latent noise for richer dynamics
        noise_sigma = self.params['z_noise_sigma']
        if dwell_limit > 0:
            noise_scale = 1.0 - min(1.0, current_dwell / max(1, dwell_limit))
        else:
            noise_scale = 1.0
        if noise_sigma > 0.0:
            noise = torch.randn_like(z_inferred) * noise_sigma * noise_scale
            z_inferred = torch.clamp(z_inferred + noise, DEFAULTS['ACTIVATION_CLIP_MIN'], DEFAULTS['ACTIVATION_CLIP_MAX'])

        # Soft WTA + global workspace broadcast (precision-gated).
        tau = max(0.25, 0.9 - (0.7 * prec_mod))
        winner = torch.softmax(z_inferred / tau, dim=0)
        workspace_gain = float(np.clip(prec_mod, 0.0, 1.0))
        z_inferred = torch.clamp(
            z_inferred + (workspace_gain * winner),
            DEFAULTS['ACTIVATION_CLIP_MIN'],
            DEFAULTS['ACTIVATION_CLIP_MAX']
        )

        alpha = self.z_ema_alpha
        aha_idx = self.ts_index.get('aha_moment')
        if aha_idx is not None and (z_inferred[aha_idx].item() > 0.6 or van_spike_detected):
            alpha = min(alpha, 0.6)
        alpha = min(alpha, 0.4 + 0.4 * progress)

        if current_dwell == 0:
            self.z_ema = z_inferred.clone()
            self.z_ema_initialized = True
        elif not self.z_ema_initialized or self.z_ema.shape != z_inferred.shape:
            self.z_ema = z_inferred.clone()
            self.z_ema_initialized = True
        else:
            self.z_ema = (alpha * self.z_ema) + ((1 - alpha) * z_inferred)

        updated_activations = torch.clamp(self.z_ema, DEFAULTS['ACTIVATION_CLIP_MIN'], DEFAULTS['ACTIVATION_CLIP_MAX'])
        
        return updated_activations

    @staticmethod
    def _compose_l1_policy(agent_bias: torch.Tensor, precision: float, transition_pressure: float) -> dict:
        """Compose Layer-1 control signals from Layer-2 state and Layer-3 modulation."""
        precision = float(np.clip(precision, 0.0, 1.0))
        transition_pressure = float(np.clip(transition_pressure, 0.0, 1.0))
        noise_reduction = float(np.clip(1.0 - (0.6 * precision), 0.4, 1.0))
        return {
            'agent_bias': agent_bias,
            'l2tol1_enactive_bias': precision,
            'noise_reduction': noise_reduction,
            'transition_drive': transition_pressure,
        }

    def prescriptive_action(self, z: torch.Tensor, vfe: torch.Tensor, current_state: str) -> dict:
        """Run hierarchical policy: L3 modulates L2, then L2 controls L1."""
        
        # Phase 4: Evaluate candidate actions via forward prediction (if enabled)
        if self.enable_forward_actions:
            candidate_states = [current_state]  # Primary: stay in current state
            # Add transition candidates based on state machine
            if current_state == 'mind_wandering':
                candidate_states.append('meta_awareness')
            elif current_state == 'meta_awareness':
                candidate_states.append('redirect_attention')
            elif current_state == 'redirect_attention':
                candidate_states.append('breath_focus')
            
            best_action = None
            min_predicted_error = float('inf')
            
            # Get current networks for forward prediction
            x_current = torch.stack([
                self.blanket.sensory_states.get(net, torch.tensor(0.0, device=z.device))
                for net in self.networks
            ])
            
            for candidate_state in candidate_states:
                # Decode candidate action to get target thoughtseed activation
                mu_candidate = self.mu_params[candidate_state]
                
                # Phase 4 FIX: Use candidate mu as the "action" input to forward model
                # This represents taking action toward candidate_state attractor
                with torch.no_grad():
                    x_next_pred = self.vae.predict_next(x_current, mu_candidate)
                
                # Decode candidate to get expected network activations
                action_candidate = self.decode_with_state(mu_candidate, candidate_state)
                
                # Evaluate predicted deviation from candidate goal
                prediction_error = torch.mean((x_next_pred - action_candidate)**2).item()
                
                if prediction_error < min_predicted_error:
                    min_predicted_error = prediction_error
                    best_action = (candidate_state, action_candidate, mu_candidate)
            
            # Use best action (or fallback to current state)
            if best_action is not None:
                _, agent_bias, selected_mu = best_action  # State not needed after selection
            else:
                selected_mu = self.mu_params[current_state]
                agent_bias = self.decode_with_state(selected_mu, current_state)
        else:
            # Ablation: forward-informed actions disabled, use current state only
            selected_mu = self.mu_params[current_state]
            agent_bias = self.decode_with_state(selected_mu, current_state)
        
        vfe_val = vfe.item() if isinstance(vfe, torch.Tensor) else float(vfe)
        self.blanket_l2l3.update_sensory_states({
            'thoughtseed_activations': z,
            'vfe': vfe_val,
            'current_state': current_state
        })
        
        l3_policy = self.monitor.evaluate_policies()

        # Use the smoothed precision stored on the L2<->L3 blanket for consistency
        # with latent dynamics updates.
        precision = self.blanket_l2l3.active_states['precision_modulation']
        if isinstance(precision, torch.Tensor):
            precision = precision.item()

        transition_pressure = l3_policy['transition_pressure']
        if isinstance(transition_pressure, torch.Tensor):
            transition_pressure = transition_pressure.item()

        prescription_l1l2 = self._compose_l1_policy(
            agent_bias=agent_bias,
            precision=float(precision),
            transition_pressure=float(transition_pressure),
        )
        
        # Phase 4: Store selected action for forward prediction
        prescription_l1l2['selected_action_mu'] = selected_mu
        
        self.blanket.update_active_states(prescription_l1l2)
        
        return prescription_l1l2
