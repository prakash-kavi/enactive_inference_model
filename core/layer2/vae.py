"""Lightweight VAE for the Layer 2 attentional model.

Role in Russian Doll Architecture:
- Encoder: Recognition model q(z|x). Maps networks (L1) -> thoughtseed strengths (z in [0,1]).
- Bottleneck: independent strengths (no simplex constraint).
- Decoder: generative model p(x|z). Predicts expected brain-network activity given thoughtseeds.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MeditationVAE(nn.Module):
    def __init__(self, input_dim=4, latent_dim=5, hidden_dim=32):
        super().__init__()

        self.input_dim = input_dim  # 4 Networks
        self.latent_dim = latent_dim  # 5 Thoughtseeds
        
        # --- Encoder (Recognition Model q(z|x)) ---
        # Maps 4 Networks -> 5 Thoughtseed Logits
        self.encoder_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim) # Output: Logits for 5 classes
        )
        
        # --- Decoder (Generative Model p(x|z)) ---
        # Maps 5 Thoughtseeds -> 4 Network Activations
        self.decoder_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim), 
            nn.Sigmoid() # Output: 0-1 Activation space
        )
        
    def encode(self, x):
        """
        Input: x (Batch, 4) - Network Activations
        Output: logits (Batch, 5) - Unnormalized thoughtseed probabilities
        """
        return self.encoder_net(x)
        
    def decode(self, z):
        """
        Input: z (Batch, 5) - Soft-one-hot thoughtseed vector
        Output: x_recon (Batch, 4) - Reconstructed Network Activations
        """
        return self.decoder_net(z)
        
    def forward(self, x):
        """
        Full VAE pass (independent strengths).
        Returns: recon_x, z, logits
        """
        logits = self.encode(x)
        z = torch.sigmoid(logits)
        recon_x = self.decode(z)
        return recon_x, z, logits

    def compute_loss(self, x, recon_x, logits):
        """
        Computes VAE Loss (VFE) for independent-strength latents.
        Loss = Reconstruction (MSE) + KL Divergence (Bernoulli vs p=0.5)
        """
        # 1. Reconstruction Loss (MSE)
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')

        # 2. KL Divergence (Bernoulli vs p=0.5)
        z = torch.sigmoid(logits)
        eps = 1e-6
        z = torch.clamp(z, eps, 1.0 - eps)
        prior = 0.5
        kl_div = torch.sum(
            z * torch.log(z / prior) +
            (1 - z) * torch.log((1 - z) / (1 - prior))
        )

        # Total VFE
        beta = 1.0
        loss = recon_loss + beta * kl_div

        return loss, recon_loss, kl_div
