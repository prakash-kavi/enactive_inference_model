"""Lightweight VAE for the Layer 2 attentional model.

Role in Russian Doll Architecture:
- Encoder: Recognition model q(z|x). Maps networks (L1) -> thoughtseed strengths (z in [0,1]).
- Bottleneck: independent strengths (no simplex constraint).
- Decoder: generative model p(x|z). Predicts expected brain-network activity given thoughtseeds.
"""

import torch
import torch.nn as nn

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
