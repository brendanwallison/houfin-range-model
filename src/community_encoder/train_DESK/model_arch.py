"""DESK autoencoder architecture, shared by the trainer (``desk_training``) and
the spacetime-cube builder (``build_final_z_cube``).
"""
import torch
import torch.nn.functional as F
from torch import nn


class BMLPBlock(nn.Module):
    def __init__(self, m, k=4, dropout=0.5):
        super().__init__()
        self.ln = nn.LayerNorm(m)
        self.fc1 = nn.Linear(m, m * k)
        self.fc2 = nn.Linear(m * k, m)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        z = self.ln(x)
        z = F.gelu(self.fc1(z))
        z = self.drop(z)
        z = self.fc2(z)
        return x + z


class MultiInputAutoencoder(nn.Module):
    def __init__(self, prism_dim, bui_dim, latent_dim):
        super().__init__()
        h = max(128, latent_dim * 4)

        self.prism_enc = nn.Sequential(
            nn.Linear(prism_dim, h), nn.GELU(),
            BMLPBlock(h), BMLPBlock(h),
        )
        self.bui_enc = nn.Sequential(
            nn.Linear(bui_dim, h), nn.GELU(),
            BMLPBlock(h), BMLPBlock(h),
        )
        self.mixer = nn.Sequential(
            nn.Linear(2 * h, 2 * h), nn.GELU(),
            nn.Linear(2 * h, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 2 * h), nn.GELU(),
            nn.Linear(2 * h, prism_dim + bui_dim),
        )

    def forward(self, prism, bui):
        h = torch.cat([self.prism_enc(prism), self.bui_enc(bui)], dim=1)
        z_pred = self.mixer(h)
        return z_pred, self.decoder(z_pred)
