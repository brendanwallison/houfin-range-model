"""DESK autoencoder architecture, shared by the trainer (``desk_training``) and
the spacetime-cube builder (``build_final_z_cube``).

Each input stream (climate, land use, soil, ...) gets its own encoder branch;
the branch codes are concatenated, mixed to the latent Z, and a single decoder
reconstructs the concatenated inputs. ``MultiStreamAutoencoder`` takes an
arbitrary list of per-stream input dims; ``MultiInputAutoencoder`` is the
2-stream (prism, bui) special case kept for backward compatibility.
"""
import torch
import torch.nn.functional as F
from torch import nn


class BMLPBlock(nn.Module):
    """Residual pre-LayerNorm MLP block: x + fc2(drop(gelu(fc1(LN(x)))))."""

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


class MultiStreamAutoencoder(nn.Module):
    """N-stream autoencoder: one encoder branch per input, shared latent + decoder.

    ``dims`` is the per-stream input width (e.g. [climate, landuse, soil]).
    ``forward(*streams)`` takes one tensor per stream (same order as ``dims``)
    and returns ``(z_pred, reconstruction)``, where the reconstruction is the
    concatenation of all streams (order = ``dims``); split it with ``self.dims``.
    """

    def __init__(self, dims, latent_dim):
        super().__init__()
        self.dims = list(dims)
        n = len(self.dims)
        h = max(128, latent_dim * 4)
        self.encoders = nn.ModuleList([
            nn.Sequential(nn.Linear(d, h), nn.GELU(), BMLPBlock(h), BMLPBlock(h))
            for d in self.dims
        ])
        self.mixer = nn.Sequential(
            nn.Linear(n * h, n * h), nn.GELU(),
            nn.Linear(n * h, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, n * h), nn.GELU(),
            nn.Linear(n * h, sum(self.dims)),
        )

    def forward(self, *streams):
        if len(streams) != len(self.encoders):
            raise ValueError(f"expected {len(self.encoders)} streams, got {len(streams)}")
        h = torch.cat([enc(s) for enc, s in zip(self.encoders, streams)], dim=1)
        z_pred = self.mixer(h)
        return z_pred, self.decoder(z_pred)


class MultiInputAutoencoder(MultiStreamAutoencoder):
    """Backward-compatible 2-stream (prism, bui) autoencoder."""

    def __init__(self, prism_dim, bui_dim, latent_dim):
        super().__init__([prism_dim, bui_dim], latent_dim)

    def forward(self, prism, bui):
        return super().forward(prism, bui)
