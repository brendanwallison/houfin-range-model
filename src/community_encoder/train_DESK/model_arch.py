"""DESK autoencoder architecture, shared by the trainer (``desk_training``) and
the spacetime-cube builder (``build_final_z_cube``).

Each input stream (climate, land use, soil, ...) gets its own per-pixel encoder
branch; the branch codes are concatenated and mixed to a per-pixel latent
``z_point``. A small, config-gated **spatial residual** then lets each cell's
code be nudged by its immediate neighbours' codes -- the only way spatial context
can enter what is otherwise a strictly point-wise map. A single decoder
reconstructs the concatenated inputs.

The map is **grid-native**: ``forward(x, mask)`` takes a covariate grid
``(B, H, W, C)`` and a validity mask ``(B, H, W)`` (1 = usable cell) and returns
``(z (B,H,W,latent), recon (B,H,W,C))``. The MLP branches act per pixel (a 1x1
map); the spatial residual is a tight ``kernel``x``kernel`` **partial** (masked)
convolution so ocean/nodata cells never leak into a coastal cell's neighbourhood.

Design safeguards (see the DESK spatial-conv plan):
- The spatial term is a **residual** scaled by a learnable ``gamma`` initialised
  to 0, so at the start of training ``z == z_point`` -- identical to the pure
  point-wise model the ESK stabilizing/metric losses are aligned to. The conv can
  only earn influence if it reduces loss.
- The conv sits on the **reduced** latent (``latent_dim`` channels), so it is tiny
  (``kernel^2 * latent^2`` params) and data-efficient -- important because the
  supervised (ESK-labelled) signal exists for only a single year's grid.

``MultiInputAutoencoder`` (the deprecated 2-stream PRISM/BUI special case) is
retained only as a constructor shim; it has no live caller.
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


class PartialConv2d(nn.Conv2d):
    """Masked convolution (Liu et al. 2018): the output at each cell is the conv
    over only its *valid* neighbours, renormalised by the valid-neighbour count,
    so zero-filled invalid (ocean/nodata) cells contribute nothing. Returns the
    convolved tensor and the propagated validity mask.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        kH, kW = self.kernel_size
        self.register_buffer("_ones", torch.ones(1, 1, kH, kW))
        self._winsize = float(kH * kW)

    def forward(self, x, mask):
        # x: (B, C, H, W); mask: (B, 1, H, W) with 1 = valid.
        with torch.no_grad():
            cnt = F.conv2d(mask, self._ones, bias=None, stride=self.stride,
                           padding=self.padding)          # valid cells per window
            ratio = self._winsize / (cnt + 1e-8)          # upscale for missing neighbours
            new_mask = (cnt > 0).float()
            ratio = ratio * new_mask
        raw = super().forward(x * mask)
        if self.bias is not None:
            b = self.bias.view(1, -1, 1, 1)
            out = (raw - b) * ratio + b
        else:
            out = raw * ratio
        return out * new_mask, new_mask


class MultiStreamAutoencoder(nn.Module):
    """N-stream grid autoencoder: per-pixel encoder branches + shared latent +
    optional spatial residual + decoder.

    ``dims`` is the per-stream input width (e.g. [climate, landuse, soil]); their
    sum is the channel count ``C`` of the input grid, split internally in ``dims``
    order. ``spatial_kernel`` > 0 enables the residual masked conv (0 disables it,
    recovering the pure point-wise model).
    """

    def __init__(self, dims, latent_dim, spatial_kernel=3):
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
        self.spatial_kernel = int(spatial_kernel)
        if self.spatial_kernel > 0:
            pad = self.spatial_kernel // 2
            self.spatial = PartialConv2d(latent_dim, latent_dim, self.spatial_kernel, padding=pad)
            self.spatial_mix = nn.Conv2d(latent_dim, latent_dim, 1)
            # gamma init 0 -> z == z_point at start (pure point-wise, ESK-aligned).
            self.gamma = nn.Parameter(torch.zeros(1))

    def _pointwise_latent(self, x):
        """Per-pixel encode+mix of a flat ``(P, C)`` covariate batch -> ``(P, latent)``."""
        parts = torch.split(x, self.dims, dim=1)
        if len(parts) != len(self.encoders):
            raise ValueError(f"expected {len(self.encoders)} streams (dims={self.dims}), "
                             f"got input width {x.shape[1]}")
        h = torch.cat([enc(p) for enc, p in zip(self.encoders, parts)], dim=1)
        return self.mixer(h)

    def forward(self, x, mask):
        """``x``: covariate grid ``(B, H, W, C)``; ``mask``: ``(B, H, W)`` (1=valid).

        Returns ``z (B, H, W, latent)`` and ``recon (B, H, W, C)``.
        """
        if x.dim() != 4:
            raise ValueError(f"expected grid input (B,H,W,C), got {tuple(x.shape)}")
        B, H, W, C = x.shape
        z_point = self._pointwise_latent(x.reshape(B * H * W, C))       # (B*H*W, L)
        L = z_point.shape[1]
        z_grid = z_point.reshape(B, H, W, L).permute(0, 3, 1, 2)         # (B, L, H, W)

        if self.spatial_kernel > 0:
            m = mask.reshape(B, 1, H, W).to(z_grid.dtype)
            r, _ = self.spatial(z_grid, m)
            r = self.spatial_mix(F.gelu(r))
            z_grid = z_grid + self.gamma * r

        z = z_grid.permute(0, 2, 3, 1).contiguous()                      # (B, H, W, L)
        recon = self.decoder(z.reshape(B * H * W, L)).reshape(B, H, W, C)
        return z, recon


class MultiInputAutoencoder(MultiStreamAutoencoder):
    """Backward-compatible 2-stream (prism, bui) constructor shim (no live caller)."""

    def __init__(self, prism_dim, bui_dim, latent_dim, spatial_kernel=3):
        super().__init__([prism_dim, bui_dim], latent_dim, spatial_kernel)
