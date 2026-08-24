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
import numpy as np
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


def resolve_hidden_widths(hidden_width, n_streams, latent_dim):
    """``hidden_width`` (scalar, sequence, or ``None``) -> a per-stream list of ``n_streams`` ints.

    One resolver because the value arrives from three places -- a config, a constructor
    argument, and ``desk_meta.npz`` -- and it has three admissible shapes. Resolving it in
    each of them separately is how the scalar/list distinction becomes a silent shape bug:
    a length-1 list would broadcast in one place and index out of range in another.

    A sequence whose length does not match the stream count is refused rather than padded
    or truncated. Widths are POSITIONAL against ``dims``, so a length mismatch means the
    widths are being applied to the wrong streams, and padding would apply a default to
    whichever streams happened to fall off the end -- exactly the kind of off-by-one that
    the species-column bug taught this project to fail loudly on.
    """
    n = int(n_streams)
    if hidden_width is None or (not isinstance(hidden_width, (list, tuple))
                                and not hidden_width):
        return [max(128, int(latent_dim) * 4)] * n           # the historical default
    if isinstance(hidden_width, (list, tuple)):
        widths = [int(v) for v in hidden_width]
        if len(widths) != n:
            raise ValueError(
                f"hidden_width has {len(widths)} entries but there are {n} streams. Widths "
                f"are positional against `dims`, so a mismatch applies them to the wrong "
                f"streams; state the width for every stream explicitly.")
        if any(w < 1 for w in widths):
            raise ValueError(f"every per-stream hidden_width must be >= 1; got {widths}")
        return widths
    return [int(hidden_width)] * n


def hidden_width_from_meta(dm):
    """Read ``hidden_width`` back out of a ``desk_meta.npz`` as a scalar, list, or ``None``.

    ``np.savez`` stores an int as a 0-d array and a per-stream list as a 1-d one, and
    ``int()`` on the latter raises only for length > 1 -- so a one-stream list would convert
    silently while a six-stream list crashed at inference, two GPU-hours after the run that
    produced it. Every reader goes through here so the scalar/list distinction is decided in
    exactly one place, and so an OLD checkpoint (scalar, or the key absent entirely) keeps
    loading unchanged.
    """
    if "hidden_width" not in dm:
        return None                                    # pre-capacity-knob checkpoint
    arr = np.asarray(dm["hidden_width"])
    if arr.ndim == 0:
        return int(arr)
    return [int(v) for v in arr.reshape(-1)]


class MultiStreamAutoencoder(nn.Module):
    """N-stream grid autoencoder: per-pixel encoder branches + shared latent +
    optional spatial residual + decoder.

    ``dims`` is the per-stream input width (e.g. [climate, landuse, soil]); their
    sum is the channel count ``C`` of the input grid, split internally in ``dims``
    order. ``spatial_kernel`` > 0 enables the residual masked conv (0 disables it,
    recovering the pure point-wise model).

    ``dropout`` is the rate inside every ``BMLPBlock`` (the only stochastic
    regularizer in the network). It is a constructor argument rather than the block
    default because the right value depends on what other regularization is active:
    with input channel-group masking (``augment.py``) supplying most of the noise, a
    much lower rate is appropriate than without it.

    ``hidden_width`` (``h``) is the per-branch width; ``None`` keeps the historical
    ``max(128, latent_dim*4)``. It is the main CAPACITY knob: branch parameters scale
    as ``h**2`` (each ``BMLPBlock`` is ``h -> h*mlp_expansion -> h``), so halving it
    quarters them. That matters because generalization here is measured along the
    SPATIAL axis -- the holdout is blocked by cell -- and there are only ~12k training
    cells against ~7.6M parameters at ``h=256``, i.e. ~620 parameters per cell.

    ``hidden_width`` may also be a PER-STREAM sequence, one width per entry of ``dims``.
    The uniform scalar spends the same capacity on every branch, and the branches are not
    the same size: only ``Linear(d, h)`` depends on the input width (128 parameters for
    elevation's ~1 channel against ~25,600 for climate's ~200), while the two
    ``BMLPBlock``s that hold ~262k parameters each at ``h=128`` are identical in every
    branch. So a scalar gives a 3-channel static stream the same 262k-parameter encoder as
    the 240-channel climate stream. A sequence lets capacity follow input width -- wide for
    climate, narrow for the static streams -- which is the cheapest form of the knob.

    ``self.hidden_widths`` is the canonical per-stream list; ``self.hidden_width`` stays an
    ``int`` when every branch shares a width (and is ``None`` when they do not), so a
    uniform net persists and reloads exactly as it did before this became a sequence.

    NOTE ``hidden_width`` and ``mlp_expansion`` change ``state_dict`` shapes, so both are
    persisted in ``desk_meta.npz`` and must be read back wherever the model is rebuilt
    for inference (``build_final_z_cube``, ``validate_spacetime``). A per-stream list has
    to survive that round-trip as a list, and every reader must still accept the scalar
    that older checkpoints hold.
    """

    def __init__(self, dims, latent_dim, spatial_kernel=3, dropout=0.5,
                 hidden_width=None, mlp_expansion=4):
        super().__init__()
        self.dims = list(dims)
        self.dropout = float(dropout)
        n = len(self.dims)
        self.hidden_widths = resolve_hidden_widths(hidden_width, n, latent_dim)
        # Kept an int for the uniform case so `desk_meta.npz` still holds a scalar and
        # nothing downstream has to change to reload an existing checkpoint; None when the
        # branches differ, because there is no single width to report and returning one of
        # them would be a number that silently misdescribes the net.
        uniform = len(set(self.hidden_widths)) == 1
        self.hidden_width = self.hidden_widths[0] if uniform else None
        self.mlp_expansion = int(mlp_expansion)
        self.encoders = nn.ModuleList([
            nn.Sequential(nn.Linear(d, h), nn.GELU(),
                          BMLPBlock(h, k=self.mlp_expansion, dropout=self.dropout),
                          BMLPBlock(h, k=self.mlp_expansion, dropout=self.dropout))
            for d, h in zip(self.dims, self.hidden_widths)
        ])
        # sum(hidden_widths), not n*h: identical to the old n*h whenever the widths are
        # uniform (so uniform state_dicts keep their exact shapes), and the only width that
        # matches the concatenated branch codes when they are not.
        hsum = int(sum(self.hidden_widths))
        self.mixer = nn.Sequential(
            nn.Linear(hsum, hsum), nn.GELU(),
            nn.Linear(hsum, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hsum), nn.GELU(),
            nn.Linear(hsum, sum(self.dims)),
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

    def __init__(self, prism_dim, bui_dim, latent_dim, spatial_kernel=3, dropout=0.5,
                 hidden_width=None, mlp_expansion=4):
        super().__init__([prism_dim, bui_dim], latent_dim, spatial_kernel, dropout,
                         hidden_width, mlp_expansion)
