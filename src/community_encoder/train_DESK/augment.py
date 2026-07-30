"""Structured channel-group masking for DESK covariate inputs (denoising augmentation).

The covariate grid has ~295 channels whose redundancy is *structured*: climate is 14 base
variables x 12 bio-year month positions x 1-3 elevation quantiles, so a channel's nearest
neighbours (adjacent month, same base; same month, another quantile) carry nearly the same
information. Independent per-channel dropout is therefore almost free to satisfy — the model
inpaints a missing January from December — and buys little. Masking whole GROUPS is what forces
the encoder to use the seasonal cycle and cross-variable structure rather than one convenient
channel.

Axes (all config-gated, sampled independently per call):

- ``p_base``    one base variable, all its months AND levels (e.g. every ``Tmax_*``)
- ``p_month``   one bio-year month position across every base and level (e.g. all ``b06m01``)
- ``span_prob`` a CONTIGUOUS run of month positions (``b03..b05``), SpecAugment-style: harder to
                inpaint from neighbours than i.i.d. months. No wrap-around — the bio-year is a
                window (Aug->Jul), not a cycle, so b12 and b01 are 11 months apart, not adjacent.
- ``p_level``   one elevation quantile (``q10``/``q90``), which exist only for temperature bases
                and are strongly correlated with ``q50``
- ``p_stream``  a whole stream (landuse / hyde / soil / elevation)

``tile_cells`` controls the SPATIAL granularity of all of the above. At 0 (the default) one draw
is shared by the whole grid, which gives temporal diversity but none spatially -- every cell sees
the same variables missing, so nothing teaches the encoder that different regions must cope with
losing different variables. That matters because the holdout is spatially blocked, so spatial
generalization is precisely what is being measured. Set it to a block size in cells for
independent draws per tile.

Three rules that are easy to get wrong:

1. **No ``1/(1-p)`` rescaling.** This is masked-input augmentation (denoising-autoencoder /
   MAE semantics), not unit dropout: surviving channels must keep their true values, because
   the decoder is asked to reconstruct the CLEAN input from the corrupted one. ``nn.Dropout``
   would rescale survivors and is deliberately not used.
2. **Masking to 0 == mean imputation, for free.** Covariates are standardized and invalid cells
   are zero-filled *after* standardization (``covariate_io.norm_grid``), so 0 already means "the
   channel mean" everywhere in this pipeline. No sentinel, no extra missingness indicator.
3. **The climate stream is never dropped as a stream.** It is ~81% of all channels (240/295);
   removing it leaves the model with almost no input. Climate is masked on the base/month/span/
   level axes instead, which is the point of having them.

Cell (spatial) and year (temporal) masking are deliberately NOT offered here: cell masking would
desync ``PartialConv2d``'s explicit validity mask (it would average zeros in as though valid),
and year masking would corrupt the causal ``OutputEMA`` scan over the year axis.
"""
import re

import numpy as np
import torch

# Climate variable token written by src/data/preprocess/climate_grid.py:
#   {base}_b{kk}m{MM}_{lvl}   e.g. Tmax_b01m08_q50
_CLIMATE_TOKEN = re.compile(r"^(?P<base>.+)_b(?P<pos>\d{2})m(?P<month>\d{2})_(?P<lvl>q\d{2})$")

CLIMATE_STREAM = "climate"


class ChannelGroupMasker:
    """Samples 0/1 keep-vectors over the ``C`` covariate channels by structured group.

    Built once from the ``state_schema.json`` dict; ``sample_keep`` is then a cheap draw.
    Group membership is precomputed as boolean ``(C,)`` arrays so a draw is a few OR-reductions.
    """

    def __init__(self, schema, cfg=None, total_dim=None):
        cfg = dict(cfg or {})
        self.enabled = bool(cfg.get("enabled", False))
        self.p_base = float(cfg.get("p_base", 0.15))
        self.p_month = float(cfg.get("p_month", 0.15))
        self.span_prob = float(cfg.get("span_prob", 0.30))
        self.span_max = int(cfg.get("span_max", 3))
        self.p_level = float(cfg.get("p_level", 0.10))
        self.p_stream = float(cfg.get("p_stream", 0.10))
        self.max_masked_frac = float(cfg.get("max_masked_frac", 0.5))
        # 0 = one mask for the whole grid (no spatial diversity); >0 = independent draws per
        # tile_cells x tile_cells block, so different regions practise on different subsets.
        self.tile_cells = int(cfg.get("tile_cells", 0))

        streams = schema["streams"]
        self.C = int(total_dim if total_dim is not None else schema.get(
            "total_dim", sum(int(s["dim"]) for s in streams)))

        self.by_base, self.by_month_pos, self.by_level = {}, {}, {}
        self.by_stream = {}
        for s in streams:
            name, lo, hi = s["name"], int(s["start"]), int(s["end"])
            m = np.zeros(self.C, dtype=bool)
            m[lo:hi] = True
            self.by_stream[name] = m
            # Static streams (soil, elevation) carry variables: [] in the schema, so they are
            # only ever maskable at stream level. Do not assume names exist.
            for j, var in enumerate(s.get("variables") or []):
                tok = _CLIMATE_TOKEN.match(str(var))
                if not tok:
                    continue
                ch = lo + j
                self.by_base.setdefault(tok.group("base"), np.zeros(self.C, bool))[ch] = True
                self.by_month_pos.setdefault(int(tok.group("pos")), np.zeros(self.C, bool))[ch] = True
                self.by_level.setdefault(tok.group("lvl"), np.zeros(self.C, bool))[ch] = True

        # q50 is the only level every base has; dropping it would remove most of climate, so it
        # is not a maskable level. Only the temperature-only q10/q90 are.
        self.level_keys = sorted(k for k in self.by_level if k != "q50")
        self.month_keys = sorted(self.by_month_pos)
        self.base_keys = sorted(self.by_base)
        self.stream_keys = [s["name"] for s in streams if s["name"] != CLIMATE_STREAM]

    def describe(self):
        """One-line summary for the training log (so a misparsed schema is visible)."""
        return (f"ChannelGroupMasker(C={self.C}, bases={len(self.base_keys)}, "
                f"month_pos={len(self.month_keys)}, levels={self.level_keys}, "
                f"streams={self.stream_keys}, tile_cells={self.tile_cells}, "
                f"enabled={self.enabled})")

    def _draw(self, rng):
        """One raw draw -> boolean (C,) 'drop' array, before the max-fraction guard."""
        drop = np.zeros(self.C, dtype=bool)
        for b in self.base_keys:
            if rng.random() < self.p_base:
                drop |= self.by_base[b]
        for p in self.month_keys:
            if rng.random() < self.p_month:
                drop |= self.by_month_pos[p]
        if self.month_keys and rng.random() < self.span_prob:
            span = int(rng.integers(1, self.span_max + 1))
            lo = int(rng.integers(0, max(1, len(self.month_keys) - span + 1)))
            for p in self.month_keys[lo:lo + span]:     # contiguous in window order, no wrap
                drop |= self.by_month_pos[p]
        for lv in self.level_keys:
            if rng.random() < self.p_level:
                drop |= self.by_level[lv]
        for st in self.stream_keys:                     # climate excluded by construction
            if rng.random() < self.p_stream:
                drop |= self.by_stream[st]
        return drop

    def sample_drop(self, rng, max_tries=8):
        """Boolean ``(C,)`` drop mask honouring ``max_masked_frac``.

        A degenerate draw (most of the input gone) teaches nothing and destabilizes the step, so
        redraw; if the configured probabilities are so high that ``max_tries`` draws all exceed
        the cap, fall back to no masking for this step rather than looping or hard-failing.
        """
        if not self.enabled or self.C == 0:
            return np.zeros(self.C, dtype=bool)
        cap = self.max_masked_frac * self.C
        for _ in range(max_tries):
            drop = self._draw(rng)
            if drop.sum() <= cap:
                return drop
        return np.zeros(self.C, dtype=bool)

    def sample_keep(self, rng, device=None, dtype=torch.float32, hw=None):
        """0/1 keep mask to broadcast-multiply into a ``(B,H,W,C)`` grid.

        ``hw=None`` (or ``tile_cells<=0``) returns ``(1,1,1,C)``: ONE draw shared by every cell.
        ``hw=(H,W)`` returns ``(1,H,W,C)`` assembled from ``tile_cells`` x ``tile_cells`` spatial
        tiles, each drawing its own channel groups.

        This distinction matters for SPATIAL generalization, which is exactly what the blocked
        holdout measures. A grid-wide mask gives temporal diversity (a fresh draw per
        year-forward) but *no spatial diversity*: every cell sees the same variables missing, so
        nothing teaches the encoder that different regions must cope with losing different
        variables. Tiled masking makes each region practise on its own subset.

        Tiles rather than independent per-cell draws because the covariates are spatially
        autocorrelated and the latent conv pools a 5x5 neighbourhood -- per-cell noise would
        largely average out within a receptive field instead of removing information.

        Strictly 0/1 -- survivors are NOT rescaled (see the module docstring).
        """
        if hw is None or self.tile_cells <= 0:
            keep = ~self.sample_drop(rng)
            t = torch.as_tensor(keep.astype("float32"), dtype=dtype, device=device)
            return t.view(1, 1, 1, self.C)

        H, W = int(hw[0]), int(hw[1])
        b = int(self.tile_cells)
        nby, nbx = (H + b - 1) // b, (W + b - 1) // b
        tiles = np.empty((nby, nbx, self.C), dtype=bool)
        for iy in range(nby):
            for ix in range(nbx):
                tiles[iy, ix] = ~self.sample_drop(rng)
        keep = np.repeat(np.repeat(tiles, b, axis=0), b, axis=1)[:H, :W]
        t = torch.as_tensor(keep.astype("float32"), dtype=dtype, device=device)
        return t.view(1, H, W, self.C)


def blocked_holdout(valid, block_cells, holdout_frac, buffer_cells, seed=0):
    """Spatially blocked train/val split with a buffer ring, as boolean grids.

    Returns ``(val, buffer)`` over the same shape as ``valid``. Whole ``block_cells`` x
    ``block_cells`` tiles are assigned to validation, then every cell within Chebyshev distance
    ``buffer_cells`` of a val cell (and not itself val) becomes buffer — excluded from BOTH train
    and val, and from the normalization fit.

    The buffer is what makes the split honest in the presence of the latent ``PartialConv2d``:
    with a k x k kernel a prediction at a val cell reads cells up to ``k//2`` away, so without a
    buffer of at least that width the val cells' receptive fields overlap training cells and the
    metric is optimistic. Callers derive ``buffer_cells`` from the kernel rather than configuring
    it separately, so the two cannot drift apart.
    """
    valid = np.asarray(valid, dtype=bool)
    H, W = valid.shape
    b = max(1, int(block_cells))
    rng = np.random.default_rng(int(seed))
    nby, nbx = (H + b - 1) // b, (W + b - 1) // b
    pick = rng.random((nby, nbx)) < float(holdout_frac)
    # Expand block decisions to cell resolution, then crop to the grid.
    val = np.repeat(np.repeat(pick, b, axis=0), b, axis=1)[:H, :W] & valid

    r = max(0, int(buffer_cells))
    if r == 0:
        return val, np.zeros_like(val)
    near = np.zeros_like(val)
    for dy in range(-r, r + 1):                       # Chebyshev dilation (square structuring)
        for dx in range(-r, r + 1):
            near |= _shift2d(val, dy, dx)
    return val, near & valid & (~val)


def _shift2d(a, dy, dx):
    """``a`` shifted by ``(dy, dx)`` with False fill (no wraparound)."""
    out = np.zeros_like(a)
    H, W = a.shape
    ys_dst = slice(max(0, dy), min(H, H + dy))
    ys_src = slice(max(0, -dy), min(H, H - dy))
    xs_dst = slice(max(0, dx), min(W, W + dx))
    xs_src = slice(max(0, -dx), min(W, W - dx))
    out[ys_dst, xs_dst] = a[ys_src, xs_src]
    return out
