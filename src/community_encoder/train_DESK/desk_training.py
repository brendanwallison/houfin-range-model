"""Train DESK: a semi-supervised autoencoder predicting ESK's Z from covariates.

DESK ("Deep ESK") learns to reconstruct the ESK kernel-PCA latent Z -- the
habitat-similarity "ground truth" -- from covariates that exist for every year, so Z
can be extrapolated across the whole timeline. Trained with three losses: a
stabilizing MSE against the ESK Z, a metric loss preserving Ruzicka-similarity
relationships, and an autoencoder reconstruction.

**The objective is the output-EMA one.** DESK predicts a per-year raw z from
(lightly-smoothed) covariates, applies a LEARNED causal EMA over the year axis
(``OutputEMA``) to get z_ema, and supervises z_ema against the per-year targets
projected from the trend point set -- every (cell, year) point equally weighted. The
EMA models demographic lag: the year-Y community is a leaky integral of past
suitability, so per-year predictions flux with the environment while the supervised
estimate stays a stable mixture of recent years.

**Grid-native.** The model (``MultiStreamAutoencoder``) maps a covariate grid
``(B,H,W,C)`` -> latent grid, so its spatial residual conv can see each cell's
neighbours. Training therefore operates on whole-year grids, not a shuffled bag of
pixels.

N-stream: reads ``state_{year}.npz`` (climate/land-use/HYDE/soil/elevation) via
``state_schema.json`` (``covariate_io``).
"""
import contextlib
import json
import os
import resource
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from .augment import ChannelGroupMasker, blocked_holdout
from .config_utils import load_config
from . import covariate_io as cio
from .model_arch import MultiStreamAutoencoder


def compute_valid_mask(target_stack, cov_stack):
    """Cells DESK can supervise: a community observed in some year, and finite covariates.

    Deliberately NOT intersected with the ESK Z mask. That mask marks cells with an
    ANCHOR-YEAR embedding, and DESK never reads the anchor-year Z values -- its per-year
    targets come from project_points_to_z over every point (see _prepare_trend_targets). Gating
    on it cost real supervision: with raw BBS it held the mask to the 2,222 cells surveyed in
    the anchor year, against the 3,902 that BBS covers.
    """
    m_obs = np.any(~np.isnan(target_stack), axis=-1)
    m_cov = np.all(~np.isnan(cov_stack), axis=-1)
    final = m_obs & m_cov
    print(f"[mask] observed {m_obs.sum()} & cov {m_cov.sum()} -> {final.sum()} supervised pixels")
    return final


def _pair_kernel_loss(zi, zj, xi, xj):
    """MSE between the dot product in Z and the Ruzicka similarity in raw X, for paired rows."""
    sum_plus = xi + xj
    diff_abs = torch.abs(xi - xj)
    numerator = 0.5 * torch.sum(sum_plus - diff_abs, dim=1)
    denominator = 0.5 * torch.sum(sum_plus + diff_abs, dim=1)
    valid = denominator > 1e-3
    if valid.sum() == 0:
        return torch.tensor(0.0, device=z_pred.device, requires_grad=True)
    sim_true = numerator[valid] / (denominator[valid] + 1e-8)
    sim_pred = (zi[valid] * zj[valid]).sum(dim=1)
    return F.mse_loss(sim_pred, sim_true)


def true_kernel_loss(z_pred, x_raw, num_pairs=4096):
    """Kernel loss over pairs drawn from one supplied (valid) pixel set."""
    B = z_pred.shape[0]
    if B < 2:
        return torch.tensor(0.0, device=z_pred.device, requires_grad=True)
    idx = torch.randint(0, B, (2, num_pairs), device=z_pred.device)
    i, j = idx[0], idx[1]
    return _pair_kernel_loss(z_pred[i], z_pred[j], x_raw[i], x_raw[j])


def spacetime_kernel_loss(z_by_t, pool_t, pool_flat, pool_x, num_pairs=4096, generator=None,
                          pool_w=None):
    """Kernel loss over pairs drawn from the whole SPACETIME pool of supervised cell-years.

    The ESK basis is one joint Ružička kernel-PCA over every ``(cell, year)`` point, so its
    contract is ``dot(z_i, z_j) ~= Ružička(x_i, x_j)`` for ANY two points -- two cells in one
    year, one cell in two years, or two different cells in two different years. This term is
    what holds DESK to that contract, so its pairs have to be drawn the same way.

    With ``pool_w=None`` pairs are uniform over the pool, which weights each year by how many
    cells it supervises and leaves the within-year share at roughly 1/n_years. That is the faithful
    reproduction of the SURVEY; deliberately over-sampling same-year pairs would be a second knob
    with no principled value.

    ``pool_w`` supplies a per-point sampling weight instead, and the distinction it turns on is
    that faithful-to-the-survey and works-everywhere are different objectives. BBS is coast- and
    present-heavy, so a uniform objective is fit where the survey is dense; the strata a range
    model most needs (the interior, the early era) contribute in proportion to how little they were
    surveyed. The contract itself is indifferent -- it says dot(z_i, z_j) must match Ruzicka for ANY
    pair, not that pairs must arrive at survey frequency. See ``stratum_weights`` for the shrinkage,
    floor and cap that keep the correction from over-fitting thin strata.

    ``z_by_t`` is ``(T, H, W, L)``; ``pool_t``/``pool_flat`` index its year and flattened cell
    axes, and ``pool_x`` holds the matching raw communities.
    """
    N = int(pool_t.shape[0])
    if N < 2:
        return torch.zeros((), device=z_by_t.device, requires_grad=True)
    T, H, W, L = z_by_t.shape
    if pool_w is None:
        idx = torch.randint(0, N, (2, num_pairs), device=z_by_t.device, generator=generator)
    else:
        # Weighted draw over the pool. This OVERRIDES the uniform-draw argument recorded above:
        # uniform is faithful to the survey's own sampling, and the survey is concentrated on the
        # coasts and in recent decades, so a uniform objective is fit where BBS happens to be dense.
        # The kernel contract governs WHICH PAIRS the model must reproduce -- any two points -- not
        # whether the loss should inherit the survey's geographic and temporal bias. The weights are
        # sqrt-shrunk, floored and capped upstream (see stratum_weights), so a thin stratum gains
        # influence without being able to dominate.
        idx = torch.multinomial(pool_w, 2 * num_pairs, replacement=True,
                                generator=generator).view(2, num_pairs)
    i, j = idx[0], idx[1]
    zf = z_by_t.reshape(T, H * W, L)
    zi = zf[pool_t[i], pool_flat[i]]
    zj = zf[pool_t[j], pool_flat[j]]
    return _pair_kernel_loss(zi, zj, pool_x[i], pool_x[j])



def stratum_weights(labels, n_min=200, cap=5.0, power=0.5):
    """Per-point sampling weight that partly corrects BBS's coast/present bias. ``(N,)`` float.

    BBS coverage is concentrated on the coasts and in recent decades, so a uniform draw over the
    supervised pool trains the model where the survey happens to be dense and leaves the interior
    and the early era fit by whatever generalises. But a naive correction is worse than none: full
    inverse-frequency would hand a stratum with a dozen noisy cell-years the same total pull as one
    with thousands, and the model would chase that noise.

    Three guards, in the order they bind:

    * ``power=0.5`` -- weight goes as ``n^-0.5``, not ``n^-1``. The standard partial correction:
      it removes most of the population tilt while leaving a thin stratum a fraction of the pull
      full inverse-frequency would give it.
    * ``n_min`` -- a stratum with fewer than this many points gets the weight computed AT the floor,
      i.e. no further uplift. This is the direct answer to "a few noisy observations in the Great
      Plains in the late 1960s": below the floor there is not enough data to support an estimate, so
      those strata are the ones that must NOT be amplified. Without it they receive the largest
      boost of all, which inverts the intent.
    * ``cap`` -- the final ratio to the median weight is clipped to ``[1/cap, cap]``, bounding the
      worst case whatever the occupancy table turns out to look like.

    Returns weights normalised to a median of 1.0, so they compose multiplicatively with the
    existing per-cell weights (notably ``first_year_weight``) without changing the effective
    learning rate on a term.
    """
    labels = np.asarray(labels)
    counts = np.bincount(labels)
    n_eff = np.maximum(counts[labels].astype("float64"), 1.0)
    w = np.power(np.maximum(n_eff, float(n_min)), -float(power))
    med = float(np.median(w))
    if med <= 0:
        return np.ones(len(labels), dtype="float64")
    w = w / med
    return np.clip(w, 1.0 / float(cap), float(cap))


def stratum_occupancy(labels, keys, pidx, top_thin=20):
    """Human-readable occupancy of the shared strata, for choosing ``n_min`` and ``cap``.

    Runs BEFORE any rebalancing so the parameters are set against the real table rather than
    guessed. The coast/present bias is asserted everywhere in this project and has never been
    quantified here; this is that number.
    """
    labels = np.asarray(labels)
    counts = np.bincount(labels)
    rows = []
    for lab in np.argsort(counts):
        m = labels == lab
        if not m.any():
            continue
        dec, rb, cb, ab = (int(v) for v in keys[m][0])
        rows.append({"stratum": f"{dec}s/tile{rb},{cb}/abund{ab}",
                     "n_cell_years": int(m.sum()),
                     "n_cells": int(len(np.unique(pidx[m][:, :2], axis=0))),
                     "n_years": int(len(np.unique(pidx[m][:, 2])))})
    return {"n_strata": int((counts > 0).sum()),
            "n_points": int(len(labels)),
            "median_per_stratum": float(np.median(counts[counts > 0])),
            "p10_per_stratum": float(np.quantile(counts[counts > 0], 0.10)),
            "max_per_stratum": int(counts.max()),
            "imbalance_ratio_max_over_median": float(counts.max()
                                                     / max(np.median(counts[counts > 0]), 1)),
            "thinnest": rows[:int(top_thin)]}


def spacetime_metric_pool(pip, Xp, sup_rows, m_tr, W, exclude_years=(), return_pidx=False):
    """``(years, flat cell indices, community rows)`` -- the pool the metric loss samples.

    One flat pool over every supervised training cell-year, NOT a per-year grouping: pairs must
    be able to span years, because that is what the joint ESK kernel encodes. Sparse rather
    than a dense ``(T,H,W,S)`` tensor, which would be ~59x the point set for no gain, since
    only surveyed cells carry a community. Held-out and buffered cells are excluded so they
    never enter the similarity target.

    Returns the 3-tuple by default. ``return_pidx=True`` appends the selected rows'
    ``(row, col, year)``, which the shared strata need and which the flat index has already lost --
    behind a flag rather than as a fourth element always, because several call sites unpack this as
    a 3-tuple and a silent arity change is a failure mode this codebase has been bitten by (see
    ``test_every_desk_z_ema_call_site_unpacks_two_values``).

    ``exclude_years`` must carry ``desk.trend.holdout_years``. Zeroing a year's train mask in
    ``targets`` keeps it out of the stabilizing loss but NOT out of this pool, which filters on
    the spatial mask -- so a temporally held-out year would still reach the objective through
    the metric term, and the temporal-extrapolation measurement would be reading years the
    model was trained on. That leak did not exist while the metric loss used the anchor year
    alone (the anchor is never withheld); it appeared when the loss went spacetime-wide.
    """
    tr_flat = m_tr.reshape(-1)
    flat_of_row = pip[:, 0].astype(np.int64) * W + pip[:, 1].astype(np.int64)
    sel = sup_rows & tr_flat[flat_of_row]
    if len(exclude_years):
        sel = sel & ~np.isin(pip[:, 2], np.asarray(list(exclude_years), dtype=pip.dtype))
    out = (pip[sel, 2].astype(np.int64), flat_of_row[sel], Xp[sel].astype("float32"))
    return out + (pip[sel],) if return_pidx else out


def assert_focal_excluded(points_meta, focal_code):
    """Refuse a point set whose community contains the FOCAL species. Pure.

    Z is a habitat proxy built from OTHER species and the downstream regresses the focal species
    on it, so a focal member makes that regression partly a readback of its own target -- and it
    would be invisible: every metric would still look plausible.

    Two independent filters already keep it out (``avonet`` drops it by ``Avibase.ID1 !=
    FOCAL_ID``; ``bbs_crosswalk.read_community_codes`` defaults ``exclude`` to
    ``focal_species_code``), and the shipped artifact is clean. But they key on DIFFERENT
    identifiers -- an Avibase ID and an eBird species code -- so a taxonomy change could retire
    one while the other kept working, and neither is covered by a test. This asserts on the
    artifact actually being trained, which is the only place the two can be checked together.

    No-ops when either the species list or the focal code is unavailable rather than guessing:
    a missing key is a different failure and should not masquerade as leakage.
    """
    sp = (points_meta or {}).get("species")
    f = str(focal_code or "").strip().lower()
    if not sp or not f:
        return None
    hit = [str(x) for x in sp if str(x).strip().lower() == f]
    if hit:
        raise ValueError(
            f"focal species {f!r} is present in the community point set ({len(sp)} species). "
            f"Z would encode the downstream's own target, making the regression circular. "
            f"Rebuild the community (src.data.identify.avonet, then select_trend_community) "
            f"and check both exclusion layers.")
    return len(sp)


def device_targets(mapping, device):
    """``{year: (zg, mask)}`` as tensors on ``device``, from numpy or tensors.

    Exists because the alternative fails ONLY on CUDA. Mixing a numpy ``zg`` into a subtraction
    against a CPU tensor works silently; against a CUDA tensor it raises. So a local test on CPU
    passes and the GPU run dies two minutes in -- which is exactly what happened to the first
    temporal-holdout run. Converting through one function makes the requirement explicit and
    checkable anywhere.
    """
    return {int(y): (torch.as_tensor(zg, device=device).float(),
                     torch.as_tensor(m, device=device).bool())
            for y, (zg, m) in (mapping or {}).items()}


def prepare_supervised(cov_stack, target_stack, mu, sd, out_dir):
    """Normalized covariate grid + cov mask + the supervised-cell mask."""
    mask_sup = compute_valid_mask(target_stack, cov_stack)
    np.save(os.path.join(out_dir, "training_mask.npy"), mask_sup)
    covn, mask_cov = cio.norm_grid(cov_stack, mu, sd)
    return covn, mask_cov, mask_sup


# --- Output-EMA: demographic lag on the predicted Z ------------------------------

class OutputEMA(torch.nn.Module):
    """Learned causal EMA over the leading (year) axis of a ``(T, ...)`` tensor.

    Models demographic lag as a leaky integral of past predictions: ``z_ema[0]=z_raw[0]``,
    ``z_ema[t]=a*z_raw[t]+(1-a)*z_ema[t-1]`` with ``a = 1 - 2^{-1/h}`` for a learned
    half-life ``h`` (years). ``h`` is bounded to ``[hl_min, hl_max]`` via a sigmoid reparam,
    so it stays a plausible community response timescale and can't run away.
    """

    def __init__(self, hl_min=1.0, hl_max=40.0, init_hl=8.0):
        super().__init__()
        self.hl_min, self.hl_max = float(hl_min), float(hl_max)
        f = min(max((init_hl - hl_min) / (hl_max - hl_min), 1e-3), 1 - 1e-3)
        self.theta = torch.nn.Parameter(torch.tensor(float(np.log(f / (1 - f)))))

    def half_life(self):
        return self.hl_min + (self.hl_max - self.hl_min) * torch.sigmoid(self.theta)

    def alpha(self):
        return 1.0 - torch.pow(torch.tensor(2.0, device=self.theta.device), -1.0 / self.half_life())

    def forward(self, z):                                   # z: (T, ...) in temporal order
        a = self.alpha()
        out = [z[0]]
        for t in range(1, z.shape[0]):
            out.append(a * z[t] + (1.0 - a) * out[-1])
        return torch.stack(out, 0)


def apply_output_ema(z_raw, half_life, valid=None):
    """Numpy twin of :class:`OutputEMA.forward` for INFERENCE over a year axis.

    ``z_raw`` is ``(T, ..., L)`` in ascending-year order; returns the same shape holding
    ``z_ema[t]``. This must stay numerically identical to the torch scan on all-valid input —
    ``tests/test_output_ema.py`` asserts exactly that, since the trainer supervises ``z_ema``
    while every downstream export is raw, so anything grading DESK against an observed
    community has to reconstruct ``z_ema`` here rather than compare the wrong quantity.

    ``valid`` (``(T, ...)`` bool, optional) marks which entries are observed each step. An
    invalid entry **persists** the prior EMA state instead of overwriting it with NaN: without
    that, one gap-year poisons the running state for every later year at that location. With
    ``valid=None`` every entry is treated as observed.

    The scan is causal and cannot be applied to an isolated ``(cell, year)`` — the caller must
    supply a contiguous ascending series starting at the burn-in year (``ema_warmup_start``).
    """
    z_raw = np.asarray(z_raw, dtype="float32")
    if z_raw.ndim < 2:
        raise ValueError(f"z_raw must be (T, ..., L); got shape {z_raw.shape}")
    hl = float(half_life)
    if not np.isfinite(hl) or hl <= 0:
        raise ValueError(f"half_life must be finite and positive; got {half_life}")
    a = 1.0 - 2.0 ** (-1.0 / hl)
    state = np.full(z_raw.shape[1:], np.nan, dtype="float32")
    out = np.empty_like(z_raw)
    for t in range(z_raw.shape[0]):
        raw = z_raw[t]
        # A location is "seeded" once it has any running EMA; before that its first observed
        # value initializes the state (z_ema[0] = z_raw[0]) rather than blending against NaN.
        seeded = ~np.isnan(state).any(axis=-1)
        blend = np.where(seeded[..., None], a * raw + (1.0 - a) * state, raw)
        if valid is None:
            state = blend
        else:
            state = np.where(np.asarray(valid[t], bool)[..., None], blend, state)
        out[t] = state
    return out


def load_point_set(points_dir):
    """Read a target point set: ``(X, pidx, weights, supervise)``.

    ``point_weights.npy`` and ``point_supervise.npy`` are written by the raw-BBS builder and
    absent from the older trend-product point sets, so both default to "everything counts,
    everything supervises". That keeps a trend-product point set loadable unchanged, which is
    what makes the A/B between the two targets possible.

    ``supervise`` exists because one cell-year can be measured by two sources. The kernel wants
    both rows -- they are two measurements of one community, and their similarity is what ties
    the two scales together -- but the supervision grid is a scatter, so a duplicate cell-year
    would overwrite silently and there would be no way to tell which source survived.
    """
    X = np.load(os.path.join(points_dir, "X_points.npy"))
    pidx = np.load(os.path.join(points_dir, "point_index.npy"))
    n = pidx.shape[0]
    wp = os.path.join(points_dir, "point_weights.npy")
    sp = os.path.join(points_dir, "point_supervise.npy")
    weights = np.load(wp).astype("float32") if os.path.exists(wp) \
        else np.ones(n, dtype="float32")
    supervise = np.load(sp).astype(bool) if os.path.exists(sp) \
        else np.ones(n, dtype=bool)
    for name, arr in (("point_weights", weights), ("point_supervise", supervise)):
        if arr.shape[0] != n:
            raise SystemExit(f"{name}.npy has {arr.shape[0]} rows but point_index.npy has "
                             f"{n}. These are parallel arrays in one row order; a mismatch "
                             f"would attach the wrong value to the wrong cell-year.")
    return X, pidx, weights, supervise


def supervised_cells(pidx, supervise, shape):
    """Boolean grid of cells that carry at least one supervised point in any year.

    The holdout has to be drawn from THESE cells, not from the eBird covariate footprint. The
    two used to be nearly the same (~17,200 cells either way) because the old target was an
    interpolated surface covering everything. A measured target does not: raw BBS reaches
    ~3,900 cells and BBS+eBird ~12,400, so blocks drawn over the full footprint would leave
    many validation blocks holding no supervised cell at all, and the reported validation
    number would rest on far fewer cells than it appears to.
    """
    out = np.zeros(shape, dtype=bool)
    out[pidx[supervise, 0], pidx[supervise, 1]] = True
    return out


def _prepare_trend_targets(config, z_dir, latent_dim, holdout, points_dir=None):
    """Per-year ESK-basis targets for EVERY supervised year, from the target point set.

    Projects ``X_points`` into the joint ESK basis (z_obs) and scatters each supervised point
    into its year's grid. Returns ``{year: (zg (H,W,L), tm_tr (H,W), tm_val (H,W),
    wg (H,W))}`` where the train/val split is the spatial ``holdout`` (val = held-out cells)
    and ``wg`` is the per-cell loss weight for that year.

    Only rows flagged ``supervise`` are scattered, so each cell-year is written exactly once.
    """
    from .esk_kernel import project_points_to_z
    from src.config_utils import target_points_dir
    zt = points_dir or target_points_dir(config)
    X, pidx, weights, supervise = load_point_set(zt)
    z_obs = project_points_to_z(X, z_dir, latent_dim)
    if z_obs is None:
        raise FileNotFoundError(f"trend targets need the ESK projection in {z_dir}; re-run spacetime-esk")
    rows, cols, yrs = pidx[:, 0], pidx[:, 1], pidx[:, 2]
    H, W = holdout.shape
    out = {}
    for y in sorted({int(v) for v in yrs}):
        sel = np.where((yrs == y) & supervise)[0]
        zg = np.zeros((H, W, latent_dim), dtype="float32")
        present = np.zeros((H, W), bool)
        wg = np.zeros((H, W), dtype="float32")
        zg[rows[sel], cols[sel]] = z_obs[sel]
        present[rows[sel], cols[sel]] = True
        wg[rows[sel], cols[sel]] = weights[sel]
        out[y] = (zg, present & (~holdout), present & holdout, wg)
    return out


def _load_year_window(states_dir, schema, mu, sd, years):
    """Ordered covariate window: normalized grids for ``years`` -> (T,H,W,C), (T,H,W), kept_years."""
    covn, masks, kept = [], [], []
    for y in years:
        try:
            cov = cio.load_state_stack(y, states_dir, schema)
        except FileNotFoundError:
            continue
        cn, m = cio.norm_grid(cov, mu, sd)
        covn.append(cn); masks.append(m); kept.append(int(y))
    return np.stack(covn), np.stack(masks), kept


# ru_maxrss is KILOBYTES on Linux but BYTES on macOS, so a single divisor prints a plausible-
# looking number that is wrong by 1024x on one of them. Resolve the unit once, here.
_RSS_TO_GIB = 1024 ** 3 if sys.platform == "darwin" else 1024 ** 2


def _max_rss_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / _RSS_TO_GIB


def median_dir_cos(dp, dt):
    """Median per-row cosine between two sets of change vectors ``(n, L)``.

    Module-level so it can be tested against known angles: ``_dir_cos`` inside the trainer is a
    closure over the year indices and cannot be called directly. Rows whose either vector is
    degenerate are skipped rather than contributing a spurious value.
    """
    den = dp.norm(dim=1) * dt.norm(dim=1)
    ok = den > 1e-12
    if not bool(ok.any()):
        return float("nan")
    return float(torch.median(torch.sum(dp * dt, dim=1)[ok] / den[ok]))




def _param_groups(modules, weight_decay):
    """Split parameters into decay / no-decay groups.

    Weight decay on LayerNorm gains, biases, and standalone scalars (``gamma``, the EMA's
    ``theta``) shrinks calibration parameters toward zero for no benefit -- ``gamma`` gates the
    spatial residual and ``theta`` sets the learned half-life, so decaying them is an implicit
    prior nobody asked for. Standard transformer practice, applied here for the same reason.
    """
    decay, no_decay = [], []
    for mod in modules:
        for name, p in mod.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim <= 1 or name.endswith("bias") or "ln" in name or name in ("gamma", "theta"):
                no_decay.append(p)
            else:
                decay.append(p)
    return [{"params": decay, "weight_decay": float(weight_decay)},
            {"params": no_decay, "weight_decay": 0.0}]


def _warmup_cosine(epochs, warmup, min_frac):
    """LR multiplier: linear warmup then cosine decay to ``min_frac``.

    The EMA trainer takes ONE optimizer step per epoch (the sequential year scan forces a
    full-window step), so with ``epochs=500`` the whole run is <=500 updates and the schedule is
    naturally expressed in epochs. Replaces ReduceLROnPlateau, whose patience of 5 against an
    early-stop patience of 50 could halve the LR ~9 times chasing noise in the val metric.
    """
    warmup = max(0, int(warmup))
    total = max(1, int(epochs))

    def fn(ep):                      # ep is 0-based step count
        if warmup and ep < warmup:
            return (ep + 1) / warmup
        prog = (ep - warmup) / max(1, total - warmup)
        prog = min(1.0, max(0.0, prog))
        return min_frac + (1.0 - min_frac) * 0.5 * (1.0 + np.cos(np.pi * prog))

    return fn


def train_model_ema(cov_window, mask_window, window_years, targets, metric_pool, m2023_tr, m2023_val,
                    stream_dims, latent_dim, ema_cfg, spatial_kernel=3, epochs=500, lr=1e-3,
                    weights=None, seed=0, patience=50, min_delta=1e-4,
                    schema=None, augment_cfg=None, dropout=0.5, weight_decay=0.0,
                    warmup_epochs=0, min_lr_frac=1.0, amp=False, eval_every=1,
                    holdout_year_targets=None, pool_w=None, direction_anchor_year=None,
                    direction_withheld_anchor_year=None,
                    hidden_width=None, mlp_expansion=4,
                    _skip_target_conversion=False):
    """Train DESK with a learned output-EMA.

    Forwards the ordered year window (per-year gradient checkpointing), applies the
    learned causal EMA over the year axis, and supervises ``z_ema`` against the per-year
    trend targets (uniform over all supervised (cell,year), train = non-held-out cells).
    Plus the 2023 Ruzicka metric loss and an autoencoder reconstruction over the window.
    Returns ``(model, ema)``.

    With ``augment_cfg`` enabled this becomes a **denoising** autoencoder: each year's forward
    sees a channel-group-masked input while the reconstruction target stays the clean grid, so
    the encoder must infer masked variables/months from the rest rather than copying them.
    """
    from torch.utils.checkpoint import checkpoint
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = weights or {"stabilizing": 64.0, "metric": 5.0, "reconstruction": 0.1}
    # Seed every RNG the run touches, not just torch's CPU generator: model init, dropout masks,
    # the metric loss's pair sampling, and the augmentation draws must all be reproducible for a
    # config change to be attributable to the config rather than to the seed.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    aug_rng = np.random.default_rng(seed)

    model = MultiStreamAutoencoder(stream_dims, latent_dim, spatial_kernel, dropout,
                                   hidden_width, mlp_expansion).to(device)
    ema = OutputEMA(ema_cfg.get("half_life_bounds", [1.0, 40.0])[0],
                    ema_cfg.get("half_life_bounds", [1.0, 40.0])[1],
                    ema_cfg.get("init_half_life", 8.0)).to(device)
    params = list(model.parameters()) + list(ema.parameters())
    opt = torch.optim.AdamW(_param_groups([model, ema], weight_decay), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, _warmup_cosine(epochs, warmup_epochs, min_lr_frac))
    # Early training is unstable (the metric loss can transiently spike), so clip the global
    # grad norm to bound the step (mirrors the MAP runner's clip_by_global_norm) and don't let
    # the volatile first few epochs set the "best" checkpoint (else a near-init epoch can win).
    grad_clip = float(ema_cfg.get("grad_clip", 1.0))
    es_warmup = int(ema_cfg.get("earlystop_warmup", 5))

    masker = None
    if schema is not None and (augment_cfg or {}).get("enabled"):
        masker = ChannelGroupMasker(schema, augment_cfg, total_dim=int(sum(stream_dims)))
        print(f"[desk] input masking: {masker.describe()}", flush=True)
    # bf16 needs no GradScaler (same exponent range as fp32); it is applied ONLY to the per-year
    # forward. The EMA scan and every loss stay fp32 -- 86 sequential bf16 accumulations in the
    # causal recurrence would visibly drift, and the losses are small differences of large sums.
    use_amp = bool(amp) and device == "cuda"
    autocast = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if use_amp \
        else contextlib.nullcontext

    cov = torch.tensor(cov_window, device=device)                 # (T,H,W,C)
    msk = torch.as_tensor(mask_window, device=device).bool()      # (T,H,W)
    yi = {y: i for i, y in enumerate(window_years)}
    # The spacetime pool the metric loss samples pairs from. Points whose year falls outside
    # the forwarded window have no z to compare against, so they are dropped here rather than
    # silently indexing the wrong year.
    _py, _pf, _px = metric_pool
    _keep = np.array([int(y) in yi for y in _py], dtype=bool)
    if not _keep.all():
        print(f"[desk] metric pool: dropped {int((~_keep).sum()):,} cell-years outside the "
              f"forwarded window {min(yi)}-{max(yi)}")
    pool_t = torch.as_tensor(np.array([yi[int(y)] for y in _py[_keep]]),
                             device=device, dtype=torch.long)
    pool_flat = torch.as_tensor(_pf[_keep], device=device, dtype=torch.long)
    pool_x = torch.as_tensor(_px[_keep], device=device)
    # Same row filter as the pool itself, or the weights would address different rows than the
    # draw -- the class of index-space bug this file has hit before.
    pool_w_t = None if pool_w is None else torch.as_tensor(
        np.asarray(pool_w)[_keep], device=device, dtype=torch.float32)
    m_tr = torch.as_tensor(m2023_tr, device=device).bool(); m_val = torch.as_tensor(m2023_val, device=device).bool()
    # supervised year targets that fall inside the forwarded window
    tgt = {y: (torch.tensor(zg, device=device),
               torch.as_tensor(tr, device=device).bool(), torch.as_tensor(va, device=device).bool(),
               torch.as_tensor(wg, device=device).float())
           for y, (zg, tr, va, wg) in targets.items() if y in yi}
    # The temporal-holdout targets arrive as NUMPY (built in run_desk_experiment alongside the
    # raw targets) but _z_mse subtracts them from a CUDA tensor, so they have to be moved here
    # like `tgt` is. This path never ran while holdout_years was empty, which is exactly why it
    # was broken: the first temporal-holdout run died on it after two minutes.
    _hy_in = {y: v for y, v in (holdout_year_targets or {}).items() if y in yi}
    # _skip_target_conversion exists only so a test can prove the _z_mse guard fires; the
    # conversion is not optional in any real run.
    hy_tgt = (_hy_in or None) if _skip_target_conversion else (device_targets(_hy_in, device)
                                                              or None)
    y2023 = int(max(tgt))                                         # anchor year index in the window

    # No-skill baselines on the held-out cells (pooled over all supervised years): the Z-MSE
    # of predicting the global mean vector, and of predicting zero. Val(all-yr) must fall well
    # below these to have any skill; Val/baseline is the fraction of held-out Z variance left
    # unexplained (targets have ||Z||^2 ~ 1 since Z.Z^T ~= Ružicka with unit self-similarity).
    with torch.no_grad():
        held = [zg[va] for _, (zg, _t, va, _w) in tgt.items() if bool(va.any())]
        if held:
            allv = torch.cat(held, 0)
            base_mean = float(torch.mean(torch.sum((allv - allv.mean(0, keepdim=True)) ** 2, dim=1)))
            base_zero = float(torch.mean(torch.sum(allv ** 2, dim=1)))
            print(f"[desk] Val(all-yr) no-skill baselines: predict-mean={base_mean:.4f}, "
                  f"predict-zero={base_zero:.4f} -- a trained model must fall WELL below these", flush=True)
    # The baseline that actually matters for a spatially blocked holdout: interpolate the
    # targets from neighbouring TRAINING cells, no covariates and no learning. Val must sit
    # clearly below this or the covariates are adding nothing over spatial smoothness.
    try:
        from .validate_baselines import (interp_year_coverage, spatial_interp_baseline,
                                         spatial_interp_dir_cos)
        b_near, b_idw = spatial_interp_baseline(tgt)
        n_used, n_tot = interp_year_coverage(tgt)
        # Which years the bar actually covers, because it silently skips any year with no
        # training cells -- i.e. every temporally withheld year. Comparing the POOLED val MSE
        # against a bar measured on a subset of years is not a comparison, and that is exactly
        # the mistake the first temporal-holdout sweep invited.
        scope = (f"{n_used}/{n_tot} supervised years"
                 + ("" if n_used == n_tot else
                    " -- withheld years have no training cells to interpolate from, so compare "
                    "va(sp), NOT the pooled va"))
        print(f"[desk] spatial-interpolation baselines (no model, no covariates): "
              f"nearest-train-cell={b_near:.4f}, inverse-distance-8={b_idw:.4f} "
              f"over {scope}", flush=True)
    except Exception as exc:                       # diagnostic only; never block training
        print(f"[desk] spatial-interpolation baseline unavailable ({exc})", flush=True)

    def _forward_window(train_mode, mask_inputs, want_raw=False):
        """Forward the whole year window -> ``(z_ema, recon_loss[, z_raw])``.

        ``train_mode`` toggles dropout; ``mask_inputs`` toggles the augmentation. The eval call
        passes (False, False) so the selection metric is measured on the deterministic, unmasked
        model -- see the note at the eval site. ``want_raw`` also returns the pre-EMA stack, which
        is what the rotation diagnostic needs (the cube and ``validate_spacetime.encode_points``
        both export raw z, so raw is what every reported turnover number measures).
        """
        model.train(train_mode)
        z_raw, rl = [], torch.zeros((), device=device)
        # Drawn ONCE for the whole window, so the groups it drops are missing in EVERY year --
        # the shape real missingness has (a CONUS-only covariate, a product that ends before the
        # timeline does). Hoisted out of the year loop for exactly that reason; a per-year draw
        # can only ever express transient noise. None when persistence is unconfigured.
        keep_persist = None
        if mask_inputs and masker is not None:
            keep_persist = masker.sample_persistent_keep(
                aug_rng, device=device, hw=(cov.shape[1], cov.shape[2]))
        for t in range(cov.shape[0]):
            xt = cov[t:t + 1]
            if mask_inputs and masker is not None:
                keep = masker.sample_keep(aug_rng, device=device,
                                          hw=(cov.shape[1], cov.shape[2]))
                if keep_persist is not None:
                    keep = keep * keep_persist       # kept only if BOTH tiers keep it
                # New tensor: cov[t:t+1] is a VIEW of the single resident (T,H,W,C) window, so an
                # in-place mask would permanently destroy that year for every later epoch.
                xt = xt * keep
            with autocast():
                if train_mode:
                    zt, rt = checkpoint(model, xt, msk[t:t + 1], use_reentrant=False)
                else:
                    zt, rt = model(xt, msk[t:t + 1])
            z_raw.append(zt[0].float())                          # fp32 before the EMA scan
            # Target is the CLEAN grid even when the input was masked -> denoising objective.
            rl = rl + F.mse_loss(rt[0][msk[t]].float(), cov[t][msk[t]])
        stack = torch.stack(z_raw, 0)
        out = (ema(stack), rl / cov.shape[0])
        return out + (stack,) if want_raw else out

    def _z_mse(z_all, sel):
        """Mean per-cell summed-over-latent Z error on the cells selected by ``sel[y]``."""
        sq, cnt = 0.0, 0
        for y, (zg, m) in sel.items():
            if not (torch.is_tensor(zg) and torch.is_tensor(m)):
                raise TypeError(
                    f"_z_mse got non-tensor targets for year {y} ({type(zg).__name__}). "
                    f"Route them through device_targets(): numpy silently works on CPU and "
                    f"raises on CUDA, so this must fail the same way everywhere.")
            if bool(m.any()):
                d = (z_all[yi[y]][m] - zg[m]).detach()
                sq += float(torch.sum(d * d)); cnt += int(m.sum())
        return sq / max(cnt, 1), cnt

    def _rotation(z_all, m, y_a=None):
        """Median ``1 - cos`` between the deep and anchor year, predicted and target.

        ``y_a`` overrides the deep year so the same diagnostic can be run on a second, WITHHELD
        pair -- the trained-era pair alone never looks at the years the temporal holdout removed,
        which is the whole point of the experiment.

        Call this on ``z_ema``, the SUPERVISED quantity, so a ratio of 1.0 means "the model
        reproduces its target's temporal change". Scoring the pre-EMA ``z_raw`` against the
        same target is wrong and inverts the metric: the EMA attenuates, so ``z_raw`` must
        *over*-rotate for ``z_ema`` to match -- at the learned ~9 yr half-life by about 1.27x
        over a 59-year span. A model improving at its actual objective would push a raw-based
        ratio past 1.0, which would read as over-prediction. ``z_raw`` is still reported
        separately because the cube exports it and the population model consumes it, but it is
        NOT a ratio-to-one.

        This is the quantity the science rests on and the trainer has never logged. A per-cell
        MSE cannot expose it: Z is dominated by the static spatial pattern (the no-change null
        reproduces ~94% of the structure), so the loss can look healthy while the *temporal*
        component is systematically shrunk -- MSE's optimum for a poorly-predicted component IS a
        shrunk one. ``rotP/rotT`` makes that shrinkage a number watched every epoch instead of a
        surprise in a figure.

        Cosine, not dot, matching ``validate_spacetime.temporal_turnover_agreement``: the raw dot
        folds in ⟨z,z⟩ calibration drift (measured ``||Z||^2`` ~ 0.73) and would report that drift
        as turnover.
        """
        y_a = y_deep if y_a is None else int(y_a)
        if y_a is None or not bool(m.any()):
            return float("nan"), float("nan")

        def med(a, b):
            num = torch.sum(a * b, dim=1)
            den = a.norm(dim=1) * b.norm(dim=1)
            ok = den > 1e-12
            if not bool(ok.any()):
                return float("nan")
            return float(torch.median(1.0 - num[ok] / den[ok]))

        zp0, zp1 = z_all[yi[y_a]][m].detach(), z_all[yi[y2023]][m].detach()
        zt0, zt1 = tgt[y_a][0][m], tgt[y2023][0][m]
        return med(zp0, zp1), med(zt0, zt1)

    def _mag_ratio(z_all, m, y_a=None):
        """Median ``||dp|| / ||dt||`` for the same pair ``_dir_cos`` scores. The MAGNITUDE half.

        ``rot``, ``dcos`` and ``cal`` are all angular. Squared error splits exactly into a
        magnitude term and an angular term, so an angle reported alone accounts for only half of
        it -- and cannot separate "moved the wrong way" from "barely moved". Those are different
        models: under-moving is the MSE-optimal answer to a poor angle (the minimiser at fixed
        cosine rho is exactly rho times the truth), so a respectable dcos with a ratio near 0 is
        a model hedging, not a model succeeding. 1.0 means reproducing the observed amount of
        change; rho would be MSE-calibrated.
        """
        y_a = y_deep if y_a is None else int(y_a)
        if y_a is None or not bool(m.any()):
            return float("nan")
        dp = (z_all[yi[y2023]][m] - z_all[yi[y_a]][m]).detach()
        dt = tgt[y2023][0][m] - tgt[y_a][0][m]
        nt = dt.norm(dim=1)
        ok = nt > 1e-12
        if not bool(ok.any()):
            return float("nan")
        return float(torch.median(dp.norm(dim=1)[ok] / nt[ok]))

    def _dir_cos(z_all, m, perm, y_a=None):
        """Median cosine between the PREDICTED and TARGET change vectors, and a permuted null.

        ``_rotation`` measures only how far z swings; two models can match its magnitude while
        moving in opposite directions. This measures the direction: the angle between
        ``z_pred(anchor) - z_pred(deep)`` and the same difference on the targets, per cell.

        The pair is what makes either interpretable. For an MSE objective the optimal magnitude
        of a prediction whose direction cosine is ``rho`` is exactly ``rho`` times the truth's
        (minimizing ``||a*u - t||^2`` gives ``a = ||t||*rho``), and since ``1 - cos ~ theta^2/2``
        the calibrated relationship is ``rot ~ dcos^2``. So ``dcos`` is the ceiling on a
        well-calibrated ``rot``: below it the model under-rotates, above it the model is moving
        further than its direction accuracy justifies, which raises error rather than lowering it.

        ``perm`` is a FIXED cell permutation, so the null is a stable reference across epochs
        rather than fresh noise: it pairs each predicted change with a different cell's target
        change, giving the cosine attainable with no per-cell information.
        """
        y_a = y_deep if y_a is None else int(y_a)
        if y_a is None or not bool(m.any()):
            return float("nan"), float("nan")
        dp = (z_all[yi[y2023]][m] - z_all[yi[y_a]][m]).detach()
        dt = tgt[y2023][0][m] - tgt[y_a][0][m]

        return median_dir_cos(dp, dt), median_dir_cos(dp, dt[perm])

    val_sel = {y: (zg, va) for y, (zg, _tr, va, _w) in tgt.items()}
    # The pooled `va` above mixes two very different questions: held-out cells in years the
    # model trained on (unseen PLACE) and held-out cells in withheld years (unseen place AND
    # year). Reporting only the pool meant the two had to be recovered by hand from three runs
    # of the sweep -- and invited comparing the pool against a bar that covers only the trained
    # years. Split it here; the pooled value is unchanged and remains the selection signal.
    # "Withheld" is derived from the TRAIN MASK, not from holdout_year_targets. The mask is what
    # the objective actually saw, so the split cannot disagree with reality -- and a divergence
    # between "what we withheld" and "what we report as withheld" is precisely the class of bug
    # this whole change exists to close. run_desk_experiment sets both consistently; deriving
    # from the mask also covers a year that has no training cells for any other reason, which is
    # equally a year the objective never saw.
    _ho_yrs = {int(y) for y, (_zg, _tr, _va, _w) in tgt.items() if not bool(_tr.any())}
    val_sel_sp = {y: v for y, v in val_sel.items() if int(y) not in _ho_yrs}
    val_sel_spt = {y: v for y, v in val_sel.items() if int(y) in _ho_yrs}
    # Train-cell selection on the SAME footing as val, evaluated on the clean forward. Without
    # it the only train-side number is `Stab`, which is (a) measured under dropout+masking and
    # so overstates the error, and (b) divided by latent_dim while Val is not -- two different
    # scales for the same quantity in one log line.
    tr_sel = {y: (zg, tr) for y, (zg, tr, _va, _w) in tgt.items()}
    # Deepest year that still has TRAINING coverage -- not simply min(tgt). A temporally
    # held-out year stays in `tgt` with an all-False train mask (that is how it is withheld
    # from the objective), so anchoring on min(tgt) would make `rot_tr` empty and the rotation
    # and direction columns would silently go dark exactly when a temporal holdout is added.
    _cand = [y for y in sorted(tgt) if y != y2023 and bool(tgt[y][1].any())]
    y_deep = _cand[0] if _cand else None
    # PINNED anchors. Auto-selecting the deepest TRAINED year silently shortened the diagnostic
    # interval across the temporal sweep (1976->2025, 1986->2025, 1996->2025 = 49/39/29 yr), so
    # most of the apparent decline in dcos was a shrinking chord rather than a worse model.
    # Pinning makes the interval a constant and the holdout width the only variable.
    def _pinned(year, label, need_train):
        if year is None:
            return None
        y = int(year)
        if y not in tgt:
            raise ValueError(f"{label}={y} is not a supervised year "
                             f"(have {min(tgt)}..{max(tgt)}); fix the overlay")
        if need_train and not bool(tgt[y][1].any()):
            raise ValueError(f"{label}={y} has no training cells (it is temporally withheld); "
                             f"the trained-era control needs a year the model trained on")
        return y

    _pin = _pinned(direction_anchor_year, "desk.trend.direction_anchor_year", True)
    if _pin is not None:
        y_deep = _pin
    # The withheld-era pair: the measurement the temporal sweep exists to produce, and absent
    # from the log until now. Pinned rather than defaulted to the deepest withheld year, which
    # is 1966 in every sweep overlay -- BBS's launch year, ~400 routes, and after intersecting
    # with the anchor year and the val split it is the ~36-val-cell case DEFAULT_EPOCHS was
    # written to avoid.
    y_ho = _pinned(direction_withheld_anchor_year,
                   "desk.trend.direction_withheld_anchor_year", False)
    # Cells supervised in BOTH the deep and anchor year, split train/val -- the diagnostic
    # needs the same cell present at both ends to measure its change.
    if y_deep is not None:
        rot_tr = tgt[y_deep][1] & tgt[y2023][1]
        rot_va = tgt[y_deep][2] & tgt[y2023][2]
        # Fixed permutations so the direction null is a stable reference, not per-epoch noise.
        g_null = torch.Generator(device="cpu").manual_seed(seed)
        perm_tr = torch.randperm(int(rot_tr.sum()), generator=g_null).to(device)
        perm_va = torch.randperm(int(rot_va.sum()), generator=g_null).to(device)
        if _pin is not None:
            print(f"[desk] rotation anchor PINNED to {y_deep} "
                  f"(desk.trend.direction_anchor_year) -- the interval is a constant across the "
                  f"sweep, so a change in this pair is not a change in chord length",
                  flush=True)
        elif int(min(tgt)) != y_deep:
            print(f"[desk] rotation anchor auto-moved to {y_deep} (min supervised year "
                  f"{int(min(tgt))} has no training cells -- temporal holdout). This SHORTENS "
                  f"the interval; pin desk.trend.direction_anchor_year to compare across runs.",
                  flush=True)
        print(f"[desk] rotation/direction diagnostic {y_deep}->{y2023}: "
              f"{int(rot_tr.sum())} train / {int(rot_va.sum())} val cells", flush=True)
        # What the covariates must beat on the TEMPORAL axis. The Z-MSE baseline above says
        # nothing about direction of change, and the permutation null only says "better than
        # a shuffled pairing" -- neither rules out the model simply reproducing what its
        # neighbours did. This does.
        _idw_dc, _idw_n = spatial_interp_dir_cos(tgt, y_deep, y2023, rot_va)
        print(f"[desk] direction baseline (inverse-distance, no covariates): "
              f"dcos={_idw_dc:.2f} on {_idw_n} val cells -- the model's dcos va must beat "
              f"this, not just the permutation null", flush=True)
        if _pin is not None:
            print(f"[desk] NOTE {y_deep}->{y2023} is the trained-era CONTROL: both ends lie in "
                  f"years the model trained on, so it measures whether losing the deep past hurt "
                  f"modern-era skill -- not extrapolation. See the withheld pair below.",
                  flush=True)
    else:
        rot_tr = rot_va = torch.zeros(1, dtype=torch.bool, device=device)
        perm_tr = perm_va = torch.zeros(0, dtype=torch.long, device=device)
    # The withheld-era pair. Its IDW bar is deliberately absent: spatial_interp_dir_cos needs
    # >=k TRAINING cells in the deep year and a withheld year has none, so it self-disables.
    # That is the honest answer -- forcing a value would hand the bar that year's truth while
    # the model saw none of it. The admissible bar is validate's spacetime_idw.
    rot_ho = perm_ho = None
    if y_ho is not None:
        rot_ho = tgt[y_ho][2] & tgt[y2023][2]
        g_ho = torch.Generator(device="cpu").manual_seed(seed + 1)
        perm_ho = torch.randperm(int(rot_ho.sum()), generator=g_ho).to(device)
        _ho_dc, _ = spatial_interp_dir_cos(tgt, y_ho, y2023, rot_ho)
        print(f"[desk] WITHHELD-era direction pair {y_ho}->{y2023}: "
              f"{int(rot_ho.sum())} val cells; idw bar "
              f"{'n/a (no training cells that year -- use validate spacetime_idw)' if not np.isfinite(_ho_dc) else f'{_ho_dc:.2f}'}",
              flush=True)
    best_val, best, bad, nonfinite = float("inf"), None, 0, 0
    print(f"--- Training DESK+outputEMA ({len(window_years)}yr window {window_years[0]}..{window_years[-1]}, "
          f"{len(tgt)} supervised years, max {epochs} ep; grad_clip={grad_clip}, es_warmup={es_warmup}, "
          f"amp={use_amp}, dropout={dropout}, wd={weight_decay}) ---")
    for ep in range(1, epochs + 1):
        t_ep = time.perf_counter()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        opt.zero_grad()
        z_ema, recon_loss = _forward_window(train_mode=True, mask_inputs=True)

        # uniform stabilizing loss over all supervised (cell,year), train cells only.
        # Divided by latent_dim so the term is a per-ELEMENT mean like the other two; the
        # config's stabilizing weight absorbs the factor (1.0 -> latent_dim), leaving the total
        # loss numerically unchanged while making the three weights directly comparable.
        # Per-cell WEIGHTS, so a cell-year surveyed by an observer on their first year at that
        # route counts less. Weighted rather than dropped: those cell-years still carry real
        # absences, and dropping them would thin the early era ~2x harder than the modern one
        # (first-year share 25.6% in 1966-1980 against 12.3% in 2001-2025), i.e. exactly where
        # the data is thinnest. The denominator is the weight SUM, not the row count, so
        # changing the weights rescales the loss consistently and does not quietly change the
        # effective learning rate on this term.
        sq_sum, n_eff = torch.zeros((), device=device), 0.0
        for y, (zg, tr, _va, wg) in tgt.items():
            zz = z_ema[yi[y]]
            s = torch.sum((zz[tr] - zg[tr]) ** 2, dim=1) * wg[tr]
            sq_sum = sq_sum + s.sum(); n_eff += float(wg[tr].sum())
        loss_stab = sq_sum / max(n_eff, 1e-8) / latent_dim
        # Pairs span the whole spacetime pool -- same year, same cell across years, and
        # different cells in different years alike. The ESK basis is ONE joint kernel-PCA over
        # every (cell, year) point, so that is the similarity structure this term has to hold
        # the model to. Restricting pairs to within a year would enforce only the spatial half
        # of a spatiotemporal kernel.
        loss_true = spacetime_kernel_loss(z_ema, pool_t, pool_flat, pool_x, pool_w=pool_w_t)
        w_stab = weights["stabilizing"] * loss_stab
        w_true = weights["metric"] * loss_true
        w_rec = weights["reconstruction"] * recon_loss
        loss = w_stab + w_true + w_rec

        # A non-finite loss here would silently poison every weight via the optimizer step, and
        # several terms mean() over masks that can in principle be empty. Skip the step, keep
        # going, and abort only if it persists -- one bad epoch is recoverable, a stuck run is not.
        if not torch.isfinite(loss):
            nonfinite += 1
            print(f"Ep {ep:03d} | NON-FINITE loss (stab={loss_stab.item():.4g} "
                  f"true={loss_true.item():.4g} rec={recon_loss.item():.4g}) -- step skipped "
                  f"({nonfinite} consecutive)", flush=True)
            opt.zero_grad(set_to_none=True)
            if nonfinite >= 5:
                raise RuntimeError("DESK loss non-finite for 5 consecutive epochs; aborting")
            continue
        nonfinite = 0

        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
        opt.step()
        sched.step()

        # Held-out-cell Z-MSE pooled over ALL supervised years -- matches the uniform all-years
        # training objective (not just the easy anchor year), and averaging over many cells/years
        # stabilizes the signal.
        #
        # This is a CLEAN re-forward: dropout off and input masking off. Reusing the training
        # step's z_ema (as this did before) measured a dropout-corrupted, input-masked model, so
        # early stopping, the LR schedule, and the reported Val were all reading noise. No grad
        # and no checkpoint recompute makes the extra pass much cheaper than the training one;
        # eval_every amortizes it further if needed.
        if ep % max(1, int(eval_every)) == 0 or ep == epochs:
            with torch.no_grad():
                z_eval, _, z_raw_eval = _forward_window(train_mode=False, mask_inputs=False,
                                                        want_raw=True)
                vs, vn = _z_mse(z_eval, val_sel)
                # E and H: the two numbers the temporal sweep is actually about.
                vs_sp = _z_mse(z_eval, val_sel_sp)[0] if val_sel_sp else float("nan")
                vs_spt = _z_mse(z_eval, val_sel_spt)[0] if val_sel_spt else float("nan")
                va_anchor = tgt[y2023][2]
                vs_anchor = (float(torch.sum((z_eval[yi[y2023]][va_anchor]
                                              - tgt[y2023][0][va_anchor]) ** 2))
                             / max(int(va_anchor.sum()), 1)) if bool(va_anchor.any()) else float("nan")
                vs_time = _z_mse(z_eval, hy_tgt)[0] if hy_tgt else float("nan")
                ts, _ = _z_mse(z_eval, tr_sel)          # clean-train MSE, same scale as vs
                # Rotation on the SUPERVISED z_ema (ratio-to-one is meaningful), plus the raw
                # pre-EMA rotation for information since that is what the cube exports.
                rotP, rotT = _rotation(z_eval, rot_tr)
                rotPv, rotTv = _rotation(z_eval, rot_va)
                rawP, _ = _rotation(z_raw_eval, rot_tr)
                rawPv, _ = _rotation(z_raw_eval, rot_va)
                # Direction of change, with a fixed-permutation null. rot is magnitude only.
                dcT, dcTn = _dir_cos(z_eval, rot_tr, perm_tr)
                dcV, dcVn = _dir_cos(z_eval, rot_va, perm_va)
                # Withheld-era pair: same two diagnostics, on years the model never trained on.
                if rot_ho is not None:
                    rotPh, rotTh = _rotation(z_eval, rot_ho, y_a=y_ho)
                    dcH, _ = _dir_cos(z_eval, rot_ho, perm_ho, y_a=y_ho)
                else:
                    rotPh = rotTh = dcH = float("nan")
                # the magnitude half of the same pairs the angles above score
                mgT, mgV = _mag_ratio(z_eval, rot_tr), _mag_ratio(z_eval, rot_va)
        dt = time.perf_counter() - t_ep
        rss = _max_rss_gib()
        vram = (f" | VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f}/"
                f"{torch.cuda.memory_reserved() / 2**30:.2f}G" if device == "cuda" else "")
        tot = float(loss.item()) or 1.0
        # rot: predicted vs target temporal rotation, and their ratio. The ratio is the headline
        # number -- 1.0 means DESK reproduces the observed amount of community change; the
        # measured value on the 27 km grid was ~0.29 (interior), i.e. a 3.4x under-prediction.
        def _ratio(p, t):
            return (p / t) if (t and np.isfinite(t) and t > 1e-9) else float("nan")
        # rot: how much of the TARGET's temporal change the supervised z_ema reproduces. 1.00 is
        # the goal. (raw ...) is the pre-EMA rotation, which must exceed the target -- it is
        # informational, not a ratio-to-one.
        rr, rrv = _ratio(rotP, rotT), _ratio(rotPv, rotTv)
        rawr, rawrv = _ratio(rawP, rotT), _ratio(rawPv, rotTv)
        rrh = _ratio(rotPh, rotTh)
        gap = (vs / ts) if ts > 1e-12 else float("nan")
        # Calibration. For an MSE objective the optimal magnitude of a prediction whose direction
        # cosine is rho is rho*truth, and since 1-cos ~ theta^2/2 that makes rot ~ dcos^2 the
        # MSE-calibrated value. cal = rot/dcos^2: ~1 is MSE-calibrated, >1 means the model swings
        # further than its direction accuracy justifies (which RAISES error), and rot -> 1 is the
        # separate thing the science wants. Reported, not optimized -- the objective is unchanged.
        cal = (rrv / (dcV ** 2)) if (np.isfinite(rrv) and np.isfinite(dcV)
                                     and abs(dcV) > 1e-6) else float("nan")
        print(f"Ep {ep:03d} | Stab {loss_stab.item():.4f} | True {loss_true.item():.4f} | "
              f"Rec {recon_loss.item():.4f} | mse tr {ts:.4f} va(pool) {vs:.4f} gap {gap:.1f}x | "
              f"va(sp) {vs_sp:.4f} va(sp+t) {vs_spt:.4f} | "
              f"Val(2025) {vs_anchor:.4f} | Val(yr-out) {vs_time:.4f} | "
              f"rot tr {rr:.2f} va {rrv:.2f} ho {rrh:.2f} (raw {rawr:.2f}/{rawrv:.2f}) | "
              f"dcos tr {dcT:.2f} va {dcV:.2f} ho {dcH:.2f} null {dcTn:+.2f}/{dcVn:+.2f} | "
              f"mag tr {mgT:.2f} va {mgV:.2f} | cal {cal:.1f} | "
              f"half-life {ema.half_life().item():.1f}y | "
              f"mix {w_stab.item()/tot:.2f}/{w_true.item()/tot:.2f}/{w_rec.item()/tot:.2f} | "
              f"lr {opt.param_groups[0]['lr']:.2e} | {dt:.1f}s{vram} | RSS {rss:.1f}G", flush=True)
        if ep <= es_warmup:
            continue                       # don't let the volatile warmup epochs set 'best'
        if vs < best_val - min_delta:
            best_val, bad = vs, 0
            best = ({k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    {k: v.detach().cpu().clone() for k, v in ema.state_dict().items()})
        else:
            bad += 1
            if bad >= patience:
                print(f"[desk] early stop at ep {ep} (best Val {best_val:.4f})"); break
    if best is not None:
        model.load_state_dict({k: v.to(device) for k, v in best[0].items()})
        ema.load_state_dict({k: v.to(device) for k, v in best[1].items()})
    return model, ema



def run_desk_experiment(config=None):
    """Driver: load N-stream states + ESK Z, prepare grids, train DESK, save model+meta.

    Supervises DESK against the trend-based spatiotemporal z-target -- the anchor-year
    community plus the backward-reconstructed historical points -- over a spatially
    BLOCKED cell holdout, with a buffer ring derived from the conv kernel so held-out
    receptive fields cannot reach training cells.

    The checkpoint filename ``env_model_semisup.pth`` is historical and kept only so
    existing artifacts stay loadable; the "semi-supervised" part is the reconstruction
    loss over unlabelled years, which the EMA objective still carries.
    """
    config = load_config(config) if not isinstance(config, dict) else config
    paths, desk_cfg = config["paths"], config["desk"]
    out_dir = paths["desk_output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    states_dir = os.path.join(paths["hist_dir"], "yearly_states")
    schema = cio.load_schema(states_dir)
    label_year = int(desk_cfg.get("label_year", 2023))
    hidden_width = desk_cfg.get("hidden_width") or None
    mlp_expansion = int(desk_cfg.get("mlp_expansion", 4))
    spatial_kernel = int(desk_cfg.get("spatial_conv", {}).get("kernel", 3)) \
        if desk_cfg.get("spatial_conv", {}).get("enabled", True) else 0

    cov_stack = cio.load_state_stack(label_year, states_dir, schema)
    # The Ružicka metric anchor is the reconstructed reference-year (anchor_year) community --
    # the EXACT vectors that seeded the ESK basis (log1p abundance, anchor-mode-agnostic).
    # Scatter X_points' anchor-year rows into an (H,W,S) grid, so DESK depends on no weekly
    # eBird product (the trends-abd anchor needs none).
    # One point set for the whole run: target.points_dir when configured (the raw-BBS +
    # eBird-window target), else trend.points_dir (the older trend-product target). Keeping
    # both loadable is what makes an A/B between the two targets possible.
    from src.config_utils import target_points_dir
    points_dir = target_points_dir(config)
    Xp, pip, p_weights, p_supervise = load_point_set(points_dir)
    pm = json.load(open(os.path.join(points_dir, "points_meta.json")))
    # Fail before training rather than after: a leaky community produces plausible-looking
    # numbers throughout, so there is no downstream symptom to catch it by.
    from src.config_utils import load_data_config as _ldc
    _n_sp = assert_focal_excluded(pm, (_ldc() or {}).get("focal_species_code"))
    print(f"[desk] target: {pm.get('target_source', 'trend_products')} from {points_dir} "
          f"({Xp.shape[0]:,} rows, {int(p_supervise.sum()):,} supervised"
          + (f"; {_n_sp} species, focal excluded" if _n_sp else "") + ")", flush=True)
    ay, S = int(pm["recent_year"]), int(pm["n_species"])
    H, W = cov_stack.shape[:2]
    # There is no anchor YEAR any more. The metric loss used to read one year's community
    # (the reconstructed target gave every cell a value in every year, so that cost nothing);
    # on raw BBS it would gate supervision to the ~2,200 cells surveyed in that one year,
    # discarding 43% of the cells BBS actually covers. Coverage is now ANY supervised year,
    # and the metric loss draws its pairs across the whole spacetime pool -- see
    # spacetime_metric_pool and spacetime_kernel_loss.
    sup_rows = p_supervise if p_supervise is not None else np.ones(len(pip), bool)
    ebird_stack = np.full((H, W, S), np.nan, dtype="float32")
    order = np.argsort(pip[sup_rows, 2])                           # latest year wins per cell
    _r, _c = pip[sup_rows][order, 0], pip[sup_rows][order, 1]
    ebird_stack[_r, _c] = Xp[sup_rows][order]                      # already log1p in X_points
    log1p_kernel = bool(pm.get("ruzicka_log1p", True))
    print(f"[desk] Ružicka metric over every supervised cell-year, pairs drawn across space "
          f"AND time (log1p={log1p_kernel}); coverage grid = any surveyed year")

    z_dir = desk_cfg["z_dir"]
    try:
        z_mask = np.load(os.path.join(z_dir, "valid_mask.npy"))
        z_flat = np.load(os.path.join(z_dir, "Z.npy"))
        with open(os.path.join(z_dir, "meta.json"), encoding="utf-8") as fh:
            z_meta = json.load(fh)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"ESK Z.npy/valid_mask.npy/meta.json not in {z_dir}") from exc
    if z_meta.get("kernel") != "ruzicka" or bool(z_meta.get("centered", True)):
        raise ValueError(f"DESK requires the uncentered Ružička ESK contract; got {z_meta}")
    # ESK saves Z at the max swept latent_dim. Optionally truncate to desk.latent_dim:
    # kernel-PCA columns are eigenvalue-ordered, so Z[:, :k] IS the exact dim-k
    # embedding (no ESK re-run needed). Unset -> use all columns.
    ld = desk_cfg.get("latent_dim")
    if ld and z_flat.shape[1] < int(ld):
        raise ValueError(f"ESK Z has {z_flat.shape[1]} dimensions, fewer than desk.latent_dim={ld}")
    if ld and z_flat.shape[1] > int(ld):
        print(f"[desk] truncating ESK Z {z_flat.shape[1]} -> {int(ld)} dims (top eigen-components)")
        z_flat = z_flat[:, :int(ld)]

    mask_sup0 = compute_valid_mask(ebird_stack, cov_stack)

    # Spatial holdout is drawn BEFORE the normalization fit: mu/sd computed over held-out cells
    # too would leak the evaluation distribution into every training input (and into the frozen
    # stats the cube reuses). Blocks + a buffer, both defined here so the fit can exclude them.
    holdout = np.zeros_like(mask_sup0)
    buffer_cells_mask = np.zeros_like(mask_sup0)
    _cfg = desk_cfg.get("trend", {})
    ebird_valid = np.any(~np.isnan(ebird_stack), axis=-1)
    # Buffer width is DERIVED from the conv kernel, never configured separately: a prediction
    # at a val cell reads cells up to kernel//2 away, so a narrower buffer would let val
    # receptive fields overlap training cells and quietly flatter the metric.
    buf = spatial_kernel // 2
    block = int(_cfg.get("block_cells", 12))
    # Draw the split over the cells that actually CARRY SUPERVISION, not over the eBird
    # covariate footprint. The two were nearly the same while the target was an interpolated
    # surface covering ~17,200 cells; a measured target is not -- raw BBS reaches ~3,900 cells
    # and BBS+eBird ~12,400. Drawing blocks over the full footprint would put many validation
    # blocks on cells with no target at all, so the reported number would rest on far fewer
    # cells than the block count suggests, and its variance would be invisible.
    sup_cells = supervised_cells(pip, p_supervise, ebird_valid.shape)
    split_domain = sup_cells & ebird_valid          # a target is useless without covariates
    holdout, buffer_cells_mask = blocked_holdout(
        split_domain, block_cells=block, holdout_frac=float(_cfg.get("holdout_frac", 0.15)),
        buffer_cells=buf, seed=int(_cfg.get("seed", 0)))
    n_val = int(holdout.sum())
    print(f"[desk] split domain: {int(sup_cells.sum())} supervised cells, "
          f"{int(split_domain.sum())} with covariates too "
          f"({int(ebird_valid.sum())} in the covariate footprint)", flush=True)
    print(f"[desk] split: {block}x{block}-cell blocks, buffer {buf} cells (from "
          f"spatial_kernel={spatial_kernel}) -> {n_val} val, "
          f"{int(buffer_cells_mask.sum())} buffer cells", flush=True)
    min_val = int(_cfg.get("min_val_cells", 200))
    if n_val < min_val:
        raise SystemExit(
            f"only {n_val} validation cells (floor {min_val}). Every reported val number, "
            f"and best-epoch selection itself, would rest on too few cells to mean anything. "
            f"Raise desk.trend.holdout_frac, lower block_cells, or lower "
            f"desk.trend.min_val_cells if you accept the noise.")

    # Normalization stats: fit on the supervised TRAINING pixels only (buffer cells are excluded
    # as well -- not evaluation data, but not clean training data either), then applied to every
    # grid (labelled + historical) and frozen for the cube.
    fit_mask = mask_sup0 & (~holdout) & (~buffer_cells_mask)
    # schema so availability channels are left un-standardized (see cio.indicator_channels)
    mu, sd = cio.fit_norm(cov_stack[fit_mask].astype("float32"), schema)

    latent_dim = z_flat.shape[1]
    if int(z_mask.sum()) < int(np.any(~np.isnan(ebird_stack), axis=-1).sum()):
        print(f"[desk] note: the ESK anchor-year Z covers {int(z_mask.sum()):,} cells, fewer "
              f"than the {int(np.any(~np.isnan(ebird_stack), axis=-1).sum()):,} with an "
              f"observed community. Not a gate -- targets are projected per point.")
    covn, mask_cov, mask_sup = prepare_supervised(cov_stack, ebird_stack, mu, sd, out_dir)
    ema_cfg = desk_cfg["output_ema"]
    tr_cfg = desk_cfg.get("trend", {})
    # holdout/buffer were drawn above (before the normalization fit). Training excludes BOTH;
    # evaluation uses the holdout only, so buffer cells appear in neither.
    m_tr = mask_sup & (~holdout) & (~buffer_cells_mask)
    m_val = mask_sup & holdout

    # Computed here, above the pool, because the pool must exclude these years -- see
    # spacetime_metric_pool. The anchor is never withheld; it carries the metric loss.
    ho_years = [int(y) for y in (tr_cfg.get("holdout_years") or []) if int(y) != label_year]
    _py, _pf, _px, _ppidx = spacetime_metric_pool(pip, Xp, sup_rows, m_tr, W,
                                                  exclude_years=ho_years, return_pidx=True)
    metric_pool = (_py, _pf, _px)
    print(f"[desk] metric loss over {len(metric_pool[0]):,} training cell-years spanning "
          f"{len(np.unique(metric_pool[0]))} years, pairs drawn across space AND time "
          f"(was one year, {int((pip[:, 2] == ay).sum()):,} rows)")

    # --- the shared stratification, and the rebalance it drives ----------------------------------
    # BBS is coast- and present-heavy, so a uniform objective is fit where the survey happens to be
    # dense. The occupancy table is printed BEFORE any correction because n_min and cap should be
    # chosen against the real numbers; the bias has been asserted throughout this project and never
    # quantified here.
    from .esk_kernel import spacetime_strata
    _bal = desk_cfg.get("balance", {}) or {}
    _sb = int(config["esk"].get("spacetime", {}).get("spatial_bins", 8))
    _ab = int(config["esk"].get("spacetime", {}).get("abundance_bins", 4))
    # Geography x time only for WEIGHTING -- abundance is a property of the place, not a
    # sampling-bias axis, and including it fragmented the pool to a median of 41 rows per
    # stratum. See spacetime_strata's include_abundance note.
    _wb = int(_bal.get("spatial_bins", _sb))
    pool_labels, pool_keys = spacetime_strata(_ppidx, _px, _wb, _ab,
                                              include_abundance=False)
    occ = stratum_occupancy(pool_labels, pool_keys, _ppidx)
    print(f"[desk] weighting strata ({_wb}x{_wb} tiles x decade, no abundance axis): "
          f"{occ['n_strata']} occupied over {occ['n_points']:,} pool rows; per-stratum "
          f"median {occ['median_per_stratum']:.0f}, p10 {occ['p10_per_stratum']:.0f}, "
          f"max {occ['max_per_stratum']:,} (max/median = "
          f"{occ['imbalance_ratio_max_over_median']:.0f}x)")
    for r in occ["thinnest"][:8]:
        print(f"[desk]   thinnest: {r['stratum']:<28} {r['n_cell_years']:>5} cell-years, "
              f"{r['n_cells']:>4} cells, {r['n_years']:>3} years")
    pool_w = None
    if bool(_bal.get("enabled", False)):
        # n_min=null derives the floor from the occupancy table rather than guessing it. The first
        # run hardcoded 200 while the median stratum held 41 and the max ~570, so the floor sat
        # ABOVE almost every stratum, every one of them got the floor weight, and the realised
        # range collapsed to 0.60-1.00 -- a downweight of the few dense strata with no uplift
        # anywhere. A quantile of the observed counts cannot make that mistake.
        _cnt = np.bincount(pool_labels)
        _cnt = _cnt[_cnt > 0]
        _nmin = _bal.get("n_min")
        if _nmin is None:
            _nmin = max(2, int(np.quantile(_cnt, float(_bal.get("n_min_quantile", 0.25)))))
            print(f"[desk] n_min derived from the occupancy table: {_nmin} "
                  f"(q{float(_bal.get('n_min_quantile', 0.25)):.2f} of stratum sizes; "
                  f"median {np.median(_cnt):.0f}, max {_cnt.max()})")
        wv = stratum_weights(pool_labels, n_min=int(_nmin),
                             cap=float(_bal.get("cap", 5.0)),
                             power=float(_bal.get("power", 0.5)))
        pool_w = torch.tensor(wv, dtype=torch.float32)
        # Realised effective-sample shares, early vs modern, before and after. first_year_weight
        # already downweights green observers and is era-correlated (25.6% of 1966-1980 cell-years
        # against 12.3% of 2001-2025), so an early-era uplift partly cancels a downweight that
        # exists for a good reason. Print both shares rather than assume they compose benignly.
        # The era split has to come from the pool, not a constant. A hardcoded 1981 reported
        # "0.000 -> 0.000" in two of three runs, because those withhold every year before 1986
        # and 1996 respectively -- the pool contains no pre-1981 rows at all to shift.
        _yy = _ppidx[:, 2]
        _cut = int(np.quantile(_yy, 0.25))
        early = _yy <= _cut
        print(f"[desk] rebalance ON (n^-{float(_bal.get('power', 0.5)):g}, n_min={int(_nmin)}, "
              f"cap={float(_bal.get('cap', 5.0)):g}): earliest-quartile (<= {_cut}) share of "
              f"pool rows {early.mean():.3f} -> effective {(wv[early].sum() / wv.sum()):.3f}; "
              f"weight range {wv.min():.2f}-{wv.max():.2f} over {len(_cnt)} strata")
    else:
        print("[desk] rebalance OFF (desk.balance.enabled=false): metric pairs drawn uniformly, "
              "so the objective inherits BBS's coast/present bias")
    np.save(os.path.join(out_dir, "holdout_cells.npy"), holdout)
    # The buffer too: validate's epoch panel must draw its interpolation sources from the
    # SAME training cells the model saw, and buffer cells are in neither set.
    np.save(os.path.join(out_dir, "buffer_cells.npy"), buffer_cells_mask)
    np.save(os.path.join(out_dir, "buffer_cells.npy"), buffer_cells_mask)
    print(f"[desk] uniform z-target over the anchor + historical trend points; "
          f"{int(holdout.sum())} cells held out for eval, "
          f"{int(buffer_cells_mask.sum())} buffered out of training")

    stream_dims = cio.stream_dims(schema)
    ema_half_life = None
    # Output-EMA objective: forward the ordered year window, apply a learned causal
    # EMA over the year axis to the predicted Z (demographic lag), and supervise the
    # EMA'd z_ema against the per-year trend targets. Replaces the direct per-year target.
    warmup_start = int(ema_cfg.get("warmup_start", 1940))
    window_years = list(range(warmup_start, label_year + 1))
    cov_win, mask_win, kept = _load_year_window(states_dir, schema, mu, sd, window_years)
    targets = _prepare_trend_targets(config, z_dir, latent_dim, holdout,
                                     points_dir=points_dir)

    # Temporal holdout: withhold a contiguous span of supervised years from the objective and
    # score them separately. The spatial split says nothing about extrapolation THROUGH TIME,
    # which is what DESK exists to do. The anchor (label_year) is never withheld -- it carries
    # the eBird metric loss. Diagnostic only; model selection stays on the spatial metric so
    # there is exactly one selection signal.
    # --- item 5: make the two loss terms agree ---------------------------------------------------
    # loss_stab was weighted (first_year_weight) and loss_true was not, so the two halves of the
    # objective disagreed about what a sparse early cell-year is worth. Compose the stratum weight
    # MULTIPLICATIVELY onto the existing per-cell weight rather than stacking it as a separate
    # factor, so first_year_weight's era-correlated downweight is respected instead of overridden.
    if pool_w is not None:
        wv_np = np.asarray(pool_w, dtype="float32")
        n_touched = 0
        # Scatter per year rather than dict-lookup per cell: 3,900 cells x ~60 years is 234k
        # Python-level lookups for a grid multiply numpy does in one pass.
        for y in sorted(targets):
            sel_y = _ppidx[:, 2] == int(y)
            if not sel_y.any():
                continue
            zg, tr_m, va_m, wg = targets[y]
            g = np.ones_like(wg)
            g[_ppidx[sel_y, 0], _ppidx[sel_y, 1]] = wv_np[sel_y]
            targets[y] = (zg, tr_m, va_m, (wg * g).astype("float32"))
            n_touched += int(sel_y.sum())
        print(f"[desk] stratum weights composed onto {n_touched:,} supervised cell-year target "
              f"weights, so loss_stab and loss_true now agree on what a cell-year is worth")

    year_val = {}
    for y in ho_years:
        if y in targets:
            zg, tr_m, va_m, wg = targets[y]
            year_val[y] = (zg, tr_m | va_m)            # score every supervised cell that year
            targets[y] = (zg, np.zeros_like(tr_m), va_m, wg)   # ...and train on none of it
    if year_val:
        print(f"[desk] temporal holdout years (diagnostic, excluded from the objective): "
              f"{sorted(year_val)}", flush=True)

    model, ema = train_model_ema(
        cov_win, mask_win, kept, targets, metric_pool, m_tr, m_val,
        stream_dims, latent_dim=latent_dim, ema_cfg=ema_cfg,
        spatial_kernel=spatial_kernel,
        epochs=desk_cfg.get("epochs", 500), lr=desk_cfg.get("lr", 1e-3),
        weights=desk_cfg.get("weights"), patience=desk_cfg.get("patience", 50),
        schema=schema, augment_cfg=config.get("augment"),
        dropout=float(desk_cfg.get("dropout", 0.5)),
        weight_decay=float(desk_cfg.get("weight_decay", 0.0)),
        warmup_epochs=int(desk_cfg.get("warmup_epochs", 0)),
        min_lr_frac=float(desk_cfg.get("min_lr_frac", 1.0)),
        amp=bool(desk_cfg.get("amp", False)),
        eval_every=int(desk_cfg.get("eval_every", 1)),
        holdout_year_targets=year_val or None,
        pool_w=pool_w,
        # Pinned so the diagnostic interval is a constant across the temporal sweep instead of
        # shrinking with the holdout. None = the historical auto-selected deepest trained year.
        direction_anchor_year=tr_cfg.get("direction_anchor_year"),
        direction_withheld_anchor_year=tr_cfg.get("direction_withheld_anchor_year"),
        hidden_width=hidden_width, mlp_expansion=mlp_expansion)
    ema_half_life = float(ema.half_life().item())
    torch.save(ema.state_dict(), os.path.join(out_dir, "output_ema.pth"))
    print(f"[desk] output-EMA learned half-life = {ema_half_life:.2f} yr")
    torch.save(model.state_dict(), os.path.join(out_dir, "env_model_semisup.pth"))
    np.savez(os.path.join(out_dir, "desk_meta.npz"),
             mu=mu, sd=sd, stream_dims=np.array(stream_dims, int),
             latent_dim=latent_dim, label_year=label_year,
             spatial_kernel=spatial_kernel,
             # Provenance for the regularization/augmentation recipe. dropout does not change
             # state_dict keys (so checkpoints stay loadable either way), but without recording
             # it a run cannot be reproduced or compared against another.
             dropout=float(desk_cfg.get("dropout", 0.5)),
             # hidden_width/mlp_expansion change state_dict SHAPES: without persisting them
             # the cube would rebuild a differently-sized net and fail to load these weights.
             hidden_width=int(model.hidden_width),
             mlp_expansion=int(model.mlp_expansion),
             augment=json.dumps(config.get("augment") or {}),
             holdout_cells=int(holdout.sum()), buffer_cells=int(buffer_cells_mask.sum()),
             schema=json.dumps(schema),
             output_ema=True,
             ema_half_life=(ema_half_life if ema_half_life is not None else np.nan),
             ema_warmup_start=int(ema_cfg.get("warmup_start", 1940)),
             kernel=str(z_meta["kernel"]), centered=bool(z_meta["centered"]),
             kernel_contract=str(z_meta.get("kernel_contract", "")))
    print(f"[desk] saved model + desk_meta.npz -> {out_dir} "
          f"(spatial_kernel={spatial_kernel})")


if __name__ == "__main__":
    run_desk_experiment()
