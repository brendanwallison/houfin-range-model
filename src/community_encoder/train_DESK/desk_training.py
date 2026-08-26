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
        # `zi`, not `z_pred`: this branch referenced a name that does not exist in this scope --
        # a leftover from `true_kernel_loss`, where the argument IS called z_pred. It therefore
        # raised NameError instead of returning 0, and was unreachable only because a real
        # community is never all-zero. A degenerate pool must yield 0, not a crash.
        return torch.zeros((), device=zi.device, requires_grad=True)
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



def fixed_kernel_pairs(n_points, num_pairs, seed):
    """A FIXED ``(2, num_pairs)`` index draw over a pool -- the pairs an eval metric scores.

    ``spacetime_kernel_loss`` redraws its pairs every call, which is right for a training
    term (fresh pairs each step are the stochastic gradient) and wrong for a metric used to
    pick an epoch. With a fresh draw the epoch-to-epoch difference mixes the model's change
    with the pair set's change, so the argmin over epochs is partly a draw of the dice --
    and best-epoch selection is exactly what this instrumentation exists to support. Drawing
    once and reusing the same pairs makes consecutive epochs differ only by the model.

    Returns ``None`` when the pool holds fewer than 2 points, which is the honest answer for
    a val pool that no held-out cell reaches (see ``NULL_IS_A_FLOOR_FOR``: a metric with no
    pairs is unavailable, not zero).
    """
    n = int(n_points)
    if n < 2:
        return None
    g = np.random.default_rng(int(seed))
    return g.integers(0, n, size=(2, int(num_pairs)), dtype=np.int64)


# Distinct seed streams per pool. A dict rather than an enumerate() so a pool added later cannot
# silently shift every other pool's draw and make new runs incomparable with old ones.
POOL_SEED_OFFSET = {"pool": 0, "sp": 1, "spt": 2}


def kernel_pair_draws(n_points, num_pairs, seed, n_draws):
    """``n_draws`` INDEPENDENT fixed pair sets, or ``None`` if the pool cannot form a pair.

    One draw gives a number with no error bar, so there is no way to say whether an 8% spread
    between configurations is above or below the noise of the thing being compared. Several
    independent draws give both: the mean is the selection signal (standard error down by
    ``sqrt(n_draws)``) and the spread across draws is the estimator's own noise floor.

    Each draw gets its own seed. The previous single-draw code passed the SAME seed for the
    pool/sp/spt pools, so those three "independent" numbers came from one RNG stream and differed
    only because the pools had different lengths -- not independent at all.
    """
    out = [fixed_kernel_pairs(n_points, num_pairs, int(seed) + 7919 * d)
           for d in range(max(1, int(n_draws)))]
    return None if any(p is None for p in out) else out


def kernel_loss_on_pairs(z_by_t, pool_t, pool_flat, pool_x, pairs, rank=None):
    """``spacetime_kernel_loss`` on a caller-supplied pair index array. Same estimand.

    Shares ``_pair_kernel_loss`` with the training term rather than reimplementing the
    Ružička ratio, so the number selected on and the number optimized cannot drift apart --
    the failure this codebase hit when two modules averaged the same width in different
    orders.

    ``rank`` truncates z to its first ``rank`` components before the dot product. That is the
    quantity a downstream consuming a rank-r prefix actually gets: ingest takes ``z[..., :M]``
    positionally (``src/data/combine/model_inputs.py``), so the rank-M kernel is the covariance
    of the GP it fits, and it is NOT the rank-64 kernel this metric otherwise reports. Reported
    as a curve over several ranks rather than at one value, because the retained M is a moving
    target and nothing here should key on today's.
    """
    T, H, W, L = z_by_t.shape
    i = torch.as_tensor(pairs[0], device=z_by_t.device, dtype=torch.long)
    j = torch.as_tensor(pairs[1], device=z_by_t.device, dtype=torch.long)
    zf = z_by_t.reshape(T, H * W, L)
    zi, zj = zf[pool_t[i], pool_flat[i]], zf[pool_t[j], pool_flat[j]]
    if rank is not None and int(rank) < L:
        zi, zj = zi[:, :int(rank)], zj[:, :int(rank)]
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


def _prepare_trend_targets(config, z_dir, latent_dim, holdout, points_dir=None,
                           exclude=None):
    """Per-year ESK-basis targets for EVERY supervised year, from the target point set.

    Projects ``X_points`` into the joint ESK basis (z_obs) and scatters each supervised point
    into its year's grid. Returns ``{year: (zg (H,W,L), tm_tr (H,W), tm_val (H,W),
    wg (H,W))}`` where the train/val split is the spatial ``holdout`` (val = held-out cells)
    and ``wg`` is the per-cell loss weight for that year.

    Only rows flagged ``supervise`` are scattered, so each cell-year is written exactly once.

    ``exclude`` drops cells from the TRAIN mask without adding them to val -- the buffer ring,
    and any block a ``train_frac`` thinning removed. Both were previously excluded from the
    metric pool (through ``m_tr``) and from the normalization fit but NOT from here, so buffer
    cells still supervised the STABILIZING term, which carries most of the loss. That silently
    voided the buffer's guarantee: a held-out cell's convolutional receptive field reaches
    ``kernel//2`` cells away, and those cells were supervised, so the held-out score was
    measured against a model trained right up to the block edge. The masks are also what makes
    a ``train_frac`` axis mean anything -- thinning the metric pool alone would leave the
    dominant term reading every cell.
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
    drop = (np.zeros_like(holdout) if exclude is None
            else np.asarray(exclude, dtype=bool))
    if drop.shape != holdout.shape:
        raise ValueError(f"exclude has shape {drop.shape} but the holdout grid is "
                         f"{holdout.shape}; these are cell-aligned masks.")
    out = {}
    for y in sorted({int(v) for v in yrs}):
        sel = np.where((yrs == y) & supervise)[0]
        zg = np.zeros((H, W, latent_dim), dtype="float32")
        present = np.zeros((H, W), bool)
        wg = np.zeros((H, W), dtype="float32")
        zg[rows[sel], cols[sel]] = z_obs[sel]
        present[rows[sel], cols[sel]] = True
        wg[rows[sel], cols[sel]] = weights[sel]
        out[y] = (zg, present & (~holdout) & (~drop), present & holdout, wg)
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
                    weights=None, seed=0, patience=50, min_delta=0.0,
                    schema=None, augment_cfg=None, dropout=0.5, weight_decay=0.0,
                    warmup_epochs=0, min_lr_frac=1.0, amp=False, eval_every=1,
                    holdout_year_targets=None, pool_w=None, direction_anchor_year=None,
                    direction_withheld_anchor_year=None,
                    hidden_width=None, mlp_expansion=4,
                    val_metric_pool=None, val_pool_holdout_years=(),
                    metric_pairs=4096,
                    eval_kernel_pairs=65536, eval_kernel_draws=1, eval_kernel_ranks=(),
                    eigenbasis_batch=0, eigenbasis_every=0, eigenbasis_ref=None,
                    eigenbasis_draws=1,
                    selection_metric="val_zmse",
                    selection_smooth=0,
                    trajectory_path=None, stop_at_epoch=None, return_info=False,
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

    ``val_metric_pool`` is the same 3-tuple as ``metric_pool`` but built over the HELD-OUT
    cells, so the kernel term -- the quantity the population model actually consumes -- can be
    scored on cells the objective never saw. Until now the kernel loss existed only on the
    train pool, so the one term downstream depends on had never been measured out of sample,
    and epoch selection ran on ``val`` z-MSE instead. Those two disagree: on an existing
    500-epoch run z-MSE was best at epoch 109 (0.2026, rising to 0.2224 by 500) while the
    train-pool kernel loss was still falling at 500 (0.00250, 39% below its epoch-109 value).
    Which of them the held-out kernel follows is the question this measures, so BOTH are
    logged whatever ``selection_metric`` picks.

    ``val_pool_holdout_years`` splits that pool the way the z-MSE report is already split: a
    held-out cell in a TRAINED year is an unseen place, a held-out cell in a WITHHELD year is
    an unseen place and time. Pooling them reports a mixture of two questions, and which one
    dominates depends on how many years the overlay withheld -- i.e. on the sweep's own
    independent variable.

    ``selection_metric`` is ``val_zmse`` (historical) or ``val_kernel``. ``stop_at_epoch``
    halts the loop early WITHOUT shortening the LR schedule, which ``epochs`` would: the
    cosine anneal is parameterised on the budget, so lowering ``epochs`` to the stopping point
    changes the learning rate at every preceding step and trains a different model rather than
    the same one stopped sooner. ``trajectory_path`` receives one JSON object per epoch.

    With ``return_info=True`` the return grows a third element, the per-run summary (best
    epoch, both metrics there, the trajectory path). Behind a flag rather than always, because
    a silent arity change is a failure mode this file has been bitten by -- see
    ``spacetime_metric_pool``'s ``return_pidx`` note.
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

    # --- the VALIDATION kernel pool: the downstream's own quantity, out of sample ----------
    # Built by the caller from the val mask, so by construction it shares no cell with the
    # train pool -- the two come from complementary masks and the buffer ring sits between
    # them. Pairs are drawn ONCE (fixed_kernel_pairs) and reused at every eval so consecutive
    # epochs differ only by the model, and the whole block runs under no_grad at the eval
    # site: it must never touch a weight.
    def _prep_val_pool(py, pf, px, label):
        keep = np.array([int(y) in yi for y in py], dtype=bool)
        if not keep.any():
            print(f"[desk] val kernel pool ({label}): EMPTY -- no held-out cell-year falls "
                  f"in the forwarded window, so this metric is UNAVAILABLE_NO_VAL_POINTS "
                  f"rather than 0", flush=True)
            return None
        t = torch.as_tensor(np.array([yi[int(y)] for y in py[keep]]),
                            device=device, dtype=torch.long)
        f = torch.as_tensor(pf[keep], device=device, dtype=torch.long)
        x = torch.as_tensor(px[keep], device=device)
        # Per-pool seed offset. Passing the same seed to every pool made the three draws share
        # one RNG stream, so they differed only by pool length -- not independent, which is the
        # whole property a spread across pools is supposed to have.
        draws = kernel_pair_draws(int(t.shape[0]), eval_kernel_pairs,
                                  seed + 104729 * (POOL_SEED_OFFSET.get(label, 0) + 1),
                                  eval_kernel_draws)
        if draws is None:
            print(f"[desk] val kernel pool ({label}): {int(t.shape[0])} point(s), too few to "
                  f"form a pair -- UNAVAILABLE_TOO_FEW_POINTS", flush=True)
            return None
        print(f"[desk] val kernel pool ({label}): {int(t.shape[0]):,} held-out cell-years "
              f"over {len(np.unique(py[keep]))} years, {len(draws)} x {eval_kernel_pairs:,} "
              f"FIXED pairs", flush=True)
        return (t, f, x, draws)

    # The train pool's own fixed eval pairs, seeded differently from the val pools so the two
    # are independent draws rather than the same index pattern over different pools.
    train_eval_pairs = kernel_pair_draws(int(pool_t.shape[0]), eval_kernel_pairs, seed + 977,
                                        eval_kernel_draws)
    # --- the eigenbasis diagnostic's fixed batch -----------------------------------------------
    # A BATCH, not pairs: the nesting objective needs Tf = Kf, an operator applied to the feature
    # map, which sampled pairs cannot supply. Drawn once with a fixed seed so the number is
    # comparable epoch to epoch and run to run, and small (a few thousand) because the Ružička
    # block is O(B^2 S).
    eig_batch = None
    if eigenbasis_batch and val_metric_pool is not None:
        _ey, _ef, _ex = val_metric_pool
        _ekeep = np.array([int(y) in yi for y in _ey], dtype=bool)
        _n = int(_ekeep.sum())
        if _n >= 8:
            _pool = np.flatnonzero(_ekeep)
            # INDEPENDENT batches, not one. A single batch gives a nesting gap with no error bar,
            # which is precisely the state the kernel metric was in before eval_kernel_draws: an
            # 18% spread across configurations that cannot be told from sampling noise. Each batch
            # is its own fixed draw, so the value stays comparable epoch to epoch while the spread
            # across batches measures how much of any between-configuration difference is real.
            eig_batch = []
            for _di in range(max(1, int(eigenbasis_draws))):
                _sel = _pool
                if _n > int(eigenbasis_batch):
                    _sel = np.random.default_rng(seed + 31337 + 7919 * _di).choice(
                        _pool, int(eigenbasis_batch), replace=False)
                eig_batch.append({
                    "t": torch.as_tensor(np.array([yi[int(_ey[k])] for k in _sel]),
                                         device=device, dtype=torch.long),
                    "flat": torch.as_tensor(_ef[_sel], device=device, dtype=torch.long),
                    "x": np.asarray(_ex[_sel], dtype="float64"),
                    # ESK's own projection of the SAME communities: an ordered eigenbasis by
                    # construction, so its values are the achievable floor at this Nyström rank
                    # and DESK's gap to them is the honest measure.
                    "ref": (None if eigenbasis_ref is None
                            else np.asarray(eigenbasis_ref)[_sel]),
                })
            print(f"[desk] eigenbasis diagnostic: {len(eig_batch)} x {len(_sel):,} held-out "
                  f"cell-years, Ružička block {len(_sel)}x{len(_sel)}, every "
                  f"{eigenbasis_every} eval(s)"
                  + ("" if eig_batch[0]["ref"] is not None else
                     " -- NO ESK reference (subspace/nesting gap unavailable)"), flush=True)
        else:
            print(f"[desk] eigenbasis diagnostic: only {_n} held-out points in the window, "
                  f"too few -- UNAVAILABLE_TOO_FEW_POINTS", flush=True)
    _eig_gram = {}          # cached: the batch is fixed, so its Ružička block never changes

    val_pools = {}
    if val_metric_pool is not None:
        _vy, _vf, _vx = val_metric_pool
        _vho = np.isin(_vy, np.asarray(list(val_pool_holdout_years) or [-1], dtype=_vy.dtype))
        # sp   = held-out cell, TRAINED year   -> unseen place
        # sp+t = held-out cell, WITHHELD year  -> unseen place AND time
        # pool = both, kept because it is the single number a sweep column can be ranked on
        val_pools["pool"] = _prep_val_pool(_vy, _vf, _vx, "pool")
        if _vho.any():
            val_pools["sp"] = _prep_val_pool(_vy[~_vho], _vf[~_vho], _vx[~_vho], "sp")
            val_pools["spt"] = _prep_val_pool(_vy[_vho], _vf[_vho], _vx[_vho], "sp+t")
        else:
            # With no withheld years, `sp` IS `pool` -- same rows, and previously the same pair
            # indices too, so the log printed the two values identically and one of the four
            # eval calls per epoch was pure duplication. Alias it instead of recomputing; the
            # saving pays for the extra independent draws.
            val_pools["sp"] = val_pools["pool"]
            val_pools["spt"] = None
            print("[desk] val kernel pool: no withheld years, so sp == pool; scoring it once",
                  flush=True)
    else:
        print("[desk] val kernel pool: NOT WIRED (no val_metric_pool passed) -- the kernel "
              "term is measured on training pairs only, so it cannot be selected on",
              flush=True)
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
        """Mean per-cell summed-over-latent Z error on the cells selected by ``sel[y]``.

        Returns ``nan`` when no cell is selected, NOT 0. The old ``sq / max(cnt, 1)`` made an
        empty selection score a PERFECT 0.0000: with no validation cells -- the production
        retrain's configuration, and any run whose val blocks happen to miss every supervised
        cell -- the best-epoch comparison then accepted epoch 1 and kept near-init weights,
        while the log reported ``va(pool) 0.0000`` and the run finished normally. The only
        thing standing between that and a shipped model was the configurable
        ``min_val_cells`` floor. An absent measurement has to be absent (see
        ``UNAVAILABLE_*``), because a zero here is indistinguishable from a perfect fit.
        """
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
        return (sq / cnt if cnt else float("nan")), cnt

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
    best_epoch, best_zmse, best_kernel = None, float("nan"), float("nan")
    best_crit_raw = float("nan")
    # Pre-bound so an eval_every > 1 epoch cannot reference an eval-only name before it is
    # first assigned: the printed line and the JSONL row are emitted EVERY epoch.
    vk, vk_sd, tk, rank_curve, eig = {}, {}, float("nan"), {}, {}
    crit_hist = []
    traj_fh = None
    if trajectory_path:
        os.makedirs(os.path.dirname(os.path.abspath(trajectory_path)), exist_ok=True)
        # "w", not "a": a resumed or rerun training writes a NEW trajectory, and appending
        # would interleave two runs' epochs in one file with nothing to separate them -- the
        # best-epoch argmin would then be taken over a mixture.
        traj_fh = open(trajectory_path, "w", encoding="utf-8")
        print(f"[desk] per-epoch trajectory -> {trajectory_path}", flush=True)
    if selection_metric not in ("val_zmse", "val_kernel"):
        raise ValueError(f"desk.selection_metric must be 'val_zmse' or 'val_kernel'; "
                         f"got {selection_metric!r}")
    # Is there a validation set AT ALL? A no-holdout production retrain has none by design, and
    # for it "nothing to select on" is the expected state, not an error -- the final weights are
    # kept and the stopping epoch comes from the sweep. Distinguishing that from a WIRING failure
    # matters both ways: raising on the deliberate case makes val_kernel unusable as the default
    # (the production run would refuse to start), and staying silent on the accidental case means
    # selecting on NaN, where every comparison is False and the warmup epoch's weights are kept
    # while the log still reports a finished run.
    _has_val = any(bool(va.any()) for _y, (_z, _t, va, _w) in tgt.items())
    if selection_metric == "val_kernel" and not val_pools.get("pool"):
        if _has_val:
            raise ValueError(
                "selection_metric='val_kernel' and there ARE validation cells, but the "
                "validation kernel pool is empty or not wired -- so there is nothing to select "
                "on and selecting on a NaN would silently keep the warmup epoch's weights. "
                "Pass val_metric_pool, or select on val_zmse.")
        print("[desk] no validation cells, so no epoch can be selected on any metric. The "
              "FINAL weights are kept; the stopping epoch must come from desk.stop_at_epoch, "
              "chosen on the sweep grid. This run has no measured skill of its own.", flush=True)
    print(f"[desk] epoch selection on {selection_metric} "
          f"({'held-out kernel term -- what the population model consumes' if selection_metric == 'val_kernel' else 'held-out z-MSE -- the historical signal'})",
          flush=True)
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
        loss_true = spacetime_kernel_loss(z_ema, pool_t, pool_flat, pool_x, pool_w=pool_w_t,
                                          num_pairs=int(metric_pairs))
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
        # Named, because three things downstream depend on it: the metric line, the JSONL row,
        # and epoch selection. Recomputing the condition at each of those is how they drift.
        evaluated = (ep % max(1, int(eval_every)) == 0 or ep == epochs)
        if evaluated:
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
                # THE selection candidate: the kernel term on held-out cells. Same estimand
                # as the training term (shared _pair_kernel_loss), on fixed pairs, under
                # no_grad, on the clean unmasked forward -- so it is comparable epoch to
                # epoch and cannot influence a weight.
                # Mean AND spread across independent draws. The mean is the selection signal
                # (standard error down by sqrt(n_draws)); the spread is the estimator's own noise
                # floor, which is the number that says whether a between-configuration difference
                # is resolvable at all. Without it, an 8% spread across 17 configurations cannot
                # be told from sampling error.
                def _pool_stats(v, zq, rank=None):
                    if v is None:
                        return float("nan"), float("nan")
                    t_, f_, x_, draws = v
                    vals = [float(kernel_loss_on_pairs(zq, t_, f_, x_, d, rank=rank))
                            for d in draws]
                    return (float(np.mean(vals)),
                            float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)

                vk, vk_sd = {}, {}
                for _k, _v in val_pools.items():
                    vk[_k], vk_sd[_k] = _pool_stats(_v, z_eval)
                # Rank curve, diagnostic only: what a downstream truncating to r would get, on
                # the supervised z_ema and on the z_raw the cube actually exports. Selection
                # stays on the full-rank z_ema value above.
                # Eigenbasis diagnostics: is Z the ORDERED eigenbasis, or merely a basis
                # reproducing the kernel? Every other metric here is a function of dot products
                # and so is blind to Z -> ZQ, which is exactly what breaks the downstream's
                # positional truncation. Computed on detached numpy, off-graph by construction,
                # on a cadence because the Ružička block is O(B^2 S).
                eig = {}
                if eig_batch is not None and eigenbasis_every and (
                        ep % int(eigenbasis_every) == 0 or ep == epochs):
                    from .eigenbasis_diag import eigenbasis_report, ruzicka_gram
                    _zf = z_eval.reshape(z_eval.shape[0], -1, z_eval.shape[-1])
                    _reps = []
                    for _bi, _b in enumerate(eig_batch):
                        if _bi not in _eig_gram:
                            _eig_gram[_bi] = ruzicka_gram(_b["x"])
                        _zb = _zf[_b["t"], _b["flat"]].detach().cpu().numpy()
                        _reps.append(eigenbasis_report(
                            _zb, _b["x"], z_ref=_b["ref"],
                            ranks=tuple(int(r) for r in eval_kernel_ranks),
                            gram=_eig_gram[_bi]))
                    _rep = _reps[0]
                    _gaps = [r["nesting_gap"] for r in _reps if "nesting_gap" in r]
                    eig = {
                        "eig_max_offdiag": _rep["orthogonality"]["max_offdiag"],
                        "eig_offdiag_mean": _rep["orthogonality"]["mean_abs_offdiag"],
                        "eig_spectrum_descending": _rep["spectrum"]["descending"],
                        "eig_spectrum_inversions": _rep["spectrum"]["inversions"],
                        "eig_first_inversion": _rep["spectrum"]["worst_inversion_at"],
                        "eig_estimator_disagreement":
                            _rep["spectrum"].get("estimator_disagreement", float("nan")),
                        "eig_nesting": _rep["nesting"]["nesting_loss"],
                        "eig_nesting_ratio": _rep["nesting"]["operator_metric_ratio"],
                    }
                    if _gaps:
                        # Mean over independent batches, plus their spread: the same treatment the
                        # kernel metric gets, so a between-configuration difference in the gap can
                        # be judged against the gap's own sampling noise instead of assumed real.
                        eig["eig_nesting_gap"] = float(np.mean(_gaps))
                        eig["eig_nesting_gap_sd"] = (float(np.std(_gaps, ddof=1))
                                                     if len(_gaps) > 1 else 0.0)
                        eig["eig_nesting_gap_draws"] = len(_gaps)
                        eig["eig_nesting_ref"] = _rep["nesting_ref"]["nesting_loss"]
                        eig.update({f"eig_subspace_r{r}": v
                                    for r, v in _rep["subspace_vs_ref"].items()})
                rank_curve = {}
                for _r in (int(r) for r in eval_kernel_ranks):
                    rank_curve[f"ema_r{_r}"] = _pool_stats(val_pools.get("pool"), z_eval,
                                                           rank=_r)[0]
                    rank_curve[f"raw_r{_r}"] = _pool_stats(val_pools.get("pool"), z_raw_eval,
                                                           rank=_r)[0]
                # The train pool scored the SAME way (fixed pairs, clean forward). Without it
                # the only kernel number was the training term measured under dropout and
                # input masking on a fresh draw, which is not on the same footing as the val
                # figure and cannot support a train/val gap.
                tk = (float(np.mean([float(kernel_loss_on_pairs(z_eval, pool_t, pool_flat,
                                                               pool_x, d))
                                    for d in train_eval_pairs]))
                      if train_eval_pairs is not None else float("nan"))
        dt = time.perf_counter() - t_ep
        rss = _max_rss_gib()
        vram = (f" | VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f}/"
                f"{torch.cuda.memory_reserved() / 2**30:.2f}G" if device == "cuda" else "")
        # The ratio/rotation/calibration columns and the main log line all read names bound
        # ONLY inside the eval branch (rotP, vs, ts, ...). With eval_every > 1 this whole block
        # raised UnboundLocalError on the first un-evaluated epoch, so the knob the config
        # documents as "amortizes the clean eval re-forward" could not be used at all. Gated
        # rather than pre-bound: pre-binding would print the PREVIOUS eval's numbers under this
        # epoch's number, which is a stale value that reads as a fresh one.
        if evaluated:
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
            # The kernel line is separate rather than appended: the line above is already at the
            # terminal width, and these are the numbers a sweep is read on, so they should not be
            # the ones that scroll off.
            #
            # Labels are parenthesis-free `k_*` tokens, NOT the `va(...)` family the z-MSE line
            # uses. Reusing those made every existing log regex match BOTH lines and return a
            # mixture of two different quantities; a `kva(...)` prefix did not fix it either, since
            # `va\(sp\+t\)` still matches inside `kva(sp+t)`. Caught by a test that greps the
            # z-MSE token, and the reason to keep the two token families disjoint by construction
            # rather than relax the pattern.
        else:
            # No eval this epoch: report only what was actually measured -- the training losses
            # and the step cost. Deliberately NOT the previous eval's metrics.
            print(f"Ep {ep:03d} | Stab {loss_stab.item():.4f} | True {loss_true.item():.4f} | "
                  f"Rec {recon_loss.item():.4f} | (no eval this epoch, eval_every="
                  f"{max(1, int(eval_every))}) | "
                  f"half-life {ema.half_life().item():.1f}y | "
                  f"lr {opt.param_groups[0]['lr']:.2e} | {dt:.1f}s{vram} | RSS {rss:.1f}G",
                  flush=True)
        if evaluated:
            print(f"Ep {ep:03d} | kernel k_tr {tk:.5f} k_val {vk.get('pool', float('nan')):.5f} "
                  f"+-{vk_sd.get('pool', float('nan')):.5f} "
                  f"k_val_sp {vk.get('sp', float('nan')):.5f} "
                  f"k_val_spt {vk.get('spt', float('nan')):.5f}", flush=True)
            if eig:
                print(f"Ep {ep:03d} | eigbasis ordered={eig['eig_spectrum_descending']} "
                      f"inv={eig['eig_spectrum_inversions']} "
                      f"offdiag={eig['eig_max_offdiag']:.3f} "
                      f"disagree={eig['eig_estimator_disagreement']:.3f} "
                      f"nest={eig['eig_nesting']:+.4f}"
                      + (f" gap={eig['eig_nesting_gap']:+.4f}"
                         if "eig_nesting_gap" in eig else " (no ESK ref)"), flush=True)
        # Written EVERY epoch, unconditionally. Every trajectory analysed so far was recovered
        # by regex-parsing job logs, which only works while the log survives and while the
        # print format holds; a JSONL costs nothing and is the artifact the sweep reads.
        # Only EVALUATED epochs get a row. An un-evaluated epoch has no measurement, and the
        # eval-only names still hold the PREVIOUS eval's values -- writing those under this
        # epoch's number would be a stale value that looks like a fresh one, which is the
        # failure mode this project has been burned by most. It is also why every eval-only
        # name is read here and nowhere outside this guard: with eval_every > 1 the whole block
        # raised UnboundLocalError on epoch 1, since `vs`, `ts`, `rotP` and the rest are bound
        # only inside the eval branch.
        if traj_fh is not None and evaluated:
            traj_fh.write(json.dumps({
                "epoch": int(ep),
                "loss_total": float(loss.item()),
                "loss_stab": float(loss_stab.item()),
                "loss_metric": float(loss_true.item()),
                "loss_recon": float(recon_loss.item()),
                "kernel_train": tk,
                "kernel_val": vk.get("pool", float("nan")),
                "kernel_val_sp": vk.get("sp", float("nan")),
                "kernel_val_spt": vk.get("spt", float("nan")),
                # The estimator's own noise, so a between-configuration gap can be compared
                # against the noise of the number the gap is measured in.
                "kernel_val_sd": vk_sd.get("pool", float("nan")),
                "kernel_val_sp_sd": vk_sd.get("sp", float("nan")),
                "kernel_val_draws": int(max(1, int(eval_kernel_draws))),
                **{f"kernel_val_{k}": v for k, v in rank_curve.items()},
                **eig,
                "zmse_train": ts, "zmse_val": vs,
                "zmse_val_sp": vs_sp, "zmse_val_spt": vs_spt,
                "zmse_val_anchor": vs_anchor, "zmse_val_yearout": vs_time,
                "rot_ratio_train": rr, "rot_ratio_val": rrv, "rot_ratio_withheld": rrh,
                "dcos_train": dcT, "dcos_val": dcV, "dcos_withheld": dcH,
                "mag_train": mgT, "mag_val": mgV,
                "half_life": float(ema.half_life().item()),
                "lr": float(opt.param_groups[0]["lr"]),
                "epoch_seconds": float(dt),
                "selection_metric": selection_metric,
                # so a reader knows the row spacing rather than assuming consecutive epochs
                "eval_every": int(max(1, int(eval_every))),
            }) + "\n")
            traj_fh.flush()          # a killed job must still leave a readable trajectory
        if ep <= es_warmup:
            continue                       # don't let the volatile warmup epochs set 'best'
        if not evaluated:
            # No fresh measurement, so nothing to select on and nothing to count toward
            # patience: doing either would compare this epoch's weights against the previous
            # eval's score.
            continue
        # ONE selection signal, named in the config. z-MSE stays logged either way so a run
        # under the new metric is still comparable against every run made under the old one.
        crit_raw = vs if selection_metric == "val_zmse" else vk.get("pool", float("nan"))
        # Optional trailing-median smoothing of the SELECTION signal only (the logged and
        # recorded per-epoch values stay raw). The argmin of a noisy series is not a property of
        # the model: measured on a 30-epoch run, kernel_val swung 2.9x between adjacent epochs
        # while the LR was near peak, and the selected epoch sat 24% below the median of the
        # converged region with its immediate neighbour 3.1x higher -- an isolated spike. Ranking
        # configurations by each one's best value then partly ranks which got the luckier spike.
        #
        # OFF by default because the noise is LR-driven and largely self-correcting: once the
        # cosine decayed below ~4e-4 the spread fell to 1.017x, so at the full 500-epoch budget
        # the argmin should land in the stable tail on its own. Turn it on if a full-length run
        # shows otherwise -- which is a thing to measure, not to assume in either direction.
        # min_delta is an ABSOLUTE epsilon and the two selection metrics live on scales two
        # orders of magnitude apart: 1e-4 is 0.05% of a val z-MSE (~0.2) and 1.4% of a val
        # kernel (~0.007). At the old 1e-4 default, moving selection to the kernel silently
        # turned "the best epoch" into "the first epoch within 1e-4 of the best", so 11 of 17
        # stage-1 runs recorded a best_epoch that was not the argmin and a best value that was
        # not the minimum -- and the cross-configuration ranking built on those values was
        # comparing a mix of true minima and early-stopped ones. Default 0 makes selection a
        # pure argmin, which is what it has to be while `patience` equals `epochs` and early
        # stopping cannot fire at all; a nonzero value is only meaningful with live early
        # stopping, and then it must be scaled to the metric.
        crit_hist.append(crit_raw)
        w = int(selection_smooth)
        crit = (float(np.median(crit_hist[-w:])) if w > 1 and len(crit_hist) >= w
                else crit_raw)
        if np.isfinite(crit) and crit < best_val - min_delta:
            best_val, bad, best_epoch = crit, 0, int(ep)
            best_zmse, best_kernel = vs, vk.get("pool", float("nan"))
            best_crit_raw = crit_raw
            best = ({k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    {k: v.detach().cpu().clone() for k, v in ema.state_dict().items()})
        else:
            bad += 1
            if bad >= patience:
                print(f"[desk] early stop at ep {ep} (best {selection_metric} "
                      f"{best_val:.5f} at ep {best_epoch})"); break
        # Halt WITHOUT having shortened the schedule. Lowering `epochs` to this value instead
        # would re-parameterise the cosine anneal (_warmup_cosine takes the BUDGET, not the
        # stopping point), changing the learning rate at every preceding step -- a different
        # model, not the same one stopped earlier.
        if stop_at_epoch is not None and ep >= int(stop_at_epoch):
            print(f"[desk] stop_at_epoch={int(stop_at_epoch)} reached; the LR schedule was "
                  f"parameterised on the full {epochs}-epoch budget and is unchanged",
                  flush=True)
            break
    if traj_fh is not None:
        traj_fh.close()
    if best is not None:
        model.load_state_dict({k: v.to(device) for k, v in best[0].items()})
        ema.load_state_dict({k: v.to(device) for k, v in best[1].items()})
        print(f"[desk] restored best epoch {best_epoch} "
              f"({selection_metric}={best_val:.5f}; val z-MSE {best_zmse:.4f}, "
              f"val kernel {best_kernel:.5f})", flush=True)
    else:
        # No checkpoint was ever taken: there is no validation set (the production retrain
        # holds nothing out) or every epoch was non-finite. Keep the FINAL weights rather
        # than failing on a None -- which is what this did, so a no-holdout run could not
        # finish at all -- and say which case it was, because "no val set" is a deliberate
        # configuration and "nothing ever improved" is a broken run.
        best_epoch = int(ep)
        why = ("no validation cells, so no epoch could be scored -- this is the expected path "
               "for a no-holdout production retrain" if not any(bool(va.any())
                                                                for _, (_z, _t, va, _w) in tgt.items())
               else "no epoch improved on the selection metric after the warmup")
        print(f"[desk] keeping the FINAL weights from epoch {best_epoch}: {why}", flush=True)
    info = {"best_epoch": best_epoch, "best_selection_value": float(best_val),
            "best_val_zmse": float(best_zmse), "best_val_kernel": float(best_kernel),
            "selection_metric": str(selection_metric),
            "epochs_run": int(ep), "epochs_budget": int(epochs),
            "stop_at_epoch": (None if stop_at_epoch is None else int(stop_at_epoch)),
            "trajectory_path": (str(trajectory_path) if trajectory_path else None),
            "eval_kernel_pairs": int(eval_kernel_pairs),
            # Recorded so a reader can tell whether the run is entitled to a statement about
            # where the optimum lies. A budget only a little longer than the warmup is almost
            # entirely LR ramp, and the minimum it reports is a transient of the schedule rather
            # than a property of the model.
            "warmup_epochs": int(warmup_epochs),
            "selection_smooth": int(selection_smooth),
            "best_selection_raw": float(best_crit_raw),
            "min_lr_frac": float(min_lr_frac),
            "restored_best": best is not None}
    return (model, ema, info) if return_info else (model, ema)



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
    # A FLOOR on that derivation, never a narrower buffer. The kernel is a swept knob, and
    # kernel//2 makes the buffer move with it (0/0/1/2 for kernel 0/1/3/5) -- so each variant
    # would be graded on a different split, with different training-cell counts and different
    # val/train separation. kernel=0 in particular would put val cells directly adjacent to
    # training cells, which is exactly the optimism the buffer exists to remove, and its
    # held-out score would not be comparable with kernel=5's. Pinning the floor at the widest
    # kernel in a sweep gives every configuration identical held-out regions. null keeps the
    # historical derived-only behaviour, so no existing run changes.
    _floor = _cfg.get("buffer_floor")
    if _floor is not None:
        if int(_floor) < spatial_kernel // 2:
            raise SystemExit(
                f"desk.trend.buffer_floor={int(_floor)} is NARROWER than this kernel needs "
                f"({spatial_kernel}//2 = {spatial_kernel // 2}). A val cell's receptive field "
                f"would reach training cells and the held-out metric would flatter itself. "
                f"The floor may only widen the buffer.")
        buf = max(buf, int(_floor))
        print(f"[desk] buffer width {buf} cells = max(kernel//2={spatial_kernel // 2}, "
              f"buffer_floor={int(_floor)}) -- pinned so every swept kernel is graded on the "
              f"SAME held-out regions", flush=True)
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
    # train_frac subsamples the TRAINING blocks after the split is drawn, so the amount of
    # training data can be varied while the validation set stays byte-identical. Doing it with
    # holdout_frac instead -- the obvious reading -- moves the val set too, and then each
    # column of a data-amount sweep reports its metric on a different, differently-sized set
    # of held-out cells, which is not a trajectory. Whole blocks, not scattered cells: a
    # thinned-at-random training set leaves every val block still ringed by training data at
    # the same distance, so it would reduce the count without reducing the reach.
    train_frac = _cfg.get("train_frac")
    train_drop = np.zeros_like(holdout)
    if train_frac is not None and float(train_frac) < 1.0:
        tf = float(train_frac)
        if not 0.0 < tf < 1.0:
            raise SystemExit(f"desk.trend.train_frac must be in (0,1] or null; got {tf}")
        avail = split_domain & (~holdout) & (~buffer_cells_mask)
        _b = max(1, block)
        _nby, _nbx = (avail.shape[0] + _b - 1) // _b, (avail.shape[1] + _b - 1) // _b
        # A SEPARATE rng stream from the split's, so changing train_frac cannot perturb which
        # blocks were held out -- the whole point of doing this after the split.
        _rng = np.random.default_rng(int(_cfg.get("seed", 0)) + 104729)
        _drop = _rng.random((_nby, _nbx)) >= tf
        train_drop = (np.repeat(np.repeat(_drop, _b, axis=0), _b, axis=1)[:avail.shape[0],
                                                                         :avail.shape[1]]
                      & avail)
        print(f"[desk] train_frac={tf:g}: dropped {int(train_drop.sum())} of "
              f"{int(avail.sum())} available training cells "
              f"({1 - int(train_drop.sum()) / max(int(avail.sum()), 1):.3f} retained) by whole "
              f"{_b}x{_b} blocks; the validation set is UNCHANGED", flush=True)
    min_val = int(_cfg.get("min_val_cells", 200))
    if float(_cfg.get("holdout_frac", 0.15)) == 0.0:
        # The production retrain holds nothing out on purpose, so there is no val number to
        # protect and the floor would simply refuse to run. Say so explicitly: a run with no
        # validation set has NO measured skill of its own -- its expected skill is inferred
        # from the sweep grid it was configured from, and must be reported that way.
        min_val = 0
        print("[desk] holdout_frac=0: NO VALIDATION SET. Epoch selection is unavailable, the "
              "final weights are kept, and this run has no measured skill -- infer it from "
              "the nearest sweep grid point and state that it was inferred.", flush=True)
    if n_val < min_val:
        raise SystemExit(
            f"only {n_val} validation cells (floor {min_val}). Every reported val number, "
            f"and best-epoch selection itself, would rest on too few cells to mean anything. "
            f"Raise desk.trend.holdout_frac, lower block_cells, or lower "
            f"desk.trend.min_val_cells if you accept the noise.")

    # Normalization stats: fit on the supervised TRAINING pixels only (buffer cells are excluded
    # as well -- not evaluation data, but not clean training data either), then applied to every
    # grid (labelled + historical) and frozen for the cube.
    # train_drop is excluded too: a cell the objective never sees must not contribute to the
    # statistics every input is standardized by, or the smaller-training-set columns of a
    # data-amount sweep would still carry the full set's normalization and the axis would be
    # only partly varied.
    fit_mask = mask_sup0 & (~holdout) & (~buffer_cells_mask) & (~train_drop)
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
    m_tr = mask_sup & (~holdout) & (~buffer_cells_mask) & (~train_drop)
    m_val = mask_sup & holdout

    # Computed here, above the pool, because the pool must exclude these years -- see
    # spacetime_metric_pool. The anchor is never withheld; it carries the metric loss.
    ho_years = [int(y) for y in (tr_cfg.get("holdout_years") or []) if int(y) != label_year]
    _py, _pf, _px, _ppidx = spacetime_metric_pool(pip, Xp, sup_rows, m_tr, W,
                                                  exclude_years=ho_years, return_pidx=True)
    metric_pool = (_py, _pf, _px)
    # The SAME builder on the val mask. exclude_years is deliberately NOT passed: a withheld
    # year is exactly what the sp+t half of the val kernel metric has to score, and excluding
    # it here would leave the temporal question unmeasurable. The years are carried through so
    # the trainer can split the pool rather than having to re-derive which is which.
    val_metric_pool = spacetime_metric_pool(pip, Xp, sup_rows, m_val, W, exclude_years=())
    # ESK's own projection of the val pool's communities: the reference the eigenbasis
    # diagnostic's gap is measured against. ESK is an explicit eigenvalue-descending
    # decomposition (see esk_kernel), so it IS the ordered eigenbasis up to Nyström error, and its
    # diagnostic values are therefore the achievable floor on this batch. Without a reference the
    # nesting scalar is uninterpretable -- it scales with the kernel and the batch -- so a missing
    # projection disables the gap rather than being silently replaced by a constant.
    eig_ref = None
    if int(desk_cfg.get("eigenbasis_batch", 0)):
        from .esk_kernel import project_points_to_z as _proj
        eig_ref = _proj(val_metric_pool[2], z_dir, latent_dim)
        if eig_ref is None:
            print("[desk] eigenbasis diagnostic: no saved ESK projection in z_dir, so the "
                  "subspace curve and nesting GAP are UNAVAILABLE_NO_ESK_REFERENCE; the "
                  "reference-free parts (spectrum, orthogonality) still run", flush=True)
        else:
            print(f"[desk] eigenbasis reference: ESK projection of {len(eig_ref):,} val "
                  f"communities into the pinned basis", flush=True)

    print(f"[desk] val metric pool: {len(val_metric_pool[0]):,} HELD-OUT cell-years spanning "
          f"{len(np.unique(val_metric_pool[0])) if len(val_metric_pool[0]) else 0} years "
          f"({int(np.isin(val_metric_pool[0], ho_years).sum()):,} in withheld years)")
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
                                     points_dir=points_dir,
                                     exclude=buffer_cells_mask | train_drop)
    _n_tr_tgt = int(sum(int(tr.sum()) for _y, (_z, tr, _v, _w) in targets.items()))
    print(f"[desk] stabilizing target: {_n_tr_tgt:,} training cell-years after excluding the "
          f"buffer ({int(buffer_cells_mask.sum())} cells) and any train_frac thinning "
          f"({int(train_drop.sum())} cells) -- the same exclusions the metric pool and the "
          f"normalization fit already apply", flush=True)

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

    model, ema, train_info = train_model_ema(
        cov_win, mask_win, kept, targets, metric_pool, m_tr, m_val,
        stream_dims, latent_dim=latent_dim, ema_cfg=ema_cfg,
        spatial_kernel=spatial_kernel,
        epochs=desk_cfg.get("epochs", 500), lr=desk_cfg.get("lr", 1e-3),
        # The TRAINING seed: model init, dropout masks, augmentation draws, and the metric loss's
        # pair sampling. It was never passed, so it took train_model_ema's default of 0 in every
        # run ever made -- meaning a training run could not be replicated, and the training-init
        # noise floor was unmeasurable. desk.trend.seed is a DIFFERENT knob: it draws the spatial
        # split, so varying it changes which cells are held out and therefore what the metric is
        # measured on. Conflating the two made a 12% spread across trend.seed look like training
        # noise when it is evaluation-set variation.
        seed=int(desk_cfg.get("seed", 0)),
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
        hidden_width=hidden_width, mlp_expansion=mlp_expansion,
        val_metric_pool=val_metric_pool, val_pool_holdout_years=ho_years,
        metric_pairs=int(desk_cfg.get("metric_pairs", 4096)),
        eval_kernel_pairs=int(desk_cfg.get("eval_kernel_pairs", 65536)),
        eval_kernel_draws=int(desk_cfg.get("eval_kernel_draws", 1)),
        eval_kernel_ranks=tuple(desk_cfg.get("eval_kernel_ranks") or ()),
        eigenbasis_batch=int(desk_cfg.get("eigenbasis_batch", 0)),
        eigenbasis_every=int(desk_cfg.get("eigenbasis_every", 0)),
        eigenbasis_ref=eig_ref,
        eigenbasis_draws=int(desk_cfg.get("eigenbasis_draws", 1)),
        min_delta=float(desk_cfg.get("min_delta", 0.0)),
        selection_metric=str(desk_cfg.get("selection_metric", "val_zmse")),
        selection_smooth=int(desk_cfg.get("selection_smooth", 0)),
        trajectory_path=os.path.join(out_dir, "train_trajectory.jsonl"),
        stop_at_epoch=desk_cfg.get("stop_at_epoch"),
        return_info=True)
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
             # Scalar when every branch shares a width, a per-stream ARRAY when they do not.
             # Scalar for the uniform case on purpose: an existing uniform run's meta stays
             # byte-comparable, and any reader that has not been taught about the list form
             # still loads it. hidden_width_from_meta reads both.
             hidden_width=(np.int64(model.hidden_widths[0])
                           if model.hidden_width is not None
                           else np.array(model.hidden_widths, dtype=np.int64)),
             mlp_expansion=int(model.mlp_expansion),
             # The best epoch was recorded NOWHERE. The weights were restored from it, but the
             # only print sat in the early-stop branch, which cannot fire while patience equals
             # epochs (both 500) -- so every run's chosen epoch had to be inferred from a log,
             # and the production retrain needs it as an input.
             best_epoch=int(train_info["best_epoch"]),
             selection_metric=str(train_info["selection_metric"]),
             best_selection_value=float(train_info["best_selection_value"]),
             best_val_zmse=float(train_info["best_val_zmse"]),
             best_val_kernel=float(train_info["best_val_kernel"]),
             epochs_run=int(train_info["epochs_run"]),
             epochs_budget=int(train_info["epochs_budget"]),
             restored_best=bool(train_info["restored_best"]),
             buffer_floor=(-1 if _cfg.get("buffer_floor") is None
                           else int(_cfg["buffer_floor"])),
             train_frac=float(_cfg.get("train_frac") or 1.0),
             train_cells=int(m_tr.sum()), val_cells=int(m_val.sum()),
             train_dropped_cells=int(train_drop.sum()),
             augment=json.dumps(config.get("augment") or {}),
             holdout_cells=int(holdout.sum()), buffer_cells=int(buffer_cells_mask.sum()),
             schema=json.dumps(schema),
             output_ema=True,
             ema_half_life=(ema_half_life if ema_half_life is not None else np.nan),
             ema_warmup_start=int(ema_cfg.get("warmup_start", 1940)),
             kernel=str(z_meta["kernel"]), centered=bool(z_meta["centered"]),
             kernel_contract=str(z_meta.get("kernel_contract", "")))
    # The manifest a sweep is read from: one small JSON per run, so the analysis never has to
    # open a .npz or parse a log. Written LAST, so its presence is what "this run finished"
    # means -- which is what makes the submit script's resume able to skip it safely.
    def _jsonable(v):
        """Non-finite floats -> None. ``json.dumps`` emits bare ``NaN``/``Infinity``, which
        Python reads back but strict JSON parsers (jq, most other languages) reject -- and a
        summary nobody outside Python can read defeats the point of writing one."""
        return None if isinstance(v, float) and not np.isfinite(v) else v

    with open(os.path.join(out_dir, "run_summary.json"), "w", encoding="utf-8") as fh:
        json.dump({**{k: _jsonable(v) for k, v in train_info.items()},
                   "desk_output_dir": out_dir,
                   "spatial_kernel": int(spatial_kernel),
                   "buffer_cells": int(buffer_cells_mask.sum()),
                   "hidden_widths": [int(v) for v in model.hidden_widths],
                   "mlp_expansion": int(model.mlp_expansion),
                   "dropout": float(desk_cfg.get("dropout", 0.5)),
                   "weight_decay": float(desk_cfg.get("weight_decay", 0.0)),
                   "metric_weight": _jsonable(float((desk_cfg.get("weights") or {})
                                                    .get("metric", float("nan")))),
                   # The settings that decide whether two runs are COMPARABLE, as opposed to
                   # merely both finished. metric_pairs changes the gradient's variance and so
                   # the optimization trajectory; the eval settings change the estimator the
                   # ranking is built from. Recorded so a resume can refuse to mix them.
                   "metric_pairs": int(desk_cfg.get("metric_pairs", 4096)),
                   "eval_kernel_pairs": int(desk_cfg.get("eval_kernel_pairs", 65536)),
                   "eval_kernel_draws": int(desk_cfg.get("eval_kernel_draws", 1)),
                   "half_life_bounds": list(ema_cfg.get("half_life_bounds", [1.0, 40.0])),
                   "ema_half_life": _jsonable(ema_half_life),
                   "holdout_frac": float(_cfg.get("holdout_frac", 0.15)),
                   "train_frac": float(_cfg.get("train_frac") or 1.0),
                   "holdout_years": sorted(ho_years),
                   "n_train_cells": int(m_tr.sum()), "n_val_cells": int(m_val.sum()),
                   "n_train_cell_years": int(len(metric_pool[0])),
                   "n_val_cell_years": int(len(val_metric_pool[0])),
                   "n_params": int(sum(p.numel() for p in model.parameters())),
                   "sweep": config.get("_sweep") or None},
                  fh, indent=2)
    print(f"[desk] saved model + desk_meta.npz + run_summary.json -> {out_dir} "
          f"(spatial_kernel={spatial_kernel}, best epoch {train_info['best_epoch']} on "
          f"{train_info['selection_metric']})")


if __name__ == "__main__":
    run_desk_experiment()
