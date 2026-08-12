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


def compute_valid_mask(ebird_stack, cov_stack, z_mask):
    """Intersect finite eBird, finite covariates (all channels), and the ESK-Z mask."""
    m_ebird = np.any(~np.isnan(ebird_stack), axis=-1)
    m_cov = np.all(~np.isnan(cov_stack), axis=-1)
    final = m_ebird & m_cov & z_mask
    print(f"[mask] eBird {m_ebird.sum()} & cov {m_cov.sum()} & Z {z_mask.sum()} "
          f"-> {final.sum()} supervised pixels")
    return final


def true_kernel_loss(z_pred, x_raw, num_pairs=4096):
    """MSE between the dot product in Z and the Ruzicka similarity in raw X, over
    ``num_pairs`` random pairs drawn from the supplied (valid) pixel set."""
    B = z_pred.shape[0]
    if B < 2:
        return torch.tensor(0.0, device=z_pred.device, requires_grad=True)
    idx = torch.randint(0, B, (2, num_pairs), device=z_pred.device)
    i, j = idx[0], idx[1]
    xi, xj = x_raw[i], x_raw[j]
    sum_plus = xi + xj
    diff_abs = torch.abs(xi - xj)
    numerator = 0.5 * torch.sum(sum_plus - diff_abs, dim=1)
    denominator = 0.5 * torch.sum(sum_plus + diff_abs, dim=1)
    valid = denominator > 1e-3
    if valid.sum() == 0:
        return torch.tensor(0.0, device=z_pred.device, requires_grad=True)
    sim_true = numerator[valid] / (denominator[valid] + 1e-8)
    zi, zj = z_pred[i][valid], z_pred[j][valid]
    sim_pred = (zi * zj).sum(dim=1)
    return F.mse_loss(sim_pred, sim_true)


def prepare_supervised(cov_stack, ebird_stack, z_flat, z_mask, mu, sd, out_dir):
    """Build the labelled year's grid tensors: normalized covariate grid + cov mask,
    supervised mask (eBird & cov & Z), ESK-Z grid, and raw eBird grid."""
    H, W, _ = cov_stack.shape
    mask_sup = compute_valid_mask(ebird_stack, cov_stack, z_mask)
    np.save(os.path.join(out_dir, "training_mask.npy"), mask_sup)
    covn, mask_cov = cio.norm_grid(cov_stack, mu, sd)
    z_grid = np.zeros((H, W, z_flat.shape[1]), dtype="float32")
    z_grid[z_mask] = z_flat
    x_grid = np.nan_to_num(ebird_stack, nan=0.0).astype("float32")
    return covn, mask_cov, mask_sup, z_grid, x_grid


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


def _prepare_trend_targets(config, z_dir, latent_dim, holdout):
    """Per-year ESK-basis targets for EVERY supervised year, from the trend points.

    Projects ``X_points`` into the joint ESK basis (z_obs) and scatters each point into
    its year's grid; returns ``{year: (zg (H,W,L), tm_tr (H,W), tm_val (H,W))}`` where the
    train/val split is the spatial ``holdout`` (val = held-out cells). Includes 2023.
    """
    from .esk_kernel import project_points_to_z
    zt = config["trend"]["points_dir"]
    X = np.load(os.path.join(zt, "X_points.npy"))
    pidx = np.load(os.path.join(zt, "point_index.npy"))
    z_obs = project_points_to_z(X, z_dir, latent_dim)
    if z_obs is None:
        raise FileNotFoundError(f"trend targets need the ESK projection in {z_dir}; re-run spacetime-esk")
    rows, cols, yrs = pidx[:, 0], pidx[:, 1], pidx[:, 2]
    H, W = holdout.shape
    out = {}
    for y in sorted({int(v) for v in yrs}):
        sel = np.where(yrs == y)[0]
        zg = np.zeros((H, W, latent_dim), dtype="float32")
        present = np.zeros((H, W), bool)
        zg[rows[sel], cols[sel]] = z_obs[sel]
        present[rows[sel], cols[sel]] = True
        out[y] = (zg, present & (~holdout), present & holdout)
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


def spatial_interp_baseline(tgt, k=8, power=2.0):
    """Predict held-out cells by inverse-distance interpolation of the TARGETS themselves.

    The existing no-skill baselines (predict-mean, predict-zero) ignore geography, but Z is
    strongly spatially autocorrelated, so a model that merely interpolates its neighbours
    already scores well. This is the honest reference for a spatially BLOCKED holdout: it uses
    no covariates and no learning, only "the answer at nearby training cells". If the trained
    model's val error is not clearly below this, the covariates are adding nothing beyond
    spatial smoothness, and further capacity or regularization work is pointless.

    The blocked split plus its buffer put every val cell at least ``kernel//2 + 1`` cells from
    any training cell, so this baseline is genuinely extrapolating into the block interior --
    which is exactly the task the model faces.

    Returns ``(nearest_mse, idw_mse)`` on the same per-cell summed scale as ``_z_mse``.

    The per-year masks matter and must not be shortcut. ``_prepare_trend_targets`` builds each
    year's grid as ``np.zeros`` and marks only the cells with a trend point that year, so a cell
    absent in year Y holds a **zero**, not a NaN, and the train/val masks are ``present &
    (~holdout)`` / ``present & holdout`` -- i.e. per-year, not constant. An earlier version of
    this function reused the first year's masks throughout, which scored interpolation against
    zero-filled targets and fed zero-filled sources into the interpolation, on a different cell
    population and a different denominator from ``_z_mse``. That made the baseline
    incomparable with the model's val. So: rebuild the neighbour index per distinct mask and
    accumulate with the per-year counts, mirroring ``_z_mse`` exactly.
    """
    from scipy.spatial import cKDTree

    years = sorted(tgt)
    if not years:
        return float("nan"), float("nan")

    def _np(a):
        return a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)

    sq_n = sq_i = 0.0
    cnt = 0
    cache = {}
    for y in years:
        zg, tr_m, va_m = tgt[y]
        tr_np, va_np = _np(tr_m), _np(va_m)
        if not va_np.any() or tr_np.sum() < k:
            continue                                       # same skip as _z_mse's m.any()
        key = (tr_np.tobytes(), va_np.tobytes())           # masks repeat across many years
        if key not in cache:
            tr_rc, va_rc = np.argwhere(tr_np), np.argwhere(va_np)
            dist, idx = cKDTree(tr_rc).query(va_rc, k=k)
            w = 1.0 / np.maximum(dist, 1e-6) ** power
            cache[key] = (tr_rc, va_rc, idx, w / w.sum(axis=1, keepdims=True))
        tr_rc, va_rc, idx, w = cache[key]

        z = _np(zg)
        src = z[tr_rc[:, 0], tr_rc[:, 1]]                  # (n_train, L), this year's values
        truth = z[va_rc[:, 0], va_rc[:, 1]]                # (n_val, L)
        pred_n = src[idx[:, 0]]                            # nearest training cell
        pred_i = np.einsum("nk,nkl->nl", w, src[idx])      # inverse-distance weighted
        sq_n += float(((pred_n - truth) ** 2).sum())
        sq_i += float(((pred_i - truth) ** 2).sum())
        cnt += len(va_rc)
    if cnt == 0:
        return float("nan"), float("nan")
    return sq_n / cnt, sq_i / cnt


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


def train_model_ema(cov_window, mask_window, window_years, targets, x2023, m2023_tr, m2023_val,
                    stream_dims, latent_dim, ema_cfg, spatial_kernel=3, epochs=500, lr=1e-3,
                    weights=None, seed=0, patience=50, min_delta=1e-4,
                    schema=None, augment_cfg=None, dropout=0.5, weight_decay=0.0,
                    warmup_epochs=0, min_lr_frac=1.0, amp=False, eval_every=1,
                    holdout_year_targets=None, hidden_width=None, mlp_expansion=4):
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
    x2023_t = torch.tensor(x2023, device=device)                  # (H,W, S) annual eBird
    m_tr = torch.as_tensor(m2023_tr, device=device).bool(); m_val = torch.as_tensor(m2023_val, device=device).bool()
    # supervised year targets that fall inside the forwarded window
    tgt = {y: (torch.tensor(zg, device=device),
               torch.as_tensor(tr, device=device).bool(), torch.as_tensor(va, device=device).bool())
           for y, (zg, tr, va) in targets.items() if y in yi}
    y2023 = int(max(tgt))                                         # anchor year index in the window

    # No-skill baselines on the held-out cells (pooled over all supervised years): the Z-MSE
    # of predicting the global mean vector, and of predicting zero. Val(all-yr) must fall well
    # below these to have any skill; Val/baseline is the fraction of held-out Z variance left
    # unexplained (targets have ||Z||^2 ~ 1 since Z.Z^T ~= Ružicka with unit self-similarity).
    with torch.no_grad():
        held = [zg[va] for _, (zg, _t, va) in tgt.items() if bool(va.any())]
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
        b_near, b_idw = spatial_interp_baseline(tgt)
        print(f"[desk] spatial-interpolation baselines (no model, no covariates): "
              f"nearest-train-cell={b_near:.4f}, inverse-distance-8={b_idw:.4f} "
              f"-- Val must beat these to justify the covariates", flush=True)
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
        for t in range(cov.shape[0]):
            xt = cov[t:t + 1]
            if mask_inputs and masker is not None:
                # New tensor: cov[t:t+1] is a VIEW of the single resident (T,H,W,C) window, so an
                # in-place mask would permanently destroy that year for every later epoch.
                xt = xt * masker.sample_keep(aug_rng, device=device,
                                             hw=(cov.shape[1], cov.shape[2]))
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
            if bool(m.any()):
                d = (z_all[yi[y]][m] - zg[m]).detach()
                sq += float(torch.sum(d * d)); cnt += int(m.sum())
        return sq / max(cnt, 1), cnt

    def _rotation(z_all, m):
        """Median ``1 - cos`` between the deepest and anchor year, predicted and target.

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
        if y_deep is None or not bool(m.any()):
            return float("nan"), float("nan")

        def med(a, b):
            num = torch.sum(a * b, dim=1)
            den = a.norm(dim=1) * b.norm(dim=1)
            ok = den > 1e-12
            if not bool(ok.any()):
                return float("nan")
            return float(torch.median(1.0 - num[ok] / den[ok]))

        zp0, zp1 = z_all[yi[y_deep]][m].detach(), z_all[yi[y2023]][m].detach()
        zt0, zt1 = tgt[y_deep][0][m], tgt[y2023][0][m]
        return med(zp0, zp1), med(zt0, zt1)

    def _dir_cos(z_all, m, perm):
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
        if y_deep is None or not bool(m.any()):
            return float("nan"), float("nan")
        dp = (z_all[yi[y2023]][m] - z_all[yi[y_deep]][m]).detach()
        dt = tgt[y2023][0][m] - tgt[y_deep][0][m]

        return median_dir_cos(dp, dt), median_dir_cos(dp, dt[perm])

    val_sel = {y: (zg, va) for y, (zg, _tr, va) in tgt.items()}
    # Train-cell selection on the SAME footing as val, evaluated on the clean forward. Without
    # it the only train-side number is `Stab`, which is (a) measured under dropout+masking and
    # so overstates the error, and (b) divided by latent_dim while Val is not -- two different
    # scales for the same quantity in one log line.
    tr_sel = {y: (zg, tr) for y, (zg, tr, _va) in tgt.items()}
    # Deepest year that still has TRAINING coverage -- not simply min(tgt). A temporally
    # held-out year stays in `tgt` with an all-False train mask (that is how it is withheld
    # from the objective), so anchoring on min(tgt) would make `rot_tr` empty and the rotation
    # and direction columns would silently go dark exactly when a temporal holdout is added.
    _cand = [y for y in sorted(tgt) if y != y2023 and bool(tgt[y][1].any())]
    y_deep = _cand[0] if _cand else None
    # Cells supervised in BOTH the deep and anchor year, split train/val -- the diagnostic
    # needs the same cell present at both ends to measure its change.
    if y_deep is not None:
        rot_tr = tgt[y_deep][1] & tgt[y2023][1]
        rot_va = tgt[y_deep][2] & tgt[y2023][2]
        # Fixed permutations so the direction null is a stable reference, not per-epoch noise.
        g_null = torch.Generator(device="cpu").manual_seed(seed)
        perm_tr = torch.randperm(int(rot_tr.sum()), generator=g_null).to(device)
        perm_va = torch.randperm(int(rot_va.sum()), generator=g_null).to(device)
        if int(min(tgt)) != y_deep:
            print(f"[desk] rotation anchor moved to {y_deep} (min supervised year "
                  f"{int(min(tgt))} has no training cells -- temporal holdout)", flush=True)
        print(f"[desk] rotation/direction diagnostic {y_deep}->{y2023}: "
              f"{int(rot_tr.sum())} train / {int(rot_va.sum())} val cells", flush=True)
    else:
        rot_tr = rot_va = torch.zeros(1, dtype=torch.bool, device=device)
        perm_tr = perm_va = torch.zeros(0, dtype=torch.long, device=device)
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
        sq_sum, n = torch.zeros((), device=device), 0
        for y, (zg, tr, _va) in tgt.items():
            zz = z_ema[yi[y]]
            s = torch.sum((zz[tr] - zg[tr]) ** 2, dim=1)
            sq_sum = sq_sum + s.sum(); n += int(tr.sum())
        loss_stab = sq_sum / max(n, 1) / latent_dim
        z_anchor = z_ema[yi[y2023]]
        loss_true = true_kernel_loss(z_anchor[m_tr], x2023_t[m_tr])
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
                va_anchor = tgt[y2023][2]
                vs_anchor = (float(torch.sum((z_eval[yi[y2023]][va_anchor]
                                              - tgt[y2023][0][va_anchor]) ** 2))
                             / max(int(va_anchor.sum()), 1)) if bool(va_anchor.any()) else float("nan")
                vs_time = (_z_mse(z_eval, holdout_year_targets)[0]
                           if holdout_year_targets else float("nan"))
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
        gap = (vs / ts) if ts > 1e-12 else float("nan")
        print(f"Ep {ep:03d} | Stab {loss_stab.item():.4f} | True {loss_true.item():.4f} | "
              f"Rec {recon_loss.item():.4f} | mse tr {ts:.4f} va {vs:.4f} gap {gap:.1f}x | "
              f"Val(2025) {vs_anchor:.4f} | Val(yr-out) {vs_time:.4f} | "
              f"rot tr {rr:.2f} va {rrv:.2f} (raw {rawr:.2f}/{rawrv:.2f}) | "
              f"dcos tr {dcT:.2f} va {dcV:.2f} null {dcTn:+.2f}/{dcVn:+.2f} | "
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
    ztz = config["trend"]["points_dir"]
    Xp = np.load(os.path.join(ztz, "X_points.npy"))
    pip = np.load(os.path.join(ztz, "point_index.npy"))
    pm = json.load(open(os.path.join(ztz, "points_meta.json")))
    ay, S = int(pm["recent_year"]), int(pm["n_species"])
    H, W = cov_stack.shape[:2]
    sel = pip[:, 2] == ay
    ebird_stack = np.full((H, W, S), np.nan, dtype="float32")
    ebird_stack[pip[sel, 0], pip[sel, 1]] = Xp[sel]                # already log1p in X_points
    log1p_kernel = bool(pm.get("ruzicka_log1p", True))
    print(f"[desk] Ružicka metric anchored on the reconstructed year-{ay} community from "
          f"X_points (anchor_mode={pm.get('anchor_mode')}, log1p={log1p_kernel})")

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

    mask_sup0 = compute_valid_mask(ebird_stack, cov_stack, z_mask)

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
    holdout, buffer_cells_mask = blocked_holdout(
        ebird_valid, block_cells=block, holdout_frac=float(_cfg.get("holdout_frac", 0.2)),
        buffer_cells=buf, seed=int(_cfg.get("seed", 0)))
    print(f"[desk] split: {block}x{block}-cell blocks, buffer {buf} cells (from "
          f"spatial_kernel={spatial_kernel}) -> {int(holdout.sum())} val, "
          f"{int(buffer_cells_mask.sum())} buffer cells", flush=True)

    # Normalization stats: fit on the supervised TRAINING pixels only (buffer cells are excluded
    # as well -- not evaluation data, but not clean training data either), then applied to every
    # grid (labelled + historical) and frozen for the cube.
    fit_mask = mask_sup0 & (~holdout) & (~buffer_cells_mask)
    mu, sd = cio.fit_norm(cov_stack[fit_mask].astype("float32"))

    covn, mask_cov, mask_sup, z_grid, x_grid = prepare_supervised(
        cov_stack, ebird_stack, z_flat, z_mask, mu, sd, out_dir)
    ema_cfg = desk_cfg["output_ema"]
    tr_cfg = desk_cfg.get("trend", {})
    # holdout/buffer were drawn above (before the normalization fit). Training excludes BOTH;
    # evaluation uses the holdout only, so buffer cells appear in neither.
    m_tr = mask_sup & (~holdout) & (~buffer_cells_mask)
    m_val = mask_sup & holdout
    np.save(os.path.join(out_dir, "holdout_cells.npy"), holdout)
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
    targets = _prepare_trend_targets(config, z_dir, z_grid.shape[2], holdout)

    # Temporal holdout: withhold a contiguous span of supervised years from the objective and
    # score them separately. The spatial split says nothing about extrapolation THROUGH TIME,
    # which is what DESK exists to do. The anchor (label_year) is never withheld -- it carries
    # the eBird metric loss. Diagnostic only; model selection stays on the spatial metric so
    # there is exactly one selection signal.
    ho_years = [int(y) for y in (tr_cfg.get("holdout_years") or []) if int(y) != label_year]
    year_val = {}
    for y in ho_years:
        if y in targets:
            zg, tr_m, va_m = targets[y]
            year_val[y] = (zg, tr_m | va_m)            # score every supervised cell that year
            targets[y] = (zg, np.zeros_like(tr_m), va_m)   # ...and train on none of it
    if year_val:
        print(f"[desk] temporal holdout years (diagnostic, excluded from the objective): "
              f"{sorted(year_val)}", flush=True)

    model, ema = train_model_ema(
        cov_win, mask_win, kept, targets, x_grid, m_tr, m_val,
        stream_dims, latent_dim=z_grid.shape[2], ema_cfg=ema_cfg,
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
        hidden_width=hidden_width, mlp_expansion=mlp_expansion)
    ema_half_life = float(ema.half_life().item())
    torch.save(ema.state_dict(), os.path.join(out_dir, "output_ema.pth"))
    print(f"[desk] output-EMA learned half-life = {ema_half_life:.2f} yr")
    torch.save(model.state_dict(), os.path.join(out_dir, "env_model_semisup.pth"))
    np.savez(os.path.join(out_dir, "desk_meta.npz"),
             mu=mu, sd=sd, stream_dims=np.array(stream_dims, int),
             latent_dim=z_grid.shape[2], label_year=label_year,
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
