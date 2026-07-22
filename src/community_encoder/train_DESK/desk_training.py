"""Train DESK: a semi-supervised autoencoder predicting ESK's Z from covariates.

DESK ("Deep ESK") learns to reconstruct the ESK kernel-PCA latent Z -- the
habitat-similarity "ground truth" (for ``off``/``validate`` this is the eBird 2023
spatial Z) -- from covariates that exist for every year, so Z can be extrapolated
across the whole timeline. Trained with three losses: a stabilizing MSE against the
ESK Z where it is known, a metric loss preserving Ruzicka-similarity relationships,
and an autoencoder reconstruction over labeled + unlabeled years.

**Grid-native.** The model (``MultiStreamAutoencoder``) maps a covariate grid
``(B,H,W,C)`` -> latent grid, so its optional spatial residual conv can see each
cell's neighbours. Training therefore operates on whole-year grids, not a shuffled
bag of pixels: the supervised losses gather the valid pixels of the single labelled
year's grid (2023), and every other year's ``state_{year}.npz`` grid feeds the
reconstruction loss (and gives the spatial conv many unlabelled spatial examples to
regularise its filters against -- the labelled signal exists for only one grid).

N-stream: reads ``state_{year}.npz`` (climate/land-use/HYDE/soil/elevation) via
``state_schema.json`` (``covariate_io``). The ``enrich`` mode's multi-year
supervised points are added in ``spacetime_enrich``.
"""
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from .config_utils import load_config
from . import covariate_io as cio
from .ebird_cache import load_ebird_stack
from .model_arch import MultiStreamAutoencoder


def compute_valid_mask(ebird_stack, cov_stack, z_mask):
    """Intersect finite eBird, finite covariates (all channels), and the ESK-Z mask."""
    m_ebird = np.any(~np.isnan(ebird_stack), axis=-1)
    m_cov = np.all(~np.isnan(cov_stack), axis=-1)
    final = m_ebird & m_cov & z_mask
    print(f"[mask] eBird {m_ebird.sum()} & cov {m_cov.sum()} & Z {z_mask.sum()} "
          f"-> {final.sum()} supervised pixels")
    return final


def _split_mask(mask, train_frac=0.8, seed=0):
    """Split a boolean grid mask's True cells into (train, val) boolean grid masks."""
    ys, xs = np.where(mask)
    g = np.random.default_rng(seed)
    perm = g.permutation(len(ys))
    cut = int(train_frac * len(ys))
    tr = np.zeros_like(mask); va = np.zeros_like(mask)
    tr[ys[perm[:cut]], xs[perm[:cut]]] = True
    va[ys[perm[cut:]], xs[perm[cut:]]] = True
    return tr, va


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


def _load_hist_grids(states_dir, schema, mu, sd, exclude_year):
    """Preload every ``state_{year}.npz`` grid (except the labelled year) as
    normalized ``(H,W,C)`` tensors + validity masks for the reconstruction loss."""
    grids, masks, years = [], [], []
    for fp in sorted(glob.glob(os.path.join(states_dir, "state_*.npz"))):
        yr = int(os.path.basename(fp).split("_")[1].split(".")[0])
        if yr == exclude_year:
            continue
        covn, m = cio.norm_grid(cio.load_state_stack(yr, states_dir, schema), mu, sd)
        grids.append(torch.tensor(covn)); masks.append(torch.tensor(m)); years.append(yr)
    if not grids:
        return None, None, []
    print(f"[Historical] {len(years)} year grids ({years[0]}..{years[-1]})")
    return torch.stack(grids), torch.stack(masks), years


def train_model_semisup(covn2023, mask_cov, mask_sup_tr, mask_sup_val, z_ref, x_raw_grid,
                        hist_grids, hist_masks, stream_dims, latent_dim, spatial_kernel=3,
                        epochs=100, lr=1e-3, batch_years=8, weights=None, seed=0,
                        enrich=None, ebird_frac=0.8, direction=None, dir_weights=None,
                        uniform_stab=False, patience=50, min_delta=1e-4):
    """Train the N-stream grid DESK autoencoder semi-supervised; return the fitted model.

    ``enrich`` (or None): tuple ``(pt_covn, pt_covmask, pt_zobs, pt_tgt)`` of per-historical-
    year grids/targets. When given, the stabilizing loss becomes eBird-heavy weighted:
    ``ebird_frac``·(recent eBird MSE) + (1-``ebird_frac``)·``w_absolute``·(historical BBS z_obs
    MSE), so reliable modern eBird dominates.

    ``direction`` (or None): per-cell direction-of-change targets (from
    ``_prepare_direction_targets``). When given, adds an up-weighted **cosine** alignment of
    the per-cell change vector ``Δ = z(2023) − weighted-mean(z over preceding years)`` (pred vs
    obs, magnitude-free) + a tiny **one-sided magnitude floor** ``relu(‖Δ_obs‖ − ‖Δ_pred‖)``
    (punishes under-shoot only), both reliability-weighted. ``dir_weights`` = {direction,
    magnitude_floor, absolute}.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = weights or {"stabilizing": 1.0, "metric": 5.0, "reconstruction": 0.1}

    model = MultiStreamAutoencoder(stream_dims, latent_dim, spatial_kernel).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)

    # Supervised (labelled) year grid: forwarded every step (small; dropout varies it).
    cov2023 = covn2023[None].to(device)                 # (1,H,W,C)
    m_cov = mask_cov[None].to(device)                   # (1,H,W)
    z_ref_t = torch.tensor(z_ref, device=device)        # (H,W,L)
    x_t = torch.tensor(x_raw_grid, device=device)       # (H,W, S*T)
    m_tr = torch.as_tensor(mask_sup_tr, device=device).bool()
    m_val = torch.as_tensor(mask_sup_val, device=device).bool()
    m_cov2023 = torch.as_tensor(mask_cov, device=device).bool()

    en = None
    if enrich is not None:
        pc, pm, pz, pt = enrich
        en = (torch.tensor(pc, device=device), torch.as_tensor(pm, device=device),
              torch.tensor(pz, device=device), torch.as_tensor(pt, device=device).bool())

    dw = dir_weights or {}
    w_dir = float(dw.get("direction", 0.0)); w_mag = float(dw.get("magnitude_floor", 0.0))
    w_abs = float(dw.get("absolute", 1.0))
    dr = None
    if direction is not None:
        dr = {"rows": torch.as_tensor(direction["rows"], device=device).long(),
              "cols": torch.as_tensor(direction["cols"], device=device).long(),
              "dobs": torch.as_tensor(direction["dobs"], device=device).float(),
              "dobs_norm": torch.as_tensor(direction["dobs_norm"], device=device).float(),
              "rel": torch.as_tensor(direction["rel"], device=device).float(),
              "pre_cell": torch.as_tensor(direction["pre_cell"], device=device).long(),
              "pre_grid": torch.as_tensor(direction["pre_grid"], device=device).long(),
              "pre_row": torch.as_tensor(direction["pre_row"], device=device).long(),
              "pre_col": torch.as_tensor(direction["pre_col"], device=device).long(),
              "pre_w": torch.as_tensor(direction["pre_w"], device=device).float()}
        n_dir = dr["rows"].shape[0]

    n_hist = 0 if hist_grids is None else hist_grids.shape[0]
    g = torch.Generator().manual_seed(seed)
    best_val, best_state, bad = float("inf"), None, 0     # early stopping on held-out Stab(val)
    print(f"--- Training grid DESK (spatial_kernel={spatial_kernel}, {n_hist} hist years, "
          f"max {epochs} ep, patience {patience}) ---")

    for ep in range(1, epochs + 1):
        model.train()
        order = torch.randperm(n_hist, generator=g) if n_hist else torch.zeros(1, dtype=torch.long)
        total_rh, total_sh, total_dir, steps = 0.0, 0.0, 0.0, 0
        for b0 in range(0, max(n_hist, 1), batch_years):
            steps += 1
            opt.zero_grad()

            # Supervised losses on the labelled (2023) grid.
            z2023, recon2023 = model(cov2023, m_cov)     # (1,H,W,L), (1,H,W,C)
            z_flat = z2023[0]
            sq_rec = torch.sum((z_flat[m_tr] - z_ref_t[m_tr]) ** 2, dim=1)   # per recent-cell
            loss_stab = sq_rec.mean()
            loss_true = true_kernel_loss(z_flat[m_tr], x_t[m_tr])
            loss_recon_s = F.mse_loss(recon2023[0][m_cov2023], cov2023[0][m_cov2023])

            # Enrich: eBird-heavy-weighted historical supervision (absolute z_obs, down-weighted)
            # + direction-of-change (cosine, up-weighted) + one-sided magnitude floor (tiny).
            loss_stab_hist = torch.zeros((), device=device)
            loss_dir = torch.zeros((), device=device)
            loss_mag = torch.zeros((), device=device)
            if en is not None:
                z_pt, _ = model(en[0], en[1])                    # (n_py,H,W,L)
                sq = torch.sum((z_pt[en[3]] - en[2][en[3]]) ** 2, dim=1)
                loss_stab_hist = sq.mean() if sq.numel() else loss_stab_hist
                if uniform_stab:
                    # Trend mode: pool the recent-anchor cells and ALL historical points,
                    # weighting every (cell,year) equally ("evenly prioritizing all
                    # locations and times") -- no eBird up-weighting, no direction split.
                    denom = int(sq_rec.numel() + sq.numel())
                    loss_stab = (sq_rec.sum() + sq.sum()) / max(denom, 1)
                else:
                    loss_stab = ebird_frac * loss_stab + (1.0 - ebird_frac) * w_abs * loss_stab_hist

                if dr is not None:
                    L = z_pt.shape[-1]
                    zp_ref = z_flat[dr["rows"], dr["cols"]]                       # (n_dir, L)
                    gathered = z_pt[dr["pre_grid"], dr["pre_row"], dr["pre_col"]] # (n_pre, L)
                    accum = torch.zeros(n_dir, L, device=device)
                    wsum = torch.zeros(n_dir, 1, device=device)
                    accum.index_add_(0, dr["pre_cell"], gathered * dr["pre_w"][:, None])
                    wsum.index_add_(0, dr["pre_cell"], dr["pre_w"][:, None])
                    dpred = zp_ref - accum / wsum.clamp_min(1e-8)                 # Δ_pred
                    rel = dr["rel"]; rsum = rel.sum().clamp_min(1e-8)
                    loss_dir = (rel * (1.0 - F.cosine_similarity(dpred, dr["dobs"], dim=1))).sum() / rsum
                    loss_mag = (rel * torch.relu(dr["dobs_norm"] - dpred.norm(dim=1))).sum() / rsum

            # Reconstruction on a batch of unlabelled year grids.
            if n_hist:
                sel = order[b0:b0 + batch_years]
                xb = hist_grids[sel].to(device); mb = hist_masks[sel].to(device).bool()
                _, recon_h = model(xb, mb)
                loss_recon_h = F.mse_loss(recon_h[mb], xb[mb])
            else:
                loss_recon_h = torch.tensor(0.0, device=device)

            loss = (weights["stabilizing"] * loss_stab
                    + weights["metric"] * loss_true
                    + weights["reconstruction"] * (loss_recon_s + loss_recon_h)
                    + w_dir * loss_dir + w_mag * loss_mag)
            loss.backward()
            opt.step()
            total_rh += loss_recon_h.item()
            total_sh += loss_stab_hist.item()
            total_dir += loss_dir.item()

        model.eval()
        with torch.no_grad():
            z2023, _ = model(cov2023, m_cov)
            zf = z2023[0]
            stab_val = torch.mean(torch.sum((zf[m_val] - z_ref_t[m_val]) ** 2, dim=1)).item()
            cos = F.cosine_similarity(zf[m_val], z_ref_t[m_val]).mean().item()
            true_val = true_kernel_loss(zf[m_val], x_t[m_val]).item()
            gpar = float(model.gamma.detach()) if spatial_kernel > 0 else 0.0
            scheduler.step(stab_val)
            sh = f" | StabHist {total_sh / max(steps,1):.4f}" if en is not None else ""
            dd = f" | Dir {total_dir / max(steps,1):.4f}" if dr is not None else ""
            print(f"Ep {ep:03d} | Stab(val) {stab_val:.4f} | True(val) {true_val:.4f} | "
                  f"Rec(H) {total_rh / max(steps,1):.4f}{sh}{dd} | Cos {cos:.3f} | gamma {gpar:+.4f}",
                  flush=True)

        # Early stopping on held-out Stab(val); keep the best weights so a long budget is safe.
        if stab_val < best_val - min_delta:
            best_val, bad = stab_val, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print(f"[desk] early stop at ep {ep} (best Stab(val) {best_val:.4f})")
                break
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model


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


def _prepare_enrich(config, states_dir, schema, mu, sd, z_dir, latent_dim, holdout, label_year):
    """Build the enrich supervised targets: project the BBS-amplitude communities into the
    SAME eBird-2023 ESK basis DESK uses (z_obs), then per historical point-year assemble a
    covariate grid, a z_obs target grid, and a target mask (point cells, cov-valid, NOT
    held-out). eBird defines the basis; BBS only says where historical cells land in it.
    """
    from .esk_kernel import project_amplitude_to_z
    zt = config["bbs"]["z_dir"]
    X = np.load(os.path.join(zt, "X_points.npy"))
    pidx = np.load(os.path.join(zt, "point_index.npy"))
    z_obs = project_amplitude_to_z(X, z_dir, latent_dim)
    if z_obs is None:
        raise FileNotFoundError(
            f"enrich needs the saved ESK projection in {z_dir} (esk_landmarks/projmat); re-run esk")
    rows, cols, yrs = pidx[:, 0], pidx[:, 1], pidx[:, 2]
    hist_years = sorted({int(y) for y in yrs if int(y) != label_year})
    covn, covm, zobs_g, tgt = [], [], [], []
    for y in hist_years:
        cn, m = cio.norm_grid(cio.load_state_stack(y, states_dir, schema), mu, sd)
        H, W = m.shape
        sel = np.where(yrs == y)[0]
        zg = np.zeros((H, W, latent_dim), dtype="float32"); tm = np.zeros((H, W), bool)
        zg[rows[sel], cols[sel]] = z_obs[sel]; tm[rows[sel], cols[sel]] = True
        tm &= m & (~holdout)
        covn.append(cn); covm.append(m); zobs_g.append(zg); tgt.append(tm)
    n_tgt = int(sum(t.sum() for t in tgt))
    print(f"[enrich] {len(hist_years)} historical target years, {n_tgt} supervised BBS points "
          f"({int(holdout.sum())} cells held out for eval)")
    return (np.stack(covn), np.stack(covm), np.stack(zobs_g), np.stack(tgt)), hist_years


def _weighted_median_cols(V, w):
    """Component-wise effort-weighted median of ``V`` (k, L) with weights ``w`` (k,) -> (L,).
    Robust central estimate of a cell's preceding-year z (BBS is noisy, per assumption 1)."""
    order = np.argsort(V, axis=0)
    Vs = np.take_along_axis(V, order, axis=0)
    Ws = np.asarray(w, float)[order]
    cum = np.cumsum(Ws, axis=0)
    idx = (cum >= 0.5 * cum[-1]).argmax(axis=0)
    return Vs[idx, np.arange(V.shape[1])]


def _prepare_direction_targets(config, z_dir, latent_dim, holdout, hist_years, recent_year,
                               reference_start, baseline_scale):
    """DEPRECATED (amplitude/enrich path only; the trend path supervises z directly).

    Per-cell direction-of-change target in the (joint) ESK basis.

    For each train cell with a 2023 anchor + >=1 preceding point (year < reference_start):
    ``Δ_obs = z_ref(2023 anchor) - weighted_median(preceding z_obs)``. Reliability
    ``r = clip((reference_start - effort-weighted preceding TCOM) / baseline_scale, 0, 1)``
    down-weights short/recent baselines. Returns the per-cell targets plus a flat map that
    lets the trainer aggregate the PREDICTED preceding mean by scatter-add over the same
    historical point-year grids (aligned to ``hist_years``), with the same effort weights.
    """
    from .esk_kernel import project_amplitude_to_z
    zt = config["bbs"]["z_dir"]
    X = np.load(os.path.join(zt, "X_points.npy"))
    pidx = np.load(os.path.join(zt, "point_index.npy"))
    z_obs = project_amplitude_to_z(X, z_dir, latent_dim)
    sf = np.load(os.path.join(zt, "support_field.npz"))
    sup, syears = sf["support"], [int(y) for y in sf["years"]]
    yr_ix = {y: i for i, y in enumerate(syears)}
    rows, cols, yrs = pidx[:, 0], pidx[:, 1], pidx[:, 2]
    hist_pos = {int(y): k for k, y in enumerate(hist_years)}

    rec = np.where(yrs == recent_year)[0]
    rec_map = {(int(rows[i]), int(cols[i])): int(i) for i in rec}
    pre = {}                                     # (r,c) -> list of (point_idx, year, weight)
    for i in np.where(yrs < reference_start)[0]:
        r, c, y = int(rows[i]), int(cols[i]), int(yrs[i])
        if (r, c) in rec_map and not holdout[r, c] and y in hist_pos and y in yr_ix:
            pre.setdefault((r, c), []).append((i, y, float(sup[yr_ix[y], r, c])))

    dir_r, dir_c, dobs, rel = [], [], [], []
    p_cell, p_grid, p_r, p_c, p_w = [], [], [], [], []
    for cpos, (r, c) in enumerate(k for k in pre if pre[k]):
        pts = pre[(r, c)]
        V = z_obs[[p[0] for p in pts]]
        W = np.array([p[2] for p in pts], float)
        Wn = W if W.sum() > 0 else np.ones_like(W)
        zpre = _weighted_median_cols(V, Wn)
        dobs.append(z_obs[rec_map[(r, c)]] - zpre)
        tcom = float(np.sum(Wn * np.array([p[1] for p in pts])) / Wn.sum())
        rel.append(min(1.0, max(0.0, (reference_start - tcom) / baseline_scale)))
        dir_r.append(r); dir_c.append(c)
        for (i, y, w) in pts:
            p_cell.append(cpos); p_grid.append(hist_pos[y]); p_r.append(r); p_c.append(c)
            p_w.append(w if W.sum() > 0 else 1.0)
    dobs = np.array(dobs, dtype="float32")
    print(f"[enrich-dir] {len(dir_r)} direction cells, {len(p_cell)} preceding points "
          f"(reference_start={reference_start}, baseline_scale={baseline_scale})")
    return dict(
        rows=np.array(dir_r, int), cols=np.array(dir_c, int), dobs=dobs,
        dobs_norm=np.linalg.norm(dobs, axis=1).astype("float32"),
        rel=np.array(rel, dtype="float32"),
        pre_cell=np.array(p_cell, int), pre_grid=np.array(p_grid, int),
        pre_row=np.array(p_r, int), pre_col=np.array(p_c, int),
        pre_w=np.array(p_w, dtype="float32"))


# --- Output-EMA path (bbs_mode=trend): demographic lag on the predicted Z --------

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


def _prepare_trend_targets(config, z_dir, latent_dim, holdout):
    """Per-year ESK-basis targets for EVERY supervised year, from the trend points.

    Projects ``X_points`` into the joint ESK basis (z_obs) and scatters each point into
    its year's grid; returns ``{year: (zg (H,W,L), tm_tr (H,W), tm_val (H,W))}`` where the
    train/val split is the spatial ``holdout`` (val = held-out cells). Includes 2023.
    """
    from .esk_kernel import project_amplitude_to_z
    zt = config["bbs"]["z_dir"]
    X = np.load(os.path.join(zt, "X_points.npy"))
    pidx = np.load(os.path.join(zt, "point_index.npy"))
    z_obs = project_amplitude_to_z(X, z_dir, latent_dim)
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


def train_model_ema(cov_window, mask_window, window_years, targets, x2023, m2023_tr, m2023_val,
                    stream_dims, latent_dim, ema_cfg, spatial_kernel=3, epochs=500, lr=1e-3,
                    weights=None, seed=0, patience=50, min_delta=1e-4):
    """Train DESK with a learned output-EMA (bbs_mode=trend).

    Forwards the ordered year window (per-year gradient checkpointing), applies the
    learned causal EMA over the year axis, and supervises ``z_ema`` against the per-year
    trend targets (uniform over all supervised (cell,year), train = non-held-out cells).
    Plus the 2023 Ruzicka metric loss and an autoencoder reconstruction over the window.
    Returns ``(model, ema)``.
    """
    from torch.utils.checkpoint import checkpoint
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = weights or {"stabilizing": 1.0, "metric": 5.0, "reconstruction": 0.1}
    torch.manual_seed(seed)

    model = MultiStreamAutoencoder(stream_dims, latent_dim, spatial_kernel).to(device)
    ema = OutputEMA(ema_cfg.get("half_life_bounds", [1.0, 40.0])[0],
                    ema_cfg.get("half_life_bounds", [1.0, 40.0])[1],
                    ema_cfg.get("init_half_life", 8.0)).to(device)
    opt = torch.optim.Adam(list(model.parameters()) + list(ema.parameters()), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)

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

    best_val, best = float("inf"), None
    print(f"--- Training DESK+outputEMA ({len(window_years)}yr window {window_years[0]}..{window_years[-1]}, "
          f"{len(tgt)} supervised years, max {epochs} ep) ---")
    for ep in range(1, epochs + 1):
        model.train(); opt.zero_grad()
        z_raw, recon_loss = [], 0.0
        for t in range(cov.shape[0]):                            # per-year forward (checkpointed)
            zt, rt = checkpoint(model, cov[t:t + 1], msk[t:t + 1], use_reentrant=False)
            z_raw.append(zt[0])
            recon_loss = recon_loss + F.mse_loss(rt[0][msk[t]], cov[t][msk[t]])
        recon_loss = recon_loss / cov.shape[0]
        z_ema = ema(torch.stack(z_raw, 0))                       # (T,H,W,L)

        # uniform stabilizing loss over all supervised (cell,year), train cells only
        sq_sum, n = torch.zeros((), device=device), 0
        for y, (zg, tr, _va) in tgt.items():
            zz = z_ema[yi[y]]
            s = torch.sum((zz[tr] - zg[tr]) ** 2, dim=1)
            sq_sum = sq_sum + s.sum(); n += int(tr.sum())
        loss_stab = sq_sum / max(n, 1)
        z_anchor = z_ema[yi[y2023]]
        loss_true = true_kernel_loss(z_anchor[m_tr], x2023_t[m_tr])
        loss = (weights["stabilizing"] * loss_stab + weights["metric"] * loss_true
                + weights["reconstruction"] * recon_loss)
        loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            # cheap val: re-forward is expensive, so eval the anchor-year held-out cells from this step's z_ema
            vs = torch.sum((z_anchor[m_val] - tgt[y2023][0][m_val]) ** 2, dim=1).mean().item() if m_val.any() else 0.0
            sched.step(vs)
            print(f"Ep {ep:03d} | Stab {loss_stab.item():.4f} | True {loss_true.item():.4f} | "
                  f"Rec {recon_loss.item():.4f} | Val(anchor) {vs:.4f} | half-life {ema.half_life().item():.1f}y", flush=True)
        if vs < best_val - min_delta:
            best_val, bad = vs, 0
            best = ({k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    {k: v.detach().cpu().clone() for k, v in ema.state_dict().items()})
        else:
            bad = bad + 1 if ep > 1 else 0
            if bad >= patience:
                print(f"[desk] early stop at ep {ep} (best Val {best_val:.4f})"); break
    if best is not None:
        model.load_state_dict({k: v.to(device) for k, v in best[0].items()})
        ema.load_state_dict({k: v.to(device) for k, v in best[1].items()})
    return model, ema


def run_desk_experiment(config=None):
    """Driver: load N-stream states + ESK Z, prepare grids, train DESK, save model+meta.

    ``off``/``validate``: eBird 2023 spatial ESK Z target (single labelled year). ``enrich``:
    additionally supervise DESK at historical (cell,year) points against the BBS-amplitude
    communities projected into the eBird ESK basis (z_obs), weighted eBird-heavy, with a
    spatial cell holdout for honest evaluation.
    """
    config = load_config(config) if not isinstance(config, dict) else config
    paths, desk_cfg = config["paths"], config["desk"]
    out_dir = paths["desk_output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    states_dir = os.path.join(paths["hist_dir"], "yearly_states")
    schema = cio.load_schema(states_dir)
    label_year = int(desk_cfg.get("label_year", 2023))
    spatial_kernel = int(desk_cfg.get("spatial_conv", {}).get("kernel", 3)) \
        if desk_cfg.get("spatial_conv", {}).get("enabled", True) else 0

    bbs_mode = config.get("bbs_mode", "validate")
    cov_stack = cio.load_state_stack(label_year, states_dir, schema)
    if bbs_mode == "trend":
        # The Ružicka metric anchor is the reconstructed reference-year (anchor_year)
        # community -- the EXACT vectors that seeded the ESK basis (log1p abundance,
        # anchor-mode-agnostic). Scatter X_points' anchor-year rows into an (H,W,S) grid,
        # so DESK depends on no weekly eBird product (trends-abd anchor needs none).
        ztz = config["bbs"]["z_dir"]
        Xp = np.load(os.path.join(ztz, "X_points.npy"))
        pip = np.load(os.path.join(ztz, "point_index.npy"))
        pm = json.load(open(os.path.join(ztz, "points_meta.json")))
        ay, S = int(pm["recent_year"]), int(pm["n_species"])
        H, W = cov_stack.shape[:2]
        sel = pip[:, 2] == ay
        ebird_stack = np.full((H, W, S), np.nan, dtype="float32")
        ebird_stack[pip[sel, 0], pip[sel, 1]] = Xp[sel]                # already log1p in X_points
        log1p_kernel = bool(pm.get("ruzicka_log1p", True))
        print(f"[desk] trend mode: Ružicka metric anchored on the reconstructed year-{ay} "
              f"community from X_points (anchor_mode={pm.get('anchor_mode')}, log1p={log1p_kernel})")
    else:
        ebird_stack, _ = load_ebird_stack(config)

    z_dir = desk_cfg["z_dir"]
    try:
        z_mask = np.load(os.path.join(z_dir, "valid_mask.npy"))
        z_flat = np.load(os.path.join(z_dir, "Z.npy"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"ESK Z.npy/valid_mask.npy not in {z_dir}") from exc
    # ESK saves Z at the max swept latent_dim. Optionally truncate to desk.latent_dim:
    # kernel-PCA columns are eigenvalue-ordered, so Z[:, :k] IS the exact dim-k
    # embedding (no ESK re-run needed). Unset -> use all columns.
    ld = desk_cfg.get("latent_dim")
    if ld and z_flat.shape[1] > int(ld):
        print(f"[desk] truncating ESK Z {z_flat.shape[1]} -> {int(ld)} dims (top eigen-components)")
        z_flat = z_flat[:, :int(ld)]

    # Normalization stats: fit on the supervised (labelled) pixels, exactly as before,
    # then applied to every grid (labelled + historical) and frozen for the cube.
    mask_sup0 = compute_valid_mask(ebird_stack, cov_stack, z_mask)
    mu, sd = cio.fit_norm(cov_stack[mask_sup0].astype("float32"))

    covn, mask_cov, mask_sup, z_grid, x_grid = prepare_supervised(
        cov_stack, ebird_stack, z_flat, z_mask, mu, sd, out_dir)
    mask_cov_t = torch.tensor(mask_cov)

    # Mode: 'trend' supervises DESK directly + uniformly against the trend-based
    # spatiotemporal z-target (recent anchor + backward-reconstructed historical points);
    # 'enrich' (deprecated amplitude path) up-weighted recent eBird + a direction-of-change
    # split; 'off'/'validate' train eBird-2023-only. bbs_mode was read above.
    enrich_data, direction, ebird_frac = None, None, 0.8
    dir_weights = {}
    uniform_stab = False
    holdout = np.zeros_like(mask_sup)
    if bbs_mode == "trend":
        tr_cfg = desk_cfg.get("trend", {})
        ebird_valid = np.any(~np.isnan(ebird_stack), axis=-1)          # spatial cell holdout
        ys, xs = np.where(ebird_valid)
        rng = np.random.default_rng(int(tr_cfg.get("seed", 0)))
        ho = rng.random(len(ys)) < float(tr_cfg.get("holdout_frac", 0.2))
        holdout[ys[ho], xs[ho]] = True
        m_tr, m_val = mask_sup & (~holdout), mask_sup & holdout          # eval on held-out cells
        # _prepare_enrich projects the trend points (year<2023) into the joint ESK basis
        # (z_obs) and assembles per-year target grids -- reused as-is, direction disabled.
        enrich_data, hist_years = _prepare_enrich(config, states_dir, schema, mu, sd, z_dir,
                                                  z_grid.shape[2], holdout, label_year)
        direction = None
        dir_weights = {"absolute": 1.0}
        uniform_stab = True
        np.save(os.path.join(out_dir, "holdout_cells.npy"), holdout)
        print(f"[desk] TREND mode: direct uniform z-target over recent + historical points; "
              f"{int(holdout.sum())} cells held out for eval")
    elif bbs_mode == "enrich":
        en_cfg = desk_cfg.get("enrich", {})
        ebird_frac = float(en_cfg.get("ebird_loss_fraction", 0.8))
        ebird_valid = np.any(~np.isnan(ebird_stack), axis=-1)          # holdout over eBird cells
        ys, xs = np.where(ebird_valid)
        rng = np.random.default_rng(int(en_cfg.get("seed", 0)))
        ho = rng.random(len(ys)) < float(en_cfg.get("holdout_frac", 0.2))
        holdout[ys[ho], xs[ho]] = True
        m_tr, m_val = mask_sup & (~holdout), mask_sup & holdout          # eval on held-out cells
        enrich_data, hist_years = _prepare_enrich(config, states_dir, schema, mu, sd, z_dir,
                                                  z_grid.shape[2], holdout, label_year)
        direction = _prepare_direction_targets(
            config, z_dir, z_grid.shape[2], holdout, hist_years, label_year,
            int(en_cfg.get("reference_start", 2014)), float(en_cfg.get("baseline_scale", 20.0)))
        dir_weights = {"direction": float(en_cfg.get("w_direction", 2.0)),
                       "magnitude_floor": float(en_cfg.get("w_magnitude_floor", 0.05)),
                       "absolute": float(en_cfg.get("w_absolute", 1.0))}
        np.save(os.path.join(out_dir, "holdout_cells.npy"), holdout)
        print(f"[desk] ENRICH mode: ebird_loss_fraction={ebird_frac}, weights={dir_weights}, "
              f"{int(holdout.sum())} cells held out")
    else:
        m_tr, m_val = _split_mask(mask_sup, desk_cfg.get("train_val_split", 0.8))

    stream_dims = cio.stream_dims(schema)
    ema_cfg = desk_cfg.get("output_ema", {})
    ema_half_life = None
    if bbs_mode == "trend" and ema_cfg.get("enabled", False):
        # Output-EMA objective: forward the ordered year window, apply a learned causal
        # EMA over the year axis to the predicted Z (demographic lag), and supervise the
        # EMA'd z_ema against the per-year trend targets. Replaces the direct per-year target.
        warmup_start = int(ema_cfg.get("warmup_start", 1940))
        window_years = list(range(warmup_start, label_year + 1))
        cov_win, mask_win, kept = _load_year_window(states_dir, schema, mu, sd, window_years)
        targets = _prepare_trend_targets(config, z_dir, z_grid.shape[2], holdout)
        model, ema = train_model_ema(
            cov_win, mask_win, kept, targets, x_grid, m_tr, m_val,
            stream_dims, latent_dim=z_grid.shape[2], ema_cfg=ema_cfg,
            spatial_kernel=spatial_kernel,
            epochs=desk_cfg.get("epochs", 500), lr=desk_cfg.get("lr", 1e-3),
            weights=desk_cfg.get("weights"), patience=desk_cfg.get("patience", 50))
        ema_half_life = float(ema.half_life().item())
        torch.save(ema.state_dict(), os.path.join(out_dir, "output_ema.pth"))
        print(f"[desk] output-EMA learned half-life = {ema_half_life:.2f} yr")
    else:
        hist_grids, hist_masks, _ = _load_hist_grids(states_dir, schema, mu, sd, label_year)
        model = train_model_semisup(
            torch.tensor(covn), mask_cov_t, m_tr, m_val, z_grid, x_grid,
            hist_grids, hist_masks, stream_dims, latent_dim=z_grid.shape[2],
            spatial_kernel=spatial_kernel,
            epochs=desk_cfg.get("epochs", 100), lr=desk_cfg.get("lr", 1e-3),
            batch_years=desk_cfg.get("batch_years", 8),
            weights=desk_cfg.get("weights"),
            enrich=enrich_data, ebird_frac=ebird_frac,
            direction=direction, dir_weights=dir_weights, uniform_stab=uniform_stab,
            patience=desk_cfg.get("patience", 50))

    torch.save(model.state_dict(), os.path.join(out_dir, "env_model_semisup.pth"))
    np.savez(os.path.join(out_dir, "desk_meta.npz"),
             mu=mu, sd=sd, stream_dims=np.array(stream_dims, int),
             latent_dim=z_grid.shape[2], label_year=label_year,
             spatial_kernel=spatial_kernel, bbs_mode=bbs_mode,
             ebird_frac=ebird_frac, schema=json.dumps(schema),
             output_ema=bool(ema_cfg.get("enabled", False)),
             ema_half_life=(ema_half_life if ema_half_life is not None else np.nan),
             ema_warmup_start=int(ema_cfg.get("warmup_start", 1940)))
    print(f"[desk] saved model + desk_meta.npz -> {out_dir} "
          f"(spatial_kernel={spatial_kernel}, mode={bbs_mode})")


if __name__ == "__main__":
    run_desk_experiment()
