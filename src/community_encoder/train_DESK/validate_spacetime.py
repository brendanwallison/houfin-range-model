"""Validate: does the eBird-only DESK already predict BBS spatiotemporal structure?

The headline test of ``bbs_mode='validate'``. The eBird-only-trained DESK gives a
predicted latent ``z(s,t)`` from that point's own-year covariates. At **held-out
historical** ``(cell, year)`` points (which the model never trained on) we ask
whether its predicted **similarities** reproduce the BBS-observed community
similarities — comparing at the **kernel level** (``⟨z_i,z_j⟩`` vs
``Ruzicka(x_i,x_j)``), never raw coordinates, because Z is basis/rotation-arbitrary.
Both live on the same eBird-unit Ruzicka scale (``x = E·anomaly`` is in eBird units,
and ``true_kernel_loss`` calibrated ``⟨z,z⟩`` to eBird Ruzicka), so the comparison is
fair. Reported per period with MSE + basis-invariant CKA/Mantel.

Strong agreement (esp. degrading gracefully, not randomly, back in time) ⇒ the
spatial→spatiotemporal extrapolation holds and no BBS-in-training is needed; weak
agreement flags where ``enrich`` is warranted.
"""
import json
import os

import numpy as np


# ----------------------------- pure metrics -----------------------------

def ruzicka_similarity_matrix(X):
    """Pairwise Ruzicka similarity ``Σmin/Σmax`` over rows of ``X (n, d)`` → ``(n, n)``."""
    X = np.asarray(X, dtype="float64")
    n = X.shape[0]
    S = np.empty((n, n))
    for i in range(n):
        mn = np.minimum(X[i], X).sum(1)
        mx = np.maximum(X[i], X).sum(1)
        S[i] = mn / np.where(mx > 0, mx, 1.0)
    return S


def _center(K):
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def linear_cka(K, L):
    """Centered-kernel-alignment between two Gram/similarity matrices (rotation-invariant)."""
    Kc, Lc = _center(np.asarray(K, float)), _center(np.asarray(L, float))
    num = (Kc * Lc).sum()
    den = np.sqrt((Kc * Kc).sum() * (Lc * Lc).sum())
    return float(num / den) if den > 0 else 0.0


def mantel_r(A, B):
    """Pearson correlation of the off-diagonal (upper-triangle) entries of two matrices."""
    iu = np.triu_indices_from(np.asarray(A), k=1)
    a, b = np.asarray(A)[iu], np.asarray(B)[iu]
    if a.size < 2 or a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def pair_sims(Z, X, pairs):
    """Predicted ``⟨z_i,z_j⟩`` and observed ``Ruzicka(x_i,x_j)`` for index pairs ``(2, m)``."""
    i, j = pairs
    sim_pred = (Z[i] * Z[j]).sum(1)
    xi, xj = X[i], X[j]
    mn = np.minimum(xi, xj).sum(1)
    mx = np.maximum(xi, xj).sum(1)
    sim_obs = mn / np.where(mx > 0, mx, 1.0)
    return sim_pred, sim_obs


def distinct_pairs(m, n, rng):
    """``n`` random index pairs into ``[0,m)`` with ``i != j`` (no self-pairs, which would
    inject artificial similarity=1). Returns two index arrays (length <= n)."""
    i = rng.integers(0, m, n)
    j = rng.integers(0, m, n)
    keep = i != j
    return i[keep], j[keep]


def _partial_corr(a, b, C):
    """Pearson correlation of ``a`` and ``b`` after linearly removing covariates ``C``
    (a (k,) list/array of columns) from both. Isolates the association not explained by C."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    A = np.column_stack([np.ones(len(a))] + [np.asarray(c, float) for c in C])
    ra = a - A @ np.linalg.lstsq(A, a, rcond=None)[0]
    rb = b - A @ np.linalg.lstsq(A, b, rcond=None)[0]
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _cka_gain_ci(Kh, Lh, Kn, rng, n_boot=200, frac=0.7):
    """Subsampling CIs for CKA(DESK,obs) and the gain CKA(DESK)-CKA(no-change) on a fixed
    precomputed Gram triple. Subsamples (without replacement) so small periods show honest
    uncertainty and duplicate-row artifacts of with-replacement bootstrap are avoided.
    Returns ((cka_lo,cka_hi), (gain_lo,gain_hi))."""
    m = Kh.shape[0]
    sub = max(8, int(frac * m))
    if m < 12:
        return (float("nan"), float("nan")), (float("nan"), float("nan"))
    ck = np.empty(n_boot); gn = np.empty(n_boot)
    for b in range(n_boot):
        s = rng.choice(m, sub, replace=False)
        ix = np.ix_(s, s); Lhs = Lh[ix]
        ck[b] = linear_cka(Kh[ix], Lhs)
        gn[b] = ck[b] - linear_cka(Kn[ix], Lhs)
    return (float(np.percentile(ck, 2.5)), float(np.percentile(ck, 97.5))), \
           (float(np.percentile(gn, 2.5)), float(np.percentile(gn, 97.5)))


# --------------------- spatiotemporal (temporal-nuance) metrics ---------------------
# All basis-invariant: they compare SIMILARITIES (<z,z> vs Ruzicka) or GEOGRAPHIC
# quantities, never the rotation-arbitrary embedding coordinates.

def ruzicka_rect(A, B):
    """Pairwise Ruzicka Σmin/Σmax between rows of ``A (n,D)`` and ``B (m,D)`` → ``(n,m)``.

    Σmin=(sa+sb−L1)/2, Σmax=(sa+sb+L1)/2 ⇒ Ruzicka=(sa+sb−L1)/(sa+sb+L1). Uses torch
    (GPU) for the L1 block if available, else scipy — same result either way.
    """
    A = np.asarray(A, "float64"); B = np.asarray(B, "float64")
    sa, sb = A.sum(1), B.sum(1)
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        L1 = torch.cdist(torch.tensor(A, device=dev), torch.tensor(B, device=dev),
                         p=1).cpu().numpy()
    except Exception:
        from scipy.spatial.distance import cdist
        L1 = cdist(A, B, "cityblock")
    denom = sa[:, None] + sb[None, :]
    return np.where((denom + L1) > 0, (denom - L1) / (denom + L1), 1.0)


def temporal_turnover_agreement(Z, X, pidx, recent_year, min_gap=5):
    """Per-site community turnover (earliest supported point → recent), pred vs obs.

    turnover = 1 − self-similarity over time (``⟨z(s,t0),z(s,rec)⟩`` for pred,
    ``Ruzicka(x(s,t0),x(s,rec))`` for obs) — basis-invariant. Spearman of the two
    per-site turnover fields answers "do the models agree on WHERE communities changed
    most" (magnitude, direction-agnostic). Returns the fields + rho.

    Each cell is anchored to its **earliest** historical point (≥ ``min_gap`` yr before
    recent) matched to its recent point — maximizing coverage across every supported
    cell (the smoothed BBS field backs far more cells than any narrow year window). Both
    ``pred`` and ``obs`` use the SAME (early, recent) year pair per cell, so a varying
    span shifts them together and barely biases their rank correlation; ``hist_year`` is
    returned so magnitude-vs-span can still be inspected.
    """
    rows, cols, yrs = pidx[:, 0], pidx[:, 1], pidx[:, 2]
    rec = yrs == recent_year
    rec_ix = {(int(r), int(c)): int(i) for r, c, i in
              zip(rows[rec], cols[rec], np.where(rec)[0])}
    best = {}                                    # (r,c) -> (year, idx): EARLIEST historical point
    for i in np.where(~rec)[0]:
        key = (int(rows[i]), int(cols[i]))
        y = int(yrs[i])
        if key in rec_ix and (int(recent_year) - y) >= min_gap \
                and (key not in best or y < best[key][0]):
            best[key] = (y, int(i))
    keys = list(best)
    if len(keys) < 4:
        return {"n_sites": len(keys), "note": "too few paired sites"}
    hi = np.array([best[k][1] for k in keys])
    ri = np.array([rec_ix[k] for k in keys])
    # COSINE self-similarity for the predicted side: a raw dot would fold in the model's
    # global ⟨z,z⟩ calibration drift (self-similarity != 1), which would show up as spurious
    # turnover. Cosine measures the angular change only. Observed side is Ruzicka (bounded).
    zi, zr = Z[hi], Z[ri]
    sim_pred = (zi * zr).sum(1) / (np.linalg.norm(zi, axis=1) * np.linalg.norm(zr, axis=1) + 1e-12)
    mn = np.minimum(X[hi], X[ri]).sum(1); mx = np.maximum(X[hi], X[ri]).sum(1)
    sim_obs = np.where(mx > 0, mn / mx, 1.0)
    tp, to = 1.0 - sim_pred, 1.0 - sim_obs
    from scipy.stats import spearmanr
    rho = float(spearmanr(tp, to).correlation)
    return {"n_sites": len(keys), "spearman_turnover": rho,
            "rows": rows[hi], "cols": cols[hi], "hist_year": yrs[hi],
            "turnover_pred": tp.astype("float32"), "turnover_obs": to.astype("float32")}


def partial_spearman(tp, to, covars):
    """Spearman(tp, to) after linearly removing ``covars`` (on ranks) from both fields.

    The raw turnover Spearman is inflated by anything that drives BOTH fields together --
    chiefly the per-site time-span (deeper history => larger pred AND obs turnover) and any
    broad shared spatial trend. Regressing those out (on ranks) and correlating the
    residuals isolates whether DESK predicts the *fine-scale* pattern of change beyond
    those trivial shared drivers. Returns NaN if degenerate.
    """
    from scipy.stats import rankdata, spearmanr
    n = len(tp)
    if n < 8:
        return float("nan")
    A = np.column_stack([np.ones(n)] + [rankdata(c) for c in covars])

    def _resid(y):
        yr = rankdata(y).astype(float)
        beta, *_ = np.linalg.lstsq(A, yr, rcond=None)
        return yr - A @ beta

    r = spearmanr(_resid(tp), _resid(to)).correlation
    return float(r) if r == r else float("nan")


def directional_change_agreement(Z, X, pidx, recent_year, rng, n_anchor=400, min_gap=5):
    """Direction (not magnitude) of each site's community change, basis-invariant.

    Turnover magnitude is direction-blind: a cell can change by the same amount toward
    opposite assemblages. Here, for each site with an early+recent point, we build its
    similarity PROFILE to a fixed anchor set (recent communities) at both times; the CHANGE
    in that profile -- 'which communities it moved toward/away from' -- is basis-invariant
    (similarities to fixed anchors, not the rotation-arbitrary z). The per-site COSINE
    between DESK's change vector (``⟨z,anchor⟩``) and BBS's (``Ruzicka(x,anchor)``) cancels
    magnitude and measures pure direction: ~0 = random/no directional skill, >0 = moves the
    right way, <0 = wrong way. ``frac_same_dir`` = share with cosine>0 (null 0.5).
    """
    rows, cols, yrs = pidx[:, 0], pidx[:, 1], pidx[:, 2]
    rec = np.where(yrs == recent_year)[0]
    if rec.size < 8:
        return {"note": "too few recent anchors", "n_sites": 0}
    anchors = rng.choice(rec, min(n_anchor, rec.size), replace=False)
    Za, Xa = Z[anchors], X[anchors]
    rec_ix = {(int(r), int(c)): int(i) for r, c, i in zip(rows[rec], cols[rec], rec)}
    best = {}
    for i in np.where(yrs != recent_year)[0]:
        key = (int(rows[i]), int(cols[i])); y = int(yrs[i])
        if key in rec_ix and (int(recent_year) - y) >= min_gap \
                and (key not in best or y < best[key][0]):
            best[key] = (y, int(i))
    keys = list(best)
    if len(keys) < 8:
        return {"note": "too few paired sites", "n_sites": len(keys)}
    hi = np.array([best[k][1] for k in keys]); ri = np.array([rec_ix[k] for k in keys])
    dp = (Z[ri] @ Za.T) - (Z[hi] @ Za.T)                 # predicted profile CHANGE (n, n_anchor)
    do = ruzicka_rect(X[ri], Xa) - ruzicka_rect(X[hi], Xa)   # observed profile CHANGE
    npv = np.linalg.norm(dp, axis=1); nov = np.linalg.norm(do, axis=1)
    cos = (dp * do).sum(1) / np.where(npv * nov > 0, npv * nov, 1.0)
    # Empirical null: pair each site's PREDICTED change with a RANDOM other site's OBSERVED
    # change. Mean cos ~0 confirms the metric's baseline; the real mean_dir_cos is meaningful
    # only relative to this (both share the anchor geometry, so the null absorbs it).
    perm = rng.permutation(len(keys))
    dop = do[perm]; nop = nov[perm]
    cos_null = (dp * dop).sum(1) / np.where(npv * nop > 0, npv * nop, 1.0)
    return {"n_sites": len(keys), "mean_dir_cos": float(np.mean(cos)),
            "median_dir_cos": float(np.median(cos)), "frac_same_dir": float(np.mean(cos > 0)),
            "mean_dir_cos_null": float(np.mean(cos_null)),
            "rows": rows[hi], "cols": cols[hi], "hist_year": yrs[hi], "dir_cos": cos.astype("float32")}


def analog_displacement(Z, X, pidx, xy, recent_year, rng, n_hist=1500, n_present=4000, topk=15):
    """Direction each historical site's community "points" toward among present cells.

    For a historical point, rank present-day cells by similarity (``⟨z,z⟩`` / Ruzicka),
    take the top-``k`` analog cells, and their mean location = the analog centroid.
    Displacement Δ = centroid − site is a GEOGRAPHIC vector (Albers x=E–W, y=N–S), so
    Δ_pred and Δ_obs ARE directly comparable across models (no rotation problem). Top-k
    is rank-based → scale-invariant between the two similarity types. Tests the
    climate-analog hypothesis: do both models send past sites toward the same present
    communities / same compass direction (poleward warming, E–W precip)?
    """
    yrs = pidx[:, 2]
    pres, hist = np.where(yrs == recent_year)[0], np.where(yrs != recent_year)[0]
    if pres.size < topk + 1 or hist.size < 4:
        return {"note": "insufficient points"}
    if pres.size > n_present:
        pres = rng.choice(pres, n_present, replace=False)
    if hist.size > n_hist:
        hist = rng.choice(hist, n_hist, replace=False)
    xyc = xy[pres]
    P_pred = Z[hist] @ Z[pres].T                 # (nh, np) dot similarities
    P_obs = ruzicka_rect(X[hist], X[pres])       # (nh, np)

    def _centroid(P):
        idx = np.argpartition(-P, kth=topk - 1, axis=1)[:, :topk]   # top-k present/site
        return xyc[idx].mean(axis=1)                                # (nh, 2)

    d_pred = _centroid(P_pred) - xy[hist]
    d_obs = _centroid(P_obs) - xy[hist]
    nrm = np.linalg.norm(d_pred, axis=1) * np.linalg.norm(d_obs, axis=1) + 1e-12
    cos = (d_pred * d_obs).sum(1) / nrm
    from scipy.stats import pearsonr
    pc = np.array([np.corrcoef(P_pred[i], P_obs[i])[0, 1] for i in range(hist.size)])
    return {"n_hist": int(hist.size), "n_present": int(pres.size), "topk": topk,
            "mean_cos_displacement": float(np.nanmean(cos)),
            "corr_disp_EW": float(pearsonr(d_pred[:, 0], d_obs[:, 0])[0]),
            "corr_disp_NS": float(pearsonr(d_pred[:, 1], d_obs[:, 1])[0]),
            "mean_profile_corr": float(np.nanmean(pc)),
            "d_pred": d_pred.astype("float32"), "d_obs": d_obs.astype("float32"),
            "xy_hist": xy[hist].astype("float32"), "hist_year": yrs[hist]}


def cell_xy(rows, cols, ref_raster):
    """Cell-center (x, y) in the ref-grid CRS (Albers: x=easting/E–W, y=northing/N–S)."""
    import rasterio
    with rasterio.open(ref_raster) as src:
        t = src.transform
    r = np.asarray(rows) + 0.5; c = np.asarray(cols) + 0.5
    return np.stack([t.c + c * t.a + r * t.b, t.f + c * t.d + r * t.e], axis=1)


# ----------------------------- orchestration -----------------------------

def _load_model(config):
    import torch
    from .model_arch import MultiStreamAutoencoder
    dm = np.load(os.path.join(config["paths"]["desk_output_dir"], "desk_meta.npz"), allow_pickle=True)
    schema = json.loads(str(dm["schema"]))
    spatial_kernel = int(dm["spatial_kernel"]) if "spatial_kernel" in dm else 0
    model = MultiStreamAutoencoder([int(d) for d in dm["stream_dims"]], int(dm["latent_dim"]),
                                   spatial_kernel)
    model.load_state_dict(torch.load(
        os.path.join(config["paths"]["desk_output_dir"], "env_model_semisup.pth"),
        map_location="cpu"))
    model.eval()
    return model, dm["mu"].astype("float32"), dm["sd"].astype("float32"), schema, int(dm["latent_dim"])


def encode_points(config, point_index):
    """Encode each ``(row,col,year)`` point with the eBird-only DESK → ``(N, latent)``.

    Returns ``(Z, ok)`` where ``ok`` masks points whose covariates were finite.
    """
    import torch
    from . import covariate_io as cio
    model, mu, sd, schema, latent = _load_model(config)
    states_dir = os.path.join(config["paths"]["hist_dir"], "yearly_states")
    rows, cols, years = point_index[:, 0], point_index[:, 1], point_index[:, 2]
    Z = np.full((len(point_index), latent), np.nan, dtype="float32")
    # Grid-native: encode each year's WHOLE grid (so the spatial residual conv sees
    # neighbours -- the same function the cube applies) and gather the points from it.
    for y in np.unique(years):
        sel = np.where(years == y)[0]
        covn, valid = cio.norm_grid(cio.load_state_stack(int(y), states_dir, schema), mu, sd)
        xg = torch.tensor(covn[None], dtype=torch.float32)
        mg = torch.tensor(valid[None])
        with torch.no_grad():
            zz, _ = model(xg, mg)                        # (1, H, W, L)
        zc = zz[0].numpy()
        for k in sel:
            if valid[rows[k], cols[k]]:
                Z[k] = zc[rows[k], cols[k]]
    return Z, ~np.isnan(Z).any(1)


def zspace_reconstruction(config, pidx, X, Z_desk, recent_year, to_rec, has_rec):
    """Per-cell reconstruction in Z-SPACE: is DESK's predicted z closer to the observed
    community's z than the no-change (2023) z is?

    Projects each observed amplitude community ``x(cell,year)`` into the SAME pinned ESK
    basis DESK was trained against (``project_into_z`` with the saved landmarks/proj_mat,
    matching the ESK's weekly smoothing), giving ``z_obs``. Then per historical point:
        err_desk     = || z_DESK(cell,year) - z_obs(cell,year) ||
        err_nochange = || z_obs(cell,recent) - z_obs(cell,year) ||   (assume it looked like 2023)
    DESK is a per-cell value-add where err_desk < err_nochange. Coordinates are comparable
    because both live in the one concrete ESK basis (no rotation freedom). Returns None if
    the ESK projection wasn't saved (i.e. ESK predates this feature -- re-run esk).
    """
    import json as _json
    from .esk_kernel import project_into_z, smooth_abundances
    zdir = config["desk"]["z_dir"]
    lmp, pmp = os.path.join(zdir, "esk_landmarks.npy"), os.path.join(zdir, "esk_projmat.npy")
    if not (os.path.exists(lmp) and os.path.exists(pmp)):
        return None
    landmarks, projmat = np.load(lmp), np.load(pmp)
    esk_meta = _json.load(open(os.path.join(zdir, "meta.json")))
    sigma, n_weeks, ld = float(esk_meta.get("sigma", 0.0)), int(esk_meta["n_weeks"]), Z_desk.shape[1]

    # Smooth (matching the ESK) + project into the basis, batched to bound memory.
    N = X.shape[0]
    z_obs = np.zeros((N, ld), dtype="float32")
    for s in range(0, N, 20000):
        e = min(s + 20000, N)
        xb = smooth_abundances(X[s:e], n_weeks, sigma) if sigma > 0 else X[s:e]
        z_obs[s:e] = project_into_z(xb, landmarks, projmat)[:, :ld]

    z_nc = np.full_like(z_obs, np.nan)
    z_nc[has_rec] = z_obs[to_rec[has_rec]]
    err_desk = np.linalg.norm(Z_desk - z_obs, axis=1)
    err_nc = np.linalg.norm(z_nc - z_obs, axis=1)
    hist = (pidx[:, 2] != recent_year) & has_rec
    # Validity: at recent points, z_obs must reproduce the ESK Z (same projection, same
    # smoothed E) -- confirms the basis truly matches. Compare to the saved ESK Z.
    resid = float("nan")
    try:
        vm = np.load(os.path.join(zdir, "valid_mask.npy"))
        ez = np.load(os.path.join(zdir, "Z.npy"))[:, :ld]
        cell_row = {(int(r), int(c)): i for i, (r, c) in enumerate(np.argwhere(vm))}
        rec = np.where(pidx[:, 2] == recent_year)[0]
        d = [np.linalg.norm(z_obs[k] - ez[cell_row[(int(pidx[k, 0]), int(pidx[k, 1]))]])
             for k in rec if (int(pidx[k, 0]), int(pidx[k, 1])) in cell_row]
        resid = float(np.median(d)) if d else float("nan")
    except Exception:
        pass
    if hist.sum() < 4:
        return {"n": int(hist.sum()), "note": "too few historical points"}
    ed, en = err_desk[hist], err_nc[hist]
    out = {"n": int(hist.sum()),
           "median_err_desk": float(np.median(ed)),
           "median_err_nochange": float(np.median(en)),
           "frac_desk_beats_nochange": float(np.mean(ed < en)),
           "recent_basis_residual": resid,
           "rows": pidx[hist, 0], "cols": pidx[hist, 1],
           "err_desk": ed.astype("float32"), "err_nochange": en.astype("float32")}
    # If enrich saved a spatial holdout, split the value-add into held-out (honest, unseen
    # cells) vs train -- held-out frac_desk_beats_nochange is the number that grades enrich.
    ho_path = os.path.join(config["paths"]["desk_output_dir"], "holdout_cells.npy")
    if os.path.exists(ho_path):
        ho = np.load(ho_path)
        is_ho = ho[pidx[hist, 0], pidx[hist, 1]]
        for lab, m in (("heldout", is_ho), ("train", ~is_ho)):
            if m.sum() >= 4:
                out[f"n_{lab}"] = int(m.sum())
                out[f"frac_desk_beats_nochange_{lab}"] = float(np.mean(ed[m] < en[m]))
                out[f"median_err_desk_{lab}"] = float(np.median(ed[m]))
                out[f"median_err_nochange_{lab}"] = float(np.median(en[m]))
    return out


def run_validate(config=None, n_pairs=20000, cka_sample=800, seed=0):
    """Compare eBird-only DESK predictions to BBS structure per period; write a report."""
    from .config_utils import load_config
    config = load_config(config) if not isinstance(config, dict) else config
    bc = config["bbs"]
    rng = np.random.default_rng(seed)

    zt = bc["z_dir"]                              # spacetime point set from build_amplitude_points
    X = np.load(os.path.join(zt, "X_points.npy"))
    pidx = np.load(os.path.join(zt, "point_index.npy"))
    meta = json.load(open(os.path.join(zt, "points_meta.json")))
    recent_year = int(meta["recent_year"])

    Z, ok = encode_points(config, pidx)
    X, pidx, Z = X[ok], pidx[ok], Z[ok]
    years = pidx[:, 2]

    # No-change null: reuse each cell's RECENT (recent_year) latent/observation for every
    # year ("assume the community never changed"). Recent anchor points cover every
    # eBird-valid cell, so map each point to its own cell's recent row. The gap between
    # DESK and this null is the only interpretable readout: most of the observed structure
    # is the persistent SPATIAL pattern, which the null already reproduces, so a high raw
    # CKA means little without it.
    rec_key = {}
    for k in np.where(years == recent_year)[0]:
        rec_key[(int(pidx[k, 0]), int(pidx[k, 1]))] = k
    to_rec = np.array([rec_key.get((int(pidx[k, 0]), int(pidx[k, 1])), -1)
                       for k in range(len(pidx))])
    has_rec = to_rec >= 0
    Zrec = np.full_like(Z, np.nan); Zrec[has_rec] = Z[to_rec[has_rec]]
    Xrec = np.full_like(X, np.nan); Xrec[has_rec] = X[to_rec[has_rec]]

    def _bucket_report(mask, label):
        idx = np.where(mask)[0]
        if idx.size < 4:
            return {"period": label, "n": int(idx.size), "note": "too few points"}
        # Cross-sectional (spatial) structure recovery for this period. DISTINCT pairs only
        # (self-pairs would inject similarity=1). CKA is scale-invariant structure; MSE is a
        # secondary calibration check on the raw dot vs Ruzicka.
        m = min(cka_sample, idx.size)
        samp = rng.choice(idx, m, replace=False)
        pi, pj = distinct_pairs(m, n_pairs, rng)
        sp, so = pair_sims(Z[samp], X[samp], (pi, pj))
        Kz = Z[samp] @ Z[samp].T
        Lx = ruzicka_similarity_matrix(X[samp])
        out = {"period": label, "n": int(idx.size), "n_sampled": int(m),
               "mse": float(np.mean((sp - so) ** 2)),
               "cka": linear_cka(Kz, Lx), "mantel": mantel_r(Kz, Lx)}
        # Baselines on the recent-anchored subset (all historical cells qualify), one common
        # subset so the gap is apples-to-apples:
        #   cka_nochange   -- no-change null (each cell's recent latent) vs THIS period observed
        #   cka_gain (+CI) -- DESK CKA minus null CKA: the value added over "assume no change".
        #                     CI from subsampling; gain CI overlapping 0 => no real added value.
        #   cka_obs_change -- observed(period) vs observed(recent). NOTE: inflated toward 1 by the
        #                     fixed-2023 intra-annual shape, so it is an UPPER BOUND on how much
        #                     structural change is even representable, not the true change.
        samp_r = samp[has_rec[samp]]
        if samp_r.size >= 12:
            Kz_h = Z[samp_r] @ Z[samp_r].T
            Lx_h = ruzicka_similarity_matrix(X[samp_r])
            Lx_r = ruzicka_similarity_matrix(Xrec[samp_r])
            Kz_null = Zrec[samp_r] @ Zrec[samp_r].T
            cka_desk = linear_cka(Kz_h, Lx_h)
            cka_null = linear_cka(Kz_null, Lx_h)
            (cka_lo, cka_hi), (gain_lo, gain_hi) = _cka_gain_ci(Kz_h, Lx_h, Kz_null, rng)
            out["cka_nochange"] = cka_null
            out["cka_gain"] = cka_desk - cka_null
            out["cka_gain_ci95"] = [gain_lo, gain_hi]
            out["cka_ci95"] = [cka_lo, cka_hi]
            out["cka_obs_change_upperbound"] = linear_cka(Lx_h, Lx_r)
        return out

    report = {"recent_control": _bucket_report(years == recent_year, f"recent({recent_year})")}
    hist_years = sorted(set(int(y) for y in years if y != recent_year))
    if hist_years:
        lo, hi = min(hist_years), max(hist_years)
        for d0 in range(lo - lo % 10, hi + 1, 10):
            report[f"{d0}s"] = _bucket_report((years >= d0) & (years < d0 + 10)
                                              & (years != recent_year), f"{d0}s")
    report["all_historical"] = _bucket_report(years != recent_year, "all_historical")

    # --- temporal-nuance metrics (turnover magnitude + spatiotemporal analog direction) ---
    from src.config_utils import load_data_config
    ref_raster = load_data_config()["grid"]["ref_raster"]
    xy = cell_xy(pidx[:, 0], pidx[:, 1], ref_raster)
    turn = temporal_turnover_agreement(Z, X, pidx, recent_year,
                                       min_gap=int(bc.get("turnover_min_gap", 5)))
    analog = analog_displacement(Z, X, pidx, xy, recent_year, rng)
    dirchg = directional_change_agreement(Z, X, pidx, recent_year, rng)
    recon = zspace_reconstruction(config, pidx, X, Z, recent_year, to_rec, has_rec)
    report["directional_change"] = {k: v for k, v in dirchg.items()
                                     if k in ("n_sites", "mean_dir_cos", "median_dir_cos",
                                              "frac_same_dir", "mean_dir_cos_null", "note")}
    report["directional_change"]["_note"] = ("DIRECTION of community change (magnitude-"
        "canceling), unlike turnover which is magnitude-only. Read mean_dir_cos RELATIVE to "
        "mean_dir_cos_null (permuted-site baseline); frac_same_dir null=0.5.")
    report["temporal_turnover"] = {k: v for k, v in turn.items()
                                   if k in ("n_sites", "spearman_turnover", "note")}
    # Magnitudes, to compare against the raw-BBS ceiling (scripts/viz/raw_bbs_turnover.py):
    # obs = observed AMPLITUDE turnover (1 - Ruzicka on E*anomaly); if this is far below the
    # raw-BBS turnover, the amplitude construction (fixed shape + cap/shrink + frozen species)
    # is flattening real change. pred = the model's own turnover (over-predicts if >> obs).
    if "turnover_obs" in turn and turn["turnover_obs"].size:
        report["temporal_turnover"]["median_turnover_obs_amplitude"] = float(np.median(turn["turnover_obs"]))
        report["temporal_turnover"]["median_turnover_pred"] = float(np.median(turn["turnover_pred"]))
    report["temporal_turnover"]["_magnitude_only_note"] = ("turnover is MAGNITUDE-only "
        "(how much a community changed, not toward what) -- see directional_change for "
        "direction. Compare median_turnover_obs_amplitude to the raw-BBS ceiling to see how "
        "much the amplitude construction flattens real change.")
    # Partial Spearman: control for per-site time-span + broad spatial trend, which inflate
    # the raw value (both pred & obs turnover rise with time-depth and share spatial trends).
    if "turnover_pred" in turn and turn["turnover_pred"].size >= 8:
        txy = cell_xy(turn["rows"], turn["cols"], ref_raster)
        span = (int(recent_year) - turn["hist_year"]).astype(float)
        report["temporal_turnover"]["spearman_turnover_partial"] = partial_spearman(
            turn["turnover_pred"], turn["turnover_obs"], [span, txy[:, 0], txy[:, 1]])
        report["temporal_turnover"]["_partial_note"] = (
            "spearman_turnover_partial removes per-site span + broad space from both fields; "
            "if it collapses toward 0 the raw value was mostly the shared time-depth artifact.")
    if "d_pred" in analog:
        dp_a, do_a, xyh = analog["d_pred"], analog["d_obs"], analog["xy_hist"]
        cx, cy = xyh[:, 0], xyh[:, 1]
        nrm = np.linalg.norm(dp_a, axis=1) * np.linalg.norm(do_a, axis=1) + 1e-12
        cos_a = (dp_a * do_a).sum(1) / nrm
        perm = rng.permutation(len(dp_a))
        nrm_n = np.linalg.norm(dp_a, axis=1) * np.linalg.norm(do_a[perm], axis=1) + 1e-12
        cos_a_null = (dp_a * do_a[perm]).sum(1) / nrm_n
        report["analog"] = {
            "n_hist": analog["n_hist"], "n_present": analog["n_present"], "topk": analog["topk"],
            "mean_cos_displacement": float(np.mean(cos_a)),
            "mean_cos_displacement_null": float(np.mean(cos_a_null)),
            "corr_disp_EW_partial": _partial_corr(dp_a[:, 0], do_a[:, 0], [cx, cy]),
            "corr_disp_NS_partial": _partial_corr(dp_a[:, 1], do_a[:, 1], [cx, cy]),
            "_note": ("displacement cosine read vs its permutation null; EW/NS correlations have "
                      "site position partialled out (raw versions were inflated by domain "
                      "geometry -- edge sites' analogs point inward for both models). "
                      "profile_corr dropped: it re-measured the static spatial structure.")}
    else:
        report["analog"] = {k: v for k, v in analog.items() if k == "note"}

    if recon is not None:
        report["zspace_reconstruction"] = {k: v for k, v in recon.items()
            if k in ("n", "median_err_desk", "median_err_nochange", "frac_desk_beats_nochange",
                     "recent_basis_residual", "note", "n_heldout", "frac_desk_beats_nochange_heldout",
                     "median_err_desk_heldout", "median_err_nochange_heldout", "n_train",
                     "frac_desk_beats_nochange_train", "median_err_desk_train",
                     "median_err_nochange_train")}
        report["zspace_reconstruction"]["_note"] = ("PER-CELL reconstruction in the pinned ESK "
            "z-basis: err_desk = ||z_DESK - z_obs||, err_nochange = ||z_obs(2023) - z_obs||. "
            "frac_desk_beats_nochange > 0.5 => DESK reconstructs the past community better than "
            "assuming 2023. recent_basis_residual ~0 confirms the basis matches (z_obs reproduces "
            "the ESK Z at recent points).")

    out_dir = config["paths"]["desk_output_dir"]
    out = os.path.join(out_dir, "validate_report.json")
    os.makedirs(out_dir, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    # Bundle the per-site/per-point arrays for visualization (turnover maps + analog arrows).
    viz = os.path.join(out_dir, "validate_spacetime.npz")
    np.savez_compressed(
        viz,
        turn_rows=turn.get("rows", np.array([])), turn_cols=turn.get("cols", np.array([])),
        turnover_pred=turn.get("turnover_pred", np.array([])),
        turnover_obs=turn.get("turnover_obs", np.array([])),
        d_pred=analog.get("d_pred", np.zeros((0, 2))), d_obs=analog.get("d_obs", np.zeros((0, 2))),
        xy_hist=analog.get("xy_hist", np.zeros((0, 2))),
        analog_hist_year=analog.get("hist_year", np.array([])),
        recon_rows=(recon or {}).get("rows", np.array([])),
        recon_cols=(recon or {}).get("cols", np.array([])),
        recon_err_desk=(recon or {}).get("err_desk", np.array([])),
        recon_err_nochange=(recon or {}).get("err_nochange", np.array([])),
        ref_raster=np.array(ref_raster))

    print("[validate] SPATIAL structure recovery per period. gain = CKA(DESK) - CKA(no-change "
          "null); gain CI overlapping 0 => DESK adds nothing over 'assume no change':")
    for k, v in report.items():
        if "cka" in v:
            if "cka_gain" in v:
                gl, gh = v["cka_gain_ci95"]
                extra = f" | gain={v['cka_gain']:+.3f} [95% {gl:+.3f},{gh:+.3f}]"
            else:
                extra = ""
            print(f"  {v['period']:<16} n={v['n']:<7} cka={v['cka']:.3f}{extra}")
    dc = report.get("directional_change", {})
    if "mean_dir_cos" in dc:
        print(f"[validate] DIRECTION of change ({dc['n_sites']} sites): mean cos={dc['mean_dir_cos']:+.3f} "
              f"vs null={dc.get('mean_dir_cos_null', float('nan')):+.3f} | frac right way="
              f"{dc['frac_same_dir']:.3f} (null 0.5)")
    if "spearman_turnover" in report["temporal_turnover"]:
        tt = report["temporal_turnover"]
        part = tt.get("spearman_turnover_partial", float("nan"))
        print(f"[validate] turnover MAGNITUDE Spearman ({turn['n_sites']} sites, cosine self-sim): "
              f"raw={tt['spearman_turnover']:+.3f} | partial(span+space out)={part:+.3f}")
    a = report.get("analog", {})
    if "mean_cos_displacement" in a:
        print(f"[validate] analog displacement ({a['n_hist']} pts): cos={a['mean_cos_displacement']:+.3f} "
              f"vs null={a['mean_cos_displacement_null']:+.3f} | EW(partial)={a['corr_disp_EW_partial']:+.3f} "
              f"NS(partial)={a['corr_disp_NS_partial']:+.3f}")
    rc = report.get("zspace_reconstruction", {})
    if "median_err_desk" in rc:
        print(f"[validate] Z-SPACE reconstruction ({rc['n']} hist pts): err DESK={rc['median_err_desk']:.4f} "
              f"vs no-change={rc['median_err_nochange']:.4f} | DESK beats no-change in "
              f"{rc['frac_desk_beats_nochange']:.1%} of cells | basis residual={rc['recent_basis_residual']:.2e}")
        if "frac_desk_beats_nochange_heldout" in rc:
            print(f"           HELD-OUT cells ({rc['n_heldout']}): DESK beats no-change in "
                  f"{rc['frac_desk_beats_nochange_heldout']:.1%}  <- the honest enrich grade "
                  f"(err DESK={rc['median_err_desk_heldout']:.4f} vs {rc['median_err_nochange_heldout']:.4f})")
    print(f"[validate] report -> {out} ; viz arrays -> {viz}")
    return report


if __name__ == "__main__":
    run_validate()
