"""Validate: do DESK's predicted similarities reproduce the target's spatiotemporal structure?

DESK gives a predicted latent ``z(s,t)`` from that point's own-year covariates. At
**held-out** ``(cell, year)`` points we ask whether its predicted **similarities**
reproduce the target community similarities -- comparing at the **kernel level**
(``⟨z_i,z_j⟩`` vs ``Ruzicka(x_i,x_j)``), never raw coordinates, because Z is
basis/rotation-arbitrary. Both live on the same Ruzicka scale (``X_points`` are the
log1p community vectors that seeded the basis, and ``true_kernel_loss`` calibrated
``⟨z,z⟩`` to exactly that), so the comparison is fair. Reported per period with MSE +
basis-invariant CKA/Mantel.

Strong agreement, especially degrading gracefully rather than randomly back in time,
means the spatial-to-spatiotemporal extrapolation holds.

SCOPE, and the reason ``validate_bbs_routes`` also exists: the target here is the
trend reconstruction, which is built from USGS's published inverse-distance BBS
surface. So this grades DESK against an interpolated quantity, and an interpolation
baseline can win it for structural reasons. ``validate_bbs_routes`` grades the same
model against RAW route counts at genuinely surveyed cell-years, where reproducing an
interpolator cannot help. Read them together.
"""
import json
import os
import time

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
            "turnover_pred": tp.astype("float32"), "turnover_obs": to.astype("float32"),
            # The two sides are on DIFFERENT scales and their ratio is meaningless. pred is
            # 1-cosine (length-invariant); obs is 1-Ruzicka, which corresponds to a DOT and so
            # carries the representation's magnitude. Because the ESK target's ||z||^2 is ~0.67
            # rather than the contract's 1.0, ~80% of `obs` is that norm deficit and only ~20%
            # is rotation -- so pred/obs looks like a 3x under-prediction when a like-for-like
            # comparison shows none. Use spearman_turnover (rank, scale-free) for agreement, and
            # compare pred only against another cosine quantity.
            "_scale_warning": ("turnover_pred is 1-cos (length-invariant); turnover_obs is "
                               "1-Ruzicka (magnitude-bearing). Their RATIO is not "
                               "interpretable; only their rank agreement is.")}


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
    # Width is a property of the TRAINED net, not a config choice here: it sets state_dict
    # shapes, so rebuilding at the config's current width would fail to load older weights.
    model = MultiStreamAutoencoder(
        [int(d) for d in dm["stream_dims"]], int(dm["latent_dim"]), spatial_kernel,
        hidden_width=(int(dm["hidden_width"]) if "hidden_width" in dm else None),
        mlp_expansion=(int(dm["mlp_expansion"]) if "mlp_expansion" in dm else 4))
    # Put the net on the accelerator. This path used to load with map_location="cpu" and never
    # call .to(device), so all 60 whole-grid forwards ran on CPU (~480 ms each, ~87% of this
    # path's measured cost) on a job submitted to a GPU queue. The CPU fallback is retained so
    # the stage still runs on a login node.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.load_state_dict(torch.load(
        os.path.join(config["paths"]["desk_output_dir"], "env_model_semisup.pth"),
        map_location=device))
    model.to(device).eval()
    return (model, dm["mu"].astype("float32"), dm["sd"].astype("float32"), schema,
            int(dm["latent_dim"]), device)


def encode_points(config, point_index):
    """Encode each ``(row,col,year)`` point with the eBird-only DESK → ``(N, latent)``.

    Returns ``(Z, ok)`` where ``ok`` masks points whose covariates were finite.
    """
    import torch
    from . import covariate_io as cio
    model, mu, sd, schema, latent, device = _load_model(config)
    states_dir = os.path.join(config["paths"]["hist_dir"], "yearly_states")
    # The model's mu/sd and input widths are positional; refuse to encode against a
    # states dir that was rebuilt with a different channel set or order.
    cio.assert_schema_compatible(schema, cio.load_schema(states_dir),
                                 context="validate_spacetime")
    rows, cols, years = point_index[:, 0], point_index[:, 1], point_index[:, 2]
    Z = np.full((len(point_index), latent), np.nan, dtype="float32")
    # Grid-native: encode each year's WHOLE grid (so the spatial residual conv sees
    # neighbours -- the same function the cube applies) and gather the points from it.
    for y in np.unique(years):
        sel = np.where(years == y)[0]
        covn, valid = cio.norm_grid(cio.load_state_stack(int(y), states_dir, schema), mu, sd)
        xg = torch.tensor(covn[None], dtype=torch.float32, device=device)
        mg = torch.tensor(valid[None], device=device)
        with torch.no_grad():
            zz, _ = model(xg, mg)                        # (1, H, W, L)
        zc = zz[0].float().cpu().numpy()
        for k in sel:
            if valid[rows[k], cols[k]]:
                Z[k] = zc[rows[k], cols[k]]
    return Z, ~np.isnan(Z).any(1)


#: Per-point arrays from ``zspace_reconstruction``: they go to the .npz for viz, never to the
#: JSON report. Everything else is a scalar figure and belongs in the report.
RECON_ARRAY_KEYS = ("rows", "cols", "err_desk", "err_nochange")


def report_scalars(recon):
    """Report-safe view of a reconstruction dict: drop the per-point arrays, keep every scalar.

    This was an ALLOW-LIST of key names, which silently dropped anything newly added -- the
    interpolation bar was computed, never listed, and so could never print, because the summary
    reads this filtered dict rather than the original. An exclusion cannot fail that way.
    """
    return {k: v for k, v in recon.items() if k not in RECON_ARRAY_KEYS}


def is_ho_hist(ho, pidx, hist):
    """Held-out flag per HISTORICAL point, in ``pidx[hist]`` row order."""
    return ho[pidx[hist, 0], pidx[hist, 1]]


def zspace_reconstruction(config, pidx, X, Z_desk, recent_year, to_rec, has_rec):
    """Per-cell reconstruction in Z-SPACE: is DESK's predicted z closer to the observed
    community's z than the no-change (2023) z is?

    Projects each observed community ``x(cell,year)`` into the SAME pinned ESK
    basis DESK was trained against (``project_into_z`` with the saved landmarks/proj_mat,
    matching the ESK's weekly smoothing), giving ``z_obs``. Then per historical point:
        err_desk     = || z_DESK(cell,year) - z_obs(cell,year) ||
        err_nochange = || z_obs(cell,recent) - z_obs(cell,year) ||   (assume it looked like 2023)
    DESK is a per-cell value-add where err_desk < err_nochange. Coordinates are comparable
    because both live in the one concrete ESK basis (no rotation freedom).
    """
    import json as _json
    from .esk_kernel import project_into_z, smooth_abundances
    zdir = config["desk"]["z_dir"]
    lmp, pmp = os.path.join(zdir, "esk_landmarks.npy"), os.path.join(zdir, "esk_projmat.npy")
    # run_spacetime_esk always writes both, so a miss is a broken or partial ESK run, not an
    # older artifact. Returning None here instead made every recon_* output silently empty.
    for pth in (lmp, pmp):
        if not os.path.exists(pth):
            raise FileNotFoundError(f"ESK projection missing: {pth}; re-run the spacetime-esk stage")
    meta_path = os.path.join(zdir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"missing ESK kernel contract: {meta_path}")
    with open(meta_path, encoding="utf-8") as fh:
        basis_meta = json.load(fh)
    if basis_meta.get("kernel") != "ruzicka" or bool(basis_meta.get("centered", True)):
        raise ValueError(f"validation requires uncentered Ružička ESK basis; got {basis_meta}")
    landmarks, projmat = np.load(lmp), np.load(pmp)
    esk_meta = _json.load(open(os.path.join(zdir, "meta.json")))
    sigma, n_weeks, ld = float(esk_meta.get("sigma", 0.0)), int(esk_meta["n_weeks"]), Z_desk.shape[1]

    # Smooth (matching the ESK) + project into the basis, batched to bound memory. The basis is
    # uploaded ONCE: this loop previously re-transferred the landmark and projection matrices to
    # the device on every outer batch (~45 times for 890k points).
    import torch as _torch
    _dev = "cuda" if _torch.cuda.is_available() else "cpu"
    _lm = _torch.tensor(np.asarray(landmarks), device=_dev, dtype=_torch.float32)
    _pm = _torch.tensor(np.asarray(projmat), device=_dev, dtype=_torch.float32)
    N = X.shape[0]
    z_obs = np.zeros((N, ld), dtype="float32")
    for s in range(0, N, 20000):
        e = min(s + 20000, N)
        xb = smooth_abundances(X[s:e], n_weeks, sigma) if sigma > 0 else X[s:e]
        z_obs[s:e] = project_into_z(xb, _lm, _pm, device=_dev)[:, :ld]

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

    # --- the error, decomposed ------------------------------------------------------------
    # ||z_desk - z_obs||^2 splits EXACTLY into a magnitude term (||z_desk|| - ||z_obs||)^2 and an
    # angular term 2||z_desk|| ||z_obs||(1 - cos). Reporting only the total (as this function did)
    # cannot say whether DESK points the wrong way or merely too short -- and those have different
    # fixes. The two halves also trade off: shrinking is the MSE-optimal answer to a poor angle,
    # which is why the norm deficit and the direction deficit are one phenomenon, not two.
    from .validate_baselines import error_decomposition
    tot_d, mag_d, ang_d, cos_d = error_decomposition(Z_desk[hist], z_obs[hist])
    n_d = np.linalg.norm(Z_desk[hist], axis=1)
    n_o = np.linalg.norm(z_obs[hist], axis=1)
    out.update({
        "err_total_sq_desk": float(np.median(tot_d)),
        "err_magnitude_sq_desk": float(np.median(mag_d)),
        "err_angular_sq_desk": float(np.median(ang_d)),
        # Shares of the MEAN, not of the medians: medians of two terms do not add to the median
        # of their sum, and a reader comparing the three medians above would otherwise conclude
        # the identity is broken. The identity holds per point; these are its expectation split.
        "err_magnitude_share": float(np.mean(mag_d) / max(np.mean(tot_d), 1e-12)),
        "err_angular_share": float(np.mean(ang_d) / max(np.mean(tot_d), 1e-12)),
        "median_cos_desk_vs_obs": float(np.nanmedian(cos_d)),
        # Self-similarity. The kernel contract pins ||z||^2 = Ruzicka(x,x) = 1 EXACTLY (Sum-min
        # over Sum-max of a vector with itself is 1), so a deficit is the model asserting a
        # community is less than fully similar to itself. Separated for z_desk and z_obs because
        # the causes differ and the fixes differ: z_desk short = MSE shrinkage (the model
        # hedging), z_obs short = basis truncation (what the finite ESK projection cannot
        # represent). The report could not previously tell them apart.
        "median_z_desk_norm2": float(np.median(n_d ** 2)),
        "median_z_obs_norm2": float(np.median(n_o ** 2)),
        # What the error distance MEANS in the units the space represents: the Ruzicka similarity
        # between the predicted and the observed community. The naive 1 - d^2/2 assumes both norms
        # are 1 and so flatters the model by exactly the norm deficit.
        "implied_ruzicka_desk_vs_obs": float(np.median((n_d ** 2 + n_o ** 2 - tot_d) / 2.0)),
        "implied_ruzicka_naive_if_unit_norm": float(np.median(1.0 - tot_d / 2.0)),
    })
    # Per-dimension shrinkage: the magnitude half resolved along the basis. The ESK is
    # kernel-PCA, so dim index is ordered by eigenvalue. FLAT means a uniform rescale, which the
    # downstream's fitted w_env absorbs and which therefore costs nothing. FALLING with the index
    # means the low-eigenvalue directions are squeezed hardest -- and since MSE shrinks whatever
    # it predicts worst, those are the temporal ones -- so the kernel would be tilted toward
    # spatial similarity and the GP prior distorted. Aggregate ||z||^2 cannot distinguish these.
    #
    # Read it as a VARIANCE ratio, so a uniform norm scale c appears as c^2: a flat profile at
    # 0.64 means every direction is scaled 0.8. If the shrinkage really is uniform then
    # shrinkage_median should agree with median_z_desk_norm2 / median_z_obs_norm2; a large
    # disagreement between those two is itself evidence of a tilt.
    #
    # Measured behaviour of this diagnostic on planted inputs: a uniform 0.8 rescale gives
    # slope +0.00000 with the error 100% in the magnitude term (a rescale cannot change a
    # direction); a linear tilt from 1.0 to 0.2 gives slope -0.031 and splits the error 53%
    # magnitude / 47% angular.
    v_d = np.var(Z_desk[hist], axis=0)
    v_o = np.var(z_obs[hist], axis=0)
    prof = np.where(v_o > 1e-12, v_d / np.maximum(v_o, 1e-12), np.nan)
    # latent_dim long (~64), so this is a figure rather than a per-point array: it stays in the
    # JSON report. As a list, not an ndarray -- report_scalars keeps it and json.dump must be
    # able to write it.
    out["shrinkage_by_dim"] = [None if not np.isfinite(v) else float(v) for v in prof]
    ok_p = np.isfinite(prof)
    if ok_p.sum() >= 3:
        kk = np.arange(len(prof))[ok_p]
        out["shrinkage_slope"] = float(np.polyfit(kk, prof[ok_p], 1)[0])
        out["shrinkage_median"] = float(np.median(prof[ok_p]))
    # DESK saves a spatial holdout; split the value-add into held-out (honest, unseen
    # cells) vs train -- held-out frac_desk_beats_nochange is the number that counts.
    ho_path = os.path.join(config["paths"]["desk_output_dir"], "holdout_cells.npy")
    if os.path.exists(ho_path):
        ho = np.load(ho_path)
        # The no-change null assumes 60 years of stasis, so it is weakest exactly where the
        # historical points are densest -- beating it is a low bar. This is the real one:
        # interpolate the OBSERVED z from training cells surveyed the SAME year. No covariates,
        # no learning. On the val MSE and the direction diagnostic this bar was not cleared.
        from .validate_baselines import zspace_idw_baseline
        e_idw = zspace_idw_baseline(pidx, z_obs, ho, hist)
        # The SAME-YEAR spatial bar above cannot run in a withheld year -- there are no training
        # cells that year, so every withheld row comes back NaN and the deep past, which is the
        # whole point of the temporal experiment, ends up with no bar at all. The spacetime
        # variant borrows across years as well as space and so does reach it.
        try:
            from .validate_baselines import spacetime_idw_baseline, spacetime_idw_z
            bf_p = os.path.join(config["paths"]["desk_output_dir"], "buffer_cells.npy")
            bf = np.load(bf_p) if os.path.exists(bf_p) else np.zeros_like(ho)
            hy = [int(y) for y in (config["desk"].get("trend", {}).get("holdout_years") or [])]
            _e, ratio = spacetime_idw_baseline(pidx, z_obs, ho, np.zeros(len(pidx), bool),
                                               buffer_mask=bf, exclude_years=hy, verbose=False)
            z_st = spacetime_idw_z(pidx, z_obs, ho, float(ratio),
                                   buffer_mask=bf, exclude_years=hy)
            e_st = np.linalg.norm(z_st[hist] - z_obs[hist], axis=1)
            fst = np.isfinite(e_st)
            if fst.sum() >= 4:
                out["spacetime_idw_ratio_cells_per_year"] = float(ratio)
                out["median_err_spacetime_idw"] = float(np.median(e_st[fst]))
                out["frac_desk_beats_spacetime_idw"] = float(np.mean(ed[fst] < e_st[fst]))
                out["n_spacetime_idw_scored"] = int(fst.sum())
                hs = fst & is_ho_hist(ho, pidx, hist)
                if hs.sum() >= 4:
                    out["frac_desk_beats_spacetime_idw_heldout"] = float(
                        np.mean(ed[hs] < e_st[hs]))
                    out["median_err_spacetime_idw_heldout"] = float(np.median(e_st[hs]))
                if hy:
                    inhy = np.isin(pidx[hist, 2], np.asarray(hy)) & fst
                    if inhy.sum() >= 4:
                        # the rows the same-year bar cannot reach at all
                        out["frac_desk_beats_spacetime_idw_withheld_years"] = float(
                            np.mean(ed[inhy] < e_st[inhy]))
                        out["n_withheld_years_scored"] = int(inhy.sum())
        except Exception as exc:                  # diagnostic only
            print(f"[validate] spacetime IDW bar on reconstruction unavailable ({exc})")
        fin = np.isfinite(e_idw)
        if fin.sum() >= 4:
            out["median_err_idw"] = float(np.median(e_idw[fin]))
            out["frac_desk_beats_idw"] = float(np.mean(ed[fin] < e_idw[fin]))
            out["n_idw_scored"] = int(fin.sum())
            hm = fin & is_ho_hist(ho, pidx, hist)
            if hm.sum() >= 4:
                out["frac_desk_beats_idw_heldout"] = float(np.mean(ed[hm] < e_idw[hm]))
                out["median_err_idw_heldout"] = float(np.median(e_idw[hm]))
        is_ho = is_ho_hist(ho, pidx, hist)
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

    from src.config_utils import target_points_dir
    zt = target_points_dir(config)                # raw-BBS target, else the trend points
    X = np.load(os.path.join(zt, "X_points.npy"))
    pidx = np.load(os.path.join(zt, "point_index.npy"))
    meta = json.load(open(os.path.join(zt, "points_meta.json")))
    recent_year = int(meta["recent_year"])

    # Phase timing. Four separate attempts to explain this stage's runtime by benchmarking
    # individual functions were all wrong, so measure it in situ instead of reasoning about it.
    _t0 = time.perf_counter()
    _marks = []

    def _phase(name):
        nonlocal _t0
        now = time.perf_counter()
        _marks.append((name, now - _t0))
        print(f"[validate:timing] {name:<26} {now - _t0:7.1f} s", flush=True)
        _t0 = now

    # The trainer supervised z_ema, not raw z, so that is what the report grades. Raw z is
    # what the CUBE exports (the population model below it supplies demographic lag), so it is
    # kept as a secondary figure -- the two answer different questions and with a learned
    # ~10 y half-life they are not close. Grading raw here was measuring a quantity the
    # objective never optimized.
    from .validate_bbs_routes import desk_z_ema
    Z_raw, ok = encode_points(config, pidx)
    _phase("encode_points (raw z)")
    Z, ema_meta = desk_z_ema(config, pidx)      # returns (Z, metadata), not a bare array
    ok = ok & np.isfinite(Z).all(axis=1)
    _phase("encode_points (z_ema)")
    # Every metric below is computed ONLY on these points. `encode_points` fills Z solely
    # where `norm_grid`'s all-channels-finite mask holds, so points over cells the covariates
    # do not cover are dropped here rather than scored -- this path reads the STATES, never
    # the gap-filled Z cube, so the cube's year-invariant stage-2 fill cannot leak in. The
    # dropped count is reported because it is large (the covariate footprint is much smaller
    # than the land mask) and it bounds what these metrics can claim to cover.
    n_points_total, n_encoded = int(len(ok)), int(ok.sum())
    print(f"[validate] encoded {n_encoded}/{n_points_total} points; "
          f"{n_points_total - n_encoded} dropped for non-finite covariates "
          f"(outside the covariate footprint)", flush=True)
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

    report = {
        "point_coverage": {
            "n_points_total": n_points_total, "n_encoded": n_encoded,
            "n_dropped_nonfinite_covariates": n_points_total - n_encoded,
            "_note": ("metrics cover only encoded points -- those inside the covariate "
                      "footprint. This path encodes from state_*.npz, never from the "
                      "gap-filled Z cube, so no fill enters these numbers."),
        },
        "recent_control": _bucket_report(years == recent_year, f"recent({recent_year})"),
    }
    hist_years = sorted(set(int(y) for y in years if y != recent_year))
    if hist_years:
        lo, hi = min(hist_years), max(hist_years)
        for d0 in range(lo - lo % 10, hi + 1, 10):
            report[f"{d0}s"] = _bucket_report((years >= d0) & (years < d0 + 10)
                                              & (years != recent_year), f"{d0}s")
    report["all_historical"] = _bucket_report(years != recent_year, "all_historical")
    _phase("bucket reports")

    # --- temporal-nuance metrics (turnover magnitude + spatiotemporal analog direction) ---
    from src.config_utils import load_data_config
    ref_raster = load_data_config()["grid"]["ref_raster"]
    xy = cell_xy(pidx[:, 0], pidx[:, 1], ref_raster)
    turn = temporal_turnover_agreement(Z, X, pidx, recent_year,
                                       min_gap=int(bc.get("turnover_min_gap", 5)))
    _phase("turnover")
    analog = analog_displacement(Z, X, pidx, xy, recent_year, rng)
    _phase("analog")
    dirchg = directional_change_agreement(Z, X, pidx, recent_year, rng)
    _phase("directional change")
    recon = zspace_reconstruction(config, pidx, X, Z, recent_year, to_rec, has_rec)
    _phase("zspace recon (projection)")

    # Direction of change, per epoch pair, against interpolation rather than a permutation null.
    # Folded in here rather than run as its own stage: it needs exactly what this function has
    # already paid for -- the encoded z and the projected z_obs.
    epochs_panel = ladder = None
    ho_path = os.path.join(config["paths"]["desk_output_dir"], "holdout_cells.npy")
    bf_path = os.path.join(config["paths"]["desk_output_dir"], "buffer_cells.npy")
    if os.path.exists(ho_path):
        from .validate_baselines import epoch_direction_panel, per_era_attenuation
        from .esk_kernel import project_points_to_z
        ho = np.load(ho_path)
        bf = np.load(bf_path) if os.path.exists(bf_path) else np.zeros_like(ho)
        z_obs_pts = project_points_to_z(X, config["desk"]["z_dir"], Z.shape[1])
        # Hoisted ABOVE the epoch panel: the panel needs it too. It used to be computed only
        # for baseline_panel below, which is why the panel's IDW bar was free to source from
        # withheld years -- it interpolated the answer's own year while the model saw none of it.
        _tr_cfg = config["desk"].get("trend", {})
        hy = [int(y) for y in (_tr_cfg.get("holdout_years") or [])]
        if z_obs_pts is not None:
            zm = {(int(r), int(c), int(y)): Z[i]
                  for i, (r, c, y) in enumerate(pidx) if ok[i]}
            # How much of the reported dir-cos is eaten by single-year measurement noise, per
            # era. Printed before the panel so the panel's numbers can be read against it.
            # Per-DIMENSION signal/noise, read against zspace_reconstruction's shrinkage profile.
            # The shrinkage tilt (leading dims ~1.11, trailing ~0.43) is only a defect if the
            # trailing directions carry temporal SIGNAL. Noise is ~half a same-cell difference and
            # is high-dimensional and low-variance per direction, so it should land in exactly those
            # trailing dims -- in which case shrinking them is correct and the tilt is the model
            # denoising. This is the measurement that separates the two readings.
            from .validate_baselines import per_dimension_signal_noise
            sn = per_dimension_signal_noise(pidx, z_obs_pts)
            if "snr_slope" in sn:
                report["per_dimension_signal_noise"] = sn
                verdict = ("signal concentrates in the LEADING dims -> the shrinkage tilt discards "
                           "little signal, so it is appropriate rather than a defect"
                           if sn["snr_slope"] < 0 else
                           "signal persists into the TRAILING dims -> the shrinkage tilt is "
                           "discarding temporal signal, which makes it a real defect")
                print(f"[validate] per-dimension signal/noise (gap {sn['gap_years']}yr, "
                      f"{sn['n_adjacent_pairs']:,} adjacent / {sn['n_gap_pairs']:,} gap pairs): "
                      f"SNR leading-8 {sn['snr_leading_8']:.3f} vs trailing-8 "
                      f"{sn['snr_trailing_8']:.3f}, slope {sn['snr_slope']:+.5f}/dim, "
                      f"{sn['signal_share_leading_half']:.0%} of signal in the leading half")
                print(f"[validate]   => {verdict}")
            else:
                print(f"[validate] per-dimension signal/noise unavailable ({sn.get('note')})")

            atten = per_era_attenuation(pidx, z_obs_pts)
            if atten:
                print("[validate] dir-cos attenuation from single-year survey noise "
                      "(raw BBS, ~1.08 routes/cell-year; 0.80 => observed 0.40 ~ true 0.50):")
                for era, a in atten.items():
                    print(f"  {era}  attenuation {a['dir_cos_attenuation']:.2f}  "
                          f"(noise is {a['noise_share_of_long_gap']:.0%} of the long-gap "
                          f"difference; n_adj={a['n_adjacent_pairs']})")
            # The spacetime bar for the epoch panel: available even for epochs inside the
            # holdout, where the per-epoch spatial bar correctly reads n/a. Built once here and
            # handed in, so the panel and the ladder below score against the same alternative.
            z_st_panel = None
            try:
                from .validate_baselines import spacetime_idw_baseline, spacetime_idw_z
                _e, _r = spacetime_idw_baseline(pidx, z_obs_pts, ho,
                                                np.zeros(len(pidx), bool),
                                                buffer_mask=bf, exclude_years=hy, verbose=False)
                z_st_panel = spacetime_idw_z(pidx, z_obs_pts, ho, float(_r),
                                             buffer_mask=bf, exclude_years=hy)
                print(f"[validate] epoch-panel spacetime bar at ratio={_r:g} cells/yr")
            except Exception as exc:
                print(f"[validate] epoch-panel spacetime bar unavailable ({exc})")
            # ONE point-denoising half-width for the whole pipeline. This used to be
            # desk.trend.direction_half_width here and bbs_routes.window_half_width there, with
            # DEFAULT_TOL reused as a third value -- three constants for one concept.
            hw = int((config.get("target", {}) or {}).get(
                "smooth_half_width", _tr_cfg.get("direction_half_width", 0)))
            # Raw-space truth averaging, if the basis spans window means. The panel gates on that
            # itself and reports which estimand it used.
            _proj_fn = (lambda A: project_points_to_z(np.asarray(A, "float32"),
                                                     config["desk"]["z_dir"], Z.shape[1]))
            print("[validate] DIRECTION of change vs inverse-distance interpolation "
                  "(z_ema; pairs share cells and nest in time, so never pooled):")
            epochs_panel = epoch_direction_panel(pidx, None, z_obs_pts, zm, ho, bf,
                                                 exclude_years=hy, z_spacetime=z_st_panel,
                                                 x_obs=X, project=_proj_fn)
            if hw:
                # Same panel, endpoints averaged over +/-hw years on model, target and bar
                # alike. The GAP between the two tables is the noise measurement.
                print(f"[validate] same panel with endpoints averaged over +/-{hw} yr "
                      f"(both sides windowed identically; the gap vs the table above is the "
                      f"noise that was eating the estimate):")
                epochs_panel = {"single_year": epochs_panel,
                                "windowed": epoch_direction_panel(
                                    pidx, None, z_obs_pts, zm, ho, bf,
                                    exclude_years=hy, half_width=hw,
                                    z_spacetime=z_st_panel,
                                    x_obs=X, project=_proj_fn),
                                "attenuation_by_era": atten}
            else:
                epochs_panel = {"single_year": epochs_panel, "attenuation_by_era": atten}

            # The full ladder, per era. Each rung is handed different information, so the row
            # says WHICH claim survives: beating no-change is nearly free, beating
            # borrowed_delta means the covariates say something neighbours' trends do not.
            from .validate_baselines import baseline_panel, spacetime_idw_z
            print("[validate] BASELINE LADDER (z_ema, held-out cells):")
            ladder = baseline_panel(pidx, z_obs_pts, Z, ho, int(recent_year), buffer_mask=bf,
                                    exclude_years=hy)

            # With a temporal holdout configured, the spatial table above answers only "unseen
            # PLACE". Add the buckets that answer "unseen YEAR" -- the backward-extrapolation
            # question, and the one the 1900 use case actually rests on.
            if hy:
                is_ho = ho[pidx[:, 0], pidx[:, 1]]
                in_hy = np.isin(pidx[:, 2], np.asarray(hy))
                # Distance from the training edge, per withheld row. This is the sweep's real
                # independent variable and it was never reported: pooling the whole block hides
                # it, and BBS coverage grows ~7x from 1966 to 1995, so each run's pooled figure
                # is dominated by its OWN shallow, cheap end (a 10.8 yr EMA half-life makes a
                # 1-2 year reach close to interpolation). That is why 0.1836/0.2050/0.2195
                # barely moved: three different target sets, each weighted to short distances.
                first_trained = int(min(y for y in np.unique(pidx[:, 2]) if int(y) not in set(hy)))
                dist = first_trained - pidx[:, 2]
                # The distance axis assumes the holdout is a contiguous block BELOW the training
                # data (backward extrapolation), which is what every sweep overlay does. A
                # withheld year ABOVE first_trained gets a negative distance and would be dropped
                # from every bin without a trace, so say so rather than quietly under-reporting.
                fwd = int(np.sum(in_hy & (dist <= 0)))
                if fwd:
                    print(f"[validate] NOTE {fwd:,} withheld rows lie at or after the first "
                          f"trained year ({first_trained}); the distance bins below cover "
                          f"BACKWARD extrapolation only and exclude them")
                # The window withheld in EVERY run of the sweep. Restricting to it makes the
                # runs compare like with like -- identical years, cells and truth -- so the
                # cross-run difference means extrapolation distance and nothing else.
                common = [int(y) for y in (_tr_cfg.get("common_holdout_years") or [])]
                # Guard, not an assumption: a common window containing a year this run TRAINED
                # on would put trained rows in a bucket labelled "withheld", and the cross-run
                # comparison would silently be measuring something else.
                leak = sorted(set(common) - set(hy))
                if leak:
                    raise ValueError(
                        f"desk.trend.common_holdout_years contains {leak}, which this run did "
                        f"NOT withhold (holdout_years {min(hy)}-{max(hy)}). The common window "
                        f"must be a subset of every run's holdout or the buckets are not common.")
                in_common = np.isin(pidx[:, 2], np.asarray(common)) if common else None
                if common:
                    print(f"[validate] common withheld window {min(common)}-{max(common)} "
                          f"(withheld in every run of the sweep); this run reaches it at "
                          f"distance {first_trained - max(common)}-{first_trained - min(common)} yr")
                else:
                    print("[validate] NOTE: desk.trend.common_holdout_years unset -- the "
                          "cross-run figures below are on this run's own block and are NOT "
                          "comparable across the sweep")
                ladder["temporal_buckets"] = {}
                buckets = [("unseen_year_seen_cell", in_hy & ~is_ho),
                           ("unseen_year_unseen_cell", in_hy & is_ho)]
                if in_common is not None:
                    buckets += [("common_window_seen_cell", in_common & ~is_ho),
                                ("common_window_unseen_cell", in_common & is_ho)]
                for lab, rows in buckets:
                    if int(rows.sum()) < 20:
                        print(f"  {lab}: {int(rows.sum())} rows, too few to report")
                        continue
                    yrs_here = pidx[rows, 2]
                    print(f"  {lab} ({int(rows.sum()):,} rows) -- years "
                          f"{int(yrs_here.min())}-{int(yrs_here.max())}, distance "
                          f"{int(dist[rows].min())}-{int(dist[rows].max())} yr:")
                    res = baseline_panel(
                        pidx, z_obs_pts, Z, ho, int(recent_year), buffer_mask=bf,
                        target_rows=rows, exclude_years=hy)
                    # Resolution INSIDE the band: one average over a 1-30 year range is a
                    # blurry measurement of the very thing being measured.
                    res["by_distance"] = {}
                    # Half-open on the LOW side: a withheld year adjacent to the training edge is
                    # distance 1, never 0, so bin "d1-d10" must be (0, 10] and not [0, 10).
                    for lo in range(0, int(dist[rows].max()), 10):
                        sub = rows & (dist > lo) & (dist <= lo + 10)
                        if int(sub.sum()) >= 20:
                            res["by_distance"][f"d{lo + 1}-{lo + 10}"] = baseline_panel(
                                pidx, z_obs_pts, Z, ho, int(recent_year), buffer_mask=bf,
                                target_rows=sub, exclude_years=hy, verbose=False)
                    ladder["temporal_buckets"][lab] = res
                ladder["first_trained_year"] = first_trained
                ladder["common_holdout_years"] = common

            # Re-run the EXISTING metrics on an interpolated stand-in for Z. Both take Z as an
            # argument, so this turns "vs null" into "vs a real alternative" without a second
            # copy of either metric. The null they already report is weak: on direction of
            # change the model beat its permutation null (0.48 vs 0.22) and still lost to
            # interpolation (0.51).
            ratio = ladder.get("spacetime_ratio_cells_per_year")
            if ratio:
                Z_idw = spacetime_idw_z(pidx, z_obs_pts, ho, float(ratio),
                                        buffer_mask=bf, exclude_years=hy)
                fin = np.isfinite(Z_idw).all(axis=1)
                if fin.sum() >= 100:
                    Zi = np.where(fin[:, None], Z_idw, 0.0).astype("float32")
                    _SCALARS = lambda d: {k: v for k, v in d.items()
                                          if np.isscalar(v) or isinstance(v, str)}
                    _di = directional_change_agreement(Zi, X, pidx, recent_year,
                                                       np.random.default_rng(seed))
                    _an = analog_displacement(Zi, X, pidx, xy, recent_year,
                                              np.random.default_rng(seed))
                    ladder["interpolated_z_metrics"] = {"n_rows": int(fin.sum()),
                                                        "directional": _SCALARS(_di),
                                                        "analog": _SCALARS(_an)}
                    print(f"  same metrics on INTERPOLATED z (spacetime IDW, ratio={ratio:g}): "
                          f"direction mean_dir_cos={_di.get('mean_dir_cos', float('nan')):+.3f}, "
                          f"analog cos={_an.get('mean_cos_displacement', float('nan')):+.3f}"
                          f"  <- the model's figures above must beat THESE, not the nulls")
        _phase("baseline ladder")
        _phase("epoch direction panel")
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
    # raw-BBS turnover, the trend reconstruction (soft caps + coverage gate)
    # is flattening real change. pred = the model's own turnover (over-predicts if >> obs).
    if "turnover_obs" in turn and turn["turnover_obs"].size:
        report["temporal_turnover"]["median_turnover_obs_target"] = float(np.median(turn["turnover_obs"]))
        report["temporal_turnover"]["median_turnover_pred"] = float(np.median(turn["turnover_pred"]))
    report["temporal_turnover"]["_magnitude_only_note"] = ("turnover is MAGNITUDE-only "
        "(how much a community changed, not toward what) -- see directional_change for "
        "direction. Compare median_turnover_obs_target to the raw-BBS ceiling "
        "(scripts/viz/raw_bbs_turnover.py) to see how much the trend reconstruction "
        "flattens real change.")
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
        report["zspace_reconstruction"] = report_scalars(recon)
        if epochs_panel is not None:
            report["epoch_directions"] = epochs_panel
        if ladder is not None:
            report["baseline_ladder"] = ladder
        report["graded_on"] = "z_ema"
        report["z_ema_meta"] = ema_meta
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
        recon_rows=recon["rows"],
        recon_cols=recon["cols"],
        recon_err_desk=recon["err_desk"],
        recon_err_nochange=recon["err_nochange"],
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
        if "frac_desk_beats_idw" in rc:
            print(f"           vs INTERPOLATION of observed z from same-year training cells "
                  f"(n={rc['n_idw_scored']}): DESK beats it in "
                  f"{rc['frac_desk_beats_idw']:.1%} of cells (err idw={rc['median_err_idw']:.4f})")
            if "frac_desk_beats_idw_heldout" in rc:
                print(f"           HELD-OUT vs interpolation: DESK beats it in "
                      f"{rc['frac_desk_beats_idw_heldout']:.1%} "
                      f"(err idw={rc['median_err_idw_heldout']:.4f})")
        if "frac_desk_beats_nochange_heldout" in rc:
            print(f"           HELD-OUT cells ({rc['n_heldout']}): DESK beats no-change in "
                  f"{rc['frac_desk_beats_nochange_heldout']:.1%}  <- the honest held-out grade "
                  f"(err DESK={rc['median_err_desk_heldout']:.4f} vs {rc['median_err_nochange_heldout']:.4f})")
    _phase("write outputs")
    print("[validate:timing] total " + "  ".join(f"{n}={t:.0f}s" for n, t in _marks)
          + f"  => {sum(t for _, t in _marks):.0f}s", flush=True)
    print(f"[validate] report -> {out} ; viz arrays -> {viz}")
    return report


if __name__ == "__main__":
    run_validate()
