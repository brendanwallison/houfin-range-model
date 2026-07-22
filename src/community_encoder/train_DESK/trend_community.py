"""Trend-product spatiotemporal community vectors for the ESK kernel.

Replaces the amplitude-modulation construction (``spacetime_community``, now
deprecated). Instead of a fixed 2023 shape modulated by a BBS anomaly, the
historical community is reconstructed by applying **published trend products**
to the modern eBird raster:

    N(cell, sp, Y) = N_2023(cell, sp) * prod_{y=Y+1..2023} (1 + r(cell, sp, y)/100)^-1

where ``N_2023`` is the 2023 eBird annual relative-abundance anchor and the
per-year percent rate is a **smooth blend** of the two trend products,

    r(cell, sp, Y) = w(Y) * ebird_ppy(cell, sp) + (1 - w(Y)) * bbs_rate(cell, sp)

heavily weighted to the eBird recent trend in its temporal domain (w(Y) -> 1 near
present) handing off smoothly to the BBS long-term trend for older years (w -> 0).
Blending the *rate* (not the endpoints) makes the trajectory continuous -- no
hinge. The BBS rate is winsorized (its inverse-distance interpolation has heavy
tails at sparse-coverage range margins); eBird's model-based rate is already tight.

The two products are near-orthogonal at the cell level (different temporal
domains: BBS 1966-2022 vs eBird 2012-2022), so this genuinely spans more of the
community-change space than either alone -- the point of the redesign.

The numerical core (``blend_weight``, ``blended_rate``, ``backward_trajectory``,
``assemble_points``) is pure (arrays in, arrays out) so it unit-tests without any
data, rasterio, or a cluster. ``build_trend_points`` is the I/O orchestration; it
emits the SAME files the ESK consumes (``X_points.npy`` ``(N, n_species)`` with
``n_weeks=1``, ``point_index.npy`` ``(N,3)`` row/col/year, ``points_meta.json``).
"""
import glob
import json
import os

import numpy as np


# --- Pure numerical core -------------------------------------------------------

def blend_weight(years, crossover, width):
    """Smooth eBird-vs-BBS weight w(Y) in (0,1): ~1 recent (eBird), ->0 old (BBS).

    Logistic ramp centred at ``crossover`` with scale ``width`` (years). Heavily
    weighting eBird in its 2012-2022 domain wants ``crossover`` a little below
    2012 so w is already high by the window's start.
    """
    y = np.asarray(years, dtype=float)
    return 1.0 / (1.0 + np.exp(-(y - crossover) / width))


def blended_rate(bbs_rate, ebird_ppy, w):
    """Per-(species,cell) %/yr at one year: ``w*ebird + (1-w)*bbs``, NaN-aware.

    ``bbs_rate``/``ebird_ppy`` are arrays (NaN = that product has no data there).
    Where both are present -> the blend; where only one -> that one; where neither
    -> NaN (caller holds abundance constant, i.e. rate 0).
    """
    bbs_rate = np.asarray(bbs_rate, dtype="float64")
    ebird_ppy = np.asarray(ebird_ppy, dtype="float64")
    hb, he = np.isfinite(bbs_rate), np.isfinite(ebird_ppy)
    b0 = np.where(hb, bbs_rate, 0.0)
    e0 = np.where(he, ebird_ppy, 0.0)
    r = np.full(bbs_rate.shape, np.nan)
    both = hb & he
    r[both] = w * e0[both] + (1.0 - w) * b0[both]
    r[hb & ~he] = b0[hb & ~he]
    r[he & ~hb] = e0[he & ~hb]
    return r


def soft_clip(x, knee, asymptote):
    """Soft version of a hard cap: identity for ``|x|<=knee``, saturating toward
    ``+/-asymptote`` beyond it.

    Below the knee the value is untransformed; above it, a tanh eases it smoothly to
    the asymptote (C1-continuous at the knee -- both pieces have unit slope there),
    so it never exceeds ``asymptote``. Unlike a hard clip it has no kink and unlike a
    global tanh it leaves ordinary change alone. Applied to the cumulative LOG
    abundance change, so ``knee``/``asymptote`` are in log-fold units
    (e.g. ln 10 -> untransformed to ~10x; ln 100 -> capped near 100x).
    """
    x = np.asarray(x, dtype="float64")
    knee = np.asarray(knee, dtype="float64")
    span = np.maximum(np.asarray(asymptote, dtype="float64") - knee, 1e-9)   # scalar or per-species
    a = np.abs(x)
    out = np.where(a > knee, knee + span * np.tanh((a - knee) / span), a)
    return np.sign(x) * out


def backward_trajectory(anchor, bbs_rate, ebird_ppy, bbs_abund, k, sample_years, anchor_year,
                        first_year, crossover, width, soft_knee, soft_asymptote,
                        abs_knee=None, abs_asy=None):
    """Method-B reconstruction: a log-space blend of two closed-form trajectories.

    ``anchor`` (E, eBird present), ``bbs_rate``/``ebird_ppy`` (%/yr), ``bbs_abund`` (B,
    BBS present abundance) share shape ``(S, M)``; ``k`` is the per-species eBird<-BBS
    unit scale ``(S, 1)``. For each sampled year the abundance is

        log N(Y) = w(Y)·[log E + dy·r_e]  +  (1-w(Y))·[log(k·B) + dy·r_b],   dy = Y - present

    where ``r_e``/``r_b`` are the eBird/BBS annual log-rates and ``w`` ramps from ~1
    (recent -> eBird's own trajectory) to ~0 (deep -> BBS's ABSOLUTE past ``k·B/f``, so
    the deep spatial pattern follows BBS, not eBird's modern map). Missing eBird -> its
    rate is 0 (held); missing BBS -> that cell falls back to the eBird trajectory. Two
    overlapping soft caps then bound the change: a RELATIVE (fold) cap on ``log(N/E)``
    (``soft_knee``/``soft_asymptote``, log units) and, if given, an ABSOLUTE cap on
    ``N-E`` (``abs_knee``/``abs_asy``, per species) -- the first protects rare-base
    cells, the second abundant-base cells. Species absent now (E<=0) stay 0.
    ``N`` is ``(len(years), S, M)`` with ``N[anchor_year] == anchor`` exactly.
    """
    E = np.asarray(anchor, dtype="float64")
    B = np.asarray(bbs_abund, dtype="float64")
    k = np.asarray(k, dtype="float64")
    re = np.log1p(np.clip(ebird_ppy, -99.0, None) / 100.0)       # NaN where eBird absent
    rb = np.log1p(np.clip(bbs_rate, -99.0, None) / 100.0)        # NaN where BBS rate absent
    has_e = np.isfinite(re)
    has_b = np.isfinite(rb) & np.isfinite(B) & (B > 0)
    posE = E > 0
    logE = np.where(posE, np.log(np.where(posE, E, 1.0)), 0.0)
    logkB = np.where(has_b, np.log(k * np.where(has_b, B, 1.0)), 0.0)
    re0 = np.where(has_e, re, 0.0)

    p = int(anchor_year)
    out = {}
    for Y in sorted({int(y) for y in sample_years}):
        if Y == p:
            out[Y] = np.where(posE, E, 0.0)                      # anchor exact
            continue
        dy = Y - p
        ebird_term = logE + dy * re0
        bbs_term = np.where(has_b, logkB + dy * rb, ebird_term)  # fall back to eBird if no BBS
        w = blend_weight(Y, crossover, width)
        logN = w * ebird_term + (1.0 - w) * bbs_term
        lr = soft_clip(logN - logE, soft_knee, soft_asymptote)   # cap 1: relative (fold) change
        Ny = np.where(posE, E * np.exp(lr), 0.0)
        if abs_knee is not None:                                 # cap 2: absolute change (overlapping)
            d = soft_clip(Ny - E, abs_knee, abs_asy)
            Ny = np.where(posE, np.clip(E + d, 0.0, None), 0.0)
        out[Y] = Ny
    years = sorted(out)
    return years, np.stack([out[y] for y in years]).astype("float32")


def _smooth_log_years(N, rr, cc, H, W, sigma):
    """Masked Gaussian smooth of log1p(N) per (year, species), applied on the grid.

    Smooths the KERNEL-space quantity (log-abundance) so per-cell interpolation noise
    doesn't survive into the community vectors; masked (Nadaraya-Watson) so nodata
    can't bleed in. Run AFTER the soft caps, so a capped extreme can't poison its
    neighbours. Returns N with the same shape.
    """
    from scipy.ndimage import gaussian_filter
    T, S, M = N.shape
    m = np.zeros((H, W)); m[rr, cc] = 1.0
    md = gaussian_filter(m, sigma, mode="constant")
    out = N.copy()
    for t in range(T):
        for s in range(S):
            g = np.zeros((H, W)); g[rr, cc] = np.log1p(np.clip(N[t, s], 0, None))
            sm = gaussian_filter(g, sigma, mode="constant")
            with np.errstate(invalid="ignore", divide="ignore"):
                lg = np.where(md > 1e-9, sm / md, 0.0)
            out[t, s] = np.expm1(lg[rr, cc])
    return out


def assemble_points(anchor, bbs_rate, ebird_ppy, bbs_abund, k, valid, years_cfg, log1p=True,
                    abs_knee=None, abs_asy=None, smooth_sigma=0.0):
    """Build ``(X, point_index, meta_years)`` from grid arrays. Pure.

    ``anchor`` (eBird present), ``bbs_rate``/``ebird_ppy`` (%/yr) and ``bbs_abund``
    (BBS present abundance) are ``(S, H, W)`` (NaN where absent); ``k`` is the
    per-species eBird<-BBS scale ``(S,)``. ``valid`` is ``(H, W)`` bool (community
    support = the anchor's footprint). ``years_cfg`` = dict(anchor_year, first_year,
    stride, crossover, width, soft_knee, soft_asymptote, min_coverage). ``abs_knee``/
    ``abs_asy`` (per species, optional) add the absolute-change soft cap;
    ``smooth_sigma`` (cells, optional) applies a post-cap masked Gaussian to the
    log-abundance. ``log1p`` emits ``log1p(abundance)`` community vectors.

    Returns ``X`` ``(N, S)`` float32 (recent anchor rows first, then each strided
    historical year), ``pidx`` ``(N,3)`` int32 row/col/year, and the year list.
    """
    S, H, W = anchor.shape
    rr, cc = np.where(valid)
    M = rr.size
    a = np.stack([anchor[s][rr, cc] for s in range(S)])       # (S, M)
    b = np.stack([bbs_rate[s][rr, cc] for s in range(S)])
    e = np.stack([ebird_ppy[s][rr, cc] for s in range(S)])
    ba = np.stack([bbs_abund[s][rr, cc] for s in range(S)])
    kk = np.asarray(k, dtype="float64").reshape(S, 1)
    ak = None if abs_knee is None else np.asarray(abs_knee, dtype="float64").reshape(S, 1)
    aa = None if abs_asy is None else np.asarray(abs_asy, dtype="float64").reshape(S, 1)

    ay, fy = int(years_cfg["anchor_year"]), int(years_cfg["first_year"])
    stride = int(years_cfg["stride"])
    sample_years = [ay] + [y for y in range(ay - 1, fy - 1, -1) if (ay - y) % stride == 0]
    years, N = backward_trajectory(a, b, e, ba, kk, sample_years, ay, fy,
                                   years_cfg["crossover"], years_cfg["width"],
                                   years_cfg["soft_knee"], years_cfg["soft_asymptote"],
                                   abs_knee=ak, abs_asy=aa)             # (T, S, M)
    if smooth_sigma and smooth_sigma > 0:
        N = _smooth_log_years(N, rr, cc, H, W, float(smooth_sigma))
    # Recent year first (ESK strata key on recent_year), then the rest ascending.
    # Coverage gate: a historical (cell,year) community vector is only meaningful if enough
    # of the cell's occupying species have a trend that actually informs that year (eBird when
    # the blend leans recent, BBS when it leans deep). Species without one are held constant,
    # so a low-coverage cell would report false stability -- drop those points. The recent
    # anchor year is always kept (it's the observed community).
    min_cov = float(years_cfg.get("min_coverage", 0.0))
    occ = a > 0                                                   # (S, M) modern occupancy
    n_occ = np.clip(occ.sum(0), 1, None)
    has_e, has_b = np.isfinite(e), np.isfinite(b)
    cross, wid = years_cfg["crossover"], years_cfg["width"]

    order = [years.index(ay)] + [i for i, y in enumerate(years) if y != ay]
    blocks_X, blocks_idx = [], []
    for i in order:
        y = years[i]
        if y == ay or min_cov <= 0:
            keep = np.ones(M, dtype=bool)
        else:
            w = blend_weight(y, cross, wid)                       # recent->eBird, deep->BBS
            contrib = (has_e & (w > 0.05)) | (has_b & ((1.0 - w) > 0.05))
            keep = ((contrib & occ).sum(0) / n_occ) >= min_cov
        blocks_X.append(N[i][:, keep].T)                          # (n_keep, S)
        blocks_idx.append(np.stack([rr[keep], cc[keep], np.full(int(keep.sum()), y)], axis=1))
    X = np.nan_to_num(np.concatenate(blocks_X, axis=0))
    if log1p:
        X = np.log1p(np.clip(X, 0.0, None))               # log-abundance community vectors
    X = X.astype("float32")
    pidx = np.concatenate(blocks_idx, axis=0).astype(np.int32)
    return X, pidx, [years[i] for i in order]


# --- I/O orchestration ---------------------------------------------------------

def _load_trend_grid(path, codes, field):
    """Load a trend .npz and reindex ``field`` to ``codes`` order -> (S, H, W).

    Species absent from the grid are filled with NaN (treated as 'product absent').
    """
    z = np.load(path, allow_pickle=True)
    grid_codes = [str(c) for c in z["species_code"]]
    idx = {c: i for i, c in enumerate(grid_codes)}
    arr = z[field]
    H, W = arr.shape[1:]
    out = np.full((len(codes), H, W), np.nan, dtype="float32")
    missing = []
    for j, c in enumerate(codes):
        if c in idx:
            out[j] = arr[idx[c]]
        else:
            missing.append(c)
    return out, missing


def _annual_anchor(weekly_dir, codes):
    """2023 eBird annual mean relative abundance per community species -> (S, H, W).

    Reads the projected weekly grids (already at the model grid) and averages over
    weeks. Errors listing any community species missing weekly abundance.
    """
    from src.community_encoder.train_DESK.ebird_io import load_tifs_structured

    stack, meta = load_tifs_structured(weekly_dir, target_res_m=None)   # (H, W, S*T)
    H, W, D = stack.shape
    S_w, T = meta["n_species"], meta["n_weeks"]
    order = meta["species"]                                             # sorted unique
    annual = stack.reshape(H, W, S_w, T).mean(axis=3)                  # (H, W, S_w)
    si = {c: i for i, c in enumerate(order)}
    missing = [c for c in codes if c not in si]
    if missing:
        raise SystemExit(f"[trend] {len(missing)} community species lack weekly eBird "
                         f"abundance in {weekly_dir}: {missing}. Download them first "
                         f"(download_ebird.py --species-list <community_trend.csv>).")
    anchor = np.stack([annual[:, :, si[c]] for c in codes]).astype("float32")  # (S, H, W)
    return anchor


def _species_scale(anchor, bbs_abund):
    """Per-species eBird<-BBS unit scale ``k = median(E/B)`` over cells where both > 0.

    Calibrates the units conversion from the present-day overlap so the method-B deep
    target ``k·B/f`` is on eBird's scale. Falls back to 1.0 where there is too little
    overlap to estimate. Returns ``(S,)``.
    """
    S = anchor.shape[0]
    k = np.ones(S, dtype="float64")
    for s in range(S):
        E, B = anchor[s], bbs_abund[s]
        m = np.isfinite(E) & (E > 0) & np.isfinite(B) & (B > 0)
        if int(m.sum()) >= 10:
            k[s] = float(np.median(E[m] / B[m]))
    return k


def build_trend_points(config=None):
    """Assemble the trend-based ESK point matrix and emit the ESK input files."""
    import pandas as pd
    from src.config_utils import load_config, load_data_config

    config = load_config(config) if not isinstance(config, dict) else config
    dcfg = load_data_config()
    tc = config.get("trend", {})
    bc = config["bbs"]

    community_csv = tc.get("community_trend_list") or dcfg["community_trend_list"]
    codes = [str(c) for c in pd.read_csv(community_csv)["species_code"].tolist()]
    S = len(codes)

    bbs_path = tc.get("bbs_trend_grid") or dcfg["trends"]["bbs_trend_grid"]
    eb_path = tc.get("ebird_trend_grid") or dcfg["trends"]["ebird_trend_grid"]
    weekly_dir = tc.get("ebird_weekly_grid") or config.get("paths", {}).get("ebird_folder")
    if not weekly_dir:
        raise SystemExit("set trend.ebird_weekly_grid (or paths.ebird_folder) to the "
                         "projected 2023 eBird weekly grid dir for the anchor.")

    ba_path = tc.get("bbs_abund_grid") or dcfg["trends"]["bbs_abund_grid"]
    bbs_rate, miss_b = _load_trend_grid(bbs_path, codes, "rate")
    ebird_ppy, miss_e = _load_trend_grid(eb_path, codes, "abd_ppy")
    bbs_abund, miss_ba = _load_trend_grid(ba_path, codes, "abund")       # method-B deep scale
    anchor = _annual_anchor(weekly_dir, codes)                          # (S, H, W)
    k = _species_scale(anchor, bbs_abund)                               # (S,) eBird<-BBS unit scale

    valid = np.any(np.isfinite(anchor), axis=0)                         # anchor footprint
    years_cfg = {
        "anchor_year": int(tc.get("anchor_year", 2023)),
        "first_year": int(tc.get("first_year", 1966)),
        "stride": int(tc.get("point_year_stride", 3)),
        "crossover": float(tc.get("handoff_crossover", 2010.0)),
        "width": float(tc.get("handoff_width", 1.5)),
        "soft_knee": float(np.log(tc.get("soft_knee_fold", 10.0))),
        "soft_asymptote": float(np.log(tc.get("soft_max_fold", 100.0))),
        "min_coverage": float(tc.get("min_coverage", 0.5)),
    }
    log1p = bool(tc.get("ruzicka_log1p", True))
    # Absolute-change soft cap: per-species knee/asymptote = multiples of that species'
    # present-abundance scale (p90 over occupied cells). Complements the relative cap.
    abs_knee = abs_asy = None
    if bool(tc.get("abs_soft_cap", True)):
        ref = np.array([float(np.nanpercentile(anchor[i][anchor[i] > 0], 99))   # best-habitat ceiling
                        if np.isfinite(anchor[i]).any() and (anchor[i] > 0).any() else 1.0
                        for i in range(S)])
        abs_knee = ref * float(tc.get("abs_soft_knee_mult", 0.3))
        abs_asy = ref * float(tc.get("abs_soft_max_mult", 1.0))
    smooth_sigma = float(tc.get("smooth_sigma_cells", 0.0))
    X, pidx, years = assemble_points(anchor, bbs_rate, ebird_ppy, bbs_abund, k, valid,
                                     years_cfg, log1p=log1p, abs_knee=abs_knee, abs_asy=abs_asy,
                                     smooth_sigma=smooth_sigma)
    n_recent = int(valid.sum())

    out_dir = bc["z_dir"]
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "X_points.npy"), X)
    np.save(os.path.join(out_dir, "point_index.npy"), pidx)
    with open(os.path.join(out_dir, "points_meta.json"), "w") as fh:
        json.dump({"n_species": S, "n_weeks": 1,
                   "n_recent": n_recent, "n_hist": int(X.shape[0] - n_recent),
                   "species": codes, "recent_year": years_cfg["anchor_year"],
                   "years": years, "handoff": years_cfg, "ruzicka_log1p": log1p,
                   "deep_target": "bbs_absolute", "k_median": float(np.median(k)),
                   "missing_bbs": miss_b, "missing_ebird": miss_e,
                   "missing_bbs_abund": miss_ba}, fh, indent=2)
    print(f"[trend] X {X.shape}: {n_recent} recent (year {years_cfg['anchor_year']}) + "
          f"{X.shape[0]-n_recent} historical over {len(years)-1} back-years "
          f"(stride {years_cfg['stride']}, {years[-1]}..{years_cfg['anchor_year']}); "
          f"{S} species -> {out_dir}")
    if miss_b or miss_e:
        print(f"[trend] note: {len(miss_b)} species w/o BBS grid, {len(miss_e)} w/o eBird grid "
              f"(held constant where absent).")
    return out_dir


if __name__ == "__main__":
    build_trend_points()
