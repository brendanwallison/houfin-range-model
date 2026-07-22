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


def backward_trajectory(anchor, bbs_rate, ebird_ppy, sample_years, anchor_year,
                        first_year, crossover, width, clip_pct):
    """Compound the blended rate backward from the anchor. Returns ``(years, N)``.

    ``anchor``/``bbs_rate``/``ebird_ppy`` share shape ``(...,)`` (e.g. ``(S, M)``).
    Integration is on the ANNUAL lattice ``first_year..anchor_year`` (so the
    trajectory is stride-independent), sampled at ``sample_years``. The rate at a
    year is winsorized to ``[-clip_pct, clip_pct]``; where NEITHER product covers a
    (species,cell) the rate is 0 (abundance held constant). ``N`` is
    ``(len(years), *anchor.shape)`` with ``N[anchor_year] == anchor``.
    """
    anchor = np.asarray(anchor, dtype="float64")
    bbs = np.clip(bbs_rate, -clip_pct, clip_pct)      # NaN-preserving
    eb = np.clip(ebird_ppy, -clip_pct, clip_pct)
    want = {int(y) for y in sample_years}

    out = {}
    if anchor_year in want:
        out[anchor_year] = anchor.copy()
    cumlog = np.zeros_like(anchor)                    # sum_{k>Y..anchor} ln(1+r_k/100)
    for Y in range(int(anchor_year) - 1, int(first_year) - 1, -1):
        r = blended_rate(bbs, eb, blend_weight(Y + 1, crossover, width))   # step Y+1 -> Y
        r = np.where(np.isfinite(r), r, 0.0)
        cumlog = cumlog + np.log1p(r / 100.0)
        if Y in want:
            out[Y] = anchor * np.exp(-cumlog)
    years = sorted(out)
    return years, np.stack([out[y] for y in years]).astype("float32")


def assemble_points(anchor, bbs_rate, ebird_ppy, valid, years_cfg):
    """Build ``(X, point_index, meta_years)`` from grid arrays. Pure.

    ``anchor``/``bbs_rate``/``ebird_ppy`` are ``(S, H, W)`` (NaN where absent).
    ``valid`` is ``(H, W)`` bool (community support = the anchor's footprint).
    ``years_cfg`` = dict(anchor_year, first_year, stride, crossover, width, clip_pct).

    Returns ``X`` ``(N, S)`` float32 (recent anchor rows first, then each strided
    historical year), ``pidx`` ``(N,3)`` int32 row/col/year, and the year list.
    """
    S, H, W = anchor.shape
    rr, cc = np.where(valid)
    M = rr.size
    a = np.stack([anchor[s][rr, cc] for s in range(S)])       # (S, M)
    b = np.stack([bbs_rate[s][rr, cc] for s in range(S)])
    e = np.stack([ebird_ppy[s][rr, cc] for s in range(S)])

    ay, fy = int(years_cfg["anchor_year"]), int(years_cfg["first_year"])
    stride = int(years_cfg["stride"])
    sample_years = [ay] + [y for y in range(ay - 1, fy - 1, -1) if (ay - y) % stride == 0]
    years, N = backward_trajectory(a, b, e, sample_years, ay, fy,
                                   years_cfg["crossover"], years_cfg["width"],
                                   years_cfg["clip_pct"])          # (T, S, M)
    # Recent year first (ESK strata key on recent_year), then the rest ascending.
    order = [years.index(ay)] + [i for i, y in enumerate(years) if y != ay]
    blocks_X, blocks_idx = [], []
    for i in order:
        y = years[i]
        blocks_X.append(N[i].T)                                   # (M, S)
        blocks_idx.append(np.stack([rr, cc, np.full(M, y)], axis=1))
    X = np.nan_to_num(np.concatenate(blocks_X, axis=0)).astype("float32")
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

    bbs_rate, miss_b = _load_trend_grid(bbs_path, codes, "rate")
    ebird_ppy, miss_e = _load_trend_grid(eb_path, codes, "abd_ppy")
    anchor = _annual_anchor(weekly_dir, codes)                          # (S, H, W)

    valid = np.any(np.isfinite(anchor), axis=0)                         # anchor footprint
    years_cfg = {
        "anchor_year": int(tc.get("anchor_year", 2023)),
        "first_year": int(tc.get("first_year", 1966)),
        "stride": int(tc.get("point_year_stride", 3)),
        "crossover": float(tc.get("handoff_crossover", 2010.0)),
        "width": float(tc.get("handoff_width", 1.5)),
        "clip_pct": float(tc.get("rate_clip_pct", 15.0)),
    }
    X, pidx, years = assemble_points(anchor, bbs_rate, ebird_ppy, valid, years_cfg)
    n_recent = int(valid.sum())

    out_dir = bc["z_dir"]
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "X_points.npy"), X)
    np.save(os.path.join(out_dir, "point_index.npy"), pidx)
    with open(os.path.join(out_dir, "points_meta.json"), "w") as fh:
        json.dump({"n_species": S, "n_weeks": 1,
                   "n_recent": n_recent, "n_hist": int(X.shape[0] - n_recent),
                   "species": codes, "recent_year": years_cfg["anchor_year"],
                   "years": years, "handoff": years_cfg,
                   "missing_bbs": miss_b, "missing_ebird": miss_e}, fh, indent=2)
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
