"""DEPRECATED (superseded by train_DESK/trend_community.py).

Amplitude-modulated spatiotemporal community vectors for the ESK kernel.

Kept for reference/comparison only -- NOT on the live pipeline. The fixed-2023-shape
amplitude construction hit an expressiveness ceiling (joint-ESK effective rank ~8-9);
the trend-product path (``trend_community.build_trend_points``) replaces it, applying
published BBS + eBird %/yr trends to the modern eBird raster for a spanning Z basis.

Original docstring follows.

Implements the approved construction (see the plan + memory
``desk-bbs-spatiotemporal-design``):

    x(s, t) = E(s) * ( B~(s, t) / B~_ref(s) )

- ``E(s)`` is the fixed 2023 eBird species×52-week community vector at cell ``s``
  (from the eBird stack cache).
- ``B~`` is a per-species, spatiotemporally kernel-smoothed BBS relative-abundance
  field; the ratio to the recent (eBird-reference) window is a **unitless** anomaly,
  so no BBS→eBird calibration is needed.
- Only BBS-matched community species are modulated; unmatched species keep ``E(s)``
  constant across years. Feed the resulting species×52 rows into the **existing raw
  Ruzicka kernel unchanged**.

The numerical core (``gaussian_nw_field``, ``reference_field``, ``robust_anomaly``,
``apply_amplitude``) is pure (arrays in, arrays out) so it unit-tests without BBS
data, rasterio, or a cluster. ``build_amplitude_points`` is the I/O orchestration.
"""
import os

import numpy as np


def gaussian_nw_field(obs_sum, weight, sigma_t, sigma_s, support_floor=0.0):
    """Separable Gaussian Nadaraya–Watson smooth of a per-(year,cell) field.

    ``obs_sum`` = effort-weighted counts (``mean_count * effort``) placed on the
    ``(T, H, W)`` grid (0 where no survey); ``weight`` = effort (0 where no survey).
    Returns ``(field, support)``: ``field = smooth(obs_sum)/smooth(weight)`` (NaN
    where smoothed support < ``support_floor``), and ``support`` = smoothed effort
    (how much real data backs each cell-year — the extrapolation guard).

    A **symmetric** temporal kernel (σ_t) is used deliberately: this smooths for
    noise/sparsity only; any ecological lag is modelled on the environment side
    (backward EMA), never by skewing this estimator.
    """
    from scipy.ndimage import gaussian_filter

    sigma = (float(sigma_t), float(sigma_s), float(sigma_s))
    num = gaussian_filter(np.asarray(obs_sum, dtype="float64"), sigma=sigma, mode="constant")
    den = gaussian_filter(np.asarray(weight, dtype="float64"), sigma=sigma, mode="constant")
    field = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
    field[den < support_floor] = np.nan
    return field, den


def reference_field(field, years, ref_years):
    """Recent-window reference ``B~_ref(cell)`` = mean of ``field`` over ``ref_years``.

    ``field`` is ``(T, H, W)`` aligned to ``years``; returns ``(H, W)`` (NaN where
    the reference window has no support at that cell).
    """
    import warnings

    years = list(years)
    idx = [i for i, y in enumerate(years) if y in set(ref_years)]
    if not idx:
        raise ValueError(f"no reference years {sorted(set(ref_years))} in field years")
    with warnings.catch_warnings():          # all-NaN cells -> NaN ref (handled as no-change)
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(field[idx], axis=0)


def robust_anomaly(field, support, ref, pseudocount=1.0, cap=np.log(10.0),
                   support_floor=1e-6, shrink_k=2.0):
    """Robust, bounded per-(year,cell) anomaly from a smoothed field.

    ``anomaly = (field+α)/(ref+α)`` in log space, then: empirical-Bayes shrink the
    log-anomaly toward the species' per-year support-weighted spatial mean
    (weight ``support/(support+shrink_k)`` → sparse cells pulled to the regional
    trend), hard-cap at ``±cap``, and force ``anomaly=1`` (no change) where support
    is below ``support_floor`` or data is missing. Returns ``(T, H, W)`` ≥ 0.
    """
    field = np.asarray(field, dtype="float64")
    ref = np.asarray(ref, dtype="float64")[None]                 # (1,H,W)
    a = float(pseudocount)
    with np.errstate(invalid="ignore", divide="ignore"):
        log_anom = np.log((field + a) / (ref + a))               # NaN where field/ref NaN
    have = np.isfinite(log_anom) & (support > 0)

    # Per-year, support-weighted spatial-mean log-anomaly (the regional trend).
    reg = np.zeros(field.shape[0])
    for t in range(field.shape[0]):
        m = have[t]
        if m.any():
            reg[t] = np.average(log_anom[t][m], weights=support[t][m])

    w = support / (support + float(shrink_k))                    # (T,H,W) in [0,1)
    shrunk = w * np.where(have, log_anom, 0.0) + (1.0 - w) * reg[:, None, None]
    shrunk = np.where(have, shrunk, 0.0)                         # missing -> log 1 = no change
    shrunk = np.clip(shrunk, -abs(cap), abs(cap))
    anomaly = np.exp(shrunk)
    anomaly[support < support_floor] = 1.0
    return anomaly


def apply_amplitude(E_cell, anomaly_by_species, n_weeks):
    """Scale a cell's species×week eBird vector by per-species anomaly scalars.

    ``E_cell`` is ``(n_species * n_weeks,)`` species-blocked (weeks contiguous per
    species — the ``esk_kernel`` layout). ``anomaly_by_species`` is ``(n_species,)``
    (1.0 for unmodulated species). Returns the modulated vector, same shape.
    """
    E = np.asarray(E_cell, dtype="float64").reshape(-1, n_weeks)   # (n_species, n_weeks)
    if E.shape[0] != len(anomaly_by_species):
        raise ValueError(f"species mismatch: E has {E.shape[0]}, anomaly has "
                         f"{len(anomaly_by_species)}")
    return (E * np.asarray(anomaly_by_species, dtype="float64")[:, None]).reshape(-1)


def _scatter_dense(values, rows, cols, t_idx, T, H, W):
    """Long-form (value, row, col, t_idx) -> dense (T, H, W) grid (0 elsewhere)."""
    g = np.zeros((T, H, W), dtype="float64")
    g[t_idx, rows, cols] = values
    return g


def build_amplitude_points(config=None):
    """Assemble the amplitude-modulated ESK point matrix ``x(s,t) = E(s)·anomaly(s,t)``.

    Streams per BBS-matched species (one dense ``(T,H,W)`` field at a time) so memory
    stays bounded. Emits, to ``bbs.z_dir``: ``X (N_points, S*n_weeks)`` (recent eBird
    anchor points + supported historical `(cell,year)` points), ``point_index
    (row,col,year)``, and a per-point valid mask. Returns the output dir.
    """
    from src.config_utils import load_config
    from src.community_encoder.train_DESK.ebird_cache import load_ebird_stack

    config = load_config(config) if not isinstance(config, dict) else config
    bc = config["bbs"]

    E_stack, meta = load_ebird_stack(config)              # (H,W,S*T), meta.species
    H, W, D = E_stack.shape
    n_species, n_weeks = meta["n_species"], meta["n_weeks"]
    e_index = {code: i for i, code in enumerate(meta["species"])}
    E_flat = E_stack.reshape(H * W, D)
    ebird_valid = np.any(~np.isnan(E_stack), axis=-1).reshape(-1)   # (H*W,)

    cm = np.load(bc["community_matrix"], allow_pickle=True)
    community_codes = [str(c) for c in cm["species_codes"]]
    # BBS-matched community species that exist as eBird blocks -> (E block index, code)
    matched = [(e_index[c], c) for c in community_codes if c in e_index]

    years = np.arange(int(cm["cov_year"].min()), int(cm["cov_year"].max()) + 1)
    yr_ix = {int(y): i for i, y in enumerate(years)}
    T = len(years)

    # Shared effort/coverage dense + smoothed support (species-independent).
    cov_t = np.array([yr_ix[int(y)] for y in cm["cov_year"]])
    weight = _scatter_dense(cm["cov_n"].astype(float), cm["cov_row"], cm["cov_col"], cov_t, T, H, W)
    _, support = gaussian_nw_field(np.zeros_like(weight), weight,
                                   bc["smooth_sigma_t"], bc["smooth_sigma_s"])
    ref_years = list(range(int(bc["anomaly_ref_years"][0]), int(bc["anomaly_ref_years"][1]) + 1))
    recent_year = int(bc["anomaly_ref_years"][1])
    # RELATIVE support floor: a fraction of the max smoothed support. An absolute floor is
    # not invariant to smooth_sigma_s -- wider spatial smoothing dilutes the per-cell
    # magnitude, so a fixed 0.25 silently over-cut coverage to a small core once sigma_s
    # was widened to 5. Fraction-of-max tracks the field's own scale. Falls back to the old
    # absolute key if support_floor_frac is absent.
    if "support_floor_frac" in bc:
        smax = float(np.nanmax(support)) if np.isfinite(support).any() else 1.0
        floor = float(bc["support_floor_frac"]) * smax
        print(f"[spacetime] support floor = {bc['support_floor_frac']} x max({smax:.4f}) = {floor:.5f}")
    else:
        floor = float(bc["support_floor"])

    # Historical point coords: supported cells at subsampled years (excl. the recent anchor year).
    stride = int(bc.get("point_year_stride", 1))
    hist_years = [y for y in years if y != recent_year and (recent_year - y) % stride == 0]
    ht, hr, hc = [], [], []
    for y in hist_years:
        t = yr_ix[int(y)]
        rr, cc = np.where(support[t] >= floor)
        keep = ebird_valid.reshape(H, W)[rr, cc]           # need E(s) too
        ht.append(np.full(keep.sum(), t)); hr.append(rr[keep]); hc.append(cc[keep])
    ht = np.concatenate(ht) if ht else np.array([], int)
    hr = np.concatenate(hr) if hr else np.array([], int)
    hc = np.concatenate(hc) if hc else np.array([], int)
    n_hist = ht.size

    # Per-species anomaly at the historical points (default 1.0 for unmatched species).
    anomaly_pts = np.ones((n_hist, n_species), dtype="float64")
    mean_t = np.array([yr_ix[int(y)] for y in cm["year"]])
    for j, code in matched:
        sel = cm["species_index"] == community_codes.index(code)
        obs_sum = _scatter_dense(
            cm["mean_count"][sel].astype(float) * _cov_at(cm, sel),   # mean * effort
            cm["row"][sel], cm["col"][sel], mean_t[sel], T, H, W)
        field, sup = gaussian_nw_field(obs_sum, weight, bc["smooth_sigma_t"], bc["smooth_sigma_s"])
        ref = reference_field(field, years, ref_years)
        anom = robust_anomaly(field, sup, ref, pseudocount=bc["anomaly_pseudocount"],
                              cap=bc["anomaly_cap_log"], support_floor=floor,
                              shrink_k=bc["anomaly_shrink_k"])
        if n_hist:
            anomaly_pts[:, j] = anom[ht, hr, hc]

    # Recent anchor points: every eBird-valid cell, x = E (anomaly == 1).
    rec_lin = np.where(ebird_valid)[0]
    rec_r, rec_c = rec_lin // W, rec_lin % W
    X_recent = E_flat[rec_lin]                             # (n_recent, D)

    # Historical points: x = E(cell) * per-species anomaly.
    hist_lin = hr * W + hc
    X_hist = np.empty((n_hist, D), dtype="float32")
    for k in range(n_hist):
        X_hist[k] = apply_amplitude(E_flat[hist_lin[k]], anomaly_pts[k], n_weeks)

    # Zero-fill eBird NaN, exactly as ESK (esk_kernel: np.nan_to_num) and DESK's x_raw
    # do -- NaN = no eBird data for that species-week => 0 abundance. Without this the
    # recent anchors (raw E) carry NaN, which breaks Ruzicka on the recent/present
    # points (NaN recent_control, and contaminated turnover/analog metrics).
    X = np.nan_to_num(np.concatenate([X_recent, X_hist], axis=0)).astype("float32")
    pidx = np.concatenate([
        np.stack([rec_r, rec_c, np.full(rec_r.size, recent_year)], axis=1),
        np.stack([hr, hc, years[ht]], axis=1),
    ], axis=0).astype(np.int32) if n_hist else \
        np.stack([rec_r, rec_c, np.full(rec_r.size, recent_year)], axis=1).astype(np.int32)

    out_dir = bc["z_dir"]
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "X_points.npy"), X)
    np.save(os.path.join(out_dir, "point_index.npy"), pidx)   # (N,3): row,col,year
    with open(os.path.join(out_dir, "points_meta.json"), "w") as fh:
        import json
        json.dump({"n_species": n_species, "n_weeks": n_weeks,
                   "n_recent": int(rec_r.size), "n_hist": int(n_hist),
                   "species": meta["species"], "recent_year": recent_year,
                   "matched_species": [c for _, c in matched]}, fh, indent=2)
    # Smoothed effort field (species-independent support = Σ K·coverage per year·cell):
    # the diagnostic for "where does BBS actually back the estimate" (validate viz).
    np.savez_compressed(os.path.join(out_dir, "support_field.npz"),
                        support=support.astype("float32"), years=years)
    print(f"[spacetime] X {X.shape}: {rec_r.size} recent + {n_hist} historical points "
          f"({len(matched)} BBS-modulated species) -> {out_dir}")
    return out_dir


def _cov_at(cm, sel):
    """Effort (cov_n) for each selected mean-count row, matched on (row,col,year)."""
    import pandas as pd
    cov = pd.DataFrame({"row": cm["cov_row"], "col": cm["cov_col"],
                        "year": cm["cov_year"], "n": cm["cov_n"]})
    q = pd.DataFrame({"row": cm["row"][sel], "col": cm["col"][sel], "year": cm["year"][sel]})
    merged = q.merge(cov, on=["row", "col", "year"], how="left")
    return merged["n"].fillna(1).to_numpy(dtype=float)
