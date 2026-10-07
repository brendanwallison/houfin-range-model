"""What predicts DESK's per-species GP skill: similarity, data, and how far it extrapolates.

Reads the outputs of ``validate_gp_species`` (``per_species.csv``, ``layout.json``,
``heldout_predictions.npz``, ``thinning.npz``) and needs no GPU or encode, so it runs anywhere the
AVONET inputs are, including a laptop after copying the gp_species directory back.

THREE PRODUCTS

1. Species similarity to the community, for EVERY evaluation species. The saved candidate pool
   covers only the 389 urban-indexed species, so it is recomputed here from AVONET and the
   phylogeny: minimum and mean patristic distance to the community, minimum and mean distance in
   standardized AVONET trait space, migration class, and urban tolerance where it exists. The
   data-driven co-occurrence measure comes from the run itself (training rows only).

2. Regressions. The response is a normalized squared-error DIFFERENCE between desk and a
   baseline, per (held-out cell, decade, species) for level and per (change cell, species) for
   change::

       r = (err_desk^2 - err_base^2) / scale(species)          negative = desk better

   ``scale`` is the species' held-out variance for level and its mean squared observed change for
   change -- properties of the DATA, identical for every baseline. The first run divided by the
   baseline's own MSE, which is near zero for a rare species a baseline predicts as all zeros and
   put coefficients in the thousands; a data-side scale cannot do that. Change regressions use
   RESOLVABLE species only (real change distinguishable from noise, ``change_noise_ceiling``),
   the population the captured-share metric is read on. Regressors are the species covariates, the extrapolation degrees
   (distance to the nearest training cell, covariate novelty, years before the modern epoch), and
   extrapolation x {similarity, data} interactions -- the question being whether desk's advantage
   grows or shrinks as extrapolation gets harder, and whether that depends on the species.
   Standard errors are two-way clustered by species AND held-out block (Cameron-Gelbach-Miller):
   species share rows, and rows in one block share a geography, so neither clustering alone is
   honest. With ~20 blocks the block dimension is thin; read the extrapolation coefficients with
   that in mind.

3. Thinning curves: per-species skill against REMAINING training detections, per predictor, from
   the fixed-test-set refits.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

#: Winsorization of the regression response. Squared-error differences are heavy-tailed, and a
#: handful of rows with huge errors would otherwise set every coefficient.
WINSOR = (0.005, 0.995)
DETECTION_BINS = (0, 1, 4, 11, 31, 101, 301, np.inf)


# ----------------------------- similarity -----------------------------

def patristic_to_set(tree, labels_from, labels_to):
    """Min and mean patristic distance from each ``labels_from`` tip to the ``labels_to`` tips.

    One single-source traversal per ``labels_to`` tip (``avonet.compute_phylo_distances``), so
    cost is ``len(labels_to)`` tree passes rather than the all-pairs matrix. Tips absent from the
    tree are skipped on the ``to`` side and give NaN on the ``from`` side.
    """
    from src.data.identify.avonet import compute_phylo_distances
    # Positional, not keyed on label: two evaluation species can share a tip (a BirdLife lump),
    # and a label-keyed dict would silently merge them and misalign every later row.
    acc = [[] for _ in labels_from]
    for t in labels_to:
        try:
            d = compute_phylo_distances(tree, t)
        except ValueError:
            continue
        for i, lab in enumerate(labels_from):
            if lab in d and lab != t:
                acc[i].append(d[lab])
    mn = np.array([min(v) if v else np.nan for v in acc])
    me = np.array([float(np.mean(v)) if v else np.nan for v in acc])
    return mn, me


def trait_distance_to_set(T_from, T_to):
    """Min and mean Euclidean distance in (already standardized) trait space. Pure."""
    A, B = np.asarray(T_from, "float64"), np.asarray(T_to, "float64")
    B = B[np.isfinite(B).all(1)]
    ok = np.isfinite(A).all(1)
    mn, me = np.full(len(A), np.nan), np.full(len(A), np.nan)
    if len(B) and ok.any():
        d = np.sqrt(((A[ok, None, :] - B[None, :, :]) ** 2).sum(-1))
        mn[ok], me[ok] = d.min(1), d.mean(1)
    return mn, me


def species_similarity(evaluation, community, datasets_root):
    """Similarity of every evaluation species to the community. ``DataFrame`` keyed on code.

    Joins eBird codes to AVONET (BirdLife taxonomy) on normalized scientific name -- the join
    ``avonet.main`` uses -- and to BirdTree tips through AVONET's crosswalk. Species the join
    misses keep NaN and are counted, not dropped: taxonomies differ and the miss is not random
    (recent splits), so the regression reports how many it lost.
    """
    import dendropy

    from src.data.identify import avonet as av
    from src.data.identify.bbs_crosswalk import load_ebird_taxonomy

    tax = load_ebird_taxonomy(os.path.join(datasets_root, "avonet", "eBird_taxonomy.csv"))
    code2sci = dict(zip(tax["species_code"], tax["sci_norm"]))
    bl = pd.read_csv(os.path.join(datasets_root, "avonet", "TraitData", "AVONET1_BirdLife.csv"),
                     encoding="latin1")
    bl["sci_norm"] = bl["Species1"].apply(av.normalize_name)
    cw = av.load_crosswalk(os.path.join(datasets_root, "avonet", "PhylogeneticData",
                                        "BirdLife-BirdTree crosswalk.csv"))
    bl = bl.merge(cw[["Species1", "Species3"]], on="Species1", how="left")
    bl["tip"] = bl["Species3"].str.replace(" ", "_")
    bl = bl.drop_duplicates("sci_norm").set_index("sci_norm")

    codes = list(evaluation) + list(community)
    rows = bl.reindex([code2sci.get(c) for c in codes])
    rows.index = codes
    traits = av.standardize(rows[av.TRAIT_COLS].apply(pd.to_numeric, errors="coerce"),
                            av.TRAIT_COLS).to_numpy()
    ne = len(evaluation)
    t_min, t_mean = trait_distance_to_set(traits[:ne], traits[ne:])

    tree = dendropy.Tree.get(path=os.path.join(datasets_root, "avonet", "PhylogeneticData",
                                               "HackettStage1_0001_1000_MCCTreeTargetHeights.nex"),
                             schema="nexus", preserve_underscores=True)
    tips_e = rows["tip"].iloc[:ne].tolist()
    tips_c = [t for t in rows["tip"].iloc[ne:].tolist() if isinstance(t, str)]
    p_min, p_mean = patristic_to_set(tree, [t if isinstance(t, str) else f"__missing{i}"
                                            for i, t in enumerate(tips_e)], tips_c)

    out = pd.DataFrame({"species_code": list(evaluation),
                        "phylo_min": p_min, "phylo_mean": p_mean,
                        "trait_min": t_min, "trait_mean": t_mean,
                        "migration": pd.to_numeric(rows["Migration"].iloc[:ne],
                                                   errors="coerce").to_numpy()})
    upath = os.path.join(datasets_root, "urban_avian", "spp_urban_indices.csv")
    if os.path.exists(upath):
        u = pd.read_csv(upath)
        u["species_code"] = u["species_code"].astype(str).str.lower()
        u = av.standardize(u.dropna(subset=av.URBAN_COLS), av.URBAN_COLS)
        u["urban_tolerance"] = u[av.URBAN_COLS].mean(1)
        out = out.merge(u[["species_code", "urban_tolerance"]].drop_duplicates("species_code"),
                        on="species_code", how="left")
    return out


# ----------------------------- regression -----------------------------

def wls_twoway_cluster(X, y, w, g1, g2):
    """Weighted least squares with Cameron-Gelbach-Miller two-way clustered covariance. Pure.

    ``V = V_g1 + V_g2 - V_g1&g2``, each a sandwich with the ``G/(G-1)`` small-sample factor. If the
    difference is not positive semi-definite (it can fail to be with few clusters), negative
    eigenvalues are clipped to zero, the standard repair.
    """
    X, y, w = np.asarray(X, "float64"), np.asarray(y, "float64"), np.asarray(w, "float64")
    Xw = X * w[:, None]
    B = np.linalg.pinv(X.T @ Xw)
    beta = B @ (Xw.T @ y)
    e = y - X @ beta
    s = Xw * e[:, None]                                       # per-row score

    def _V(g):
        _, inv = np.unique(g, return_inverse=True, axis=0)
        G = inv.max() + 1
        S = np.zeros((G, X.shape[1]))
        np.add.at(S, inv, s)
        return (G / max(G - 1, 1)) * (B @ (S.T @ S) @ B), G

    g1, g2 = np.asarray(g1), np.asarray(g2)
    V1, G1 = _V(g1)
    V2, G2 = _V(g2)
    V12, _ = _V(np.column_stack([g1, g2]))
    V = V1 + V2 - V12
    ev, Q = np.linalg.eigh((V + V.T) / 2)
    V = (Q * np.clip(ev, 0.0, None)) @ Q.T
    se = np.sqrt(np.diag(V))
    return {"beta": beta, "se": se, "n_clusters": [int(G1), int(G2)]}


def _z(v):
    v = np.asarray(v, "float64")
    sd = np.nanstd(v)
    return (v - np.nanmean(v)) / (sd if sd > 0 else 1.0)


def design(frame, species_cols, extrap_cols):
    """Standardized main effects plus extrapolation x species interactions. ``(X, names)``. Pure."""
    cols, names = [np.ones(len(frame))], ["intercept"]
    zs = {c: _z(frame[c]) for c in species_cols + extrap_cols}
    for c in species_cols + extrap_cols:
        cols.append(zs[c])
        names.append(c)
    for e in extrap_cols:
        for s in species_cols:
            cols.append(zs[e] * zs[s])
            names.append(f"{e} x {s}")
    return np.column_stack(cols), names


def _winsorize(v):
    lo, hi = np.nanquantile(v, WINSOR)
    return np.clip(v, lo, hi)


def level_frame(H, base, species_tab):
    """Long table of normalized level-error differences per (cell, decade, species). Pure."""
    y = H["y"].astype("float64")
    ed = (H["mean_desk"] - y) ** 2
    eb = (H[f"mean_{base}"] - y) ** 2
    scale = y.var(0)
    keep = scale > 0
    r = (ed[:, keep] - eb[:, keep]) / scale[keep]
    keys = H["keys"]
    decade = (keys[:, 2] // 10) * 10
    grp = pd.DataFrame({"r": keys[:, 0], "c": keys[:, 1], "dec": decade})
    gid = grp.groupby(["r", "c", "dec"]).ngroup().to_numpy()
    G = gid.max() + 1
    cnt = np.bincount(gid, minlength=G).astype("float64")

    def gmean(v):
        return np.bincount(gid, weights=v, minlength=G) / cnt

    R = np.zeros((G, keep.sum()))
    np.add.at(R, gid, r)
    R /= cnt[:, None]
    first = np.zeros(G, int)
    first[gid[::-1]] = np.arange(len(gid))[::-1]
    sp = species_tab.loc[keep].reset_index(drop=True)
    n_sp = len(sp)
    frame = pd.DataFrame({
        "response": R.T.ravel(),
        "weight": np.tile(cnt, n_sp),
        "block": np.tile(H["block_id"][first], n_sp),
        "species": np.repeat(np.arange(n_sp), G),
        "dist_to_train_km": np.tile(gmean(H["dist_to_train_km"]), n_sp),
        "cov_novelty": np.tile(gmean(np.nan_to_num(H["cov_novelty"])), n_sp),
        "years_before_modern": np.tile(gmean(H["years_before_modern"]), n_sp)})
    for c in sp.columns:
        if c != "species_code":
            frame[c] = np.repeat(sp[c].to_numpy(), G)
    return frame


def change_frame(H, base, species_tab, include=None):
    """Long table of normalized change-error differences per (change cell, species). Pure.

    ``include`` (bool per species) restricts to a subset -- the resolvable species."""
    y = H["y"].astype("float64")
    er, mr = H["change_early_rows"], H["change_modern_rows"]
    if len(er) == 0:
        return None

    def delta(v):
        return (np.stack([v[np.asarray(m, int)].mean(0) for m in mr])
                - np.stack([v[np.asarray(e, int)].mean(0) for e in er]))
    d_obs = delta(y)
    ed = (delta(H["mean_desk"].astype("float64")) - d_obs) ** 2
    eb = (delta(H[f"mean_{base}"].astype("float64")) - d_obs) ** 2
    scale = (d_obs ** 2).mean(0)
    keep = scale > 0
    if include is not None:
        keep &= np.asarray(include, bool)
    r = (ed[:, keep] - eb[:, keep]) / scale[keep]
    sp = species_tab.loc[keep].reset_index(drop=True)
    G, n_sp = r.shape[0], len(sp)
    frame = pd.DataFrame({
        "response": r.T.ravel(), "weight": 1.0,
        "block": np.tile(H["change_cell_block"], n_sp),
        "species": np.repeat(np.arange(n_sp), G),
        "dist_to_train_km": np.tile(H["change_cell_dist_km"], n_sp),
        "cov_novelty": np.tile(np.nan_to_num(H["change_cell_novelty"]), n_sp)})
    for c in sp.columns:
        if c != "species_code":
            frame[c] = np.repeat(sp[c].to_numpy(), G)
    return frame


def fit_regression(frame, species_cols, extrap_cols):
    """Two-way clustered WLS of the winsorized response on the design. ``dict``. Pure."""
    need = species_cols + extrap_cols
    f = frame.dropna(subset=need + ["response"])
    if len(f) < 50 or f["species"].nunique() < 10:
        return {"note": f"too few rows ({len(f)}) or species ({f['species'].nunique()})"}
    X, names = design(f, species_cols, extrap_cols)
    res = wls_twoway_cluster(X, _winsorize(f["response"].to_numpy()), f["weight"].to_numpy(),
                             f["species"].to_numpy(), f["block"].to_numpy())
    t = res["beta"] / np.where(res["se"] > 0, res["se"], np.nan)
    return {"n_rows": int(len(f)), "n_species": int(f["species"].nunique()),
            "n_blocks": int(f["block"].nunique()), "clusters": res["n_clusters"],
            "coefficients": {n: {"beta": float(b), "se": float(s), "t": float(tt)}
                             for n, b, s, tt in zip(names, res["beta"], res["se"], t)},
            "note": ("response is (err_desk^2 - err_base^2)/scale: NEGATIVE coefficients mean "
                     "desk gains on the baseline as the regressor rises. Regressors are "
                     "standardized (per SD).")}


# ----------------------------- thinning -----------------------------

def thinning_curves(T, bins=DETECTION_BINS, noise=None):
    """Median per-species skill by remaining-detection bin, per predictor and fraction. Pure.

    Change skill is against no_change (zero predicted change, so the reference needs no refit);
    level skill is against the intercept refitted at the same fraction. Given ``noise`` -- the
    ``change_noise_ceiling`` per-species table, in the same species order -- also the CAPTURED
    share of available change on resolvable species, the metric the change result is read on.
    """
    det = T["n_train_detections"]                                     # (F, S)
    preds = sorted({k.split("_sse_", 1)[1] for k in T.files if "_sse_" in k} - {"zero"})
    labels = [f"{int(lo)}-{'inf' if hi == np.inf else int(hi) - 1}"
              for lo, hi in zip(bins[:-1], bins[1:])]
    out = {"bins": labels, "fractions": T["fractions"].tolist(), "change": {}, "level": {}}

    def _bin_medians(sk):
        """Per FRACTION, the median over species in each remaining-detection bin. Each fraction is
        its own refit, so pooling fractions into one bin would mix different models."""
        b = np.digitize(det, bins[1:-1], right=False)
        res = []
        for f in range(sk.shape[0]):
            row = []
            for k in range(len(labels)):
                v = sk[f][(b[f] == k) & np.isfinite(sk[f])]
                row.append({"median": float(np.median(v)) if len(v) else None, "n": int(len(v))})
            res.append(row)
        return res

    if "change_sse_zero" in T.files:
        z = T["change_sse_zero"][None, :]
        for p in preds:
            if f"change_sse_{p}" in T.files:
                with np.errstate(invalid="ignore", divide="ignore"):
                    sk = np.where(z > 0, 1 - np.sqrt(T[f"change_sse_{p}"] / z), np.nan)
                out["change"][p] = _bin_medians(sk)
    if noise is not None and "change_sse_zero" in T.files:
        z = T["change_sse_zero"]
        n_cells = np.where(noise["ms_obs_change"].to_numpy() > 0,
                           z / np.maximum(noise["ms_obs_change"].to_numpy(), 1e-300), 0)
        avail = z - noise["noise_change"].to_numpy() * n_cells
        res = noise["resolvable"].to_numpy(bool)
        out["captured"] = {}
        for p in preds:
            if f"change_sse_{p}" in T.files:
                with np.errstate(invalid="ignore", divide="ignore"):
                    c = np.where(res[None, :], (z[None, :] - T[f"change_sse_{p}"]) / avail, np.nan)
                pooled = [float(np.nansum(np.where(res, z - row, 0))
                                / max(np.sum(np.where(res, avail, 0)), 1e-300))
                          for row in T[f"change_sse_{p}"]]
                out["captured"][p] = {"by_detection_bin": _bin_medians(c), "pooled": pooled}
    if "level_sse_intercept" in T.files:
        ref = T["level_sse_intercept"]
        for p in preds:
            if p != "intercept" and f"level_sse_{p}" in T.files:
                with np.errstate(invalid="ignore", divide="ignore"):
                    sk = np.where(ref > 0, 1 - np.sqrt(T[f"level_sse_{p}"] / ref), np.nan)
                out["level"][p] = _bin_medians(sk)
    return out


# ----------------------------- observation noise -----------------------------

def abba_halves(rows, years):
    """Split rows into two year-balanced halves by an ABBA pattern over year order. Pure.

    Rows are sorted by year and assigned A, B, B, A, A, B, B, A, ... A random split can put more
    early years in one half and more late years in the other, so a trend INSIDE the epoch would
    read as disagreement between halves and be counted as noise. Plain alternation (ABAB) still
    leaves B half a step later than A. ABBA gives both halves the same mean year in every complete
    block of four, so a linear within-epoch trend cancels exactly and only year-to-year variation
    about it counts as noise. A trailing unpaired year is dropped from both halves; when the
    paired count is not a multiple of four, the halves' mean years differ by one step over n/2.
    """
    rows = np.asarray(rows, int)
    order = rows[np.argsort(np.asarray(years), kind="stable")]
    n = (len(order) // 2) * 2
    pattern = np.array([0, 1, 1, 0])
    lab = pattern[np.arange(n) % 4]
    # When n % 4 == 2 the last pair is (A, B) and B ends up one year-step later in total, i.e.
    # 1/(n/2) steps on the half means. No assignment of two rows to two halves can avoid that
    # short of dropping the pair, and dropping it costs a third of the data at n = 6.
    return order[:n][lab == 0], order[:n][lab == 1]


def split_half_change(y, early_rows, modern_rows, years=None):
    """Per-species change from two DISJOINT, year-balanced halves of each cell's survey years.

    The per-species counterpart of ``validate_bbs_routes.split_half_groups``. Each change cell's
    early-epoch years and modern-epoch years are split into halves A and B by ``abba_halves``, so
    ``d_a`` and ``d_b`` are two independent observations of the SAME cell's change with the same
    mean year. Their disagreement is measurement noise; their covariance is the real change. Rows
    are cell-years (routes already averaged), so the split is over years, and the noise is route
    sampling, observer and interannual variation about any within-epoch trend -- everything a
    smooth predictor of the epoch change cannot see.

    ``years`` is the year of each row of ``y``; without it rows are taken to be in year order (the
    order ``epoch_gate`` emits). Returns ``(d_full, d_a, d_b)``, each ``(n_cells, S)``. The epoch
    gate guarantees >= 3 years per epoch, so every half has at least one year.
    """
    y = np.asarray(y, "float64")
    yr = np.arange(len(y)) if years is None else np.asarray(years)
    full, da, db = [], [], []
    for e, m in zip(early_rows, modern_rows):
        e, m = np.asarray(e, int), np.asarray(m, int)
        ea, eb = abba_halves(e, yr[e])
        ma, mb = abba_halves(m, yr[m])
        full.append(y[m].mean(0) - y[e].mean(0))
        da.append(y[ma].mean(0) - y[ea].mean(0))
        db.append(y[mb].mean(0) - y[eb].mean(0))
    return np.stack(full), np.stack(da), np.stack(db)


def change_noise_ceiling(H, n_boot=400, seed=0):
    """How much of observed per-species change is noise, and what share of the REAL change each
    predictor captures. ``(summary, per_species DataFrame)``. Pure given the inputs.

    Per species, over change cells:

    * ``ms_obs = mean(d_full^2)``, the no_change predictor's MSE (it predicts zero change).
    * ``noise = mean((d_a - d_b)^2) / 4``, the noise variance of ``d_full``. Two independent half
      estimates differ by twice a half's noise variance, and the full estimate averages both
      halves, halving it again. The halves are year-balanced (``abba_halves``), so a linear trend
      inside an epoch is NOT counted as noise. Slightly CONSERVATIVE still: an unpaired year left
      out of the halves is in ``d_full``, and curvature inside an epoch counts as noise.
    * ``signal_share = 1 - noise/ms_obs``, the share of observed squared change that is real.
    * ``ceiling_skill = 1 - sqrt(noise/ms_obs)``, the change skill of a predictor that knows the
      true change exactly: no predictor can beat it.
    * ``captured_<p> = (ms_obs - MSE_p) / (ms_obs - noise)``, each predictor's improvement on
      no_change as a share of the improvement available. This is the per-species analogue of the
      route suite's "share of available temporal signal".

    A species is RESOLVABLE when the bootstrap over cells puts ``ms_obs - noise`` above zero, the
    same viability rule as ``validate_bbs_routes.stratum_viable``. For an unresolvable species the
    observed change is consistent with pure noise and nothing can be graded on it; its
    ``captured`` values are left out of the pooled medians, not averaged in.
    """
    y = H["y"].astype("float64")
    er, mr = H["change_early_rows"], H["change_modern_rows"]
    years = H["keys"][:, 2] if "keys" in H.files else None
    d_full, d_a, d_b = split_half_change(y, er, mr, years)
    sq = d_full ** 2
    nz = (d_a - d_b) ** 2 / 4.0
    ms_obs, noise = sq.mean(0), nz.mean(0)
    rng = np.random.default_rng(seed + 1)
    nc = sq.shape[0]
    boot = np.empty((int(n_boot), sq.shape[1]))
    for b in range(int(n_boot)):
        i = rng.integers(0, nc, nc)
        boot[b] = sq[i].mean(0) - nz[i].mean(0)
    lo = np.quantile(boot, 0.025, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(ms_obs > 0, 1 - noise / ms_obs, np.nan)
        ceil = np.where(ms_obs > 0, 1 - np.sqrt(np.minimum(noise / ms_obs, 1.0)), np.nan)
    resolvable = (ms_obs > 0) & (lo > 0)
    tab = pd.DataFrame({"species_code": H["species"], "ms_obs_change": ms_obs,
                        "noise_change": noise, "signal_share": share,
                        "ceiling_skill": ceil, "resolvable": resolvable})

    def _delta(v):
        v = np.asarray(v, "float64")
        return (np.stack([v[np.asarray(m, int)].mean(0) for m in mr])
                - np.stack([v[np.asarray(e, int)].mean(0) for e in er]))

    preds = [k[len("mean_"):] for k in H.files if k.startswith("mean_")]
    avail = ms_obs - noise
    captured = {}
    for p in preds:
        mse = ((_delta(H[f"mean_{p}"]) - d_full) ** 2).mean(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            c = np.where(resolvable, (ms_obs - mse) / avail, np.nan)
        tab[f"captured_{p}"] = c
        captured[p] = {"median": float(np.nanmedian(c)) if np.isfinite(c).any() else None,
                       "share_above_zero": float(np.nanmean(c[np.isfinite(c)] > 0))
                       if np.isfinite(c).any() else None,
                       "pooled": float(np.nansum(np.where(resolvable, ms_obs - mse, 0))
                                       / max(np.sum(np.where(resolvable, avail, 0)), 1e-300))}
    ok = ms_obs > 0
    summary = {
        "n_cells": int(nc), "n_species_with_change": int(ok.sum()),
        "n_resolvable": int(resolvable.sum()),
        "median_signal_share": float(np.nanmedian(share[ok])),
        "signal_share_quartiles": [float(np.nanquantile(share[ok], q)) for q in (0.25, 0.75)],
        "pooled_signal_share": float(1 - noise[ok].sum() / ms_obs[ok].sum()),
        "median_ceiling_skill_resolvable": (float(np.median(ceil[resolvable]))
                                            if resolvable.any() else None),
        "captured_share_of_available_signal": captured,
        "note": ("signal_share is the fraction of observed squared change that is real; "
                 "ceiling_skill is the best change skill any predictor could reach; captured is "
                 "each predictor's gain on no_change as a share of the available gain, over "
                 "resolvable species only. Noise is split-half over years, slightly conservative.")}
    return summary, tab


# ----------------------------- run -----------------------------

SPECIES_COLS = ["phylo_min", "trait_min", "cooc_max", "log_train_detections"]
LEVEL_EXTRAP = ["dist_to_train_km", "cov_novelty", "years_before_modern"]
CHANGE_EXTRAP = ["dist_to_train_km", "cov_novelty"]


def run_noise(gp_dir):
    """The observation-noise read alone: no AVONET, no regressions, seconds to run."""
    H = np.load(os.path.join(gp_dir, "heldout_predictions.npz"), allow_pickle=True)
    summary, tab = change_noise_ceiling(H)
    tab.to_csv(os.path.join(gp_dir, "change_noise_per_species.csv"), index=False)
    with open(os.path.join(gp_dir, "change_noise.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[gp-noise] {summary['n_cells']} change cells, {summary['n_species_with_change']} "
          f"species with observed change, {summary['n_resolvable']} resolvable above noise")
    print(f"  signal share of observed squared change: median "
          f"{summary['median_signal_share']:+.3f} (IQR {summary['signal_share_quartiles'][0]:+.3f}"
          f"..{summary['signal_share_quartiles'][1]:+.3f}), pooled "
          f"{summary['pooled_signal_share']:+.3f}")
    print(f"  ceiling change skill (resolvable species, median): "
          f"{summary['median_ceiling_skill_resolvable']}")
    print("  share of AVAILABLE change captured (resolvable species): median / share>0 / pooled")
    for p, v in summary["captured_share_of_available_signal"].items():
        if v["median"] is not None:
            print(f"    {p:14s} {v['median']:+.3f}   {v['share_above_zero']:.2f}   "
                  f"{v['pooled']:+.3f}")
    return summary


def run(gp_dir, datasets_root=None):
    from src.config_utils import load_data_config
    datasets_root = datasets_root or load_data_config()["datasets_root"]
    tab = pd.read_csv(os.path.join(gp_dir, "per_species.csv"))
    with open(os.path.join(gp_dir, "layout.json"), encoding="utf-8") as fh:
        lay = json.load(fh)
    sim = species_similarity(lay["evaluation"], lay["community"], datasets_root)
    tab = tab.merge(sim, on="species_code", how="left")
    tab["log_train_detections"] = np.log1p(tab["n_train_detections"])
    tab["migratory"] = (tab["migration"] == 3).astype(float).where(tab["migration"].notna())
    tab.to_csv(os.path.join(gp_dir, "per_species_with_similarity.csv"), index=False)

    rep = {"similarity_coverage": {c: int(tab[c].notna().sum()) for c in
                                   ["phylo_min", "trait_min", "cooc_max", "migration",
                                    "urban_tolerance"] if c in tab},
           "n_species": int(len(tab)), "level": {}, "change": {}}
    H = np.load(os.path.join(gp_dir, "heldout_predictions.npz"), allow_pickle=True)
    stab = tab.drop(columns=[c for c in tab.columns if c.startswith(("change_skill", "level_skill",
                                                                     "direction_skill", "lpd_"))])
    noise_tab = None
    if len(H["change_early_rows"]):
        rep["change_noise"], noise_tab = change_noise_ceiling(H)
    resolvable = (None if noise_tab is None else noise_tab["resolvable"].to_numpy(bool))
    bases = [b for b in ("no_change", "spacetime", "covariate") if f"mean_{b}" in H.files]
    for base in bases:
        lf = level_frame(H, base, stab)
        rep["level"][f"desk_vs_{base}"] = {
            "main": fit_regression(lf, SPECIES_COLS + ["migratory"], LEVEL_EXTRAP)}
        # Urban tolerance exists for a subset only, so it gets its own fit on that subset rather
        # than shrinking the main one to it.
        if "urban_tolerance" in lf:
            rep["level"][f"desk_vs_{base}"]["with_urban"] = fit_regression(
                lf, SPECIES_COLS + ["urban_tolerance"], LEVEL_EXTRAP)
        cf = change_frame(H, base, stab, include=resolvable)
        if cf is not None:
            rep["change"][f"desk_vs_{base}"] = {
                "main": fit_regression(cf, SPECIES_COLS + ["migratory"], CHANGE_EXTRAP)}
    tp = os.path.join(gp_dir, "thinning.npz")
    if os.path.exists(tp):
        rep["thinning_curves"] = thinning_curves(np.load(tp), noise=noise_tab)
    with open(os.path.join(gp_dir, "analysis.json"), "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=2)
    _print(rep)
    return rep


def _print(rep):
    for sec in ("level", "change"):
        for comp, fits in rep[sec].items():
            m = fits["main"]
            if "coefficients" not in m:
                print(f"[gp-analysis] {sec} {comp}: {m.get('note')}")
                continue
            print(f"\n[gp-analysis] {sec} {comp}: {m['n_rows']:,} rows, {m['n_species']} species, "
                  f"{m['n_blocks']} blocks  (negative = desk gains)")
            for n, c in m["coefficients"].items():
                flag = " *" if abs(c["t"]) > 2 else ""
                print(f"    {n:44s} {c['beta']:+.4f}  se {c['se']:.4f}  t {c['t']:+.2f}{flag}")
    cap = (rep.get("thinning_curves") or {}).get("captured")
    if cap:
        tc = rep["thinning_curves"]
        print(f"\n[gp-analysis] thinning: POOLED share of available change captured, resolvable "
              f"species, by training fraction {tc['fractions']}")
        for p, v in cap.items():
            print(f"    {p:14s} {['%+.3f' % x for x in v['pooled']]}")
        print(f"  median captured by remaining-detection bin {tc['bins']}, per fraction:")
        for p, v in cap.items():
            row = ["  ".join("   .  " if b["median"] is None else f"{b['median']:+.3f}"
                             for b in frac) for frac in v["by_detection_bin"]]
            print(f"    {p}")
            for f, r in zip(tc["fractions"], row):
                print(f"      f={f:<5g} {r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gp_dir", help="the gp_species output directory of validate_gp_species")
    ap.add_argument("--datasets-root", default=None, help="default: data_config datasets_root")
    ap.add_argument("--noise-only", action="store_true",
                    help="only the split-half observation-noise read (seconds; no AVONET)")
    args = ap.parse_args()
    if args.noise_only:
        run_noise(args.gp_dir)
    else:
        run(args.gp_dir, args.datasets_root)


if __name__ == "__main__":
    main()
