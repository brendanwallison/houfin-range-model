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

from .validation_core import captured_share, change_noise, epoch_values, split_half_change

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


def level_frame(H, base, species_tab, group=None):
    """Long table of normalized level-error differences per (cell, decade, species). Pure.

    ``group`` restricts to one held-out row group (``row_group`` code); in-sample rows (code 0)
    are never scored."""
    rg = H["row_group"] if "row_group" in H.files else np.ones(len(H["y"]), int)
    sel = (rg == group) if group is not None else (rg > 0)
    H = {k: (H[k][sel] if k in ("y", "keys", "block_id", "dist_to_train_km", "cov_novelty",
                                "years_before_modern", "years_beyond_training_edge")
                         or k.startswith("mean_") else H[k]) for k in H.files}
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
        "years_before_modern": np.tile(gmean(H["years_before_modern"]), n_sp),
        "years_beyond_training_edge": np.tile(gmean(H.get("years_beyond_training_edge",
                                                          np.zeros(len(y)))), n_sp)})
    for c in sp.columns:
        if c != "species_code":
            frame[c] = np.repeat(sp[c].to_numpy(), G)
    return frame


def change_frame(H, base, species_tab, sname, include=None):
    """Long table of normalized change-error differences per (change cell, species). Pure.

    ``sname`` is the change set; ``include`` (bool per species) restricts to a subset -- the
    resolvable species."""
    er, mr = H[f"change_early_rows_{sname}"], H[f"change_modern_rows_{sname}"]
    if len(er) == 0:
        return None
    d_obs, preds = _set_deltas(H, sname, ("desk", base))
    ed = (preds["desk"] - d_obs) ** 2
    eb = (preds[base] - d_obs) ** 2
    scale = (d_obs ** 2).mean(0)
    keep = scale > 0
    if include is not None:
        keep &= np.asarray(include, bool)
    r = (ed[:, keep] - eb[:, keep]) / scale[keep]
    sp = species_tab.loc[keep].reset_index(drop=True)
    G, n_sp = r.shape[0], len(sp)
    frame = pd.DataFrame({
        "response": r.T.ravel(), "weight": 1.0,
        "block": np.tile(H[f"change_cell_block_{sname}"], n_sp),
        "species": np.repeat(np.arange(n_sp), G),
        "dist_to_train_km": np.tile(H[f"change_cell_dist_km_{sname}"], n_sp),
        "cov_novelty": np.tile(np.nan_to_num(H[f"change_cell_novelty_{sname}"]), n_sp),
        "years_beyond_training_edge": np.tile(H[f"change_cell_reach_{sname}"], n_sp)})
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
        # A species with no remaining detections at a fraction is predicted by its training mean
        # by EVERY model; its exact-zero skill says nothing and is left out.
        sk = np.where(det > 0, sk, np.nan)
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

def _set_deltas(H, sname, predictors=None):
    """Observed and predicted per-cell change for one change set, on the abundance estimand.

    ``(d_obs, {predictor: d_pred})``. Truth is log1p of the epoch-mean count (the saved ``y`` is
    log1p of the cell-year mean, so ``expm1`` recovers the count exactly); a prediction's count is
    ``exp(mean + var/2) - 1`` (``validate_gp_species.predicted_raw``). Both go through
    ``validation_core.epoch_values``, the one definition of the estimand.
    """
    from .validate_gp_species import predicted_raw
    er = [np.asarray(r, int) for r in H[f"change_early_rows_{sname}"]]
    mr = [np.asarray(r, int) for r in H[f"change_modern_rows_{sname}"]]
    oe, om = epoch_values(np.expm1(H["y"].astype("float64")), er, mr)
    names = predictors or [k[len("mean_"):] for k in H.files if k.startswith("mean_")]
    out = {}
    for p in names:
        pe, pm = epoch_values(predicted_raw(H[f"mean_{p}"], H[f"var_{p}"]), er, mr)
        out[p] = pm - pe
    return om - oe, out


def change_noise_ceiling(H, sname, n_boot=400, seed=0):
    """How much of observed per-species change is noise, and what share of the REAL change each
    predictor captures. ``(summary, per_species DataFrame)``.

    A wrapper over ``validation_core.split_half_change`` / ``change_noise`` / ``captured_share``,
    the same functions the run itself reports with, on the same abundance estimand -- so this
    re-derives nothing. See those functions for the definitions and caveats.
    """
    er = [np.asarray(r, int) for r in H[f"change_early_rows_{sname}"]]
    mr = [np.asarray(r, int) for r in H[f"change_modern_rows_{sname}"]]
    years = H["keys"][:, 2] if "keys" in H.files else np.arange(len(H["y"]))
    d_full, d_a, d_b = split_half_change(np.expm1(H["y"].astype("float64")), er, mr, years)
    nz = change_noise(d_full, d_a, d_b, n_boot=n_boot, seed=seed)
    tab = pd.DataFrame({"species_code": H["species"], "ms_obs_change": nz["ms_obs"],
                        "noise_change": nz["noise"], "signal_share": nz["signal_share"],
                        "ceiling_skill": nz["ceiling_skill"], "resolvable": nz["resolvable"]})
    _, deltas = _set_deltas(H, sname)
    captured = {}
    for p, dp in deltas.items():
        per, pooled = captured_share(d_full, dp, nz)
        tab[f"captured_{p}"] = per
        v = per[np.isfinite(per)]
        captured[p] = {"median": float(np.median(v)) if len(v) else None,
                       "share_above_zero": float((v > 0).mean()) if len(v) else None,
                       "pooled": pooled}
    ok = nz["ms_obs"] > 0
    summary = {
        "set": sname, "n_cells": int(d_full.shape[0]), "n_species_with_change": int(ok.sum()),
        "n_resolvable": int(nz["resolvable"].sum()),
        "median_signal_share": float(np.nanmedian(nz["signal_share"][ok])),
        "signal_share_quartiles": [float(np.nanquantile(nz["signal_share"][ok], q))
                                   for q in (0.25, 0.75)],
        "pooled_signal_share": float(1 - nz["noise"][ok].sum() / nz["ms_obs"][ok].sum()),
        "median_ceiling_skill_resolvable": (float(np.median(nz["ceiling_skill"][nz["resolvable"]]))
                                            if nz["resolvable"].any() else None),
        "captured_share_of_available_signal": captured}
    return summary, tab


SPECIES_COLS = ["phylo_min", "trait_min", "cooc_max", "log_train_detections"]
LEVEL_EXTRAP = ["dist_to_train_km", "cov_novelty", "years_before_modern"]
CHANGE_EXTRAP = ["dist_to_train_km", "cov_novelty"]


def run_noise(gp_dir):
    """The observation-noise read alone, per change set: no AVONET, no regressions, seconds."""
    H = np.load(os.path.join(gp_dir, "heldout_predictions.npz"), allow_pickle=True)
    out = {}
    for sname in [str(x) for x in H["change_set_names"]]:
        summary, tab = change_noise_ceiling(H, sname)
        out[sname] = summary
        tab.to_csv(os.path.join(gp_dir, f"change_noise_per_species_{sname}.csv"), index=False)
        print(f"\n[gp-noise] set {sname}: {summary['n_cells']} change cells, "
              f"{summary['n_species_with_change']} species with observed change, "
              f"{summary['n_resolvable']} resolvable above noise")
        print(f"  signal share of observed squared change: median "
              f"{summary['median_signal_share']:+.3f} (IQR "
              f"{summary['signal_share_quartiles'][0]:+.3f}.."
              f"{summary['signal_share_quartiles'][1]:+.3f}), pooled "
              f"{summary['pooled_signal_share']:+.3f}")
        print(f"  ceiling change skill (resolvable species, median): "
              f"{summary['median_ceiling_skill_resolvable']}")
        print("  share of AVAILABLE change captured (resolvable species): median / share>0 / "
              "pooled")
        for p, v in summary["captured_share_of_available_signal"].items():
            if v["median"] is not None:
                print(f"    {p:22s} {v['median']:+.3f}   {v['share_above_zero']:.2f}   "
                      f"{v['pooled']:+.3f}")
    with open(os.path.join(gp_dir, "change_noise.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    return out


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
    bases = [b for b in ("no_change", "spacetime", "covariate") if f"mean_{b}" in H.files]
    groups = {1: "space", 2: "time", 3: "space_time"}
    rg = H["row_group"]
    for g, gname in groups.items():
        if not (rg == g).any():
            continue
        extrap = LEVEL_EXTRAP + (["years_beyond_training_edge"] if g in (2, 3) else [])
        for base in bases:
            lf = level_frame(H, base, stab, group=g)
            fits = {"main": fit_regression(lf, SPECIES_COLS + ["migratory"], extrap)}
            # Urban tolerance exists for a subset only, so it gets its own fit on that subset
            # rather than shrinking the main one to it.
            if "urban_tolerance" in lf:
                fits["with_urban"] = fit_regression(lf, SPECIES_COLS + ["urban_tolerance"],
                                                    extrap)
            rep["level"][f"{gname}: desk_vs_{base}"] = fits
    noise_by_set = {}
    for sname in [str(x) for x in H["change_set_names"]]:
        summary, ntab = change_noise_ceiling(H, sname)
        rep.setdefault("change_noise", {})[sname] = summary
        noise_by_set[sname] = ntab
        resolvable = ntab["resolvable"].to_numpy(bool)
        extrap = CHANGE_EXTRAP + (["years_beyond_training_edge"] if sname != "space" else [])
        for base in bases:
            cf = change_frame(H, base, stab, sname, include=resolvable)
            if cf is not None:
                rep["change"][f"{sname}: desk_vs_{base}"] = {
                    "main": fit_regression(cf, SPECIES_COLS + ["migratory"], extrap)}
    tp = os.path.join(gp_dir, "thinning.npz")
    if os.path.exists(tp):
        T = np.load(tp)
        tset = str(T["change_set"]) if "change_set" in T.files else None
        rep["thinning_curves"] = thinning_curves(T, noise=noise_by_set.get(tset))
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


#: The comparisons the summary prints, in reading order. Everything else is in report.json.
KEY_PAIRS = ("desk_vs_no_change", "desk_raw_vs_no_change_raw", "desk_vs_spacetime",
             "desk_vs_covariate", "desk_raw_vs_covariate_raw", "desk_pooled_vs_spacetime_pooled",
             "desk_vs_esk_oracle_independent", "spacetime_vs_no_change",
             "covariate_vs_no_change", "esk_oracle_independent_vs_no_change")
KEY_PREDICTORS = ("desk", "desk_raw", "desk_pooled", "spacetime", "covariate", "covariate_raw",
                  "esk_oracle_independent")


def summarize(gp_dir):
    """Print the headline numbers of a finished run, per held-out group and change set."""
    with open(os.path.join(gp_dir, "report.json"), encoding="utf-8") as fh:
        r = json.load(fh)
    f = lambda v: "  n/a " if v is None or (isinstance(v, float) and v != v) else f"{v:+.3f}"
    print(f"run {r['run_dir']}\nwithheld years: {r.get('withheld_years') or 'none'}  "
          f"(common {r.get('common_holdout_years') or 'none'}); buffer: {r.get('buffer')}")
    print("rows:", r["rows"], "| dropped species:", r["dropped_zero_training_detections"]["n"])
    print("estimand:", r.get("estimand"))
    print("CAVEAT:", r.get("seed_caveat"))
    for k, v in r.get("baseline_shapes", {}).items():
        print(f"  shape {k:14s} converged={v['converged']}  at_bound={v.get('at_bound')}  "
              f"lengthscales[:4]={[round(x, 2) for x in v['lengthscales'][:4]]}")
    for k, v in (r.get("unavailable") or {}).items():
        print(f"  unavailable: {k}: {v}")
    if r.get("completeness_gaps"):
        print(f"  GAPS (no result, no reason): {r['completeness_gaps']}")
    print(f"  oracle: {r.get('oracle')}")
    p = r["primary"]
    print("\nPRIMARY:", p.get("metric", p.get("note")))
    if "median" in p:
        print(f"  median {f(p['median'])} CI [{f(p['median_ci'][0])}, {f(p['median_ci'][1])}]  "
              f"share>0 {p['share_above_zero']:.2f}  balanced {f(p.get('balanced_median'))}  "
              f"viable={p.get('viable')}  ({p['n_species_defined']} species)")
    for sec in ("change", "level"):
        for grp, pairs in r[sec].items():
            extra = ""
            if sec == "change":
                v = r.get("change_viability", {}).get(grp, {})
                extra = f"   viable={v.get('qualified')} {v.get('reason') or ''}"
            print(f"\n{sec.upper()} -- {grp}{extra}\n  {'pair':34s} median [CI]  share>0  "
                  "balanced  pop-weighted  n")
            for k in KEY_PAIRS:
                v = pairs.get(k)
                if v and "median" in v:
                    print(f"  {k:34s} {f(v['median'])} [{f(v['median_ci'][0])}, "
                          f"{f(v['median_ci'][1])}]  {v['share_above_zero']:.2f}  "
                          f"{f(v.get('balanced_median'))}  "
                          f"{f(v.get('population_weighted_median'))}  {v['n_species_defined']}")
    for grp, wins in r.get("level_by_window", {}).items():
        print(f"\nLEVEL by window -- {grp}: desk_vs_no_change / desk_vs_spacetime medians")
        for w, pairs in wins.items():
            a, b = pairs.get("desk_vs_no_change", {}), pairs.get("desk_vs_spacetime", {})
            print(f"  {w:7s} {f(a.get('median'))}  {f(b.get('median'))}")
    for sname, nz in r.get("change_noise", {}).items():
        print(f"\nNOISE -- {sname}: signal share median {f(nz.get('median_signal_share'))}, "
              f"pooled {f(nz.get('pooled_signal_share'))}; {nz.get('n_resolvable')} resolvable; "
              f"ceiling skill {f(nz.get('median_ceiling_skill_resolvable'))}")
        cap = r.get("captured", {}).get(sname, {})
        dec = r.get("decomposition", {}).get(sname, {})
        print(f"  {'predictor':24s} captured: median  share>0  pooled | change direction cos  "
              "norm ratio  overmove")
        for k in KEY_PREDICTORS:
            c, d = cap.get(k), dec.get(k, {})
            if c and c.get("median") is not None:
                print(f"  {k:24s} {f(c['median'])}  {c['share_above_zero']:.2f}  "
                      f"{f(c['pooled'])} | {f(d.get('median_cos'))}  "
                      f"{f(d.get('median_norm_ratio'))}  {f(d.get('median_overmove'))}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gp_dir", help="the gp_species output directory of validate_gp_species")
    ap.add_argument("--datasets-root", default=None, help="default: data_config datasets_root")
    ap.add_argument("--noise-only", action="store_true",
                    help="only the split-half observation-noise read (seconds; no AVONET)")
    ap.add_argument("--summary", action="store_true",
                    help="print the headline numbers of a finished run and exit")
    args = ap.parse_args()
    if args.summary:
        summarize(args.gp_dir)
    elif args.noise_only:
        run_noise(args.gp_dir)
    else:
        run(args.gp_dir, args.datasets_root)


if __name__ == "__main__":
    main()
