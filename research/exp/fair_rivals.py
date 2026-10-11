"""E039i: does DESK beat FAIR linear rivals in time, or only the pooled (space-for-time) map E039h used?

    python research/exp/fair_rivals.py --cache <tempho research cache> --out <dir> [--rank 24]
    python research/exp/fair_rivals.py --summarize <dir>

Same estimand, cells, truth halves and bootstrap draws as rrr_backcast.py (E039h): held-out cells' community change
between the withheld early epoch (1966-86) and 2005-25, noise-free centred corr with the observed ESK change (r24,
ABBA halves), pooled over dimensions. Every map is fitted on the same rows (training cells x trained years, finite);
cell means are over a cell's trained-year rows. Arms:
  desk            raw z (z_raw_all), as E039h
  pooled_ridge    E039h's linear map (standardized F_cov_raw, ridge) -- reproduction
  pooled_blr      one BLR coefficient set on 64 PCs (E016's pooled)
  within_raw      Mundlak within-cell block on 64 PCs of F_cov_raw (each year's state, no output EMA, as raw z)
  within_ema      the same on F_cov (output-EMA'd covariates, as E016)
  placebo_300/600 no covariates: position RFF fields x (year - cell mean year), fitted on within-cell ESK deviations
  kindxtime_8/32  "kind of place x time": [1, top-k PCs of the cell's mean covariates] x (year - cell mean year),
                  fitted on within-cell ESK deviations -- a LINEAR model of the mechanism E039h proposed
  desk_within     DESK's own raw z deviations -> ESK deviations (within-cell BLR), applied to DESK's change
  desk_pooled     DESK's raw z -> ESK through one pooled BLR (the readout form the age model uses)
Also per-dimension mean corr (M28: a pooled corr can net out signals of both signs) and a SPATIAL level check on the
same metric family: held-out cells x trained years, pooled-readout predictions of ESK r24 (desk_pooled vs pooled_ridge).
E039j (--incremental): content BEYOND the trend placebo -- placebo_300 + X fitted jointly on within-cell ESK deviations
(two iid blocks), X = DESK z deviations / F_cov_raw PC deviations / kind-x-time features; gain = combo - placebo on the
same draws. And the LOCAL part of change: truth and every prediction minus their leave-one-out Gaussian-weighted
average over the other held-out cells (150 and 300 km), so a regional trend scores ~0 and only place-specific change
counts.
E039k (--n-pos 96 --pos-seeds 1 2 3 4 5, M20): the placebo with an adequate basis and several draws; every placebo-based
number (the placebo itself, combos, gains, local) is the mean over draws, paired on the same bootstrap draws.
E039l (--rich-base): DESK's gain beyond a RICH linear base -- placebo_300 + linear kind x time (32 PCs) + the
covariate within-map, each its own iid block: rich = that base, rich+desk_within = base + DESK z deviations. If DESK's
gain survives, it is not a linear kind-of-place clock.
E039m (--attr <attribution_cells.npz of the same model>): which covariate part of DESK's change carries its gain
beyond the trend? In the joint fits (placebo+desk_within, and rich+desk_within with --rich-base) DESK's change is
replaced by its change MINUS one part (backcast_attribution's swap parts: land, climate, luh, luh_mgmt, luh_class,
luh_uniform = the continent-wide LUH shift (a clock), luh_pattern = where land use changed more or less, ...), same
coefficients; loss = corr(full) - corr(dropped), paired, and its share of the gain over the base.
E043 check (--ref-cache <control cache>, same keys): the control model's DESK arms (desk_ref, desk_ref_within,
placebo+desk_ref_within) on the same cells and draws, and paired differences treatment - control.
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))


def wc(x, y, w):
    xm = (w[:, None] * x).sum(0) / w.sum()
    ym = (w[:, None] * y).sum(0) / w.sum()
    return (w[:, None] * (x - xm) * (y - ym)).sum(0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache")
    ap.add_argument("--out")
    ap.add_argument("--rank", type=int, default=24)
    ap.add_argument("--n-pcs", type=int, default=64)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--summarize", default=None)
    ap.add_argument("--incremental", action="store_true", help="E039j: gains beyond the placebo, local change")
    ap.add_argument("--n-pos", type=int, default=24, help="position RFF features in the placebo")
    ap.add_argument("--pos-seeds", type=int, nargs="+", default=[1], help="placebo draws (results are draw means)")
    ap.add_argument("--tag", default="", help="suffix of the output json")
    ap.add_argument("--rich-base", action="store_true", help="E039l: gains beyond placebo + kind x time + within map")
    ap.add_argument("--attr", default=None, help="E039m: attribution_cells.npz of this model (drop-part losses)")
    ap.add_argument("--ref-cache", default=None, help="control model's cache (same keys): paired treatment - control")
    a = ap.parse_args()
    if a.summarize:
        return summarize(a.summarize)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    from scipy import sparse
    from change_oracle import desk_epoch_z, group_rows
    from lib import blr
    os.makedirs(a.out, exist_ok=True)
    c = lambda f: os.path.join(a.cache, f)
    keys = np.load(c("keys.npy"))
    N = len(keys)
    split = np.load(c("split.npz"))
    withheld = [int(y) for y in split["withheld"]]
    r = a.rank
    E, full, half_a, half_b = group_rows(a.cache, keys, withheld)
    idx = {}
    for j, (cell, ep, wh) in enumerate(zip(map(tuple, E["cells"]), E["epoch"], E["withheld"])):
        if ep == 0 and bool(wh):
            idx.setdefault(cell, [None, None])[0] = j
        elif ep == 1 and not wh:
            idx.setdefault(cell, [None, None])[1] = j
    pairs = np.array([(e, m) for e, m in idx.values() if e is not None and m is not None])
    cells = E["cells"][pairs[:, 0]]
    held = split["holdout"][cells[:, 0], cells[:, 1]]
    ev = np.where(held)[0]
    Za, Zb = E["z_a"][:, :r].astype("float64"), E["z_b"][:, :r].astype("float64")
    Ta = (Za[pairs[:, 1]] - Za[pairs[:, 0]])[ev]
    Tb = (Zb[pairs[:, 1]] - Zb[pairs[:, 0]])[ev]
    early_rows = [np.asarray(full[pairs[j, 0]]) for j in ev]
    mod_rows = [np.asarray(full[pairs[j, 1]]) for j in ev]
    zr_all = np.load(c("z_raw_all.npy"), mmap_mode="r")
    ci_, y0 = np.load(c("key_cell_index.npy")), int(np.load(c("years.npy"))[0])
    D_desk = (desk_epoch_z(zr_all, ci_, y0, keys, mod_rows, r) - desk_epoch_z(zr_all, ci_, y0, keys, early_rows, r))

    def gmat(groups):
        rows = np.concatenate([np.full(len(g), i) for i, g in enumerate(groups)])
        cols = np.concatenate([np.asarray(g, int) for g in groups])
        vals = np.concatenate([np.full(len(g), 1.0 / len(g)) for g in groups])
        return sparse.csr_matrix((vals, (rows, cols)), shape=(len(groups), N))
    Ge, Gm = gmat(early_rows), gmat(mod_rows)
    dchg = lambda X: np.asarray(Gm @ np.asarray(X, "float64")) - np.asarray(Ge @ np.asarray(X, "float64"))

    # rows (E039h's): training cells x trained years, finite covariates and ESK
    F_raw = np.asarray(np.load(c("F_cov_raw.npy"), mmap_mode="r"), "float64")
    Eann = np.load(c("esk_annual.npy")).astype("float64")
    yr = keys[:, 2].astype("float64")
    ho = split["holdout"][keys[:, 0], keys[:, 1]]
    bf = split["buffer"][keys[:, 0], keys[:, 1]]
    ty = ~np.isin(keys[:, 2], withheld)
    trn_mask = ~ho & ~bf & ty & np.isfinite(F_raw).all(1) & np.isfinite(Eann).all(1)
    trn = np.where(trn_mask)[0]
    Z = Eann[:, :r]
    cid = keys[:, 0].astype(np.int64) * 100000 + keys[:, 1]
    uc, inv = np.unique(cid, return_inverse=True)
    w_ty = (ty & np.isfinite(Eann).all(1)).astype(float)

    def cmean(X, w):
        S = sparse.csr_matrix((w, (inv, np.arange(N))), shape=(len(uc), N))
        cnt = np.asarray(S.sum(1)).ravel()
        return (np.asarray(S @ X) / np.maximum(cnt, 1e-12)[:, None])[inv]

    def std(F):
        mu, sd = F[trn].mean(0), F[trn].std(0)
        keep = sd > 1e-6
        return np.where(np.isfinite(F[:, keep]), (F[:, keep] - mu[keep]) / sd[keep], 0.0)

    def pcs(S, n):
        _, _, Vt = np.linalg.svd(S[trn][:: max(1, len(trn) // 40000)] - S[trn].mean(0), full_matrices=False)
        return S @ Vt[:n].T

    preds, notes = {"desk": D_desk}, {}
    # E039h's pooled ridge, reproduced exactly
    X = std(F_raw)
    Xc, Yc = X[trn] - X[trn].mean(0), Eann[trn, :64] - Eann[trn, :64].mean(0)
    lam = 1e-3 * np.trace(Xc.T @ Xc) / Xc.shape[1]
    B = np.linalg.solve(Xc.T @ Xc + lam * np.eye(Xc.shape[1]), Xc.T @ Yc)
    preds["pooled_ridge"] = (dchg(X) @ B)[:, :r]
    Zbar = cmean(Z, w_ty)
    Zd = Z - Zbar
    t = yr - cmean(yr[:, None], w_ty)[:, 0]
    dyr = dchg(yr[:, None])[:, 0]
    k = a.n_pcs
    keep_feats = {}
    for tag, fname in (("raw", "F_cov_raw.npy"), ("ema", "F_cov.npy")):
        F = F_raw if tag == "raw" else np.asarray(np.load(c(fname), mmap_mode="r"), "float64")
        P = pcs(std(F), k)
        Pbar = cmean(P, w_ty)
        mm = blr.fit(np.hstack([Pbar, P - Pbar])[trn], Z[trn], [(0, k), (k, 2 * k)])
        dP = dchg(P)
        preds[f"within_{tag}"] = dP @ mm["coef"][:, k:].T
        notes[f"within_{tag}_amp_b_w"] = [float(np.median(mm["a"][:, 0])), float(np.median(mm["a"][:, 1]))]
        if tag == "raw":
            keep_feats["within_raw"] = (P - Pbar, dP)
            preds["pooled_blr"] = dP @ blr.fit(P[trn], Z[trn], [(0, k)])["coef"].T
            # kind of place x time: cell-mean covariate PCs x (year - cell mean year), on within-cell deviations
            Pbar_ev = np.asarray(Gm @ Pbar)                       # constant within a cell
            for kk in (8, 32):
                KT = np.hstack([t[:, None], Pbar[:, :kk] * t[:, None]]) / 10.0
                g = blr.fit(KT[trn], Zd[trn], [(0, 1), (1, kk + 1)])["coef"].T
                KT_ev = np.hstack([np.ones((len(ev), 1)), Pbar_ev[:, :kk]]) * (dyr / 10.0)[:, None]
                preds[f"kindxtime_{kk}"] = KT_ev @ g
                keep_feats[f"kindxtime_{kk}"] = (KT, KT_ev)
    # placebo: position x time, no covariates
    cev = cells[ev].astype(float) * 27.0
    for ell in (300.0, 600.0):
        for sd in a.pos_seeds:
            rg = np.random.default_rng(sd)
            m = a.n_pos
            om, ph = rg.normal(size=(2, m)) / ell, rg.uniform(0, 2 * np.pi, m)
            R = np.sqrt(2.0 / m) * np.cos(keys[:, :2].astype(float) * 27.0 @ om + ph)
            b_pl = blr.fit((R * (t / 10.0)[:, None])[trn], Zd[trn], [(0, m)])["coef"].T
            Rc = np.sqrt(2.0 / m) * np.cos(cev @ om + ph)
            preds[f"placebo_{ell:.0f}#{sd}"] = (Rc * (dyr / 10.0)[:, None]) @ b_pl
            keep_feats[f"placebo_{ell:.0f}#{sd}"] = (R * (t / 10.0)[:, None], Rc * (dyr / 10.0)[:, None])
    # DESK through linear readouts fitted on the same rows (z_raw at keys)
    zk = np.asarray(np.load(c("z_raw.npy"), mmap_mode="r")[:, :r], "float64")
    dz = dchg(zk)
    notes["desk_keys_vs_epoch_z_maxabs"] = float(np.abs(dz - D_desk).max())
    preds["desk_within"] = dz @ blr.fit((zk - cmean(zk, w_ty))[trn], Zd[trn], [(0, r)])["coef"].T
    keep_feats["desk_within"] = (zk - cmean(zk, w_ty), dz)
    within_arms = ["desk_within"]
    if a.ref_cache:
        rc = lambda f: os.path.join(a.ref_cache, f)
        assert np.array_equal(np.load(rc("keys.npy")), keys), "ref cache keys differ"
        assert np.array_equal(np.load(rc("key_cell_index.npy")), ci_), "ref cache cell index differs"
        zr_ref = np.load(rc("z_raw_all.npy"), mmap_mode="r")
        preds["desk_ref"] = (desk_epoch_z(zr_ref, ci_, y0, keys, mod_rows, r)
                             - desk_epoch_z(zr_ref, ci_, y0, keys, early_rows, r))
        zkr = np.asarray(np.load(rc("z_raw.npy"), mmap_mode="r")[:, :r], "float64")
        dzr = dchg(zkr)
        preds["desk_ref_within"] = dzr @ blr.fit((zkr - cmean(zkr, w_ty))[trn], Zd[trn], [(0, r)])["coef"].T
        keep_feats["desk_ref_within"] = (zkr - cmean(zkr, w_ty), dzr)
        within_arms.append("desk_ref_within")
        notes["ref_cache"] = a.ref_cache
    parts_ev = {}
    if a.attr:
        az = np.load(a.attr)
        posn = {tuple(map(int, cc)): i for i, cc in enumerate(az["cells"])}
        order = np.array([posn[tuple(map(int, cc))] for cc in cells[ev]])
        notes["attr_full_vs_desk_maxabs"] = float(np.abs(az["part__full"][order] - D_desk).max())
        notes["attr_truth_vs_maxabs"] = float(max(np.abs(az["Ta"][order] - Ta).max(), np.abs(az["Tb"][order] - Tb).max()))
        for nm in ("land", "climate", "luh", "luh_mgmt", "luh_class", "luh_uniform", "luh_pattern", "temperature",
                   "moisture", "land_local", "land_regional"):
            if f"part__{nm}_part" in az.files:
                parts_ev[nm] = az[f"part__{nm}_part"][order]
    incr = {}
    if a.incremental:
        for sd in a.pos_seeds:
            PL, PLev = keep_feats[f"placebo_300#{sd}"]
            for nm in tuple(within_arms) + ("within_raw", "kindxtime_8"):
                Xf, Xev = keep_feats[nm]
                q = PL.shape[1]
                b = blr.fit(np.hstack([PL, Xf])[trn], Zd[trn], [(0, q), (q, q + Xf.shape[1])])["coef"].T
                incr[f"placebo+{nm}#{sd}"] = np.hstack([PLev, Xev]) @ b
                if nm == "desk_within":
                    for pn, pv in parts_ev.items():
                        incr[f"placebo+desk_within-{pn}#{sd}"] = np.hstack([PLev, Xev - pv]) @ b
            if a.rich_base:
                parts = [(PL, PLev), keep_feats["kindxtime_32"], keep_feats["within_raw"]]
                for tag, extra in (("rich", []), ("rich+desk_within", [keep_feats["desk_within"]])):
                    ps = parts + extra
                    edges = np.cumsum([0] + [f.shape[1] for f, _ in ps])
                    blocks = [(int(edges[i]), int(edges[i + 1])) for i in range(len(ps))]
                    b = blr.fit(np.hstack([f for f, _ in ps])[trn], Zd[trn], blocks)["coef"].T
                    incr[f"{tag}#{sd}"] = np.hstack([e for _, e in ps]) @ b
                    if extra:
                        for pn, pv in parts_ev.items():
                            incr[f"{tag}-{pn}#{sd}"] = np.hstack([e for _, e in parts] + [ps[-1][1] - pv]) @ b
        preds.update(incr)
    m_dp = blr.fit(zk[trn], Z[trn], [(0, r)])
    preds["desk_pooled"] = dz @ m_dp["coef"].T

    # grading (E039h's estimand and bootstrap draws)
    blk = (cells[ev, 0] // 6) * 100000 + cells[ev, 1] // 6
    ub, binv = np.unique(blk, return_inverse=True)
    rng = np.random.default_rng(11)
    W = np.vstack([np.ones(len(ev)), rng.multinomial(len(ub), np.full(len(ub), 1.0 / len(ub)), size=a.n_boot)[:, binv]])

    def grade(D, w):
        vp, vt = wc(D, D, w), wc(Ta, Tb, w)
        cv = 0.5 * (wc(D, Ta, w) + wc(D, Tb, w))
        corr = cv.sum() / np.sqrt(vp.sum() * vt.sum()) if vp.sum() > 0 and vt.sum() > 0 else np.nan
        ok = (vt > 0) & (vp > 0)
        per_dim = float(np.mean(cv[ok] / np.sqrt(vp[ok] * vt[ok]))) if ok.any() else np.nan
        return corr, (cv.sum() / vp.sum() if vp.sum() > 0 else np.nan), per_dim
    res = {"cache": a.cache, "n_cells": int(len(ev)), "rank": r, "withheld": [min(withheld), max(withheld)],
           "n_fit_rows": int(len(trn)), "notes": notes, "models": {}}
    members = {}
    for name in preds:                                  # placebo-based arms: one member per draw, graded as the mean
        members.setdefault(name.split("#")[0], []).append(name)
    G = {name: np.array([grade(D, w) for w in W]) for name, D in preds.items()}
    boots = {}
    for base, ns in members.items():
        g = np.mean([G[n] for n in ns], 0)
        boots[base] = g[:, 0]
        res["models"][base] = {"corr": float(g[0, 0]), "k": float(g[0, 1]), "per_dim_corr": float(g[0, 2]),
                               "corr_ci": [float(np.nanpercentile(g[1:, 0], q)) for q in (2.5, 97.5)]}
        if "#" in ns[0]:
            res["models"][base]["draws"] = [float(G[n][0, 0]) for n in ns]
    res["placebo"] = {"n_pos": a.n_pos, "seeds": list(a.pos_seeds)}
    for name in members:
        if name != "desk":
            d = boots["desk"] - boots[name]
            res["models"][name]["desk_minus_corr"] = float(d[0])
            res["models"][name]["desk_minus_ci"] = [float(np.nanpercentile(d[1:], q)) for q in (2.5, 97.5)]

    if a.incremental:
        if parts_ev:
            res["drop"] = {}
            for combo, base in (("placebo+desk_within", "placebo_300"), ("rich+desk_within", "rich")):
                if combo not in boots:
                    continue
                gain = boots[combo] - boots[base]
                res["drop"][combo] = {"gain": float(gain[0]), "parts": {}}
                for pn in parts_ev:
                    loss = boots[combo] - boots[f"{combo}-{pn}"]
                    res["drop"][combo]["parts"][pn] = {
                        "loss": float(loss[0]), "ci": [float(np.nanpercentile(loss[1:], q)) for q in (2.5, 97.5)],
                        "share": float(loss[0] / gain[0]) if gain[0] != 0 else None}
        if a.rich_base:
            d = boots["rich+desk_within"] - boots["rich"]
            res["rich_gain"] = {"desk_within": float(d[0]),
                                "ci": [float(np.nanpercentile(d[1:], q)) for q in (2.5, 97.5)]}
        for nm in sorted({n.split("#")[0] for n in incr if "-" not in n}):
            d = boots[nm] - boots["placebo_300"]
            res["models"][nm]["minus_placebo"] = float(d[0])
            res["models"][nm]["minus_placebo_ci"] = [float(np.nanpercentile(d[1:], q)) for q in (2.5, 97.5)]
        xy = cells[ev].astype(float) * 27.0
        D2 = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)
        res["local"] = {}
        for ell in (150.0, 300.0):
            K = np.exp(-D2 / (2 * ell ** 2))
            np.fill_diagonal(K, 0.0)
            K /= np.maximum(K.sum(1, keepdims=True), 1e-300)
            loc = lambda v: v - K @ v
            La, Lb = loc(Ta), loc(Tb)
            share = float(wc(La, Lb, W[0]).sum() / wc(Ta, Tb, W[0]).sum())

            def lgrade(D, w):
                Dl = loc(D)
                vp, vt = wc(Dl, Dl, w).sum(), wc(La, Lb, w).sum()
                cv = 0.5 * (wc(Dl, La, w).sum() + wc(Dl, Lb, w).sum())
                return cv / np.sqrt(vp * vt) if vp > 0 and vt > 0 else np.nan
            out = {"truth_local_share": share, "models": {}}
            lb_ = {}
            for nm in (("desk", "desk_within", "within_raw", "kindxtime_8", "placebo_300", "pooled_ridge")
                       + tuple(sorted({n.split("#")[0] for n in incr if "-" not in n}))):
                g = np.mean([np.array([lgrade(preds[n], w) for w in W]) for n in members[nm]], 0)
                lb_[nm] = g
                out["models"][nm] = {"corr": float(g[0]),
                                     "ci": [float(np.nanpercentile(g[1:], q)) for q in (2.5, 97.5)]}
            for nm in ("within_raw", "kindxtime_8", "placebo_300"):
                d = lb_["desk"] - lb_[nm]
                out["models"][nm]["desk_minus"] = float(d[0])
                out["models"][nm]["desk_minus_ci"] = [float(np.nanpercentile(d[1:], q)) for q in (2.5, 97.5)]
            res["local"][f"{ell:.0f}"] = out
    if a.ref_cache:
        res["vs_ref"] = {}
        for x, y in (("desk", "desk_ref"), ("desk_within", "desk_ref_within"),
                     ("placebo+desk_within", "placebo+desk_ref_within")):
            if x in boots and y in boots:
                d = boots[x] - boots[y]
                res["vs_ref"][f"{x} - {y}"] = {"diff": float(d[0]),
                                               "ci": [float(np.nanpercentile(d[1:], q)) for q in (2.5, 97.5)]}
    # spatial level on the same metric family: held-out cells x trained years, pooled readouts of ESK r24
    lev = np.where(ho & ty & np.isfinite(F_raw).all(1) & np.isfinite(Eann).all(1))[0]
    lv = {"desk_pooled": (zk[lev] - m_dp["xbar"]) @ m_dp["coef"].T,
          "pooled_ridge": ((X[lev] - X[trn].mean(0)) @ B)[:, :r]}
    lblk = (keys[lev, 0] // 6) * 100000 + keys[lev, 1] // 6
    lub, linv = np.unique(lblk, return_inverse=True)
    rng = np.random.default_rng(12)
    LW = np.vstack([np.ones(len(lev)), rng.multinomial(len(lub), np.full(len(lub), 1.0 / len(lub)),
                                                       size=a.n_boot)[:, linv]])
    lcorr = lambda p, w: float(wc(p, Z[lev], w).sum() / np.sqrt(wc(p, p, w).sum() * wc(Z[lev], Z[lev], w).sum()))
    lb = {n: np.array([lcorr(p, w) for w in LW]) for n, p in lv.items()}
    dl = lb["desk_pooled"] - lb["pooled_ridge"]
    res["space_level"] = {"n_rows": int(len(lev)),
                          **{n: float(v[0]) for n, v in lb.items()},
                          "desk_minus_linear": float(dl[0]),
                          "desk_minus_linear_ci": [float(np.percentile(dl[1:], q)) for q in (2.5, 97.5)]}
    fn = ("fair_rivals_incr" if a.incremental else "fair_rivals") + a.tag + ".json"
    with open(os.path.join(a.out, fn), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    summarize(a.out, fn)


def summarize(out, fn=None):
    if fn is None:
        import glob
        for f in sorted(glob.glob(os.path.join(out, "fair_rivals*.json"))):
            summarize(out, os.path.basename(f))
        return
    r = json.load(open(os.path.join(out, fn), encoding="utf-8"))
    pl = r.get("placebo", {"n_pos": 24, "seeds": [1]})
    print(f"[{fn}; placebo {pl['n_pos']} features x {len(pl['seeds'])} draws]")
    print(f"{os.path.basename(os.path.normpath(r['cache']))}: {r['n_cells']} held-out cells, withheld {r['withheld']}, "
          f"r{r['rank']}, {r['n_fit_rows']:,} fit rows; desk keys-vs-epoch max|diff| "
          f"{r['notes']['desk_keys_vs_epoch_z_maxabs']:.2g}")
    for name, v in r["models"].items():
        if "-" in name:
            continue
        extra = (f" | desk - it {v['desk_minus_corr']:+.3f} [{v['desk_minus_ci'][0]:+.3f},{v['desk_minus_ci'][1]:+.3f}]"
                 if "desk_minus_corr" in v else "")
        if "minus_placebo" in v:
            extra += (f" | - placebo_300 {v['minus_placebo']:+.3f} [{v['minus_placebo_ci'][0]:+.3f},"
                      f"{v['minus_placebo_ci'][1]:+.3f}]")
        print(f"  {name:13s} corr {v['corr']:+.3f} [{v['corr_ci'][0]:+.3f},{v['corr_ci'][1]:+.3f}] k {v['k']:5.2f} "
              f"per-dim {v['per_dim_corr']:+.3f}" + extra)
    for ell, o in r.get("local", {}).items():
        print(f"  LOCAL {ell} km (truth's noise-free change energy that is local: {o['truth_local_share']:.2f})")
        for nm, v in o["models"].items():
            extra = (f" | desk - it {v['desk_minus']:+.3f} [{v['desk_minus_ci'][0]:+.3f},{v['desk_minus_ci'][1]:+.3f}]"
                     if "desk_minus" in v else "")
            print(f"    {nm:24s} corr {v['corr']:+.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]" + extra)
    for combo, o in r.get("drop", {}).items():
        print(f"  DROP-PART losses in {combo} (gain over its base {o['gain']:+.3f}):")
        for pn, v in o["parts"].items():
            sh = f"{v['share']:+.2f}" if v["share"] is not None else "  n/a"
            print(f"    -{pn:14s} loss {v['loss']:+.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]  share of gain {sh}")
    if "rich_gain" in r:
        v = r["rich_gain"]
        print(f"  DESK gain beyond placebo + kind x time 32 + within map: {v['desk_within']:+.3f} "
              f"[{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]")
    for nm, v in r.get("vs_ref", {}).items():
        print(f"  TREATMENT - CONTROL {nm:46s} {v['diff']:+.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]")
    s = r["space_level"]
    print(f"  SPACE level ({s['n_rows']:,} held-out rows, trained years): desk_pooled {s['desk_pooled']:+.3f} "
          f"pooled_ridge {s['pooled_ridge']:+.3f}; desk - linear {s['desk_minus_linear']:+.3f} "
          f"[{s['desk_minus_linear_ci'][0]:+.3f},{s['desk_minus_linear_ci'][1]:+.3f}]")


if __name__ == "__main__":
    main()
