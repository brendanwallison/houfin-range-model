"""E044b: is the covariates' alignment with the regional community trend SPECIFIC, or would any equally smooth field do?

    python research/exp/shift_null.py --cache <cache> --out <dir>      (normalization from the cache's meta run_dir)
    python research/exp/shift_null.py --summarize <dir>

E044 found that covariate-only maps predict nearly all of the held-out community change that extrapolating the
observed regional trend predicts. A smooth covariate-change field can act as a position x time basis, though, so
part of that overlap could be any two smooth fields lining up. Here the covariate (and DESK) fields are moved
on the map -- reflected east-west / north-south / both, or shifted 650-1200 km (each target cell mapped to its
nearest valid cell) -- which keeps their smoothness and temporal structure but breaks their alignment with the
real places, and the same maps are REFIT on the moved fields:
    within_sup   supervised ridge of within-cell ESK deviations on all standardized covariate deviations (24 out),
                 cross-fitted by training cell, applied to the held-out cells' covariate change (E039n)
    desk_within  within-cell BLR of ESK deviations on DESK raw z deviations (DESK's own field moved; nearest
                 SURVEYED cell, since DESK z exists for the 3,902 surveyed cells only)
Same estimand, cells, truth halves and bootstrap draws as fair_rivals.py. specificity = corr(real) - mean over moves
corr(moved), paired. The identity move rebuilds F_cov_raw from the yearly states (check: max |diff|).
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

MOVES = {"identity": None, "flip_ew": "ew", "flip_ns": "ns", "rot180": "both",
         "shift_e45": (0, 45), "shift_w45": (0, -45), "shift_n25": (-25, 0), "shift_s25": (25, 0)}


def wc(x, y, w):
    xm = (w[:, None] * x).sum(0) / w.sum()
    ym = (w[:, None] * y).sum(0) / w.sum()
    return (w[:, None] * (x - xm) * (y - ym)).sum(0)


def move_map(valid_rc, H, W, how):
    """For every cell (r, c) of the H x W grid: the valid cell its moved field comes from (index into valid_rc)."""
    from scipy.spatial import cKDTree
    rr, cc = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    if how is None:
        tr, tc = rr, cc
    elif how == "ew":
        tr, tc = rr, (W - 1) - cc
    elif how == "ns":
        tr, tc = (H - 1) - rr, cc
    elif how == "both":
        tr, tc = (H - 1) - rr, (W - 1) - cc
    else:
        tr, tc = rr + how[0], cc + how[1]
    _, j = cKDTree(valid_rc.astype(float)).query(np.column_stack([tr.ravel(), tc.ravel()]).astype(float))
    return j.reshape(H, W)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache")
    ap.add_argument("--out")
    ap.add_argument("--rank", type=int, default=24)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--summarize", default=None)
    a = ap.parse_args()
    if a.summarize:
        return summarize(a.summarize)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    from scipy import sparse
    from change_oracle import desk_epoch_z, group_rows
    from lib import blr
    from src.config_utils import load_config
    from src.community_encoder.train_DESK import covariate_io as cio
    os.makedirs(a.out, exist_ok=True)
    c = lambda f: os.path.join(a.cache, f)
    keys = np.load(c("keys.npy"))
    N = len(keys)
    split = np.load(c("split.npz"))
    withheld = [int(y) for y in split["withheld"]]
    r = a.rank
    E, full, _, _ = group_rows(a.cache, keys, withheld)
    idx = {}
    for j, (cell, ep, wh) in enumerate(zip(map(tuple, E["cells"]), E["epoch"], E["withheld"])):
        if ep == 0 and bool(wh):
            idx.setdefault(cell, [None, None])[0] = j
        elif ep == 1 and not wh:
            idx.setdefault(cell, [None, None])[1] = j
    pairs = np.array([(e, m) for e, m in idx.values() if e is not None and m is not None])
    cells = E["cells"][pairs[:, 0]]
    ev = np.where(split["holdout"][cells[:, 0], cells[:, 1]])[0]
    Za, Zb = E["z_a"][:, :r].astype("float64"), E["z_b"][:, :r].astype("float64")
    Ta = (Za[pairs[:, 1]] - Za[pairs[:, 0]])[ev]
    Tb = (Zb[pairs[:, 1]] - Zb[pairs[:, 0]])[ev]
    early_rows = [np.asarray(full[pairs[j, 0]]) for j in ev]
    mod_rows = [np.asarray(full[pairs[j, 1]]) for j in ev]

    def gmat(groups):
        rows = np.concatenate([np.full(len(g), i) for i, g in enumerate(groups)])
        cols = np.concatenate([np.asarray(g, int) for g in groups])
        vals = np.concatenate([np.full(len(g), 1.0 / len(g)) for g in groups])
        return sparse.csr_matrix((vals, (rows, cols)), shape=(len(groups), N))
    Ge, Gm = gmat(early_rows), gmat(mod_rows)
    dchg = lambda X: np.asarray(Gm @ np.asarray(X, "float64")) - np.asarray(Ge @ np.asarray(X, "float64"))

    F_raw = np.asarray(np.load(c("F_cov_raw.npy"), mmap_mode="r"), "float64")
    Eann = np.load(c("esk_annual.npy")).astype("float64")
    ho = split["holdout"][keys[:, 0], keys[:, 1]]
    bf = split["buffer"][keys[:, 0], keys[:, 1]]
    ty = ~np.isin(keys[:, 2], withheld)
    trn = np.where(~ho & ~bf & ty & np.isfinite(F_raw).all(1) & np.isfinite(Eann).all(1))[0]
    Z = Eann[:, :r]
    cid = keys[:, 0].astype(np.int64) * 100000 + keys[:, 1]
    uc, inv = np.unique(cid, return_inverse=True)
    w_ty = (ty & np.isfinite(Eann).all(1)).astype(float)

    def cmean(X, w):
        S = sparse.csr_matrix((w, (inv, np.arange(N))), shape=(len(uc), N))
        cnt = np.asarray(S.sum(1)).ravel()
        return (np.asarray(S @ X) / np.maximum(cnt, 1e-12)[:, None])[inv]
    Zd = Z - cmean(Z, w_ty)
    cells_tr = np.unique(inv[trn])
    rgf = np.random.default_rng(7)
    fold_of = dict(zip(cells_tr.tolist(), rgf.integers(2, size=len(cells_tr)).tolist()))
    fold = np.array([fold_of.get(int(cc), -1) for cc in inv])

    def within_sup(F):
        mu, sd = F[trn].mean(0), F[trn].std(0)
        keep = sd > 1e-6
        X = np.where(np.isfinite(F[:, keep]), (F[:, keep] - mu[keep]) / sd[keep], 0.0)
        Sdev, dS = X - cmean(X, w_ty), dchg(X)
        Xc = Sdev[trn]
        Bw = np.linalg.solve(Xc.T @ Xc + 1e-2 * np.trace(Xc.T @ Xc) / Xc.shape[1] * np.eye(Xc.shape[1]),
                             Xc.T @ Zd[trn])
        return dS @ Bw            # the held-out prediction uses the full-fit map (as fair_rivals' within_sup)

    def desk_within(zk, dz):
        return dz @ blr.fit((zk - cmean(zk, w_ty))[trn], Zd[trn], [(0, r)])["coef"].T

    # moved covariate fields, rebuilt from the yearly states
    cfg = load_config()
    states_dir = os.path.join(cfg["paths"]["hist_dir"], "yearly_states")
    run_dir = json.load(open(c("meta.json"), encoding="utf-8"))["run_dir"]
    dm = np.load(os.path.join(run_dir, "desk_meta.npz"), allow_pickle=True)
    schema = json.loads(str(dm["schema"]))
    mu_d, sd_d = dm["mu"].astype("float32"), dm["sd"].astype("float32")
    yrs = np.unique(keys[:, 2])
    maps = None
    Fm = {m: np.full((N, F_raw.shape[1]), np.nan, "float32") for m in MOVES}
    for y in yrs:
        covn, mask = cio.norm_grid(cio.load_state_stack(int(y), states_dir, schema), mu_d, sd_d)
        if maps is None:
            H, Wd = mask.shape
            valid_rc = np.argwhere(mask)
            maps = {m: valid_rc[move_map(valid_rc, H, Wd, how)] for m, how in MOVES.items()}
        sel = np.where(keys[:, 2] == y)[0]
        for m, mp in maps.items():
            src = mp[keys[sel, 0], keys[sel, 1]]
            Fm[m][sel] = covn[src[:, 0], src[:, 1]]
    fin_id = np.isfinite(F_raw) & np.isfinite(Fm["identity"])
    check = float(np.abs(Fm["identity"][fin_id] - F_raw[fin_id]).max())
    # moved DESK fields: nearest SURVEYED cell (DESK z exists for the cache's cells only)
    from scipy.spatial import cKDTree
    zr_all = np.load(c("z_raw_all.npy"), mmap_mode="r")
    zcells = np.load(c("cells.npy"))
    y0 = int(np.load(c("years.npy"))[0])
    ci_ = np.load(c("key_cell_index.npy"))
    tree = cKDTree(zcells.astype(float))
    preds, zk_id = {}, None
    for m, mp in maps.items():
        Fmm = np.asarray(Fm[m], "float64")
        Fmm[~np.isfinite(Fmm)] = np.nan
        preds[f"within_sup:{m}"] = within_sup(Fmm)
        src = mp[keys[:, 0], keys[:, 1]]
        _, zi = tree.query(src.astype(float))
        zk = np.asarray(zr_all[zi, keys[:, 2] - y0, :r], "float64")
        if m == "identity":
            zk_id = zk
        # held-out change from the moved field, over the same epoch rows
        dz = (desk_epoch_z(zr_all, zi, y0, keys, mod_rows, r) - desk_epoch_z(zr_all, zi, y0, keys, early_rows, r))
        preds[f"desk_within:{m}"] = desk_within(zk, dz)
        print(f"  move {m} done", flush=True)
    zk0 = np.asarray(zr_all[ci_, keys[:, 2] - y0, :r], "float64")

    blk = (cells[ev, 0] // 6) * 100000 + cells[ev, 1] // 6
    ub, binv = np.unique(blk, return_inverse=True)
    rng = np.random.default_rng(11)
    Wb = np.vstack([np.ones(len(ev)), rng.multinomial(len(ub), np.full(len(ub), 1.0 / len(ub)),
                                                      size=a.n_boot)[:, binv]])

    def corr(D, w):
        vp, vt = wc(D, D, w).sum(), wc(Ta, Tb, w).sum()
        cv = 0.5 * (wc(D, Ta, w).sum() + wc(D, Tb, w).sum())
        return cv / np.sqrt(vp * vt) if vp > 0 and vt > 0 else np.nan
    boots = {k: np.array([corr(D, w) for w in Wb]) for k, D in preds.items()}
    res = {"cache": a.cache, "n_cells": int(len(ev)), "identity_vs_F_cov_raw_maxabs": check,
           "desk_identity_matches": bool(np.allclose(zk0, zk_id)),
           "arms": {k: {"corr": float(v[0]), "ci": [float(np.nanpercentile(v[1:], q)) for q in (2.5, 97.5)]}
                    for k, v in boots.items()}, "specificity": {}}
    moved = [m for m in MOVES if m != "identity"]
    for arm in ("within_sup", "desk_within"):
        null = np.mean([boots[f"{arm}:{m}"] for m in moved], 0)
        d = boots[f"{arm}:identity"] - null
        res["specificity"][arm] = {"real": float(boots[f"{arm}:identity"][0]), "moved_mean": float(null[0]),
                                   "moved_each": {m: float(boots[f"{arm}:{m}"][0]) for m in moved},
                                   "diff": float(d[0]), "ci": [float(np.nanpercentile(d[1:], q)) for q in (2.5, 97.5)]}
    with open(os.path.join(a.out, "shift_null.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    summarize(a.out)


def summarize(out):
    r = json.load(open(os.path.join(out, "shift_null.json"), encoding="utf-8"))
    print(f"{os.path.basename(os.path.normpath(r['cache']))}: {r['n_cells']} held-out cells; identity rebuild vs "
          f"F_cov_raw max|diff| {r['identity_vs_F_cov_raw_maxabs']:.2g}")
    for arm, v in r["specificity"].items():
        each = " ".join(f"{m} {x:+.3f}" for m, x in v["moved_each"].items())
        print(f"  {arm:12s} real {v['real']:+.3f}  moved mean {v['moved_mean']:+.3f}  real - moved {v['diff']:+.3f} "
              f"[{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]  | {each}")


if __name__ == "__main__":
    main()
