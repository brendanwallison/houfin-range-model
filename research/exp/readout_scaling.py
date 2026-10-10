"""E033: can a readout that scales DESK's change by how reliable it is -- learned on TRAINED years, in a form the age
model's BLR could adopt -- stop the harm into withheld decades, and does a shrink learned in the BBS era carry over?

    ESK_DESK_CONFIG=<overlay> python research/exp/readout_scaling.py --cache <dir> --out <dir> --t0 1996
    python research/exp/readout_scaling.py --summarize <out>

Readouts of DESK's change (held-out cells, withheld early epoch vs the modern epoch, species log1p epoch means):
  R0 level      coefficients fitted on training cells' modern LEVELS (the current readout; E013/E030)
  R1 calibrated R0 x k, k = truth-on-prediction slope of R0 on TRAINING cells' trained-era change ([T0, T0+9] vs
                2016-2025): one k pooled over species, and one per prevalence tier
  R2 split      a BLR on training cells' 5-year windows inside the trained years with two blocks, each with its own
                learned amplitude: the cell's mean z (place) and the window's deviation from it (change). Change is
                read through the deviation block only -- the form the downstream contract allows (a block with its own
                iid amplitude); the BLR itself decides how much to trust within-place change.
Each is scored like E030: correlation (information), k on the withheld change (scaling), squared-error skill against
no change, centred (place-specific) and uncentred, pooled over resolvable species and by prevalence tier. The best any
rescaling can reach is corr^2 (M1), so the question is mostly whether a readout learned on trained years avoids harm.
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
from lib import blr  # noqa: E402

TIERS = (0.0, 0.02, 0.1, 0.3, 1.01)


def ccov(x, y):
    return ((x - x.mean(0)) * (y - y.mean(0))).mean(0)


def cmom(x, y):
    return (x * y).mean(0)


def score(pred, da, db, dfull, m):
    """E030 decomposition for a prediction with no survey noise, pooled over species mask m."""
    out = {}
    for form, f in (("centred", ccov), ("uncentred", cmom)):
        V = float(f(da, db)[m].sum())
        C = float(f(pred, dfull)[m].sum())
        P = float(f(pred, pred)[m].sum())
        if V <= 0 or P <= 0:
            continue
        corr = C / np.sqrt(P * V)
        out[form] = {"corr": corr, "k": C / P, "skill": (2 * C - P) / V, "best": corr ** 2 if corr > 0 else 0.0,
                     "scale": float(np.sqrt(P / V))}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache")
    ap.add_argument("--out")
    ap.add_argument("--t0", type=int, default=None, help="first trained year of the model in the cache (required to run)")
    ap.add_argument("--rank", type=int, default=24)
    ap.add_argument("--summarize", default=None)
    a = ap.parse_args()
    if a.summarize:
        return summarize(a.summarize)
    from change_oracle import desk_epoch_z, group_rows
    from shared_change import groups_for
    from src.community_encoder.train_DESK.validate_bbs_routes import epoch_mean_observed
    from src.community_encoder.train_DESK.validation_core import change_noise
    os.makedirs(a.out, exist_ok=True)
    c = lambda f: os.path.join(a.cache, f)
    keys = np.load(c("keys.npy"))
    split = np.load(c("split.npz"))
    yr = keys[:, 2]
    cid = keys[:, 0].astype(np.int64) * 100000 + keys[:, 1]
    E, full, half_a, half_b = group_rows(a.cache, keys, [int(y) for y in split["withheld"]])
    Xd = np.load(c("X_dev.npy")).astype("float64")
    idx = {}
    for j, (cell, ep, wh) in enumerate(zip(map(tuple, E["cells"]), E["epoch"], E["withheld"])):
        if ep == 0 and bool(wh):
            idx.setdefault(cell, [None, None])[0] = j
        elif ep == 1 and not wh:
            idx.setdefault(cell, [None, None])[1] = j
    pairs = np.array([(e, m) for e, m in idx.values() if e is not None and m is not None])
    cells = E["cells"][pairs[:, 0]]
    held = split["holdout"][cells[:, 0], cells[:, 1]]
    buf = split["buffer"][cells[:, 0], cells[:, 1]]
    tr, ev = np.where(~held & ~buf)[0], np.where(held)[0]
    Y = {h: epoch_mean_observed(Xd, g).astype("float64") for h, g in (("full", full), ("a", half_a), ("b", half_b))}
    keep = (Y["full"] > 0).any(0)
    Y = {h: v[:, keep] for h, v in Y.items()}
    Xk = Xd[:, keep]
    d = {h: (v[pairs[:, 1]] - v[pairs[:, 0]]) for h, v in Y.items()}
    da, db, dfull = d["a"][ev], d["b"][ev], d["full"][ev]
    nz = change_noise(dfull, da, db)
    resm = nz["resolvable"]
    prev = (Xk > 0).mean(0)
    r = a.rank
    zr = np.load(c("z_raw_all.npy"), mmap_mode="r")
    ci, y0 = np.load(c("key_cell_index.npy")), int(np.load(c("years.npy"))[0])
    Fd = desk_epoch_z(zr, ci, y0, keys, full, r)
    dZ = (Fd[pairs[:, 1]] - Fd[pairs[:, 0]])[ev]

    # R0: level readout (training cells' modern levels)
    lvl = np.unique(pairs[tr, 1])
    beta0 = blr.fit(Fd[lvl], Y["b"][lvl], [(0, r)])["coef"].T
    pred0 = dZ @ beta0

    # R1: calibrate R0 on training cells' trained-era change [t0, t0+9] -> 2016-2025
    train_rows = ~split["holdout"][keys[:, 0], keys[:, 1]] & ~split["buffer"][keys[:, 0], keys[:, 1]]
    G = groups_for(keys, cid, train_rows, a.t0, a.t0 + 9, 2016, 2025)
    z1 = desk_epoch_z(zr, ci, y0, keys, [g[1] for g in G], r)
    z2 = desk_epoch_z(zr, ci, y0, keys, [g[2] for g in G], r)
    ptr = (z2 - z1) @ beta0
    m = lambda rl: epoch_mean_observed(Xk, rl)
    tfull = m([g[2] for g in G]) - m([g[1] for g in G])
    ta = m([g[4][0] for g in G]) - m([g[3][0] for g in G])
    tb = m([g[4][1] for g in G]) - m([g[3][1] for g in G])
    rtr = change_noise(tfull, ta, tb)["resolvable"]
    kcov, kvar = cmom(ptr, tfull), cmom(ptr, ptr)
    k_pool = float(kcov[rtr].sum() / kvar[rtr].sum())
    k_tier = np.full(Xk.shape[1], k_pool)
    tiers_k = {}
    for lo, hi in zip(TIERS[:-1], TIERS[1:]):
        mt = rtr & (prev >= lo) & (prev < hi)
        if mt.sum() >= 3 and kvar[mt].sum() > 0:
            kt = float(kcov[mt].sum() / kvar[mt].sum())
            k_tier[(prev >= lo) & (prev < hi)] = kt
            tiers_k[f"{lo:g}-{hi:g}"] = kt
    pred1p = pred0 * max(k_pool, 0.0)
    pred1t = pred0 * np.maximum(k_tier, 0.0)[None, :]

    # R2: split-block BLR on 5-year windows of the trained years (training cells)
    wins = [(y, min(y + 4, 2025)) for y in range(a.t0, 2026, 5)]
    rows_w, cell_w = [], []
    by = {}
    for i in np.where(train_rows & (yr >= a.t0))[0]:
        w = (yr[i] - a.t0) // 5
        by.setdefault((cid[i], w), []).append(i)
    for (cc, w), rl in by.items():
        if len(rl) >= 2:
            rows_w.append(np.array(rl))
            cell_w.append(cc)
    cell_w = np.array(cell_w)
    Zw = desk_epoch_z(zr, ci, y0, keys, rows_w, r)
    Yw = epoch_mean_observed(Xk, rows_w)
    uc, inv = np.unique(cell_w, return_inverse=True)
    cnt = np.bincount(inv).astype(float)
    zs = np.zeros((len(uc), r))
    np.add.at(zs, inv, Zw)
    zbar = (zs / cnt[:, None])[inv]
    multi = cnt[inv] >= 2
    fit2 = blr.fit(np.hstack([zbar, Zw - zbar])[multi], Yw[multi], [(0, r), (r, 2 * r)])
    beta2 = fit2["coef"][:, r:].T
    amp = np.asarray(fit2["a"])                                      # (S, 2) block amplitudes
    pred2 = dZ @ beta2

    # E033b placebos: the change block replaced (R2p) or extended (R2+p) by position x time, or a continental time (R2t)
    wyear = np.array([yr[rl].mean() for rl in rows_w])
    ys = np.zeros(len(uc))
    np.add.at(ys, inv, wyear)
    tdev = wyear - (ys / cnt)[inv]
    rng = np.random.default_rng(0)
    Wf, bf = rng.normal(size=(2, 24)) / 300.0, rng.uniform(0, 2 * np.pi, 24)
    pos = lambda cc: np.sqrt(2.0 / 24) * np.cos(np.stack([cc // 100000, cc % 100000], 1).astype("float64") * 27.0 @ Wf + bf)
    Xp = pos(cell_w) * tdev[:, None]
    fit_p = blr.fit(np.hstack([zbar, Xp])[multi], Yw[multi], [(0, r), (r, r + 24)])
    fit_pp = blr.fit(np.hstack([zbar, Zw - zbar, Xp])[multi], Yw[multi], [(0, r), (r, 2 * r), (2 * r, 2 * r + 24)])
    fit_t = blr.fit(np.hstack([zbar, tdev[:, None]])[multi], Yw[multi], [(0, r), (r, r + 1)])
    # held-out cells: position features and the gap between the two epochs' mean years
    ev_cid = (cells[ev, 0].astype(np.int64) * 100000 + cells[ev, 1])
    dyear = np.array([yr[full[pairs[j, 1]]].mean() - yr[full[pairs[j, 0]]].mean() for j in ev])
    Pev = pos(ev_cid) * dyear[:, None]
    pred2p = Pev @ fit_p["coef"][:, r:].T
    pred2pp = dZ @ fit_pp["coef"][:, r:2 * r].T + Pev @ fit_pp["coef"][:, 2 * r:].T
    pred2t = dyear[:, None] * fit_t["coef"][:, r][None, :]

    def all_scores(mask):
        return {"n_species": int(mask.sum()),
                "R0_level": score(pred0, da, db, dfull, mask),
                "R1_cal_pooled": score(pred1p, da, db, dfull, mask),
                "R1_cal_tier": score(pred1t, da, db, dfull, mask),
                "R2_split": score(pred2, da, db, dfull, mask),
                "R2p_placebo": score(pred2p, da, db, dfull, mask),
                "R2pp_desk_plus_placebo": score(pred2pp, da, db, dfull, mask),
                "R2t_time_only": score(pred2t, da, db, dfull, mask)}
    res = {"cache": a.cache, "t0": a.t0, "rank": r, "n_eval_cells": int(len(ev)), "n_cal_cells": len(G),
           "k_trained_pooled": k_pool, "k_trained_tiers": tiers_k, "n_windows_r2": int(multi.sum()),
           "r2_amplitude_ratio_median": float(np.median(amp[:, 1] / np.maximum(amp[:, 0], 1e-12))),
           "all": all_scores(resm), "tiers": {}}
    for lo, hi in zip(TIERS[:-1], TIERS[1:]):
        res["tiers"][f"{lo:g}-{hi:g}"] = all_scores(resm & (prev >= lo) & (prev < hi))
    with open(os.path.join(a.out, "readout_scaling.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    summarize(a.out)


def summarize(out):
    r = json.load(open(os.path.join(out, "readout_scaling.json"), encoding="utf-8"))
    print(f"readout scaling [{os.path.basename(os.path.dirname(r['cache'] + '/'))}, T0 {r['t0']}]: {r['n_eval_cells']} "
          f"held-out cells; k learned on trained-era change (training cells, {r['n_cal_cells']}): pooled "
          f"{r['k_trained_pooled']:.2f}, tiers " + ", ".join(f"{k} {v:.2f}" for k, v in r["k_trained_tiers"].items())
          + f"; split readout: change-block / place-block amplitude, median {r['r2_amplitude_ratio_median']:.3f}")
    for name, s in [("all", r["all"])] + [(f"prev {k}", v) for k, v in r["tiers"].items()]:
        for form in ("centred", "uncentred"):
            parts = []
            for arm in ("R0_level", "R1_cal_pooled", "R2_split", "R2p_placebo", "R2pp_desk_plus_placebo", "R2t_time_only"):
                v = s.get(arm, {}).get(form)
                if v:
                    parts.append(f"{arm} corr {v['corr']:+.3f} k {v['k']:.2f} skill {v['skill']:+.3f}")
            print(f"  {name:13s} n={s['n_species']:3d} [{form:9s}] " + " | ".join(parts))


if __name__ == "__main__":
    main()
