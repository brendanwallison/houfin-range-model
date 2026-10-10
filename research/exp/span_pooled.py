"""E024b: raw DESK backcast of short-span vs long-span models, per window and POOLED over the pre-T0 windows, with ONE
6x6-block bootstrap shared by every model and window (paired) -- the statistic span_reach.py cannot give, since its raw
arms sit in different span rows with different draws. Truth exactly as span_reach (noise-free centered corr of
held-out cells' change, window vs the first ten trained years). Adapted from the skeptic's e024_pooled.py.

    ESK_DESK_CONFIG=<overlay> python research/exp/span_pooled.py --out <dir> --t0 1986 \
        --short desk_span_1986_2005 desk_span_1986_2005_s1 --long desk_tempho_1985_s1 desk_tempho_1985_s2 [--tag late]
    python research/exp/span_pooled.py --summarize <dir>
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
ROOT = os.path.expanduser("~/houfin/work/houfin/research_cache")
WINDOWS = [(1966, 1971), (1972, 1977), (1978, 1983), (1984, 1989), (1990, 1995)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out")
    ap.add_argument("--t0", type=int)
    ap.add_argument("--short", nargs="+", help="cache subdirs of the short-span models")
    ap.add_argument("--long", nargs="+", help="cache subdirs of the long-span models (same T0)")
    ap.add_argument("--pool-windows", type=int, default=3, help="pool the FIRST n pre-T0 windows (pre-declared: 3)")
    ap.add_argument("--rank", type=int, default=24)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--tag", default="")
    ap.add_argument("--summarize", default=None)
    a = ap.parse_args()
    if a.summarize:
        return summarize(a.summarize)
    from src.config_utils import load_config
    from src.community_encoder.train_DESK.esk_kernel import project_points_to_z
    from src.community_encoder.train_DESK.validate_bbs_routes import epoch_mean_observed
    from src.community_encoder.train_DESK.validation_core import split_half_groups
    os.makedirs(a.out, exist_ok=True)
    base = os.path.join(ROOT, "gp_species_base")
    keys = np.load(os.path.join(base, "keys.npy"))
    split = np.load(os.path.join(base, "split.npz"))
    yr = keys[:, 2]
    held = split["holdout"][keys[:, 0], keys[:, 1]]
    cid = keys[:, 0].astype(np.int64) * 100000 + keys[:, 1]
    r, t0 = a.rank, a.t0
    ref = (t0, t0 + 9)
    wins = [w for w in WINDOWS if w[1] < t0]
    groups, gkey = [], []
    for (lo, hi) in wins + [ref]:
        sel = np.where(held & (yr >= lo) & (yr <= hi))[0]
        sel = sel[np.lexsort((yr[sel], cid[sel]))]
        cs, st = np.unique(cid[sel], return_index=True)
        for c_, s, e in zip(cs, st, list(st[1:]) + [len(sel)]):
            if e - s >= 4:
                groups.append(sel[s:e])
                gkey.append((int(c_), lo, hi))
    A, B, ok = split_half_groups(groups, years=yr)
    Xc = np.load(os.path.join(base, "X_comm.npy")).astype("float64")
    cfg = load_config()
    proj = lambda v: np.asarray(project_points_to_z(np.asarray(v, "float32"), cfg["desk"]["z_dir"], 64),
                                "float64")[:, :r]
    T = {"full": proj(epoch_mean_observed(Xc, groups)), "a": proj(epoch_mean_observed(Xc, A)),
         "b": proj(epoch_mean_observed(Xc, B))}
    gidx = {k: j for j, k in enumerate(gkey) if ok[j]}
    gmean = lambda M, g: np.asarray(M[g], "float64").mean(0)
    names = list(a.short) + list(a.long)
    Z = [np.asarray(np.load(os.path.join(ROOT, s, "z_raw.npy"), mmap_mode="r")[:, :r], "float64") for s in names]
    allblk = np.unique((keys[held, 0] // 6) * 100000 + keys[held, 1] // 6)
    bpos = {b: i for i, b in enumerate(allblk)}
    Wb = np.random.default_rng(11).multinomial(len(allblk), np.full(len(allblk), 1.0 / len(allblk)),
                                               size=a.n_boot).astype(float)
    ns = len(a.short)
    res = {"t0": t0, "tag": a.tag, "short": list(a.short), "long": list(a.long), "pool_windows": a.pool_windows,
           "windows": []}
    per = []
    for w in wins:
        cl = [c_ for (c_, l_, h_) in gidx if (l_, h_) == w and (c_, *ref) in gidx]
        jw = np.array([gidx[(c_, *w)] for c_ in cl])
        jr = np.array([gidx[(c_, *ref)] for c_ in cl])
        gw, gr = [groups[j] for j in jw], [groups[j] for j in jr]
        dT = {h: T[h][jw] - T[h][jr] for h in T}
        bi = np.array([bpos[(c_ // 100000 // 6) * 100000 + (c_ % 100000) // 6] for c_ in cl])
        ws = np.vstack([np.ones(len(cl))[None], Wb[:, bi]])

        def corr(p, wt):
            cen = lambda v: v - (wt[:, None] * v).sum(0) / wt.sum()
            p_, f_, a_, b_ = cen(p), cen(dT["full"]), cen(dT["a"]), cen(dT["b"])
            return (wt[:, None] * p_ * f_).sum() / np.sqrt((wt[:, None] * p_ * p_).sum() * (wt[:, None] * a_ * b_).sum())
        P = [np.stack([gmean(z, g1) - gmean(z, g2) for g1, g2 in zip(gw, gr)]) for z in Z]
        C = np.array([[corr(p, wt) for wt in ws] for p in P])          # (models, n_boot + 1)
        per.append(C)
        res["windows"].append({"window": list(w), "n_cells": len(cl), "corr": {n: float(C[i, 0]) for i, n in enumerate(names)}})
    res["pooled"] = {}
    C = np.mean(per[: a.pool_windows], axis=0)

    def stat(v):
        return {"value": float(v[0]), "ci": [float(np.percentile(v[1:], 2.5)), float(np.percentile(v[1:], 97.5))]}
    long_mean = C[ns:].mean(0)
    res["pooled"]["corr"] = {n: float(C[i, 0]) for i, n in enumerate(names)}
    for i, n in enumerate(a.short):
        res["pooled"][f"{n} - mean(long)"] = stat(C[i] - long_mean)
    res["pooled"]["mean(short) - mean(long)"] = stat(C[:ns].mean(0) - long_mean)
    if len(a.long) == 2:
        res["pooled"]["long seed gap"] = stat(C[ns] - C[ns + 1])
    if ns == 2:
        res["pooled"]["short seed gap"] = stat(C[0] - C[1])
    tag = f"_{a.tag}" if a.tag else ""
    with open(os.path.join(a.out, f"span_pooled_{t0}{tag}.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    summarize(a.out)


def summarize(out):
    for f in sorted(os.listdir(out)):
        if not (f.startswith("span_pooled_") and f.endswith(".json")):
            continue
        r = json.load(open(os.path.join(out, f), encoding="utf-8"))
        print(f"T0 {r['t0']} {r['tag'] or 'selected'}: short {r['short']} vs long {r['long']}")
        for w in r["windows"]:
            print(f"  W {w['window'][0]}-{w['window'][1] % 100:02d} n={w['n_cells']:3d} " +
                  " ".join(f"{k} {v:+.3f}" for k, v in w["corr"].items()))
        for k, v in r["pooled"].items():
            if k == "corr":
                continue
            print(f"  POOLED (first {r['pool_windows']} windows) {k}: {v['value']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]")


if __name__ == "__main__":
    main()
