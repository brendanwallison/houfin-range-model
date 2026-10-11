"""E038b: paired comparison of two models' backcast attribution on the SAME held-out cells.

    python research/exp/attr_pair.py --a <attribution dir of model A> --b <dir of model B> --out <json> [--parts full ...]
    python research/exp/attr_pair.py --summarize <json> [...]

Reads each run's attribution_cells.npz (backcast_attribution.py) and, with one shared 6x6-cell block bootstrap,
reports A - B for: noise-free centred corr of a part with the observed change (truth halves Ta/Tb), and the part's
alignment with the analog direction (each run's own Dir -- run both with the same --dir-cache so the analog CELLS
are chosen by the same covariates). The runs must cover the same cells in the same order.
"""
import argparse
import json
import os

import numpy as np


def wc(x, y, w):
    xm = (w[:, None] * x).sum(0) / w.sum()
    ym = (w[:, None] * y).sum(0) / w.sum()
    return float((w[:, None] * (x - xm) * (y - ym)).sum())


def corr(D, Ta, Tb, w):
    vp, vt = wc(D, D, w), wc(Ta, Tb, w)
    return 0.5 * (wc(D, Ta, w) + wc(D, Tb, w)) / np.sqrt(vp * vt) if vp > 0 and vt > 0 else np.nan


def align(B, D, w):
    vb, vd = wc(B, B, w), wc(D, D, w)
    return wc(B, D, w) / np.sqrt(vb * vd) if vb > 0 and vd > 0 else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--a")
    ap.add_argument("--b")
    ap.add_argument("--out")
    ap.add_argument("--parts", nargs="*", default=["full", "land_part", "climate_part", "luh_part", "moisture_part"])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--summarize", nargs="*", default=None)
    a = ap.parse_args()
    if a.summarize:
        return summarize(a.summarize)
    A = np.load(os.path.join(a.a, "attribution_cells.npz"))
    B = np.load(os.path.join(a.b, "attribution_cells.npz"))
    if not np.array_equal(A["cells"], B["cells"]):
        raise SystemExit("the two runs cover different cells")
    cells = A["cells"]
    blk = (cells[:, 0] // 6) * 100000 + cells[:, 1] // 6
    ub, inv = np.unique(blk, return_inverse=True)
    rng = np.random.default_rng(11)
    W = rng.multinomial(len(ub), np.full(len(ub), 1.0 / len(ub)), size=a.n_boot).astype(float)[:, inv]
    W = np.vstack([np.ones(len(cells)), W])
    Ta, Tb = A["Ta"], A["Tb"]
    res = {"a": a.a, "b": a.b, "n_cells": int(len(cells)), "parts": {}}
    for p in a.parts:
        ka = "part__" + p
        if ka not in A.files or ka not in B.files:
            continue
        Da, Db = A[ka], B[ka]
        ca = np.array([corr(Da, Ta, Tb, w) for w in W])
        cb = np.array([corr(Db, Ta, Tb, w) for w in W])
        la = np.array([align(-Da, A["Dir"], w) for w in W])
        lb = np.array([align(-Db, B["Dir"], w) for w in W])
        q = lambda v: [float(np.nanpercentile(v[1:], 2.5)), float(np.nanpercentile(v[1:], 97.5))]
        res["parts"][p] = {"corr_a": float(ca[0]), "corr_b": float(cb[0]), "corr_diff": float(ca[0] - cb[0]),
                           "corr_diff_ci": q(ca - cb), "lean_a": float(la[0]), "lean_b": float(lb[0]),
                           "lean_diff": float(la[0] - lb[0]), "lean_diff_ci": q(la - lb)}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    summarize([a.out])


def summarize(paths):
    for f in paths:
        r = json.load(open(f, encoding="utf-8"))
        print(f"{os.path.basename(f)}: A = {os.path.basename(os.path.normpath(r['a']))}, "
              f"B = {os.path.basename(os.path.normpath(r['b']))}, {r['n_cells']} cells (paired block bootstrap)")
        for p, v in r["parts"].items():
            print(f"  {p:14s} corr {v['corr_a']:+.3f} vs {v['corr_b']:+.3f}: A-B {v['corr_diff']:+.3f} "
                  f"[{v['corr_diff_ci'][0]:+.3f},{v['corr_diff_ci'][1]:+.3f}] | lean {v['lean_a']:+.3f} vs "
                  f"{v['lean_b']:+.3f}: A-B {v['lean_diff']:+.3f} [{v['lean_diff_ci'][0]:+.3f},{v['lean_diff_ci'][1]:+.3f}]")


if __name__ == "__main__":
    main()
