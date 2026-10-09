"""E020: how much real ESK content does DESK carry in each component -- spatially and temporally, trained vs backcast?

    python research/exp/rank_fidelity.py --cache <dir> --out <dir> [--population trained|withheld]
    python research/exp/rank_fidelity.py --summarize <out>

The readout uses the first 24 of DESK's 64 dims because DESK seemed to carry nothing meaningful beyond (maybe
beyond ~12). Per ESK component k, on HELD-OUT cells (DESK never saw them):
    spatial   modern-epoch levels across cells          R2_k = cov(D, T)^2 / (var(D) cov(T_A, T_B))
    temporal  early -> modern change across cells       same, on the change
i.e. the noise-free share of the component's TRUE variance that DESK's own component k explains linearly (the
halves' covariance is the true variance; DESK carries no survey noise). Plus each component's share of the true
spatial / temporal variance, and the cumulative variance DESK captures up to rank r (sum of R2_k x true var_k)
against what is available. A component is "meaningful" where R2_k is clearly above 0 and its variance share is
not negligible. Centered across cells, so the continental mean (shift) does not count.
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

RANKS = (6, 12, 24, 32, 48, 64)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache")
    ap.add_argument("--out")
    ap.add_argument("--population", default="trained", choices=("trained", "withheld"))
    ap.add_argument("--summarize", default=None)
    a = ap.parse_args()
    if a.summarize:
        return summarize(a.summarize)
    from change_oracle import group_rows
    os.makedirs(a.out, exist_ok=True)
    c = lambda f: os.path.join(a.cache, f)
    keys = np.load(c("keys.npy"))
    split = np.load(c("split.npz"))
    E, full, _, _ = group_rows(a.cache, keys, [int(y) for y in split["withheld"]])
    want_wh = a.population == "withheld"
    idx = {}
    for j, (cell, ep, wh) in enumerate(zip(map(tuple, E["cells"]), E["epoch"], E["withheld"])):
        if ep == 0 and bool(wh) == want_wh:
            idx.setdefault(cell, [None, None])[0] = j
        elif ep == 1 and not wh:
            idx.setdefault(cell, [None, None])[1] = j
    pairs = np.array([(e, m) for e, m in idx.values() if e is not None and m is not None])
    cells = E["cells"][pairs[:, 0]]
    ev = np.where(split["holdout"][cells[:, 0], cells[:, 1]])[0]
    gm = lambda A: np.stack([np.asarray(A[np.asarray(g)], "float64").mean(0) for g in full])
    res = {"cache": a.cache, "population": a.population, "n_cells": int(len(ev)), "features": {}}
    T = {h: E[f"z_{h}"].astype("float64") for h in ("full", "a", "b")}
    for feat in ("raw", "ema"):
        D = gm(np.load(c(f"z_{feat}.npy"), mmap_mode="r"))
        out = {}
        for kind in ("spatial", "temporal"):
            if kind == "spatial":
                d = D[pairs[ev, 1]]
                t = {h: T[h][pairs[ev, 1]] for h in T}
            else:
                d = D[pairs[ev, 1]] - D[pairs[ev, 0]]
                t = {h: T[h][pairs[ev, 1]] - T[h][pairs[ev, 0]] for h in T}
            cen = lambda v: v - v.mean(0)
            dc, tf, ta, tb = cen(d), cen(t["full"]), cen(t["a"]), cen(t["b"])
            cov = (dc * tf).mean(0)
            vd = (dc * dc).mean(0)
            vt = (ta * tb).mean(0)                          # true variance per component (noise-free)
            with np.errstate(invalid="ignore", divide="ignore"):
                r2 = np.where((vd > 0) & (vt > 0), cov ** 2 / (vd * vt), np.nan)
                calib = np.where(vd > 0, cov / vd, np.nan)      # truth on DESK, per component
            share = np.clip(vt, 0, None) / np.clip(vt, 0, None).sum()
            cum = {}
            for r in RANKS:
                avail = float(np.clip(vt[:r], 0, None).sum())
                capt = float(np.nansum(np.clip(r2[:r], 0, 1) * np.clip(vt[:r], 0, None)))
                cum[str(r)] = {"true_var_share_in_first_r": float(share[:r].sum()),
                               "desk_captured_share_of_first_r": capt / avail if avail > 0 else None,
                               "desk_captured_share_of_all64": capt / float(np.clip(vt, 0, None).sum())}
            out[kind] = {"r2": r2.tolist(), "calibration": calib.tolist(), "true_var_share": share.tolist(),
                         "cumulative": cum}
        res["features"][feat] = out
    with open(os.path.join(a.out, "rank_fidelity.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    summarize(a.out)


def summarize(out):
    r = json.load(open(os.path.join(out, "rank_fidelity.json"), encoding="utf-8"))
    print(f"rank fidelity [{r['population']}] {r['n_cells']} held-out cells: per-component R2 of DESK vs the true "
          f"(noise-free) ESK component, and cumulative captured variance")
    for feat, o in r["features"].items():
        for kind in ("spatial", "temporal"):
            k = o[kind]
            r2 = np.array(k["r2"], float)
            bands = [(0, 6), (6, 12), (12, 24), (24, 32), (32, 48), (48, 64)]
            med = "  ".join(f"{lo + 1}-{hi}: {np.nanmedian(r2[lo:hi]):.2f}" for lo, hi in bands)
            sh = np.array(k["true_var_share"])
            shs = "  ".join(f"{sh[lo:hi].sum():.2f}" for lo, hi in bands)
            print(f"  {feat:3s} {kind:8s} median R2 by band  {med}")
            print(f"  {'':3s} {'':8s} true variance share   {shs}")
            print(f"  {'':3s} {'':8s} cumulative: " + "  ".join(
                f"r{rk}: {v['desk_captured_share_of_all64']:.3f}" for rk, v in k["cumulative"].items())
                  + "   (share of all 64 dims' true variance that DESK captures using the first r)")


if __name__ == "__main__":
    main()
