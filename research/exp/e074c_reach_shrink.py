"""E074c (peer suggestion after E074): does any of the place-dependent readout survive if its interaction block is
shrunk toward zero with REACH (years before the trained span)?

    python research/exp/e074c_reach_shrink.py --cache <research cache> --cbc cbc_circles_comm.npz --out DIR

The E074 interaction readout (context = the cell-mean z_ema's first 4 dims; refitted here identically), with the
interaction features of each prediction scaled by exp(-reach / L), reach = max(0, 1996 - year); L in {0 (interaction
off: the readout's own between + within part), 5, 10, 20, inf (= E074)}. Species offsets refitted per L on 2005-2024
lists (reach 0, so every L shares them up to the fit). Per era and L: backcast - modern [95% bootstrap over circles],
log-likelihood with no shift. Reference: the split readout from E069a2/E074 (same lists).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e069_presence_score import fit_delta, interact_ridge_cv, list_ll  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", required=True); ap.add_argument("--cbc", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    C = a.cache
    meta = json.load(open(os.path.join(C, "meta.json")))
    keys = np.load(os.path.join(C, "keys.npy")); Xc = np.load(os.path.join(C, "X_comm.npy"))
    zk = np.load(os.path.join(C, "z_ema.npy")); Zall = np.load(os.path.join(C, "z_ema_all.npy"), mmap_mode="r")
    cells = np.load(os.path.join(C, "cells.npy")); years = np.load(os.path.join(C, "years.npy"))
    spl = np.load(os.path.join(C, "split.npz")); kci = np.load(os.path.join(C, "key_cell_index.npy"))
    D = np.load(a.cbc, allow_pickle=False)
    comm, cs = list(meta["community"]), list(D["species"])
    sp = [s for s in comm if s in cs]
    assert "houfin" not in sp
    tr = spl["is_train"] & (keys[:, 2] >= 1996)
    T = np.log1p(Xc[:, [comm.index(s) for s in sp]])
    trained = (years >= 1996) & (years <= 2025)
    zbar = np.asarray(Zall[:, trained].mean(1), float)
    ro, lam = interact_ridge_cv(zbar[kci[tr]], zk[tr].astype(float), T[tr], spl["block_id"][tr], [1e-3, 1e-2, 1e-1, 1, 10])
    cix = {(int(r), int(c)): i for i, (r, c) in enumerate(cells)}
    ci = np.array([cix.get((int(r), int(c)), -1) for r, c in zip(D["row"], D["col"])])
    yr = D["year_end"]
    P = D["Y"][:, [cs.index(s) for s in sp]] > 0
    ok = (ci >= 0) & (yr >= int(years[0])) & (yr <= 2024) & (P.sum(1) >= 2)
    zt = lambda r: np.asarray(Zall[ci[r], yr[r] - int(years[0])], float)
    zmod = np.asarray(Zall[:, (years >= 2005) & (years <= 2024)].mean(1), float)
    rm = np.where(ok & (yr >= 2005))[0]
    res = {"ridge_lambda": lam, "species": len(sp)}
    for L in (0.0, 5.0, 10.0, 20.0, float("inf")):
        sc = lambda y: np.ones(len(y)) if L == float("inf") else (np.zeros(len(y)) if L == 0 else
                                                                   np.exp(-np.maximum(0, 1996 - y) / L))
        delta = fit_delta(ro(zbar[ci[rm]], zt(rm), None, sc(yr[rm])), P[rm])
        out = {}
        for era, e0, e1 in (("1953_1965", 1953, 1965), ("1966_1995", 1966, 1995)):
            r = np.where(ok & (yr >= e0) & (yr <= e1))[0]
            eb = ro(zbar[ci[r]], zt(r), None, sc(yr[r]))
            em = ro(zbar[ci[r]], zmod[ci[r]], None, sc(yr[r]))
            llb, llm = list_ll(eb, delta, P[r]), list_ll(em, delta, P[r])
            d = llb - llm
            circ = D["circle_idx"][r]; uc = np.unique(circ); idx = {c: np.where(circ == c)[0] for c in uc}
            rng = np.random.default_rng(69)
            bs = [d[np.concatenate([idx[c] for c in rng.choice(uc, len(uc))])].mean() for _ in range(a.boot)]
            out[era] = {"backcast_minus_modern": float(d.mean()), "ci95": [float(x) for x in np.percentile(bs, [2.5, 97.5])],
                        "ll_modern": float(llm.mean())}
        res[f"L={L}"] = out
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "e074c.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
