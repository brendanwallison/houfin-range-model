"""E069b: does a DESK trained with the CBC presence term predict HELD-OUT CBC lists better than its control?

    python research/exp/e069b_heldout_lists.py --arm <cache> --control <cache> --cbc cbc_circles_comm.npz
        --similarity species_similarity.csv --out DIR [--label NAME]

Lists in DESK's HELD-OUT cells only (split.npz holdout grid: the presence term never saw them), resident reference
species (the term's species; all 83 also reported). For each cache separately, as E069a: a level ridge readout of its
own z_ema (training cell-years, trained years >= 1996), species offsets from its own 2005-2024 lists (training cells),
per-list log P(species | record) for the BACKCAST (z at the list's year) and the MODERN placebo (the cell's mean z
2005-2024). Paired over the same lists: (arm backcast - control backcast), and (arm [backcast - modern] - control
[backcast - modern]), 95% bootstrap over circles; eras 1953-1965 and 1966-1995 (1940-1952 reported).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e069_presence_score import fit_delta, list_ll, ridge_cv  # noqa: E402

ERAS = (("1940_1952", 1940, 1952), ("1953_1965", 1953, 1965), ("1966_1995", 1966, 1995))


def per_list(cache, D, cols_sp, heldout_rows_fn):
    meta = json.load(open(os.path.join(cache, "meta.json")))
    keys = np.load(os.path.join(cache, "keys.npy")); Xc = np.load(os.path.join(cache, "X_comm.npy"))
    zk = np.load(os.path.join(cache, "z_ema.npy")); Zall = np.load(os.path.join(cache, "z_ema_all.npy"), mmap_mode="r")
    cells = np.load(os.path.join(cache, "cells.npy")); years = np.load(os.path.join(cache, "years.npy"))
    spl = np.load(os.path.join(cache, "split.npz"))
    comm, cs = list(meta["community"]), list(D["species"])
    tr = spl["is_train"] & (keys[:, 2] >= 1996)
    T = np.log1p(Xc[:, [comm.index(s) for s in cols_sp]])
    ro, _ = ridge_cv(zk[tr].astype(float), T[tr], spl["block_id"][tr], [1e-4, 1e-3, 1e-2, 1e-1, 1, 10])
    cix = {(int(r), int(c)): i for i, (r, c) in enumerate(cells)}
    ci = np.array([cix.get((int(r), int(c)), -1) for r, c in zip(D["row"], D["col"])])
    yr = D["year_end"]
    P = D["Y"][:, [cs.index(s) for s in cols_sp]] > 0
    ho = spl["holdout"]
    inho = np.array([bool(ho[r, c]) if r >= 0 else False for r, c in zip(D["row"], D["col"])])
    ok = (ci >= 0) & (yr >= int(years[0])) & (yr <= 2024) & (P.sum(1) >= 2)
    zt = lambda r: np.asarray(Zall[ci[r], yr[r] - int(years[0])], float)
    zmod = np.asarray(Zall[:, (years >= 2005) & (years <= 2024)].mean(1), float)
    rm = np.where(ok & ~inho & (yr >= 2005))[0]
    delta = fit_delta(ro(zt(rm)), P[rm])
    out = {}
    for era, e0, e1 in ERAS:
        r = np.where(ok & inho & (yr >= e0) & (yr <= e1))[0]
        out[era] = (r, list_ll(ro(zt(r)), delta, P[r]), list_ll(ro(zmod[ci[r]]), delta, P[r]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", required=True); ap.add_argument("--control", required=True)
    ap.add_argument("--cbc", required=True); ap.add_argument("--similarity", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--label", default="arm")
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    import pandas as pd
    D = np.load(a.cbc, allow_pickle=False)
    meta = json.load(open(os.path.join(a.arm, "meta.json")))
    sim = pd.read_csv(a.similarity); mig = dict(zip(sim.species_code, sim.migration))
    cs = list(D["species"])
    allsp = [s for s in meta["community"] if s in cs]
    res_sp = [s for s in allsp if mig.get(s) in (1.0, 2.0)]
    assert "houfin" not in allsp
    res = {"label": a.label, "arm": a.arm, "control": a.control}
    for name, sp in (("residents", res_sp), ("all", allsp)):
        A, C = per_list(a.arm, D, sp, None), per_list(a.control, D, sp, None)
        out = {"species": len(sp)}
        for era, _, _ in ERAS:
            ra, lba, lma = A[era]; rc, lbc, lmc = C[era]
            assert (ra == rc).all()
            d_back = lba - lbc
            d_gain = (lba - lma) - (lbc - lmc)
            circ = D["circle_idx"][ra]; uc = np.unique(circ)
            if len(uc) < 10:
                out[era] = {"lists": int(len(ra)), "circles": int(len(uc))}
                continue
            idx = {c: np.where(circ == c)[0] for c in uc}
            rng = np.random.default_rng(69)
            bb, bg = [], []
            for _ in range(a.boot):
                pk = np.concatenate([idx[c] for c in rng.choice(uc, len(uc))])
                bb.append(d_back[pk].mean()); bg.append(d_gain[pk].mean())
            out[era] = {"lists": int(len(ra)), "circles": int(len(uc)),
                        "arm_backcast_minus_modern": float((lba - lma).mean()),
                        "control_backcast_minus_modern": float((lbc - lmc).mean()),
                        "arm_minus_control_backcast": float(d_back.mean()),
                        "arm_minus_control_backcast_ci95": [float(x) for x in np.percentile(bb, [2.5, 97.5])],
                        "arm_minus_control_gain": float(d_gain.mean()),
                        "arm_minus_control_gain_ci95": [float(x) for x in np.percentile(bg, [2.5, 97.5])]}
        res[name] = out
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, f"e069b_lists_{a.label}.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
