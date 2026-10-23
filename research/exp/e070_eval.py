"""E070 evaluation: which early-CBC route recovers the known 1966-85 -> 2005-24 change under realistic early damage?

    python research/exp/e070_eval.py --similarity species_similarity.csv
        --truth VARIANT=PATH ... --fit VARIANT:HANDLING:COPY=PATH ... --out DIR

Truth for a variant = its ridge fit on the REAL data (all circles). Degraded fits: the same variant on a pseudo-early2
copy (rich or random place selection; cbc_pseudo_early2.py), with a circle HANDLING: ridge (as E065), neighbour (circle
effects shrunk toward nearby circles), persistent (only circles with >= 5 counts in both windows).
Per (variant, handling, copy) and species stratum (SEDENTARY primary, AVONET 1; RESIDENTS, 1-2):
  SPECIES ROUTE  change per stratum x species, log(mean index 2005-24) - log(mean index 1966-85), strata with >= 3 fitted
                 circle-years in both windows in both fits; species level = mean over shared strata (>= 3).
                 rel_bias = median |species bias| / sd(true species change); species corr; stratum corr (species-centred).
  RUZICKA ROUTE  per BBS stratum, the community similarity between the two windows' mean indices (Ruzicka on log1p of
                 the index, over the stratum's species with support): degraded vs truth across strata -- Spearman, and
                 rel_bias = median |diff| / sd(true similarity).
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e065_eval import aligned, change  # noqa: E402

W = (1966, 1985, 2005, 2024)


def window_means(f):
    I, years, sup = f["index"].astype(float), f["years"], f["support"]
    m0, m1 = (years >= W[0]) & (years <= W[1]), (years >= W[2]) & (years <= W[3])
    ok = (sup[:, m0].sum(1) >= 3) & (sup[:, m1].sum(1) >= 3)
    return I[:, m0].mean(1), I[:, m1].mean(1), ok


def ruz(a, b):
    a, b = np.log1p(a), np.log1p(b)
    den = np.maximum(a, b).sum(-1)
    return np.where(den > 0, np.minimum(a, b).sum(-1) / np.maximum(den, 1e-12), np.nan)


def species_route(fd, ft, cols):
    dt, okt = change(ft, *W)
    dd, okd = change(fd, *W)
    dd = aligned(ft, fd, dd); okd = aligned(ft, fd, okd.astype(float)) == 1
    ok = okt & okd
    ok[:, ~cols] = False
    x, y = np.where(ok, dd, np.nan), np.where(ok, dt, np.nan)
    sx, sy = np.nanmean(x, 0), np.nanmean(y, 0)
    k = np.isfinite(sx) & np.isfinite(sy) & (ok.sum(0) >= 3)
    if k.sum() < 5:
        return {"species": int(k.sum())}
    b = sx[k] - sy[k]
    xc, yc = x - sx, y - sy
    m = ok & np.isfinite(xc) & np.isfinite(yc) & k[None, :]
    return {"species": int(k.sum()), "pairs": int(m.sum()), "median_bias": float(np.median(b)),
            "rel_bias": float(np.median(np.abs(b)) / np.std(sy[k])), "species_corr": float(np.corrcoef(sx[k], sy[k])[0, 1]),
            "stratum_corr_centred": float(np.corrcoef(xc[m], yc[m])[0, 1]) if m.sum() > 20 else None}


def ruzicka_route(fd, ft, cols):
    a0, a1, okt = window_means(ft)
    b0, b1, okd = window_means(fd)
    st, sd_ = list(ft["strata"]), list(fd["strata"])
    sp_t, sp_d = list(ft["species"]), list(fd["species"])
    ci = [j for j, s in enumerate(sp_t) if cols[j] and s in sp_d]
    cd = [sp_d.index(sp_t[j]) for j in ci]
    tv, dv = [], []
    for r, s in enumerate(st):
        if s in sd_ and okt[r] and okd[sd_.index(s)]:
            rd = sd_.index(s)
            tv.append(float(ruz(a0[r, ci], a1[r, ci]))); dv.append(float(ruz(b0[rd, cd], b1[rd, cd])))
    tv, dv = np.array(tv), np.array(dv)
    k = np.isfinite(tv) & np.isfinite(dv)
    if k.sum() < 10:
        return {"strata": int(k.sum())}
    return {"strata": int(k.sum()), "spearman": float(spearmanr(dv[k], tv[k]).correlation),
            "median_shift": float(np.median(dv[k] - tv[k])),
            "rel_bias": float(np.median(np.abs(dv[k] - tv[k])) / np.std(tv[k]))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--similarity", required=True)
    ap.add_argument("--truth", action="append", required=True)
    ap.add_argument("--fit", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import pandas as pd
    sim = pd.read_csv(a.similarity); mig = dict(zip(sim.species_code, sim.migration))
    truth = {k: np.load(p, allow_pickle=False) for k, p in (s.split("=", 1) for s in a.truth)}
    res = {}
    for s in a.fit:
        key, p = s.split("=", 1)
        v, h, cp = key.split(":")
        fd, ft = np.load(p, allow_pickle=False), truth[v]
        spc = list(ft["species"])
        strata = {"sedentary": np.array([mig.get(x) == 1.0 for x in spc]),
                  "residents": np.array([mig.get(x) in (1.0, 2.0) for x in spc])}
        res[key] = {st: {"species_route": species_route(fd, ft, cols), "ruzicka_route": ruzicka_route(fd, ft, cols)}
                    for st, cols in strata.items()}
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "e070.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
