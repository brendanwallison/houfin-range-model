"""M15: DESK's raw backcast skill by Great Plains zone (1 west, 2 Great Plains barrier, 3 east).

    ESK_DESK_CONFIG=<overlay> python research/exp/zone_breakdown.py --out <dir> --t0 1996 --models desk_tempho_1995_s1 ...
    python research/exp/zone_breakdown.py --summarize <dir>

The plan requires every metric to be reported separately for the Great Plains zone, where DESK is weakest and the age
model's pseudo-zeros bite; nothing had been. Truth as span_reach / span_pooled (noise-free centred corr of held-out
cells' community change, window vs the first ten trained years); pooled over the first three pre-T0 windows; one
shared 6x6-block bootstrap; zones from processed/regions/great_plains_zones_27km.tif. Also the trend placebo (smooth
position x year fitted on training cells over the longest span) per zone, as the reference.
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

ROOT = os.path.expanduser("~/houfin/work/houfin/research_cache")
WINDOWS = [(1966, 1971), (1972, 1977), (1978, 1983), (1984, 1989), (1990, 1995)]
ZONES = {1: "west", 2: "great_plains", 3: "east"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out")
    ap.add_argument("--t0", type=int)
    ap.add_argument("--models", nargs="+", default=())
    ap.add_argument("--rank", type=int, default=24)
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--summarize", default=None)
    a = ap.parse_args()
    if a.summarize:
        return summarize(a.summarize)
    import rasterio
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
    train_cell = ~held & ~split["buffer"][keys[:, 0], keys[:, 1]]
    cid = keys[:, 0].astype(np.int64) * 100000 + keys[:, 1]
    cfg = load_config()
    zpath = os.path.expandvars(os.path.join(os.environ.get("HOUFIN_PROCESSED", ""), "regions", "great_plains_zones_27km.tif"))
    with rasterio.open(zpath) as src:
        zgrid = src.read(1)
    r, t0 = a.rank, a.t0
    ref = (t0, t0 + 9)
    wins = [w for w in WINDOWS if w[1] < t0][:3]
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
    proj = lambda v: np.asarray(project_points_to_z(np.asarray(v, "float32"), cfg["desk"]["z_dir"], 64), "float64")[:, :r]
    T = {"full": proj(epoch_mean_observed(Xc, groups)), "a": proj(epoch_mean_observed(Xc, A)),
         "b": proj(epoch_mean_observed(Xc, B))}
    gidx = {k: j for j, k in enumerate(gkey) if ok[j]}
    gmean = lambda M, g: np.asarray(M[g], "float64").mean(0)
    # trend placebo on training cells, longest span [t0, 2025]
    rng = np.random.default_rng(0)
    Wf, bf = rng.normal(size=(2, 24)) / 300.0, rng.uniform(0, 2 * np.pi, 24)
    pos = lambda rc: np.sqrt(2.0 / 24) * np.cos(rc.astype("float64") * 27.0 @ Wf + bf)
    Zobs = np.load(os.path.join(base, "esk_annual.npy"))[:, :r].astype("float64")
    rows = np.where(train_cell & (yr >= t0))[0]
    uc, inv = np.unique(cid[rows], return_inverse=True)
    cnt = np.bincount(inv).astype(float)

    def demean(M):
        s = np.zeros((len(uc), M.shape[1]))
        np.add.at(s, inv, M)
        return M - (s / cnt[:, None])[inv]
    tdev = demean(yr[rows, None].astype(float))[:, 0]
    btr = blr.fit(pos(keys[rows, :2]) * tdev[:, None], demean(Zobs[rows]), [(0, 24)])["coef"].T
    Z = {m: np.asarray(np.load(os.path.join(ROOT, m, "z_raw.npy"), mmap_mode="r")[:, :r], "float64") for m in a.models}
    allblk = np.unique((keys[held, 0] // 6) * 100000 + keys[held, 1] // 6)
    bpos = {b: i for i, b in enumerate(allblk)}
    Wb = np.random.default_rng(11).multinomial(len(allblk), np.full(len(allblk), 1.0 / len(allblk)),
                                               size=a.n_boot).astype(float)
    res = {"t0": t0, "models": list(a.models), "windows": [list(w) for w in wins], "zones": {}}
    for zc, zname in list(ZONES.items()) + [(0, "all")]:
        per = {m: [] for m in list(a.models) + ["trend_placebo"]}
        ncell = 0
        for w in wins:
            cl = [c_ for (c_, l_, h_) in gidx if (l_, h_) == w and (c_, *ref) in gidx
                  and (zc == 0 or zgrid[c_ // 100000, c_ % 100000] == zc)]
            if len(cl) < 10:
                continue
            ncell += len(cl)
            jw = np.array([gidx[(c_, *w)] for c_ in cl])
            jr = np.array([gidx[(c_, *ref)] for c_ in cl])
            gw, gr = [groups[j] for j in jw], [groups[j] for j in jr]
            dT = {h: T[h][jw] - T[h][jr] for h in T}
            bi = np.array([bpos[(c_ // 100000 // 6) * 100000 + (c_ % 100000) // 6] for c_ in cl])
            ws = np.vstack([np.ones(len(cl))[None], Wb[:, bi]])

            def corr(p, wt):
                cen = lambda v: v - (wt[:, None] * v).sum(0) / wt.sum()
                p_, f_, a_, b_ = cen(p), cen(dT["full"]), cen(dT["a"]), cen(dT["b"])
                den = (wt[:, None] * p_ * p_).sum() * (wt[:, None] * a_ * b_).sum()
                return (wt[:, None] * p_ * f_).sum() / np.sqrt(den) if den > 0 else np.nan
            for m in a.models:
                P = np.stack([gmean(Z[m], g1) - gmean(Z[m], g2) for g1, g2 in zip(gw, gr)])
                per[m].append(np.array([corr(P, wt) for wt in ws]))
            dyear = np.array([yr[g1].mean() - yr[g2].mean() for g1, g2 in zip(gw, gr)])
            rc = np.array([(c_ // 100000, c_ % 100000) for c_ in cl])
            Pt = (pos(rc) * dyear[:, None]) @ btr
            per["trend_placebo"].append(np.array([corr(Pt, wt) for wt in ws]))
        out = {"n_cell_windows": ncell}
        for m, lst in per.items():
            if not lst:
                continue
            C = np.nanmean(np.vstack(lst), axis=0)
            out[m] = {"corr": float(C[0]), "ci": [float(np.nanpercentile(C[1:], 2.5)), float(np.nanpercentile(C[1:], 97.5))]}
        res["zones"][zname] = out
    with open(os.path.join(a.out, f"zone_breakdown_{t0}.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    summarize(a.out)


def summarize(out):
    for f in sorted(os.listdir(out)):
        if not f.startswith("zone_breakdown_"):
            continue
        r = json.load(open(os.path.join(out, f), encoding="utf-8"))
        print(f"T0 {r['t0']}: raw backcast corr by Great Plains zone, pooled over {r['windows']}")
        for z, v in r["zones"].items():
            parts = [f"{m} {v[m]['corr']:+.3f} [{v[m]['ci'][0]:+.3f},{v[m]['ci'][1]:+.3f}]"
                     for m in list(r["models"]) + ["trend_placebo"] if m in v]
            print(f"  {z:13s} (cell-windows {v['n_cell_windows']:3d}) " + " | ".join(parts))


if __name__ == "__main__":
    main()
