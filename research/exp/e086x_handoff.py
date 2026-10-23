"""E086x: does handing supervision from CBC-only (withheld years) to BBS + CBC (trained years) leave an ARTIFACT in z?

    python research/exp/e086x_handoff.py --caches name=<cache> ... --similarity species_similarity.csv --out <dir>

In the tempho design the withheld span (1966-95 at T0 1996) is supervised only by CBC in arms that include withheld
winters (ruz, ruzt), and by nothing in ruzx / controls; from T0 on, by BBS (+ CBC). A target-switch artifact would be
(a) a STEP in z at the boundary and (b) a common SHIFT in predicted change -- invisible to the correlation yardsticks
used so far (fair_rivals pools noise-free corr, centred per species).
  step   per year t: common shift |mean over cells of z(t+1) - z(t)| and mean per-cell |z(t+1) - z(t)|, raw z and z_ema,
         all cells of the cache; reported at the boundary (T0-1 -> T0) as a ratio to the median over the other years.
  bias   per dev species, the readouts of e086t (level BLR on cell means, within BLR on deviations; r24; training cells
         x trained years) applied to held-out cells' z change (withheld early epoch -> modern): mean predicted change
         over held-out cells minus mean observed change (all BBS years), and the slope of the species' mean predicted
         on mean observed change; sedentary common / sedentary / migrants.
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from lib import blr  # noqa: E402


def one(cache, strata_fn, k):
    j = lambda f: os.path.join(cache, f)
    meta = json.load(open(j("meta.json"))); dev = list(meta["dev_species"])
    keys = np.load(j("keys.npy")); r, c, y = keys[:, 0], keys[:, 1], keys[:, 2]
    sp = np.load(j("split.npz")); ho, bf = sp["holdout"].astype(bool), sp["buffer"].astype(bool)
    wh = np.array(sorted(int(v) for v in sp["withheld"])); t0 = int(wh[-1]) + 1
    years = np.load(j("years.npy"))
    out = {"t0": t0, "step": {}}
    for nm in ("z_raw_all", "z_ema_all"):
        Zc = np.load(j(nm + ".npy"), mmap_mode="r")                     # (cells, years, L)
        dz = np.diff(np.asarray(Zc[:, :, :k], np.float64), axis=1)      # (cells, years-1, k)
        ok = np.isfinite(dz).all((0, 2))
        common = np.linalg.norm(np.nanmean(dz, 0), axis=1); percell = np.nanmean(np.linalg.norm(dz, axis=2), 0)
        tb = int(np.where(years == t0)[0][0]) - 1                        # transition (t0-1 -> t0)
        oth = np.array([i for i in range(len(common)) if ok[i] and abs(i - tb) > 2 and years[i] >= 1966])
        out["step"][nm] = {"common_at_boundary": float(common[tb]), "common_median_other": float(np.median(common[oth])),
                           "common_ratio": float(common[tb] / np.median(common[oth])),
                           "percell_ratio": float(percell[tb] / np.median(percell[oth])),
                           "common_series": {int(years[i + 1]): float(common[i]) for i in range(len(common)) if years[i] >= 1960}}
    Xd = np.asarray(np.load(j("X_dev.npy"), mmap_mode="r")); Ld = np.log1p(Xd.astype(np.float64)); S = Ld.shape[1]
    Z = np.load(j("z_ema.npy")).astype(np.float64)[:, :k]
    is_tr = ~ho[r, c] & ~bf[r, c] & ~np.isin(y, wh)
    strata = strata_fn(dev, (Xd[is_tr] > 0).mean(0))
    cid = r.astype(np.int64) * 1000 + c
    rows = np.where(is_tr)[0]; u, inv = np.unique(cid[rows], return_inverse=True); n = np.bincount(inv)
    mean = lambda V: (lambda s: (np.add.at(s, inv, V[rows]), s / n[:, None])[1])(np.zeros((len(u), V.shape[1])))
    Zm, Ym = mean(Z), mean(Ld)
    g2 = n[inv] >= 2
    Bl = np.asarray(blr.fit(Zm, Ym, [(0, k)])["coef"])
    Bw = np.asarray(blr.fit((Z[rows] - Zm[inv])[g2], (Ld[rows] - Ym[inv])[g2], [(0, k)])["coef"])
    E = dict(np.load(j("esk_epochs.npz"))); ptr, flat = E["rows_ptr"], E["rows_flat"]
    idx = {}
    for i, (cell, ep, w) in enumerate(zip(map(tuple, E["cells"]), E["epoch"], E["withheld"])):
        if ep == 0 and bool(w):
            idx.setdefault(cell, [None, None])[0] = i
        elif ep == 1 and not w:
            idx.setdefault(cell, [None, None])[1] = i
    pairs = np.array([(e, m) for e, m in idx.values() if e is not None and m is not None])
    pc = E["cells"][pairs[:, 0]]; hp = pairs[ho[pc[:, 0], pc[:, 1]]]
    hmean = lambda V, i: V[flat[ptr[i]:ptr[i + 1]]].mean(0)
    dZ = np.stack([hmean(Z, m) - hmean(Z, e) for e, m in hp]); dY = np.stack([hmean(Ld, m) - hmean(Ld, e) for e, m in hp])
    out["bias"] = {}
    for ro, B in (("level", Bl), ("within", Bw)):
        mp, mo = (dZ @ B.T).mean(0), dY.mean(0)
        out["bias"][ro] = {st: {"mean_pred_minus_obs": float((mp - mo)[m].mean()), "mean_abs_err": float(np.abs(mp - mo)[m].mean()),
                                "mean_pred": float(mp[m].mean()), "mean_obs": float(mo[m].mean()),
                                "slope_pred_on_obs": float(np.polyfit(mo[m], mp[m], 1)[0]) if m.sum() > 2 else None}
                           for st, m in strata.items()}
    out["n_heldout_cells"] = int(len(hp))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--caches", nargs="+", required=True); ap.add_argument("--similarity", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--rank", type=int, default=24)
    a = ap.parse_args()
    import pandas as pd
    sim = pd.read_csv(a.similarity); mig = dict(zip(sim.species_code, sim.migration))
    def strata_fn(dev, prev):
        assert "houfin" not in dev
        sed = np.array([mig.get(s) == 1.0 for s in dev])
        return {"sed_common": sed & (prev >= 0.08), "sedentary": sed, "migrants": ~sed}
    res = {}
    os.makedirs(a.out, exist_ok=True)
    for spec in a.caches:
        name, cache = spec.split("=", 1)
        res[name] = one(cache, strata_fn, a.rank)
        s, b = res[name]["step"], res[name]["bias"]
        print(f"[e086x] {name:10s} step at {res[name]['t0']}: common raw x{s['z_raw_all']['common_ratio']:.2f} ema "
              f"x{s['z_ema_all']['common_ratio']:.2f} | per-cell raw x{s['z_raw_all']['percell_ratio']:.2f} | bias sed_common "
              f"level {b['level']['sed_common']['mean_pred_minus_obs']:+.4f} within {b['within']['sed_common']['mean_pred_minus_obs']:+.4f} "
              f"(obs {b['level']['sed_common']['mean_obs']:+.4f})", flush=True)
        json.dump(res, open(os.path.join(a.out, "e086x.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
