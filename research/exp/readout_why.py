"""E030: WHY the community -> species readout of change does no better than no change -- for the TRUE community (the
oracle) and for DESK. A noise-free decomposition on the same cells, species and betas as E013 (readout_corr.py).

    ESK_DESK_CONFIG=<overlay> python research/exp/readout_why.py --cache <dir> --out <dir> [--population withheld]
    python research/exp/readout_why.py --summarize <out>

Squared-error skill against "no change" for a prediction p of change t (per species, centred over held-out cells):
    skill = 1 - E(t - p)^2 / E t^2 = (2 cov(p, t) - var(p)) / var(t) = corr^2 * (2 - 1/k) / k ... written below as
    skill = (var(p) / var(t)) * (2k - 1),   k = cov(p, t) / var(p)   (truth-on-prediction slope, "calibration")
so a prediction beats no change only if k > 1/2, whatever its correlation, and the best any rescaling of p can do is
corr^2 (at k = 1). Three factors therefore decide the score:
    information  corr_true(p, t)^2      how much of the species' change goes with the predictor at all
    scaling      k                      how big the predicted changes are for their accuracy
    noise        var(p_A) / var_true(p) survey noise in the ORACLE's own features, amplified by the betas
All covariances are noise-free cross-half estimates (halves A / B of every cell-epoch, as E013); pooled over resolvable
species by summing numerators and denominators. Also: the share of species change a smooth spatial field of the change
itself explains (position RFF fitted on training cells' change -- a ceiling for "regional", not usable for 1900-1940).
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
    """Per-species covariance over cells (centred: place-specific change)."""
    return ((x - x.mean(0)) * (y - y.mean(0))).mean(0)


def cmom(x, y):
    """Per-species second moment over cells (uncentred: includes the species' mean change, as the suite scores it)."""
    return (x * y).mean(0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache")
    ap.add_argument("--out")
    ap.add_argument("--population", default="withheld", choices=("trained", "withheld"))
    ap.add_argument("--rank", type=int, default=24)
    ap.add_argument("--summarize", default=None)
    a = ap.parse_args()
    if a.summarize:
        return summarize(a.summarize)
    from change_oracle import desk_epoch_z, group_rows
    from src.community_encoder.train_DESK.validate_bbs_routes import epoch_mean_observed
    from src.community_encoder.train_DESK.validation_core import change_noise
    os.makedirs(a.out, exist_ok=True)
    c = lambda f: os.path.join(a.cache, f)
    keys = np.load(c("keys.npy"))
    split = np.load(c("split.npz"))
    E, full, half_a, half_b = group_rows(a.cache, keys, [int(y) for y in split["withheld"]])
    Xd = np.load(c("X_dev.npy")).astype("float64")
    want_wh = a.population == "withheld"
    idx = {}
    for j, (cell, ep, wh) in enumerate(zip(map(tuple, E["cells"]), E["epoch"], E["withheld"])):
        if ep == 0 and bool(wh) == want_wh:
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
    d = {h: (v[pairs[:, 1]] - v[pairs[:, 0]]) for h, v in Y.items()}
    nz = change_noise(d["full"][ev], d["a"][ev], d["b"][ev])
    res_mask = nz["resolvable"]
    prev = (Xd[:, keep] > 0).mean(0)
    r = a.rank
    lvl_groups = np.unique(pairs[tr, 1]) if want_wh else np.unique(np.concatenate([pairs[tr, 1], pairs[tr, 0]]))
    da, db, dfull = d["a"][ev], d["b"][ev], d["full"][ev]

    # the ORACLE: true community change through level betas (as E013)
    Za, Zb = E["z_a"][:, :r].astype("float64"), E["z_b"][:, :r].astype("float64")
    beta = blr.fit(Za[lvl_groups], Y["b"][lvl_groups], [(0, r)])["coef"].T
    pa = (Za[pairs[:, 1]] - Za[pairs[:, 0]])[ev] @ beta
    pb = (Zb[pairs[:, 1]] - Zb[pairs[:, 0]])[ev] @ beta
    # DESK through its own level betas (no survey noise in the prediction)
    zr = np.load(c("z_raw_all.npy"), mmap_mode="r")
    ci, y0 = np.load(c("key_cell_index.npy")), int(np.load(c("years.npy"))[0])
    Fd = desk_epoch_z(zr, ci, y0, keys, full, r)
    beta_d = blr.fit(Fd[lvl_groups], Y["b"][lvl_groups], [(0, r)])["coef"].T
    pdk = (Fd[pairs[:, 1]] - Fd[pairs[:, 0]])[ev] @ beta_d
    # a smooth spatial field of the CHANGE itself (training cells' change -> held-out cells): how regional it is
    rng = np.random.default_rng(0)
    xy = cells.astype("float64") * 27.0
    R = np.sqrt(2.0 / 48) * np.cos(xy @ (rng.normal(size=(2, 48)) / 300.0) + rng.uniform(0, 2 * np.pi, 48))
    beta_xy = blr.fit(R[tr], d["b"][tr], [(0, 48)])["coef"].T
    px = R[ev] @ beta_xy

    def parts(f):
        """(cov, var_true(pred), var_as_emitted(pred)) per species for each arm, and var_true(change)."""
        return {"vt": f(da, db),
                "oracle": (0.5 * (f(pa, db) + f(pb, da)), f(pa, pb), 0.5 * (f(pa, pa) + f(pb, pb))),
                "desk": (f(pdk, dfull), f(pdk, pdk), f(pdk, pdk)),
                "regional_field": (f(px, dfull), f(px, px), f(px, px))}
    P = {"centred": parts(ccov), "uncentred": parts(cmom)}

    def pooled(m):
        out = {"n_species": int(m.sum())}
        for form, q in P.items():
            V = float(q["vt"][m].sum())
            o = {"var_change": V}
            for name in ("oracle", "desk", "regional_field"):
                cov, vpt, vpu = (float(x[m].sum()) for x in q[name])
                if vpt <= 0 or V <= 0:
                    continue
                corr = cov / np.sqrt(vpt * V)
                o[name] = {"corr": corr, "info_r2": corr * abs(corr), "k_true": cov / vpt, "k_used": cov / vpu,
                           "noise_inflation": vpu / vpt, "skill_as_scored": (2 * cov - vpu) / V,
                           "skill_noise_free": (2 * cov - vpt) / V,
                           "skill_best_rescaled": corr ** 2 if corr > 0 else 0.0}
            out[form] = o
        return out
    res = {"cache": a.cache, "population": a.population, "rank": r, "n_eval_cells": int(len(ev)),
           "n_resolvable": int(res_mask.sum()), "all": pooled(res_mask), "tiers": {}}
    for lo, hi in zip(TIERS[:-1], TIERS[1:]):
        res["tiers"][f"{lo:g}-{hi:g}"] = pooled(res_mask & (prev >= lo) & (prev < hi))
    with open(os.path.join(a.out, "readout_why.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    summarize(a.out)


def summarize(out):
    r = json.load(open(os.path.join(out, "readout_why.json"), encoding="utf-8"))
    print(f"readout why [{r['population']}] r{r['rank']}: {r['n_eval_cells']} held-out cells, {r['n_resolvable']} "
          f"resolvable species; skill vs no change = (var p / var t)(2k - 1), so k must exceed 1/2")
    for name, s0 in [("all", r["all"])] + [(f"prev {k}", v) for k, v in r["tiers"].items()]:
      for form in ("centred", "uncentred"):
        s = s0.get(form, {})
        print(f"  {name:13s} n={s0['n_species']:3d} [{form}]")
        for arm in ("oracle", "desk", "regional_field"):
            if arm not in s:
                continue
            v = s[arm]
            print(f"     {arm:15s} corr {v['corr']:+.3f} (info {v['info_r2']:+.3f})  k {v['k_true']:.2f}"
                  f" (as emitted {v['k_used']:.2f}, noise x{v['noise_inflation']:.2f})  skill as scored "
                  f"{v['skill_as_scored']:+.3f}, noise-free {v['skill_noise_free']:+.3f}, best rescaled "
                  f"{v['skill_best_rescaled']:+.3f}")


if __name__ == "__main__":
    main()
