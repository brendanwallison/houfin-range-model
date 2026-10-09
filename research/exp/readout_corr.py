"""E013: does the TRUE community change, read through LEVEL betas, track species change? Noise-free, by split halves.

    python research/exp/readout_corr.py --cache <dir> --out <dir> [--population trained|withheld] [--rank 24]
    python research/exp/readout_corr.py --summarize <out>

The level readout of community change fails for real dev species (E011: worse than no change into withheld
decades) but recovers 98% of the change of species planted as linear functions of z. Two explanations:
  (i)  DESK's change is too wrong -- the true community change, read through the same betas, would work;
  (ii) space-for-time fails in nature -- even the true community change, read through spatial betas, does not
       track how species change.
Observed communities are noisy, so their readout cannot be used directly. With two independent halves A, B of
every cell-epoch, the cross-half covariances are noise-free estimates of the true ones:
    pred_h = dz_h . beta (h = A, B; beta fitted on training cells' LEVELS, features half A, targets half B)
    cov_true(pred, d)  ~ [cov(pred_A, d_B) + cov(pred_B, d_A)] / 2
    var_true(pred)     ~ cov(pred_A, pred_B)          var_true(d) ~ cov(d_A, d_B)
    corr_true = cov_true(pred, d) / sqrt(var_true(pred) var_true(d))      (direction; scale-free, so the
                errors-in-variables attenuation of beta does not enter)
For DESK the prediction carries no survey noise: cov(pred, d_full) is already unbiased. Covariances are taken
over held-out cells per species (centred: place-specific), then pooled over resolvable species by summing
numerators and denominators. Also: corr between the true-community readout and DESK's readout in species space.
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
    """Per-column covariance over rows (centred)."""
    return ((x - x.mean(0)) * (y - y.mean(0))).mean(0)


def pooled(num, den_a, den_b, mask):
    n, da, db = num[mask].sum(), den_a[mask].sum(), den_b[mask].sum()
    return float(n / np.sqrt(da * db)) if da > 0 and db > 0 else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache")
    ap.add_argument("--out")
    ap.add_argument("--population", default="trained", choices=("trained", "withheld"))
    ap.add_argument("--rank", type=int, default=24)
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--species-regex", default=None, help="planted caches: grade one generator, e.g. '^z_'")
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
    Y = {h: epoch_mean_observed(Xd, g).astype("float64") for h, g in (("full", full), ("a", half_a),
                                                                       ("b", half_b))}
    keep = (Y["full"] > 0).any(0)
    if a.species_regex:
        import re
        meta = json.load(open(c("meta.json"), encoding="utf-8"))
        keep &= np.array([bool(re.search(a.species_regex, s)) for s in meta["dev_species"]])
    Y = {h: v[:, keep] for h, v in Y.items()}
    d = {h: (v[pairs[:, 1]] - v[pairs[:, 0]]) for h, v in Y.items()}
    nz = change_noise(d["full"][ev], d["a"][ev], d["b"][ev])
    res_mask = nz["resolvable"]
    prev = (Xd[:, keep] > 0).mean(0)
    r = a.rank
    lvl_groups = np.unique(np.concatenate([pairs[tr, 1]] + ([pairs[tr, 0]] if not want_wh else [])))

    # true-community readout (cross halves)
    Za, Zb = E["z_a"][:, :r].astype("float64"), E["z_b"][:, :r].astype("float64")
    beta_esk = blr.fit(Za[lvl_groups], Y["b"][lvl_groups], [(0, r)])["coef"].T          # (r, S)
    dza, dzb = (Za[pairs[:, 1]] - Za[pairs[:, 0]])[ev], (Zb[pairs[:, 1]] - Zb[pairs[:, 0]])[ev]
    pa, pb = dza @ beta_esk, dzb @ beta_esk
    da, db, dfull = d["a"][ev], d["b"][ev], d["full"][ev]
    esk_num = 0.5 * (ccov(pa, db) + ccov(pb, da))
    esk_vp = ccov(pa, pb)
    vd = ccov(da, db)
    # DESK readout: no survey noise in the prediction
    zr = np.load(c("z_raw_all.npy"), mmap_mode="r")
    ci, y0 = np.load(c("key_cell_index.npy")), int(np.load(c("years.npy"))[0])
    Fd = desk_epoch_z(zr, ci, y0, keys, full, r)
    beta_desk = blr.fit(Fd[lvl_groups], Y["b"][lvl_groups], [(0, r)])["coef"].T
    pd_ = (Fd[pairs[:, 1]] - Fd[pairs[:, 0]])[ev] @ beta_desk
    desk_num = ccov(pd_, dfull)
    desk_vp = ccov(pd_, pd_)
    # does DESK's readout follow the true-community readout (species space)?
    cross_num = 0.5 * (ccov(pd_, pa) + ccov(pd_, pb))

    blocks = (cells[ev, 0] // 6) * 100000 + cells[ev, 1] // 6
    ub, inv = np.unique(blocks, return_inverse=True)
    rng = np.random.default_rng(0)

    def stats(mask):
        out = {"n_species": int(mask.sum())}
        if mask.sum() < 3:
            return out
        out["corr_true_esk"] = pooled(esk_num, esk_vp, vd, mask)
        out["corr_true_desk"] = pooled(desk_num, desk_vp, vd, mask)
        out["corr_desk_vs_esk_readout"] = pooled(cross_num, desk_vp, esk_vp, mask)
        out["slope_true_esk"] = (float(esk_num[mask].sum() / vd[mask].sum())
                                 if vd[mask].sum() > 0 else None)
        out["slope_true_desk"] = (float(desk_num[mask].sum() / vd[mask].sum())
                                  if vd[mask].sum() > 0 else None)
        # block bootstrap over held-out blocks: recompute the cell covariances on resampled cells
        boots = {k: [] for k in ("corr_true_esk", "corr_true_desk")}
        for _ in range(a.n_boot):
            w = np.bincount(rng.integers(0, len(ub), len(ub)), minlength=len(ub))[inv].astype(float)
            if w.sum() == 0:
                continue
            wc = lambda x, y: (((x - np.average(x, 0, w)) * (y - np.average(y, 0, w))) * w[:, None]
                               ).sum(0) / w.sum()
            vdb = wc(da, db)
            boots["corr_true_esk"].append(pooled(0.5 * (wc(pa, db) + wc(pb, da)), wc(pa, pb), vdb, mask))
            boots["corr_true_desk"].append(pooled(wc(pd_, dfull), wc(pd_, pd_), vdb, mask))
        for k, v in boots.items():
            v = np.array([x for x in v if x is not None and np.isfinite(x)])
            out[k + "_ci"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else None
        return out

    res = {"cache": a.cache, "population": a.population, "rank": r, "n_eval_cells": int(len(ev)),
           "n_species": int(keep.sum()), "n_resolvable": int(res_mask.sum()),
           "all": stats(res_mask), "tiers": {}}
    for lo, hi in zip(TIERS[:-1], TIERS[1:]):
        res["tiers"][f"{lo:g}-{hi:g}"] = stats(res_mask & (prev >= lo) & (prev < hi))
    with open(os.path.join(a.out, "readout_corr.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    summarize(a.out)


def summarize(out):
    r = json.load(open(os.path.join(out, "readout_corr.json"), encoding="utf-8"))
    f = lambda v: "   -  " if v is None else f"{v:+.3f}"
    fci = lambda v: "" if not v else f"[{v[0]:+.3f},{v[1]:+.3f}]"
    print(f"readout corr [{r['population']}] r{r['rank']}: {r['n_eval_cells']} held-out cells, "
          f"{r['n_resolvable']}/{r['n_species']} species resolvable. Noise-free correlation of the LEVEL "
          f"readout of community change with species change (place-specific, pooled):")
    print(f"  {'stratum':14s} {'n':>4s}  {'true community':>26s}  {'DESK':>26s}  DESK~true readout")
    for name, s in [("all", r["all"])] + [(f"prev {k}", v) for k, v in r["tiers"].items()]:
        if "corr_true_esk" not in s:
            continue
        print(f"  {name:14s} {s['n_species']:4d}  {f(s['corr_true_esk'])} {fci(s.get('corr_true_esk_ci')):19s}"
              f"  {f(s['corr_true_desk'])} {fci(s.get('corr_true_desk_ci')):19s}  "
              f"{f(s['corr_desk_vs_esk_readout'])}")


if __name__ == "__main__":
    main()
