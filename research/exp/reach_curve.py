"""E019: how fast does DESK's backcast degrade with distance from its training years? (S1, S3, T9)

    ESK_DESK_CONFIG=<overlay> python research/exp/reach_curve.py --out <dir> [--caches base tempho_1975 ...]
    python research/exp/reach_curve.py --summarize <out>

Four DESK models share the same held-out cells: base (trained on every year) and the tempho models trained from 1976,
1986 and 1996 on. For each held-out cell and each 6-year window from 1966 on, the observed community (log1p of the
window's mean counts, two ABBA halves) is projected into ESK; each model's z is averaged over the same surveyed rows.
Per model and window, against the cell's modern epoch (2005-2025), in ESK coordinates (first r), noise-free by halves
and centered across cells:
    corr         DESK change vs true change (direction)
    calibration  truth-on-DESK slope (1 = moves the right amount for its accuracy)
    mse_ratio    E|dD - dT|^2 / E|dT|^2   (< 1: the backcast state is closer to the truth than the modern state is)
    z_error      E|zD - zT|^2 per window (noise-free; uncentered) -- the size of the feature error the age model
                 would have to carry as uncertainty, against the true change energy
reach = (first training year of the model) - (window centre); <= 0 means the window was in the training years.
The curve stops at ~28 years of reach; 1902-1939 is 26-64 years before the BBS. Anything beyond the measured range is
extrapolation, and the summary says so.
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

CACHE_ROOT = os.path.expanduser("~/houfin/work/houfin/research_cache")
MODELS = {"base": ("gp_species_base", 1966), "tempho_1975": ("desk_tempho_1975", 1976),
          "tempho_1985": ("desk_tempho_1985", 1986), "tempho_1995": ("desk_tempho_1995", 1996)}
WINDOWS = [(1966, 1971), (1972, 1977), (1978, 1983), (1984, 1989), (1990, 1995), (1996, 2001)]
MODERN = (2005, 2025)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out")
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--extra", nargs="*", default=(),
                    help="more models as name=cache_subdir:first_training_year (e.g. seed replicates)")
    ap.add_argument("--ranks", type=int, nargs="+", default=(12, 24, 64))
    ap.add_argument("--min-years", type=int, default=4)
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--summarize", default=None)
    a = ap.parse_args()
    if a.summarize:
        return summarize(a.summarize)
    from src.config_utils import load_config
    from src.community_encoder.train_DESK.esk_kernel import project_points_to_z
    from src.community_encoder.train_DESK.validate_bbs_routes import epoch_mean_observed
    from src.community_encoder.train_DESK.validation_core import split_half_groups
    os.makedirs(a.out, exist_ok=True)
    for spec in a.extra:
        name, rest = spec.split("=")
        sub, first = rest.split(":")
        MODELS[name] = (sub, int(first))
        a.models = list(a.models) + [name]
    base_dir = os.path.join(CACHE_ROOT, MODELS["base"][0])
    keys = np.load(os.path.join(base_dir, "keys.npy"))
    for m in a.models:
        k2 = np.load(os.path.join(CACHE_ROOT, MODELS[m][0], "keys.npy"))
        assert np.array_equal(k2, keys), f"{m}: keys differ from base"
    ho = np.ones(np.load(os.path.join(base_dir, "split.npz"))["holdout"].shape, bool)
    for m in a.models:
        ho &= np.load(os.path.join(CACHE_ROOT, MODELS[m][0], "split.npz"))["holdout"]
    Xc = np.load(os.path.join(base_dir, "X_comm.npy")).astype("float64")
    yr = keys[:, 2]
    held = ho[keys[:, 0], keys[:, 1]]
    cid = keys[:, 0].astype(np.int64) * 100000 + keys[:, 1]
    # groups: (cell, window) and (cell, modern) for held-out cells with >= min-years surveyed
    groups, gcell, gwin = [], [], []
    for wi, (lo, hi) in enumerate(WINDOWS + [MODERN]):
        sel = np.where(held & (yr >= lo) & (yr <= hi))[0]
        order = np.lexsort((yr[sel], cid[sel]))
        sel = sel[order]
        cs, starts = np.unique(cid[sel], return_index=True)
        for c_, s, e in zip(cs, starts, list(starts[1:]) + [len(sel)]):
            g = sel[s:e]
            if len(g) >= a.min_years:
                groups.append(g)
                gcell.append(c_)
                gwin.append(wi)
    gcell, gwin = np.array(gcell), np.array(gwin)
    A, B, ok = split_half_groups(groups, years=yr)
    xf = epoch_mean_observed(Xc, groups)
    xa = epoch_mean_observed(Xc, A)
    xb = epoch_mean_observed(Xc, B)
    cfg = load_config()
    L = max(a.ranks)
    proj = lambda v: np.asarray(project_points_to_z(np.asarray(v, "float32"), cfg["desk"]["z_dir"], L), "float64")
    T = {"full": proj(xf), "a": proj(xa), "b": proj(xb)}
    mod_ix = {c_: j for j, (c_, w) in enumerate(zip(gcell, gwin)) if w == len(WINDOWS)}
    res = {"windows": WINDOWS, "modern": MODERN, "n_heldout_cells": int(len(np.unique(cid[held]))), "rows": []}
    rng = np.random.default_rng(0)
    for m in a.models:
        z = np.asarray(np.load(os.path.join(CACHE_ROOT, MODELS[m][0], "z_raw.npy"), mmap_mode="r"), "float64")
        D = np.stack([z[g].mean(0) for g in groups])
        for wi, (lo, hi) in enumerate(WINDOWS):
            js = [j for j in np.where((gwin == wi) & ok)[0] if gcell[j] in mod_ix and ok[mod_ix[gcell[j]]]]
            if len(js) < 20:
                continue
            js = np.array(js)
            ms = np.array([mod_ix[gcell[j]] for j in js])
            cells_rc = np.column_stack([gcell[js] // 100000, gcell[js] % 100000])
            blk = (cells_rc[:, 0] // 6) * 100000 + cells_rc[:, 1] // 6
            ub, binv = np.unique(blk, return_inverse=True)
            W = rng.multinomial(len(ub), np.full(len(ub), 1.0 / len(ub)), size=a.n_boot)
            row = {"model": m, "window": [lo, hi], "reach": float(MODELS[m][1] - (lo + hi) / 2.0),
                   "n_cells": int(len(js)), "ranks": {}}
            for r in a.ranks:
                dD = D[js, :r] - D[ms, :r]
                dTa, dTb = T["a"][js, :r] - T["a"][ms, :r], T["b"][js, :r] - T["b"][ms, :r]
                dTf = T["full"][js, :r] - T["full"][ms, :r]
                zerr = ((D[js, :r] - T["a"][js, :r]) * (D[js, :r] - T["b"][js, :r])).sum(1)

                def stats(w):
                    cen = lambda v: v - (w[:, None] * v).sum(0) / w.sum()
                    d_, f_, a_, b_ = cen(dD), cen(dTf), cen(dTa), cen(dTb)
                    cov = (w[:, None] * d_ * f_).sum()
                    vd = (w[:, None] * d_ * d_).sum()
                    vt = (w[:, None] * a_ * b_).sum()
                    if vd <= 0 or vt <= 0:
                        return np.nan, np.nan, np.nan, np.nan
                    return cov / np.sqrt(vd * vt), cov / vd, (vd - 2 * cov + vt) / vt, np.sqrt(vd / vt)
                corr, calib, mse, amp = stats(np.ones(len(js)))
                bs = np.array([stats(w[binv].astype(float)) for w in W])
                ci = lambda k: [float(np.nanpercentile(bs[:, k], 2.5)), float(np.nanpercentile(bs[:, k], 97.5))]
                true_energy = float((dTa * dTb).sum(1).mean())
                row["ranks"][str(r)] = {"corr": float(corr), "corr_ci": ci(0), "calibration": float(calib),
                                        "mse_ratio": float(mse), "mse_ratio_ci": ci(2), "amplitude": float(amp),
                                        "z_error": float(zerr.mean()), "true_change_energy": true_energy}
            res["rows"].append(row)
    with open(os.path.join(a.out, "reach_curve.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    summarize(a.out)


def summarize(out):
    r = json.load(open(os.path.join(out, "reach_curve.json"), encoding="utf-8"))
    print(f"reach curve: {r['n_heldout_cells']} held-out cells; change vs the modern epoch {r['modern']}; reach = years "
          f"before the model's first training year (<= 0: trained)")
    ranks = sorted({k for row in r["rows"] for k in row["ranks"]}, key=int)
    for rk in ranks:
        print(f"  r{rk}:  model        window      reach  cells   corr [95% CI]          calib  mse_ratio [95% CI]"
              f"        z_error / true change energy")
        for row in sorted(r["rows"], key=lambda x: (x["reach"], x["model"])):
            v = row["ranks"].get(rk)
            if not v:
                continue
            print(f"        {row['model']:12s} {row['window'][0]}-{row['window'][1]}  {row['reach']:+6.1f}  "
                  f"{row['n_cells']:5d}   {v['corr']:+.3f} [{v['corr_ci'][0]:+.2f},{v['corr_ci'][1]:+.2f}]   "
                  f"{v['calibration']:.2f}   {v['mse_ratio']:.3f} [{v['mse_ratio_ci'][0]:.2f},{v['mse_ratio_ci'][1]:.2f}]"
                  f"   {v['z_error']:.3f} / {v['true_change_energy']:.3f}")
    print("  (measured reach stops near +28 years; 1902-1939 lies 26-64 years before the BBS -- beyond it is "
          "extrapolation)")


if __name__ == "__main__":
    main()
