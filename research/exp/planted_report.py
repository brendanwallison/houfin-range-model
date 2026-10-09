"""Grade a fast-arm run on a PLANTED cache against the noise-free truth, by generator and prevalence.

    python research/exp/planted_report.py --planted <planted cache> --arm <fast arm out (--dump-change)>

Two questions per generator x prevalence tier:
  1. What can the instrument show at best? For the ``z`` generator the readout is exactly right in form
     (log lambda is linear in the features the BLR sees), so its level skill and change attenuation are
     the ceiling every real-species number must be read against.
  2. Is the noise-corrected attenuation estimator unbiased? The suite's slope divides by
     ``var(observed change) - split-half noise``; here the true per-cell change is known (route-mean NB
     counts have mean lambda, so the epoch change of the expectation is the epoch-mean lambda change),
     so the estimated slope can be set against the true slope cov(pred, true) / var(true), and the
     estimated real-change variance against var(true).
"""
import argparse
import glob
import json
import os

import numpy as np


def epoch_mean(v, flat, ptr):
    """Mean of rows ``flat[ptr[i]:ptr[i+1]]`` of ``v`` for each cell i."""
    s = np.add.reduceat(v[flat], ptr[:-1], axis=0)
    return s / np.diff(ptr)[:, None]


def slope(x, y):
    """Per-column slope of x on y, cov(x, y) / var(y), over rows."""
    xc, yc = x - x.mean(0), y - y.mean(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (xc * yc).mean(0) / (yc * yc).mean(0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--planted", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--tiers", type=float, nargs="+", default=(0.02, 0.1, 0.3),
                    help="prevalence cut points (share of cell-years with a detection)")
    a = ap.parse_args()
    import pandas as pd
    truth = np.load(os.path.join(a.planted, "truth.npz"))
    lam = np.exp(truth["log_lambda"].astype("float64"))
    gen_all = truth["generator"]
    X = np.load(os.path.join(a.planted, "X_dev.npy"), mmap_mode="r")
    meta = json.load(open(os.path.join(a.planted, "meta.json"), encoding="utf-8"))
    name_to_col = {s: i for i, s in enumerate(meta["dev_species"])}
    per = pd.read_csv(os.path.join(a.arm, "per_species.csv"))
    cols = np.array([name_to_col[s] for s in per["species"]])
    per["generator"] = gen_all[cols]
    per["prevalence"] = (np.asarray(X)[:, cols] > 0).mean(0)
    cuts = [0.0] + list(a.tiers) + [1.01]
    per["tier"] = pd.cut(per["prevalence"], cuts, right=False,
                         labels=[f"[{lo:g},{hi:g})" for lo, hi in zip(cuts[:-1], cuts[1:])])

    out = {"tiers": cuts, "change": {}}
    for f in sorted(glob.glob(os.path.join(a.arm, "change_*.npz"))):
        sname = os.path.basename(f)[len("change_"):-len(".npz")]
        d = np.load(f)
        sp_cols = np.array([name_to_col[s] for s in d["species"]])
        assert (sp_cols == cols).all(), "change dump and per-species table disagree on species order"
        L = lam[:, sp_cols]
        # the suite's estimand is log1p of the epoch-MEAN abundance (validation_core.epoch_values)
        d_true = (np.log1p(epoch_mean(L, d["modern_rows"], d["modern_ptr"]))
                  - np.log1p(epoch_mean(L, d["early_rows"], d["early_ptr"])))
        dp, dfull = d["dp_model"], d["d_full"]
        static = d_true.std(0) < 1e-9 * np.maximum(np.abs(d_true).mean(0), 1e-12)
        d_true[:, static] = np.nan                                   # mix_0: no true change to grade
        per[f"att_true_{sname}"] = slope(dp, d_true)
        per[f"corr_true_{sname}"] = [np.corrcoef(dp[:, j], d_true[:, j])[0, 1]
                                     if dp[:, j].std() > 0 and d_true[:, j].std() > 0 else np.nan
                                     for j in range(dp.shape[1])]
        dfc = dfull - dfull.mean(0)
        per[f"realvar_est_over_true_{sname}"] = (((dfc ** 2).mean(0) - d["noise"])
                                                 / np.maximum(d_true.var(0), 1e-300))
        per[f"true_share_{sname}"] = d_true.var(0) / np.maximum((dfc ** 2).mean(0), 1e-300)
        out["change"][sname] = {"n_cells": int(dp.shape[0])}

    per.to_csv(os.path.join(a.arm, "planted_per_species.csv"), index=False)
    rows = []
    for (g, t), grp in per.groupby(["generator", "tier"], observed=True):
        r = {"generator": g, "tier": str(t), "n": int(len(grp)),
             "prev_median": float(grp["prevalence"].median())}
        for c in [c for c in per.columns if c.startswith("level_") and c.endswith(("vs_intercept",
                                                                                    "vs_persistence"))]:
            r[c] = float(grp[c].median())
        for sname in out["change"]:
            res = grp[f"resolvable_{sname}"].astype(bool)
            r[f"{sname}_n_resolvable"] = int(res.sum())
            r[f"{sname}_att_est_resolvable"] = (float(grp.loc[res, f"attenuation_{sname}"].median())
                                                if res.any() else None)
            r[f"{sname}_att_true_resolvable"] = (float(grp.loc[res, f"att_true_{sname}"].median())
                                                 if res.any() else None)
            r[f"{sname}_att_true_all"] = float(grp[f"att_true_{sname}"].median())
            r[f"{sname}_corr_true_all"] = float(grp[f"corr_true_{sname}"].median())
            r[f"{sname}_true_share"] = float(grp[f"true_share_{sname}"].median())
            r[f"{sname}_realvar_est_over_true"] = float(grp[f"realvar_est_over_true_{sname}"].median())
        rows.append(r)
    out["table"] = rows
    with open(os.path.join(a.arm, "planted_report.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    summarize(a.arm)


def summarize(arm):
    r = json.load(open(os.path.join(arm, "planted_report.json"), encoding="utf-8"))
    sets = list(r["change"])
    g = lambda v: "   -  " if v is None or v != v else f"{v:+.3f}"
    print("planted readout ceilings (medians; att = change attenuation slope, est = the suite's "
          "noise-corrected estimate, true = against the noise-free change)")
    for s in sets:
        print(f"  change set: {s} ({r['change'][s]['n_cells']} cells)")
        print("    generator  prevalence    n  lvl_space  res  att_est  att_true(res)  att_true(all)  "
              "corr_true  true_share  est/true var")
        for row in r["table"]:
            print(f"    {row['generator']:9s}  {row['tier']:11s} {row['n']:3d}  "
                  f"{g(row.get('level_space_vs_intercept'))}  {row[f'{s}_n_resolvable']:4d}  "
                  f"{g(row[f'{s}_att_est_resolvable'])}  {g(row[f'{s}_att_true_resolvable'])}         "
                  f"{g(row[f'{s}_att_true_all'])}         {g(row[f'{s}_corr_true_all'])}     "
                  f"{g(row[f'{s}_true_share'])}      {g(row[f'{s}_realvar_est_over_true'])}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3 and sys.argv[1] == "--summarize":
        summarize(sys.argv[2])
    else:
        main()
