"""E064 gates: does our raw-CBC effort model reproduce Audubon's trends (C-a), and does counters-only effort agree with
party-hours effort where both exist, and predict well before 1940 (C-b)?

    python research/exp/cbc_effort_gates.py --data cbc_circles.npz --fit-ph fit_ph.npz --fit-counters fit_cnt.npz
        --products-csv <Audubon indices csv> --out <dir>
    python research/exp/cbc_effort_gates.py --summarize <out>

C-a (validation, Audubon's own era): for each species and stratum present in both, the change log(mean index, winters
ending 2005-2024) - log(mean index, 1967-1986), from our party-hours model vs Audubon's 2023.0 product. Strata need >= 3
fitted circle-years in each window in our data. Reported: pooled stratum-level corr (each species' mean removed:
WHERE change happened) and species-level corr (mean change over strata per species). Also the corr of annual-index
TRAJECTORIES within stratum, 1967-2024, after removing each stratum's mean (median over species x strata).
C-b (i) agreement of the counters-only and party-hours models where both effort measures exist: stratum change
1950-1957 -> 1958-1965 and 1967-1986 -> 2005-2024, pooled corr. (ii) predictive check on held-out circle-years: per
species Spearman corr of predicted with observed counts, by era (< 1940, 1940-1965, >= 1966), median over species with
>= 20 non-zero held-out counts in the era.
"""
import argparse
import json
import os
import sys

import numpy as np


def _wmean(index, years, y0, y1):
    m = (years >= y0) & (years <= y1)
    return index[:, m].mean(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data"); ap.add_argument("--fit-ph"); ap.add_argument("--fit-counters")
    ap.add_argument("--products-csv"); ap.add_argument("--out"); ap.add_argument("--summarize", default=None)
    ap.add_argument("--min-support", type=int, default=3)
    a = ap.parse_args()
    if a.summarize:
        return summarize(a.summarize)
    import pandas as pd
    from scipy.stats import spearmanr
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from src.config_utils import load_data_config
    os.makedirs(a.out, exist_ok=True)
    D = np.load(a.data, allow_pickle=False)
    fp, fc = np.load(a.fit_ph, allow_pickle=False), np.load(a.fit_counters, allow_pickle=False)
    years, strata, species = fp["years"], list(fp["strata"]), list(fp["species"])
    res = {}

    def change(fit, w0, w1, v0, v1):
        I, sup = fit["index"], fit["support"]
        e = np.log(_wmean(I, years, w0, w1)); l = np.log(_wmean(I, years, v0, v1))
        se = sup[:, (years >= w0) & (years <= w1)].sum(1); sl = sup[:, (years >= v0) & (years <= v1)].sum(1)
        ok = (se >= a.min_support) & (sl >= a.min_support)          # support is per stratum (all species share it)
        return l - e, np.broadcast_to(ok[:, None], l.shape)

    # ---- C-a vs Audubon
    tax = pd.read_csv(os.path.join(load_data_config()["datasets_root"], "avonet", "eBird_taxonomy.csv"),
                      usecols=["SCIENTIFIC_NAME", "SPECIES_CODE", "CATEGORY"])
    tax = tax[tax["CATEGORY"] == "species"]
    prod = pd.read_csv(a.products_csv, encoding="latin-1")
    prod = prod[(prod["region_type"] == "stratum") & prod["survey_suitability"].isin(["medium", "high"])]
    prod["code"] = prod["scientific_name"].map(dict(zip(tax["SCIENTIFIC_NAME"], tax["SPECIES_CODE"])))
    prod = prod[prod["code"].isin(set(species))]
    pm = lambda y0, y1: prod[(prod.year_end >= y0) & (prod.year_end <= y1)].groupby(["code", "region"])["annual_index"].mean()
    aud = (np.log(pm(2005, 2024)) - np.log(pm(1967, 1986))).dropna()
    ours, ok = change(fp, 1967, 1986, 2005, 2024)
    rows = []
    for (code, reg), v in aud.items():
        if reg in strata and code in species:
            r, s = strata.index(reg), species.index(code)
            if ok[r, s] and np.isfinite(ours[r, s]) and np.isfinite(v):
                rows.append((code, reg, float(ours[r, s]), float(v)))
    df = pd.DataFrame(rows, columns=["code", "region", "ours", "audubon"])
    dc = df.copy()
    dc[["ours", "audubon"]] = dc.groupby("code")[["ours", "audubon"]].transform(lambda z: z - z.mean())
    sl = df.groupby("code")[["ours", "audubon"]].mean()
    res["Ca_pairs"] = int(len(df)); res["Ca_species"] = int(df.code.nunique())
    res["Ca_stratum_corr_centred"] = float(np.corrcoef(dc["ours"], dc["audubon"])[0, 1])
    res["Ca_stratum_corr_raw"] = float(np.corrcoef(df["ours"], df["audubon"])[0, 1])
    res["Ca_species_corr"] = float(np.corrcoef(sl["ours"], sl["audubon"])[0, 1])
    res["Ca_per_species_median"] = float(df.groupby("code").apply(
        lambda g: np.corrcoef(g["ours"], g["audubon"])[0, 1] if len(g) >= 5 else np.nan).median())
    # trajectories within stratum, 1967-2024
    traj = []
    ym = (years >= 1967) & (years <= 2024)
    for (code, reg), g in prod[(prod.year_end >= 1967) & (prod.year_end <= 2024)].groupby(["code", "region"]):
        if reg not in strata:
            continue
        r, s = strata.index(reg), species.index(code)
        if fp["support"][r, ym].sum() < 20:
            continue
        o = pd.Series(np.log(fp["index"][r, ym, s]), index=years[ym])
        au = np.log(g.set_index("year_end")["annual_index"].reindex(years[ym]))
        m = au.notna()
        if m.sum() >= 20:
            traj.append(np.corrcoef(o[m] - o[m].mean(), au[m] - au[m].mean())[0, 1])
    res["Ca_trajectory_corr_median"] = float(np.nanmedian(traj)); res["Ca_trajectory_n"] = len(traj)
    # ---- C-b (i) counters vs party-hours
    for lab, w in (("1950s", (1950, 1957, 1958, 1965)), ("1966_2024", (1967, 1986, 2005, 2024))):
        x1, ok1 = change(fp, *w); x2, ok2 = change(fc, *w)
        m = ok1 & ok2 & np.isfinite(x1) & np.isfinite(x2)
        xc1 = x1 - np.nanmean(np.where(m, x1, np.nan), 0); xc2 = x2 - np.nanmean(np.where(m, x2, np.nan), 0)
        res[f"Cb_counters_vs_ph_{lab}_n"] = int(m.sum())
        res[f"Cb_counters_vs_ph_{lab}_corr"] = float(np.corrcoef(x1[m], x2[m])[0, 1]) if m.sum() > 10 else None
        res[f"Cb_counters_vs_ph_{lab}_corr_centred"] = float(np.corrcoef(xc1[m], xc2[m])[0, 1]) if m.sum() > 10 else None
    # ---- C-b (ii) held-out prediction by era
    Y, yr = D["Y"], D["year_end"]
    for lab, fit in (("ph", fp), ("counters", fc)):
        rows_h, pred = fit["hold_rows"], fit["hold_pred"]
        for era, (y0, y1) in (("pre1940", (1901, 1939)), ("1940_1965", (1940, 1965)), ("post1966", (1966, 2025))):
            m = (yr[rows_h] >= y0) & (yr[rows_h] <= y1)
            vals = []
            for s in range(Y.shape[1]):
                obs = Y[rows_h[m], s]
                if (obs > 0).sum() >= 20:
                    vals.append(spearmanr(pred[m, s], obs).correlation)
            res[f"Cb_holdout_spearman_{lab}_{era}"] = float(np.nanmedian(vals)) if vals else None
            res[f"Cb_holdout_species_{lab}_{era}"] = len(vals)
    res["p_ph_median"] = float(np.median(fp["p"])); res["p_counters_median"] = float(np.median(fc["p"]))
    json.dump(res, open(os.path.join(a.out, "cbc_gates.json"), "w"), indent=1)
    summarize(a.out)


def summarize(out):
    r = json.load(open(os.path.join(out, "cbc_gates.json")))
    for k, v in r.items():
        print(f"  {k:40s} {v:.3f}" if isinstance(v, float) else f"  {k:40s} {v}")


if __name__ == "__main__":
    main()
