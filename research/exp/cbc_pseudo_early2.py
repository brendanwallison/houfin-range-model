"""E070: a pseudo-early CBC copy calibrated to the REAL early data, with the PLACE problem included.

    python research/exp/cbc_pseudo_early2.py --raw-dir <CBC request dir> --out-dir <new dir> --place rich|random

Builds on cbc_pseudo_early.py (E065), which degraded 1966-1985 counts by one mechanism at a time inside modern circles.
Here, for every count in --years:
  MIXED DEGRADATION -- each species record is kept with probability q (omission: incomplete reporting) and its
  individuals are binomially thinned with probability pi (less effort); per count, log q and log pi share one
  normal draw (sd --sd), so a poor count is poor both ways. pi is calibrated so the median NON-ZERO count falls to
  --nonzero-ratio of the undegraded median (0.45 = real pre-1940 vs post-1966 at <= 6 field counters, E065 F); q is then
  calibrated so median benchmark completeness falls to --target (0.26 = the 1920-39 median).
  PLACE -- circles are made EARLY-ONLY (every count after --years[1] removed) until they hold --early-only-share of the
  degraded window's counts (0.6 ~ the real share of pre-1940 counts in circles without >= 5 modern counts, E065:
  54-66%). --place rich picks circles with probability proportional to the rank of their later (> window) mean list
  length (richer circles more likely, as early circles near population centres were, Dunn et al. 2005; the inflation
  E068c found); --place random picks uniformly.
  DISTANCE MISSINGNESS -- party-distance is removed for --distance-missing of the degraded counts (0.40 = the real
  1920-39 share without distance); hours are removed for all of them (real: ~6%).
Field counters are untouched. Writes a fresh directory: degraded Circle_Species_Report and Effort_Many_Types_Report,
the other reports symlinked; pseudo_early2.json records the calibration.
"""
import argparse
import json
import os

import numpy as np

PRE = "20250205_FullSet_AllCountries_AllYears_CBC_"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw-dir", required=True); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--place", choices=["rich", "random"], required=True)
    ap.add_argument("--years", type=int, nargs=2, default=[1966, 1985])
    ap.add_argument("--target", type=float, default=0.26)
    ap.add_argument("--nonzero-ratio", type=float, default=0.45)
    ap.add_argument("--early-only-share", type=float, default=0.6)
    ap.add_argument("--distance-missing", type=float, default=0.40)
    ap.add_argument("--sd", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=70)
    a = ap.parse_args()
    import pandas as pd
    os.makedirs(a.out_dir, exist_ok=False)
    for n in ("Effort_Summary_Report", "Weather_Report"):
        os.symlink(os.path.join(os.path.abspath(a.raw_dir), PRE + n + ".csv"), os.path.join(a.out_dir, PRE + n + ".csv"))
    rng = np.random.default_rng(a.seed)
    sp = pd.read_csv(os.path.join(a.raw_dir, PRE + "Circle_Species_Report.csv"), encoding="utf-8-sig", low_memory=False)
    yr = 1900 + sp["count_yr"]
    win = yr.between(*a.years)
    tgt = win & (sp["how_many"] > 0)
    real = sp[(sp["how_many"] > 0) & ~sp["com_name"].astype(str).str.contains(r" sp\.|/| x ", regex=True)]
    late = real[1900 + real["count_yr"] > a.years[1]]
    n_cnt = late.groupby("abbrev")["count_yr"].nunique()
    share = late.groupby(["abbrev", "com_name"])["count_yr"].nunique().div(n_cnt, level="abbrev")
    bench = share[share >= 0.5].reset_index()[["abbrev", "com_name"]]
    bench = bench[bench["abbrev"].isin(n_cnt[n_cnt >= 5].index)]
    cy = sp.loc[tgt, ["abbrev", "count_yr"]].drop_duplicates().reset_index(drop=True)
    cy["z"] = rng.standard_normal(len(cy))
    rec = sp.loc[tgt, ["abbrev", "count_yr", "com_name", "how_many"]].merge(cy, on=["abbrev", "count_yr"], how="left")
    n_ind = np.round(rec["how_many"].to_numpy()).astype(np.int64)
    zr = rec["z"].to_numpy()
    sub = rng.random(len(rec)) < min(1.0, 400000 / len(rec))
    # (1) thinning level from the non-zero count ratio (records that survive thinning)

    def nz_ratio(mu):
        pi = np.minimum(1.0, np.exp(mu + a.sd * zr[sub]))
        y = rng.binomial(n_ind[sub], pi)
        return float(np.median(y[y > 0]) / np.median(n_ind[sub][n_ind[sub] > 0]))
    lo, hi = -8.0, 1.0
    for _ in range(30):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if nz_ratio(mid) < a.nonzero_ratio else (lo, mid)
    mu_pi = (lo + hi) / 2
    # (2) omission level from benchmark completeness, given thinning
    rb = rec.merge(bench.assign(b=1), on=["abbrev", "com_name"], how="left")
    bn = bench.groupby("abbrev").size().rename("bn")
    br = rb[rb["b"] == 1]
    nb_, zb = np.round(br["how_many"].to_numpy()).astype(np.int64), br["z"].to_numpy()

    def completeness(mu_q):
        pi = np.minimum(1.0, np.exp(mu_pi + a.sd * zb)); q = np.minimum(1.0, np.exp(mu_q + a.sd * zb))
        keep = q * (1 - (1 - pi) ** nb_)
        comp = pd.Series(keep).groupby([br["abbrev"].to_numpy(), br["count_yr"].to_numpy()]).sum()
        return float(np.median(comp.to_numpy() / bn.reindex(comp.index.get_level_values(0)).to_numpy()))
    lo, hi = -6.0, 0.5
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if completeness(mid) < a.target else (lo, mid)
    mu_q = (lo + hi) / 2
    # apply
    pi = np.minimum(1.0, np.exp(mu_pi + a.sd * zr)); q = np.minimum(1.0, np.exp(mu_q + a.sd * zr))
    new = np.where(rng.random(len(rec)) < q, rng.binomial(n_ind, pi), 0)
    sp.loc[tgt, "how_many"] = new
    drop = tgt.to_numpy() & (sp["how_many"].to_numpy() == 0)
    # PLACE: early-only circles
    wrows = sp.loc[win & ~pd.Series(drop, index=sp.index), ["abbrev", "count_yr"]].drop_duplicates()
    per_c = wrows.groupby("abbrev").size()
    circles = per_c.index.to_numpy()
    if a.place == "rich":
        ll = late.groupby(["abbrev", "count_yr"]).size().groupby("abbrev").mean().reindex(circles)
        w = ll.rank(pct=True).fillna(0.5).to_numpy()
    else:
        w = np.ones(len(circles))
    order = rng.choice(len(circles), len(circles), replace=False, p=w / w.sum())
    cum = np.cumsum(per_c.to_numpy()[order]) / per_c.sum()
    # E086v: --early-only-share 0 leaves every circle's later counts untouched
    chosen = set(circles[order[: int(np.searchsorted(cum, a.early_only_share)) + 1]]) if a.early_only_share > 0 else set()
    drop |= (sp["abbrev"].isin(chosen) & (yr > a.years[1])).to_numpy()
    sp[~drop].to_csv(os.path.join(a.out_dir, PRE + "Circle_Species_Report.csv"), index=False, encoding="utf-8-sig")
    # DISTANCE / HOURS missingness in the degraded window
    ef = pd.read_csv(os.path.join(a.raw_dir, PRE + "Effort_Many_Types_Report.csv"), encoding="utf-8-sig", low_memory=False)
    ew = (1900 + ef["count_yr"]).between(*a.years)
    ecy = ef.loc[ew, ["abbrev", "count_yr"]].drop_duplicates()
    miss = ecy[rng.random(len(ecy)) < a.distance_missing]
    mk = ef.set_index(["abbrev", "count_yr"]).index.isin(miss.set_index(["abbrev", "count_yr"]).index)
    ef.loc[ew, "hours"] = 0
    ef.loc[mk, "distance"] = 0
    ef.to_csv(os.path.join(a.out_dir, PRE + "Effort_Many_Types_Report.csv"), index=False, encoding="utf-8-sig")
    kept = sp[~drop & tgt.to_numpy()]
    info = {"place": a.place, "years": a.years, "mu_pi": mu_pi, "median_pi": float(np.exp(mu_pi)), "mu_q": mu_q,
            "median_q": float(np.exp(mu_q)), "nonzero_ratio_achieved": nz_ratio(mu_pi),
            "completeness_expected": completeness(mu_q), "early_only_circles": len(chosen),
            "early_only_share_of_window_counts": float(per_c.loc[list(chosen)].sum() / per_c.sum()),
            "records_removed": int(drop.sum()), "distance_missing_circle_years": int(len(miss))}
    json.dump(info, open(os.path.join(a.out_dir, "pseudo_early2.json"), "w"), indent=1)
    print(f"[E070 pseudo-early2 {a.place}] {json.dumps(info)}", flush=True)


if __name__ == "__main__":
    main()
