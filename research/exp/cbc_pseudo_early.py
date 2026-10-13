"""E065 known-truth test: degrade modern CBC counts to early-era completeness, keeping their field counters.

    python research/exp/cbc_pseudo_early.py --raw-dir <CBC request dir> --out-dir <new dir> --mechanism thin|omit
        [--years 1966 1985] [--target 0.26] [--bench-from 1986]

Writes a copy of the raw data request whose Circle_Species_Report has every count in --years degraded (the other three
reports are symlinked unchanged, so field counters, party-hours etc. stay as recorded). Each degraded circle-year gets
its own severity, log-normal across counts (sd --sd), with the median calibrated so that the median BENCHMARK
COMPLETENESS of the degraded counts (share of the circle's benchmark species -- recorded on >= 50% of its counts from
--bench-from on -- that are still recorded) equals --target (0.26 = the 1920-1939 median, cbc_completeness.py):
  thin: every individual is kept with probability pi (binomial thinning of how_many) -- LESS EFFORT: rare species
        vanish first, flocks survive, counts shrink;
  omit: every species record is kept with probability q regardless of its count -- INCOMPLETE REPORTING: counts of
        the species still listed are untouched.
Count-week records (how_many == 0) are left as they are. Records thinned to 0 individuals are removed.
Downstream (unchanged code): cbc_prepare.py -> cbc_completeness.py --bench-from 1986 -> cbc_effort_model.py --ref-from 1986.
The truth for a variant is its own fit on the undegraded data; the test is whether the degraded fit recovers the
1966-85 -> 2005-24 change.
"""
import argparse
import json
import os

import numpy as np

PRE = "20250205_FullSet_AllCountries_AllYears_CBC_"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mechanism", choices=["thin", "omit"], required=True)
    ap.add_argument("--years", type=int, nargs=2, default=[1966, 1985])
    ap.add_argument("--target", type=float, default=0.26)
    ap.add_argument("--sd", type=float, default=0.8)
    ap.add_argument("--bench-from", type=int, default=1986)
    ap.add_argument("--seed", type=int, default=65)
    a = ap.parse_args()
    import pandas as pd
    os.makedirs(a.out_dir, exist_ok=False)                      # never overwrite: a fresh directory per scenario
    for n in ("Effort_Many_Types_Report", "Effort_Summary_Report", "Weather_Report"):
        os.symlink(os.path.join(os.path.abspath(a.raw_dir), PRE + n + ".csv"), os.path.join(a.out_dir, PRE + n + ".csv"))
    src = os.path.join(a.raw_dir, PRE + "Circle_Species_Report.csv")
    sp = pd.read_csv(src, encoding="utf-8-sig", low_memory=False)
    yr = 1900 + sp["count_yr"]
    tgt = yr.between(*a.years) & (sp["how_many"] > 0)
    # benchmark sets from the undegraded later years (same rule as cbc_completeness.py)
    real = sp[(sp["how_many"] > 0) & ~sp["com_name"].astype(str).str.contains(r" sp\.|/| x ", regex=True)]
    good = real[1900 + real["count_yr"] >= a.bench_from]
    n_cnt = good.groupby("abbrev")["count_yr"].nunique()
    share = good.groupby(["abbrev", "com_name"])["count_yr"].nunique().div(n_cnt, level="abbrev")
    bench = share[share >= 0.5].reset_index()[["abbrev", "com_name"]]
    bench = bench[bench["abbrev"].isin(n_cnt[n_cnt >= 5].index)]
    # per circle-year severity: log-normal around a calibrated median
    cy = sp.loc[tgt, ["abbrev", "count_yr"]].drop_duplicates().reset_index(drop=True)
    rng = np.random.default_rng(a.seed)
    cy["z"] = rng.standard_normal(len(cy))
    rows = sp.loc[tgt, ["abbrev", "count_yr", "com_name", "how_many"]].merge(cy, on=["abbrev", "count_yr"])
    rows = rows.merge(bench.assign(b=1), on=["abbrev", "com_name"], how="left")
    bn = bench.groupby("abbrev").size().rename("bn")
    rows = rows.merge(bn, left_on="abbrev", right_index=True, how="left")
    br = rows[rows["b"] == 1]
    n_ind = br["how_many"].to_numpy(float)

    def median_completeness(mu):
        p = np.minimum(1.0, np.exp(mu + a.sd * br["z"].to_numpy()))
        keep = 1 - (1 - p) ** n_ind if a.mechanism == "thin" else p
        comp = pd.Series(keep).groupby([br["abbrev"].to_numpy(), br["count_yr"].to_numpy()]).sum()
        nb = bn.reindex(comp.index.get_level_values(0)).to_numpy()
        return float(np.median(comp.to_numpy() / nb))

    lo, hi = -12.0, 2.0
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if median_completeness(mid) < a.target else (lo, mid)
    mu = (lo + hi) / 2
    before = median_completeness(np.log(1e9))  # severity off: every record kept
    p_row = np.minimum(1.0, np.exp(mu + a.sd * sp.loc[tgt, ["abbrev", "count_yr"]].merge(
        cy, on=["abbrev", "count_yr"], how="left")["z"].to_numpy()))
    hm = sp.loc[tgt, "how_many"].to_numpy()
    if a.mechanism == "thin":
        new = rng.binomial(np.round(hm).astype(np.int64), p_row)
    else:
        new = np.where(rng.random(len(hm)) < p_row, hm, 0)
    sp.loc[tgt, "how_many"] = new
    drop = tgt.to_numpy() & (sp["how_many"].to_numpy() == 0)
    out = sp[~drop]
    out.to_csv(os.path.join(a.out_dir, PRE + "Circle_Species_Report.csv"), index=False, encoding="utf-8-sig")
    info = {"mechanism": a.mechanism, "years": a.years, "target": a.target, "sd": a.sd, "mu": mu,
            "median_severity": float(np.exp(mu)), "median_completeness_before": before,
            "median_completeness_after_expected": median_completeness(mu), "records_removed": int(drop.sum()),
            "records_in_window": int(tgt.sum()), "circle_years_in_window": int(len(cy))}
    json.dump(info, open(os.path.join(a.out_dir, "pseudo_early.json"), "w"), indent=1)
    print(f"[E065 pseudo-early {a.mechanism}] {json.dumps(info)}", flush=True)


if __name__ == "__main__":
    main()
