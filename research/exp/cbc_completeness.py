"""E065 data step: per circle-year sampling-completeness covariates for the raw CBC (aligned to cbc_prepare.py's rows).

    python research/exp/cbc_completeness.py --raw-dir <CBC request dir> --prepared cbc_circles_v2.npz --out completeness.npz

Early counts at the SAME circle and the SAME number of field counters record far fewer species (within 382 circles:
x0.31 in 1901-19, x0.54 in 1920-39, x0.78 in 1940-65 relative to 1966+), so field counters miss most early effort. Two
covariates estimated from the lists themselves (unstructured-records literature: list-length methods, Isaac et al. 2014;
Frescalo's benchmark species, Hill 2012):
  list_len[i]      species recorded on count i (count-day > 0; all real species, not only the modelled ones);
  bench_n[i]       size of the circle's BENCHMARK set: species recorded on >= --bench-share of that circle's counts in
                   well-sampled years (>= --bench-from); circles with < 5 such counts borrow the set of the nearest
                   circle that has one (within 100 km);
  bench_hit[i]     benchmark species recorded on count i;
  in_bench[c, s]   whether modelled species s is in circle c's benchmark set.
The model uses them LEAVE-ONE-OUT per species (the target species' own detection removed), so a species' count cannot
predict itself. Benchmark completeness is robust to real changes in total richness, list length is not.
"""
import argparse
import json
import os

import numpy as np

PRE = "20250205_FullSet_AllCountries_AllYears_CBC_"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--prepared", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bench-from", type=int, default=1950)
    ap.add_argument("--bench-share", type=float, default=0.5)
    ap.add_argument("--taxonomy", required=True, help="eBird taxonomy CSV (species common names -> codes)")
    ap.add_argument("--species-allowlist", default=None,
                    help="optional CSV species_code whitelist; unmapped raw names are excluded")
    ap.add_argument("--audit-sealed-cache", default=None,
                    help="cache meta.json: count recognized sealed/focal contributions to the unchanged benchmark")
    a = ap.parse_args()
    import pandas as pd
    from scipy.spatial import cKDTree
    P = np.load(a.prepared, allow_pickle=False)
    circles, species = list(P["circles"]), list(P["species"])
    ci = {c: i for i, c in enumerate(circles)}
    sp = pd.read_csv(os.path.join(a.raw_dir, PRE + "Circle_Species_Report.csv"), encoding="utf-8-sig",
                     usecols=["abbrev", "count_yr", "com_name", "how_many"])
    sp = sp[sp["abbrev"].isin(ci) & (sp["how_many"] > 0)
            & ~sp["com_name"].str.contains(r" sp\.|/| x ", regex=True, na=False)]
    sp = sp.drop_duplicates(["abbrev", "count_yr", "com_name"])
    sp["year"] = 1900 + sp["count_yr"]
    before = len(sp)
    if a.species_allowlist:
        tax = pd.read_csv(a.taxonomy)
        tax = tax[tax["CATEGORY"] == "species"]
        mapping = dict(zip(tax["COMMON_NAME"], tax["SPECIES_CODE"]))
        codes = sp["com_name"].map(lambda n: mapping.get(n) or mapping.get(n.split(" (")[0].strip()))
        allowed = set(pd.read_csv(a.species_allowlist)["species_code"])
        assert "houfin" not in allowed
        sp = sp[codes.isin(allowed)].copy()
    # rows of the prepared file
    rk = pd.DataFrame({"abbrev": np.array(circles)[P["circle_idx"]], "year": P["year_end"]})
    rk["row"] = np.arange(len(rk))
    ll = sp.groupby(["abbrev", "year"]).size().rename("list_len").reset_index()
    # benchmark sets from well-sampled years
    good = sp[sp["year"] >= a.bench_from]
    n_cnt = good.groupby("abbrev")["year"].nunique()
    freq = good.groupby(["abbrev", "com_name"])["year"].nunique().div(n_cnt, level="abbrev")
    bench = {c: set(freq.loc[c][freq.loc[c] >= a.bench_share].index) for c in n_cnt[n_cnt >= 5].index}
    lat = pd.Series(P["lat"]).groupby(P["circle_idx"]).first(); lon = pd.Series(P["lon"]).groupby(P["circle_idx"]).first()
    have = [c for c in circles if c in bench]
    tree = cKDTree(np.column_stack([lat[[ci[c] for c in have]], lon[[ci[c] for c in have]]]))
    borrowed = 0
    for c in circles:
        if c not in bench:
            d, j = tree.query([lat[ci[c]], lon[ci[c]]])
            if d <= 1.0:                                   # ~100 km in degrees, rough
                bench[c] = bench[have[j]]; borrowed += 1
    sp_bench = sp.merge(pd.DataFrame([(c, n) for c, s in bench.items() for n in s], columns=["abbrev", "com_name"]),
                        on=["abbrev", "com_name"])
    audit = {}
    if a.audit_sealed_cache:
        sealed = set(json.load(open(a.audit_sealed_cache))["sealed_species"])
        tax = pd.read_csv(a.taxonomy)
        tax = tax[tax["CATEGORY"] == "species"]
        cmap_audit = dict(zip(tax["COMMON_NAME"], tax["SPECIES_CODE"]))
        resolve = lambda n: cmap_audit.get(n) or cmap_audit.get(n.split(" (")[0].strip())
        for label, selected in [("sealed", sealed), ("focal", {"houfin"})]:
            counts = sp_bench[sp_bench["com_name"].map(resolve).isin(selected)].groupby(["abbrev", "year"]).size()
            audit[label + "_benchmark_hits"] = np.array([counts.get((c, y), 0) for c, y in zip(rk["abbrev"], rk["year"])], dtype="int32")
            audit[label + "_benchmark_n"] = np.array([sum(resolve(n) in selected for n in bench.get(c, ())) for c in rk["abbrev"]], dtype="int32")
    bh = sp_bench.groupby(["abbrev", "year"]).size().rename("bench_hit").reset_index()
    rk = rk.merge(ll, on=["abbrev", "year"], how="left").merge(bh, on=["abbrev", "year"], how="left")
    rk["bench_n"] = [len(bench.get(c, ())) for c in rk["abbrev"]]
    rk = rk.sort_values("row")
    # in_bench for modelled species: a species is in the set if any of its raw names is (base common name match)
    names = {}
    raw_names = sp[["com_name"]].drop_duplicates()["com_name"]
    t = pd.read_csv(a.taxonomy, usecols=["COMMON_NAME", "SPECIES_CODE", "CATEGORY"])
    t = t[t["CATEGORY"] == "species"]
    cmap = dict(zip(t["COMMON_NAME"], t["SPECIES_CODE"]))
    for n in raw_names:
        code = cmap.get(n) or cmap.get(n.split(" (")[0].strip())
        if code:
            names.setdefault(code, set()).add(n)
    in_bench = np.zeros((len(circles), len(species)), dtype=bool)
    for c, s in bench.items():
        for j, code in enumerate(species):
            if names.get(code, set()) & s:
                in_bench[ci[c], j] = True
    np.savez_compressed(a.out, list_len=rk["list_len"].fillna(0).to_numpy(np.float32),
                        bench_hit=rk["bench_hit"].fillna(0).to_numpy(np.float32),
                        bench_n=rk["bench_n"].to_numpy(np.float32), in_bench=in_bench, **audit)
    with open(a.out + ".provenance.json", "w") as fh:
        json.dump({"bench_from": a.bench_from, "bench_share": a.bench_share,
                   "species_allowlist": a.species_allowlist, "records_before": before,
                   "records_after": len(sp), "prepared": a.prepared,
                   "audit_sealed_cache": a.audit_sealed_cache}, fh, indent=2)
    era = np.select([P["year_end"] < 1920, P["year_end"] < 1940, P["year_end"] < 1966], [0, 1, 2], 3)
    comp = np.where(rk["bench_n"] > 0, rk["bench_hit"].fillna(0) / rk["bench_n"].clip(lower=1), np.nan)
    print(f"[E065] benchmark sets: {len(bench)} circles ({borrowed} borrowed); modelled species mapped to names: "
          f"{len(set(names) & set(species))} of {len(species)}; median benchmark size {int(np.median(rk['bench_n'][rk['bench_n'] > 0]))}")
    for e, lab in enumerate(["1901-19", "1920-39", "1940-65", "1966+"]):
        m = era == e
        print(f"  {lab}: median list length {np.nanmedian(rk['list_len'].to_numpy()[m]):.0f}, median benchmark "
              f"completeness {np.nanmedian(comp[m]):.2f}")


if __name__ == "__main__":
    main()
