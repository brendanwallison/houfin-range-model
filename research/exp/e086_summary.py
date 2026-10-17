"""E086 summary: readouts of the same model compared per species against its z-r24 iso readout, by residency.

    python research/exp/e086_summary.py --model NAME=readout:DIR[,readout:DIR...] [--model ...]
        --ref iso --similarity species_similarity.csv --out DIR

Each DIR is a fast_arm.py output (per_species.csv). For each model and residency class (SEDENTARY primary,
residents, migratory): median over species of spatial level skill, temporal level skill vs no-change, change skill and
attenuation slope (change / attenuation on resolvable species) per readout, and the paired per-species difference
readout - ref (median, 95% bootstrap over species). Also prints the mean over models of each paired median.
"""
import argparse
import json
import os

import numpy as np

COLS = {"space": "level_space_vs_intercept", "time_level": "level_time_vs_no_change",
        "change": "change_skill_time", "attenuation": "attenuation_time"}
CLASSES = (("sedentary", {1.0}), ("residents", {1.0, 2.0}), ("migratory", {3.0}))


def compare(A, B, sp, k, seed=86):
    """Median of B - A over species (resolvable ones for change / attenuation) with a 95% bootstrap."""
    c = COLS[k]
    ok = [s for s in sp if (k not in ("change", "attenuation")) or bool(A.loc[s, "resolvable_time"])]
    d = (B.loc[ok, c] - A.loc[ok, c]).to_numpy(float)
    rng = np.random.default_rng(seed)
    bs = [np.nanmedian(d[rng.integers(0, len(d), len(d))]) for _ in range(2000)]
    return float(np.nanmedian(B.loc[ok, c])), [float(np.nanmedian(d))] + [float(x) for x in np.nanpercentile(bs, [2.5, 97.5])]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--ref", default="iso"); ap.add_argument("--similarity", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import pandas as pd
    sim = pd.read_csv(a.similarity); mig = dict(zip(sim.species_code, sim.migration))
    res = {}
    for m in a.model:
        name, spec = m.split("=", 1)
        dirs = dict(x.split(":", 1) for x in spec.split(","))
        tabs = {r: pd.read_csv(os.path.join(d, "per_species.csv")).set_index("species") for r, d in dirs.items()
                if os.path.exists(os.path.join(d, "per_species.csv"))}
        if a.ref not in tabs:
            print(f"[e086-summary] {name}: no ref readout; skipped"); continue
        A = tabs[a.ref]
        out = {}
        for cname, want in CLASSES:
            sp = [s for s in A.index if mig.get(s) in want]
            r = {"species": len(sp)}
            for rd, B in tabs.items():
                common = [s for s in sp if s in B.index]
                for k in COLS:
                    lvl, d = compare(A, B, common, k)
                    r[f"{rd}_{k}"] = lvl
                    if rd != a.ref:
                        r[f"d_{rd}_{k}"] = d
            out[cname] = r
        res[name] = out
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "e086_summary.json"), "w"), indent=1)
    rds = sorted({k[2:-len("_change")] for v in res.values() for k in v["sedentary"] if k.startswith("d_") and k.endswith("_change")})
    for cname, _ in CLASSES:
        print(f"== {cname}: paired median (readout - {a.ref}) per model; space / change / attenuation")
        for rd in rds:
            row = []
            for name, v in res.items():
                r = v[cname]
                if f"d_{rd}_change" in r:
                    row.append((name, r[f"d_{rd}_space"][0], r[f"d_{rd}_change"][0], r[f"d_{rd}_attenuation"][0]))
            if not row:
                continue
            ms = np.mean([x[1] for x in row]); mc = np.mean([x[2] for x in row]); ma = np.mean([x[3] for x in row])
            npos = sum(x[2] > 0 for x in row)
            print(f"  {rd:18s} mean space {ms:+.4f}  change {mc:+.4f} ({npos}/{len(row)} > 0)  atten {ma:+.4f}   "
                  + "  ".join(f"{x[0]}:{x[2]:+.4f}" for x in row))


if __name__ == "__main__":
    main()
