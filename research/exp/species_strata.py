"""E059g: score the out-of-community species target by STRATUM of similarity to the focal species (House Finch).

    python research/exp/species_strata.py --similarity <species_similarity.csv> --runs <fair_rivals out dir> ...
        [--tag _p96dev] [--out <json>]

Reads each run's species_components<tag>.npz (fair_rivals --save-species; per-species cv / vp / vt for every bootstrap
draw) and pools them over the species of each stratum: pooled noise-free corr = sum cv / sqrt(sum vp * sum vt), per
draw, so CIs and paired differences use the run's own draws. Placebo / strong arms are the mean over their placebo
draws, as in fair_rivals.
Strata (pre-registered in E059.md before any stratified number existed; terciles of the DEV species' own distribution):
    all           every dev species
    phylo_near / phylo_far     lowest / highest tercile of patristic distance to the focal
    morph_near / morph_far     lowest / highest tercile of AVONET trait distance to the focal
    sedentary / migratory      AVONET migration 1 / 3
    hofi_like     phylo <= median AND trait <= median AND migration <= 2 (close on both selection axes, not a migrant)
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np


def strata(sim, cols):
    import pandas as pd
    d = sim[sim["group"] == "dev"].set_index("species_code").reindex(list(cols))
    q = lambda c, p: d[c].quantile(p)
    S = {"all": np.ones(len(d), bool),
         "phylo_near": (d.phylo_distance <= q("phylo_distance", 1 / 3)).values,
         "phylo_far": (d.phylo_distance >= q("phylo_distance", 2 / 3)).values,
         "morph_near": (d.trait_distance <= q("trait_distance", 1 / 3)).values,
         "morph_far": (d.trait_distance >= q("trait_distance", 2 / 3)).values,
         "sedentary": (d.migration == 1).values,
         "migratory": (d.migration == 3).values,
         "hofi_like": ((d.phylo_distance <= q("phylo_distance", 0.5)) & (d.trait_distance <= q("trait_distance", 0.5))
                       & (d.migration <= 2)).values}
    return {k: np.asarray(v, bool) for k, v in S.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--similarity", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--tag", default="_p96dev")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import pandas as pd
    sim = pd.read_csv(a.similarity)
    res = {}
    for run in a.runs:
        z = np.load(os.path.join(run, f"species_components{a.tag}.npz"))
        cols = [str(c) for c in z["columns"]]
        St = strata(sim, cols)
        vt = z["vt"].astype("float64")
        arms = defaultdict(list)
        for k in z.files:
            if k.endswith("__cv"):
                arms[k[:-4].split("#")[0]].append(k[:-4])

        def corr(member, m):
            cv, vp = z[f"{member}__cv"][:, m].astype("float64"), z[f"{member}__vp"][:, m].astype("float64")
            return cv.sum(1) / np.sqrt(vp.sum(1) * vt[:, m].sum(1))
        out = {}
        for sname, m in St.items():
            if m.sum() < 5:
                continue
            B = {arm: np.mean([corr(mb, m) for mb in mbs], axis=0) for arm, mbs in arms.items()}
            ci = lambda v: [float(np.nanpercentile(v[1:], q)) for q in (2.5, 97.5)]
            o = {"n_species": int(m.sum()), "arms": {k: {"corr": float(v[0]), "ci": ci(v)} for k, v in B.items()}}
            for x, y in (("desk_within", "strong"), ("desk_within", "placebo_300"), ("strong+desk_within", "strong"),
                         ("desk_within", "desk_ref_within"), ("desk_within", "desk_pooled")):
                if x in B and y in B:
                    d = B[x] - B[y]
                    o[f"{x} - {y}"] = {"diff": float(d[0]), "ci": ci(d)}
            out[sname] = o
        res[os.path.basename(os.path.normpath(run))] = out
        print(f"== {os.path.basename(os.path.normpath(run))}")
        f_ = lambda d: f"{d['diff']:+.3f} [{d['ci'][0]:+.3f},{d['ci'][1]:+.3f}]"
        for sname, o in out.items():
            A = o["arms"]
            line = (f"  {sname:11s} n {o['n_species']:3d} | desk {A['desk_within']['corr']:.3f} placebo "
                    f"{A['placebo_300']['corr']:.3f} strong {A['strong']['corr']:.3f} | desk-strong {f_(o['desk_within - strong'])}"
                    f" | +desk beyond strong {f_(o['strong+desk_within - strong'])}")
            if "desk_within - desk_ref_within" in o:
                line += f" | vs ref {f_(o['desk_within - desk_ref_within'])}"
            print(line, flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
