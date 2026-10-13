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
    ap.add_argument("--species-filter", default=None,
                    help="E067: a file of species codes (one per line); strata use only these (an evaluation half)")
    ap.add_argument("--contrast", action="append", default=[],
                    help="E067: RUN_A:RUN_B (basenames). Paired per-draw difference A - B of the DESK readout (change) "
                         "and of the pooled level readout, per stratum. Valid when both runs share the same draws "
                         "(same held-out cells, same seed)")
    ap.add_argument("--level", action="store_true",
                    help="E059h: also stratify SPATIAL level (species_level<tag>.npz: DESK pooled readout vs linear map)")
    ap.add_argument("--seed-summary", action="store_true",
                    help="E059h: mean and SD over runs named s<k>_t<T>[_late] / ens4_t<T>[_late] of DESK - strong per "
                         "stratum, by T0 and state")
    a = ap.parse_args()
    import pandas as pd
    sim = pd.read_csv(a.similarity)
    keep = None
    if a.species_filter:
        keep = set(l.strip() for l in open(a.species_filter) if l.strip())
    res, draws = {}, {}
    for run in a.runs:
        z = np.load(os.path.join(run, f"species_components{a.tag}.npz"))
        cols = [str(c) for c in z["columns"]]
        St = strata(sim, cols)
        if keep is not None:
            kmask = np.array([c in keep for c in cols])
            St = {k: v & kmask for k, v in St.items()}
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
            draws.setdefault(os.path.basename(os.path.normpath(run)), {})[sname] = {"change": B["desk_within"]}
            ci = lambda v: [float(np.nanpercentile(v[1:], q)) for q in (2.5, 97.5)]
            o = {"n_species": int(m.sum()), "arms": {k: {"corr": float(v[0]), "ci": ci(v)} for k, v in B.items()}}
            for x, y in (("desk_within", "strong"), ("desk_within", "placebo_300"), ("strong+desk_within", "strong"),
                         ("desk_within", "desk_ref_within"), ("desk_within", "desk_pooled")):
                if x in B and y in B:
                    d = B[x] - B[y]
                    o[f"{x} - {y}"] = {"diff": float(d[0]), "ci": ci(d)}
            out[sname] = o
        if a.level and os.path.exists(os.path.join(run, f"species_level{a.tag}.npz")):
            zl = np.load(os.path.join(run, f"species_level{a.tag}.npz"))
            vtl = zl["vt"].astype("float64")
            lc = lambda nm, m: (zl[f"{nm}__cv"][:, m].astype("float64").sum(1)
                                / np.sqrt(zl[f"{nm}__vp"][:, m].astype("float64").sum(1) * vtl[:, m].sum(1)))
            for sname, m in St.items():
                if sname in out:
                    dp, pr = lc("desk_pooled", m), lc("pooled_ridge", m)
                    draws[os.path.basename(os.path.normpath(run))][sname]["level"] = dp
                    ci = lambda v: [float(np.nanpercentile(v[1:], q)) for q in (2.5, 97.5)]
                    out[sname]["level"] = {"desk_pooled": float(dp[0]), "pooled_ridge": float(pr[0]),
                                           "desk - linear": {"diff": float((dp - pr)[0]), "ci": ci(dp - pr)}}
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
            if "level" in o:
                line += f" | LEVEL desk {o['level']['desk_pooled']:.3f} lin {o['level']['pooled_ridge']:.3f} " \
                        f"{f_(o['level']['desk - linear'])}"
            print(line, flush=True)
    if a.contrast:
        print("== CONTRASTS (paired per draw): A - B")
        res["contrasts"] = {}
        for ctr in a.contrast:
            A_, B_ = ctr.split(":")
            if A_ not in draws or B_ not in draws:
                print(f"  {ctr}: missing run")
                continue
            for sname in draws[A_]:
                if sname not in draws[B_]:
                    continue
                o = {}
                for what in ("change", "level"):
                    if what in draws[A_][sname] and what in draws[B_][sname]:
                        d = draws[A_][sname][what] - draws[B_][sname][what]
                        o[what] = {"diff": float(d[0]), "ci": [float(np.nanpercentile(d[1:], q)) for q in (2.5, 97.5)]}
                res["contrasts"].setdefault(ctr, {})[sname] = o
                txt = " | ".join(f"{w} {v['diff']:+.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]" for w, v in o.items())
                print(f"  {ctr:34s} {sname:11s} {txt}")
    if a.seed_summary:
        import re
        groups = defaultdict(list)
        for name, out in res.items():
            mm = re.match(r"(s\d+|ens4)_t(\d{4})(_late)?$", name)
            if not mm:
                continue
            kind = "ensemble" if mm.group(1) == "ens4" else "single"
            groups[(mm.group(2), "late" if mm.group(3) else "selected", kind)].append(out)
        summ = {}
        print("== SEED SUMMARY (DESK - strong, change; DESK - linear, level): mean +- SD over runs")
        for key in sorted(groups):
            outs = groups[key]
            for sname in outs[0]:
                d = [o[sname]["desk_within - strong"]["diff"] for o in outs if sname in o]
                lv = [o[sname]["level"]["desk - linear"]["diff"] for o in outs if sname in o and "level" in o[sname]]
                summ.setdefault("|".join(key), {})[sname] = {"n_runs": len(d), "change_mean": float(np.mean(d)),
                                                              "change_sd": float(np.std(d)),
                                                              "level_mean": float(np.mean(lv)) if lv else None}
                print(f"  T0 {key[0]} {key[1]:8s} {key[2]:8s} {sname:11s} runs {len(d)}: change {np.mean(d):+.3f} "
                      f"+- {np.std(d):.3f}" + (f" | level {np.mean(lv):+.3f}" if lv else ""))
        res["seed_summary"] = summ
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
