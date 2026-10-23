"""E085 summary: sum-of-Ruzickas readout (z + head blocks) vs z-only readout on the SAME trained models, by residency.

    python research/exp/e085_summary.py --root <job out dir> --similarity species_similarity.csv --out DIR

Reads fast_arm.py outputs <root>/<model>/{iso,sum}/per_species.csv (sum = --head-ckpt). Per model and species class
(SEDENTARY primary; residents; migratory): medians over species of spatial level skill (level_space_vs_intercept),
temporal level skill vs no-change (level_time_vs_no_change), change skill (change_skill_time) and attenuation slope
(attenuation_time, resolvable species), for both readouts; paired difference sum - iso per species, median, 95%
bootstrap over species. Controls (z only) are listed for reference.
"""
import argparse
import glob
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True); ap.add_argument("--similarity", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import pandas as pd
    sim = pd.read_csv(a.similarity); mig = dict(zip(sim.species_code, sim.migration))
    cols = {"space": "level_space_vs_intercept", "time_level": "level_time_vs_no_change",
            "change": "change_skill_time", "attenuation": "attenuation_time"}
    res = {}
    for mdir in sorted(glob.glob(os.path.join(a.root, "*"))):
        name = os.path.basename(mdir)
        iso_f, sum_f = os.path.join(mdir, "iso", "per_species.csv"), os.path.join(mdir, "sum", "per_species.csv")
        if not os.path.exists(iso_f):
            continue
        A = pd.read_csv(iso_f).set_index("species")
        Bm = pd.read_csv(sum_f).set_index("species") if os.path.exists(sum_f) else None
        out = {}
        if Bm is not None:
            amps = json.load(open(os.path.join(mdir, "sum", "results.json")))["block_amplitude_medians"]
            out["block_amplitude_medians"] = amps
        for cname, want in (("sedentary", {1.0}), ("residents", {1.0, 2.0}), ("migratory", {3.0})):
            sp = [s for s in A.index if mig.get(s) in want]
            r = {"species": len(sp)}
            for k, c in cols.items():
                ok = A.loc[sp, "resolvable_time"].astype(bool) if k in ("change", "attenuation") else pd.Series(True, index=sp)
                s_ok = [s for s in sp if ok[s]]
                r[f"iso_{k}"] = float(np.nanmedian(A.loc[s_ok, c]))
                if Bm is not None:
                    d = (Bm.loc[s_ok, c] - A.loc[s_ok, c]).to_numpy()
                    rng = np.random.default_rng(85)
                    bs = [np.nanmedian(d[rng.integers(0, len(d), len(d))]) for _ in range(2000)]
                    r[f"sum_{k}"] = float(np.nanmedian(Bm.loc[s_ok, c]))
                    r[f"d_{k}"] = [float(np.nanmedian(d))] + [float(x) for x in np.nanpercentile(bs, [2.5, 97.5])]
            out[cname] = r
        res[name] = out
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "e085.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
