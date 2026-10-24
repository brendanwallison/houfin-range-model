"""E086ab data step: K SPECIES FOLDS for the all-species basis -- per fold, the kernel species are all non-sealed species
OUTSIDE the fold; the fold's species are the evaluation targets (scored by a basis / DESK that never saw them).

    python research/exp/e086ab_folds.py --cache <tempho cache> --similarity species_similarity.csv
        --persist winter_persist.csv --species community_all_nonsealed.csv --bbs-points <all369 points dir>
        --cbc cbc_circles_all.npz --out-root <dir> [--k 3]

Folds: stratified by SEDENTARY COMMON (corrected rule: AVONET migration 1 AND winter persistence >= 0.5 where measured;
BBS prevalence >= 0.08) vs the rest, seeded. For fold f writes <out-root>/fold{f}/:
  community.csv           kernel species (species_code), in all-species order minus fold f -> trend.community_trend_list
  bbs_points/             the all-species BBS points with fold f's columns removed (X_points, point_index,
                          point_weights, points_meta.json with the reduced species list)
  cbc_winter_half.npz     CBC winter cell-years (1966-2024, party-hours^0.8) of the kernel species, raw counts
                          (year, row, col, X, species) for the identity winter-half metric head
  fold_species.json       the fold's target species and strata
"""
import argparse
import json
import os
import shutil

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for k in ("--cache", "--similarity", "--persist", "--species", "--bbs-points", "--cbc", "--out-root"):
        ap.add_argument(k, required=True)
    ap.add_argument("--k", type=int, default=3); ap.add_argument("--seed", type=int, default=86)
    a = ap.parse_args()
    import pandas as pd
    codes = [str(c).lower() for c in pd.read_csv(a.species)["species_code"]]
    meta = json.load(open(os.path.join(a.cache, "meta.json")))
    assert "houfin" not in codes and not set(codes) & set(meta["sealed_species"])
    names = list(meta["community"]) + list(meta["dev_species"])
    X = np.concatenate([np.load(os.path.join(a.cache, "X_comm.npy")),
                        np.asarray(np.load(os.path.join(a.cache, "X_dev.npy"), mmap_mode="r"))], 1)
    prev = dict(zip(names, (X > 0).mean(0)))
    sim = pd.read_csv(a.similarity); mig = dict(zip(sim.species_code, sim.migration))
    per = pd.read_csv(a.persist); wp = dict(zip(per.sp, per.winter_persist))
    sedentary = {s for s in codes if mig.get(s) == 1.0 and (s not in wp or wp[s] >= 0.5)}
    sc = {s for s in sedentary if prev.get(s, 0) >= 0.08}
    rng = np.random.default_rng(a.seed); fold = {}
    for grp in (sorted(sc), sorted(set(codes) - sc)):
        for i, s in enumerate(rng.permutation(grp)):
            fold[s] = i % a.k
    P = os.path.join
    Xp = np.load(P(a.bbs_points, "X_points.npy"), mmap_mode="r")
    pm = json.load(open(P(a.bbs_points, "points_meta.json")))
    psp = [str(s).lower() for s in pm["species"]]
    assert psp == codes, "points columns must be the all-species list order"
    D = np.load(a.cbc, allow_pickle=False); cs = list(D["species"])
    ph, yr = D["party_hours"].astype(float), D["year_end"]
    okc = (D["row"] >= 0) & (yr >= 1966) & (yr <= 2024) & np.isfinite(ph) & (ph > 0)
    for f in range(a.k):
        out = P(a.out_root, f"fold{f}"); os.makedirs(P(out, "bbs_points"), exist_ok=True)
        ks = [s for s in codes if fold[s] != f]; tg = [s for s in codes if fold[s] == f]
        pd.DataFrame({"species_code": ks}).to_csv(P(out, "community.csv"), index=False)
        col = [codes.index(s) for s in ks]
        np.save(P(out, "bbs_points", "X_points.npy"), np.asarray(Xp[:, col], np.float32))
        for fn in ("point_index.npy", "point_weights.npy"):
            shutil.copy(P(a.bbs_points, fn), P(out, "bbs_points", fn))
        m2 = dict(pm); m2["species"] = ks; m2["n_species"] = len(ks); m2["e086ab_fold"] = f
        if "bbs" in m2 and isinstance(m2["bbs"], dict):
            m2["bbs"] = dict(m2["bbs"]); m2["bbs"]["n_species"] = len(ks)
        json.dump(m2, open(P(out, "bbs_points", "points_meta.json"), "w"), indent=1)
        wk = [s for s in ks if s in cs]
        Y = D["Y"][okc][:, [cs.index(s) for s in wk]].astype(float) * ((np.median(ph[okc]) / ph[okc])[:, None] ** 0.8)
        df = pd.DataFrame(Y); df["r"], df["c"], df["y"] = D["row"][okc], D["col"][okc], yr[okc]
        g = df.groupby(["r", "c", "y"]).mean()
        Xw = g.to_numpy().astype(np.float32); nz = Xw.sum(1) > 0
        np.savez_compressed(P(out, "cbc_winter_half.npz"), year=np.array([q[2] for q in g.index], np.int32)[nz],
                            row=np.array([q[0] for q in g.index], np.int32)[nz], col=np.array([q[1] for q in g.index], np.int32)[nz],
                            X=Xw[nz], species=np.array(wk))
        json.dump({"fold": f, "k": a.k, "targets": tg, "kernel_species": len(ks), "winter_species": len(wk),
                   "targets_sed_common": sorted(set(tg) & sc), "targets_sedentary": sorted(set(tg) & sedentary),
                   "sedentary_rule": "AVONET migration 1 AND winter persistence >= 0.5 where measured"},
                  open(P(out, "fold_species.json"), "w"), indent=1)
        print(f"[e086ab] fold {f}: kernel {len(ks)} (winter {len(wk)}), targets {len(tg)} "
              f"(sedentary common {len(set(tg) & sc)}, sedentary {len(set(tg) & sedentary)}) -> {out}", flush=True)
    json.dump({"fold": fold, "sed_common": sorted(sc), "sedentary": sorted(sedentary)},
              open(P(a.out_root, "folds.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
