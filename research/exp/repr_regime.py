"""E039b: DESK features vs covariates under the age model's readout, in the regime it actually uses -- low rank, little
data -- by species prevalence.

    ESK_DESK_CONFIG=<overlay> python research/exp/repr_regime.py --cache <research cache> --out <dir>
    python research/exp/repr_regime.py --summarize <dir>

Readout: gp_kernels.fit / predict (the suite's desk arm: per-species BLR with a linear kernel, REML scales). Rows as
the suite's SPACE set: training cells x trained years -> held-out cells x trained years. Representations: raw z at 24
dims (what the age model reads) and 64; whitened principal components of the raw-matched covariates (F_cov_raw,
input-side EMA only) at 24, 64 and all channels (validate_gp_species.covariate_design). Training rows thinned at
random to each fraction (validate_gp_species.thin_rows). Metric: per-species level skill on held-out rows, 1 -
SSE / SSE(training mean), log1p counts; dev species only, by training-prevalence tier; the paired difference desk -
covariates per species (median and share of species favouring desk).
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

TIERS = [(0.0, 0.01), (0.01, 0.05), (0.05, 0.20), (0.20, 1.01)]


def skill(y, p, ybar):
    sse = ((y - p) ** 2).sum(0)
    sse0 = ((y - ybar) ** 2).sum(0)
    return np.where(sse0 > 0, 1.0 - sse / np.maximum(sse0, 1e-12), np.nan)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache")
    ap.add_argument("--out")
    ap.add_argument("--fracs", type=float, nargs="+", default=[1.0, 0.1, 0.03])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--bioclim-dir", default=None,
                    help="E039c: research/exp/bioclim.py output for this cache (19 BIOCLIM + 12 LUH classes, same EMA)")
    ap.add_argument("--summarize", default=None)
    a = ap.parse_args()
    if a.summarize:
        return summarize(a.summarize)
    from src.community_encoder.train_DESK import gp_kernels as gpk
    from src.community_encoder.train_DESK.validate_gp_species import covariate_design, thin_rows
    os.makedirs(a.out, exist_ok=True)
    c = lambda f: os.path.join(a.cache, f)
    keys = np.load(c("keys.npy"))
    split = np.load(c("split.npz"))
    wh = np.isin(keys[:, 2], np.asarray(split["withheld"], int)) if "withheld" in split.files else np.zeros(len(keys), bool)
    ho = split["holdout"][keys[:, 0], keys[:, 1]]
    bf = split["buffer"][keys[:, 0], keys[:, 1]]
    Zr = np.load(c("z_raw.npy")).astype("float64")
    F = np.asarray(np.load(c("F_cov_raw.npy"), mmap_mode="r"), "float64")
    Y = np.log1p(np.load(c("X_dev.npy")).astype("float64"))
    ok = np.isfinite(F).all(1) & np.isfinite(Zr).all(1)
    tr_all = np.where(~ho & ~bf & ~wh & ok)[0]
    te = np.where(ho & ~wh & ok)[0]
    prev = (Y[tr_all] > 0).mean(0)
    keep = prev > 0
    Y, prev = Y[:, keep], prev[keep]
    tier = np.digitize(prev, [t[1] for t in TIERS[:-1]])
    P_all = covariate_design(F, tr_all, F.shape[1])            # PCA on all training rows (unsupervised)
    reps = {"desk_raw_r24": Zr[:, :24], "desk_raw_r64": Zr[:, :64],
            "cov_pc24": P_all[:, :24], "cov_pc64": P_all[:, :64], "cov_pc_all": P_all}
    if a.bioclim_dir:
        # the standard SDM inputs: 19 BIOCLIM (+ the 12 LUH class fractions), standardized on training rows; matched-N
        # (19) versions of desk and the covariate PCs for an exact comparison
        def zs(M):
            M = np.asarray(M, "float64")
            mu, sd = np.nanmean(M[tr_all], 0), np.nanstd(M[tr_all], 0) + 1e-12
            return np.where(np.isfinite(M), (M - mu) / sd, 0.0)
        Bc = zs(np.load(os.path.join(a.bioclim_dir, "bioclim_ema.npy")))
        Lc = zs(np.load(os.path.join(a.bioclim_dir, "luh_classes_ema.npy")))
        reps.update({"desk_raw_r19": Zr[:, :19], "cov_pc19": P_all[:, :19], "bioclim19": Bc,
                     "bioclim19_luh12": np.column_stack([Bc, Lc])})
    res = {"cache": a.cache, "n_train_full": int(len(tr_all)), "n_test": int(len(te)), "n_species": int(keep.sum()),
           "tier_edges": TIERS, "tier_counts": [int((tier == k).sum()) for k in range(len(TIERS))],
           "n_cov_pcs_all": int(P_all.shape[1]), "runs": {}}
    for frac in a.fracs:
        for seed in (a.seeds if frac < 1.0 else a.seeds[:1]):
            tr = thin_rows(tr_all, frac, np.random.default_rng(seed)) if frac < 1.0 else tr_all
            ybar = Y[tr].mean(0)
            sk = {}
            for name, X in reps.items():
                m = gpk.fit(X[tr], Y[tr])
                sk[name] = skill(Y[te], gpk.predict(m, X[te])[0], ybar)
            res["runs"][f"{frac}_{seed}"] = {"frac": frac, "seed": seed, "n_train": int(len(tr)),
                                             "skill": {k: v.tolist() for k, v in sk.items()}}
            print(f"[regime] frac {frac} seed {seed}: {len(tr):,} training rows done", flush=True)
    res["tier"] = tier.tolist()
    with open(os.path.join(a.out, "repr_regime.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh)
    summarize(a.out)


def summarize(out):
    r = json.load(open(os.path.join(out, "repr_regime.json"), encoding="utf-8"))
    tier = np.asarray(r["tier"])
    first = next(iter(r["runs"].values()))["skill"]
    names = [n for n in ["desk_raw_r19", "desk_raw_r24", "desk_raw_r64", "cov_pc19", "cov_pc24", "cov_pc64", "cov_pc_all",
                         "bioclim19", "bioclim19_luh12"] if n in first]
    print(f"{os.path.basename(os.path.normpath(r['cache']))}: {r['n_species']} dev species, tiers (prevalence) "
          f"{[f'{lo:.0%}-{hi:.0%}' for lo, hi in r['tier_edges']]} n={r['tier_counts']}; covariate PCs (all) "
          f"{r['n_cov_pcs_all']}; held-out rows {r['n_test']:,}")
    fracs = sorted({v["frac"] for v in r["runs"].values()}, reverse=True)
    for frac in fracs:
        runs = [v for v in r["runs"].values() if v["frac"] == frac]
        S = {n: np.nanmean(np.array([v["skill"][n] for v in runs]), axis=0) for n in names}
        print(f"  training fraction {frac} ({runs[0]['n_train']:,} rows; {len(runs)} draw(s)) -- median level skill by "
              f"tier [all, <1%, 1-5%, 5-20%, >=20%]:")
        for n in names:
            row = [np.nanmedian(S[n])] + [np.nanmedian(S[n][tier == k]) for k in range(4)]
            print(f"    {n:14s} " + " ".join(f"{x:+.3f}" for x in row))
        for d, cmp_ in (("desk_raw_r24", "cov_pc24"), ("desk_raw_r24", "cov_pc64"), ("desk_raw_r24", "cov_pc_all"),
                        ("desk_raw_r64", "cov_pc64"), ("desk_raw_r19", "bioclim19"), ("cov_pc19", "bioclim19"),
                        ("desk_raw_r24", "bioclim19_luh12"), ("bioclim19_luh12", "bioclim19")):
            if d not in S or cmp_ not in S:
                continue
            D = S[d] - S[cmp_]
            row = [np.nanmedian(D[tier == k]) for k in range(4)]
            fav = [np.mean(D[(tier == k) & np.isfinite(D)] > 0) for k in range(4)]
            print(f"    {d} - {cmp_:10s} by tier " + " ".join(f"{x:+.3f}" for x in row)
                  + "  | share favouring desk " + " ".join(f"{x:.0%}" for x in fav))


if __name__ == "__main__":
    main()
