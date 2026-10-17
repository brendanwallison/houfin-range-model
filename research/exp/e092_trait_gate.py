"""E092 stage 0 (gate): do AVONET traits predict each species' fitted readout prior shape?

    python research/exp/e092_trait_gate.py --run NAME=<fast_arm split out dir> [--run ...]
        --avonet <AVONET1_BirdLife.csv> --taxonomy <eBird_taxonomy.csv> --out <dir> [--perms 500]

Per species (fast_arm --kernel split: block 0 = the cell's reference-window mean z, BETWEEN places; block 1 = the
deviation from it, WITHIN places), targets: log a_between, log a_within, log(a_within / a_between). Traits (AVONET,
BirdLife taxonomy via the eBird scientific name): log mass, hand-wing index, log beak length / width / depth, log tarsus,
log tail, Habitat, Habitat.Density, Migration, Trophic.Niche, Primary.Lifestyle (one-hot). Base covariate: log
prevalence (rare species have small, poorly fitted amplitudes, and prevalence covaries with body size).
Score: 10-fold CV R^2 of ridge (lambda by inner CV) for base = [log prevalence] and full = [log prevalence + traits];
TRAIT GAIN = R2(full) - R2(base); null = the same gain with trait rows permuted across species (--perms), so the gain
must beat what any 30-odd random features would add. Reported per model; and each species' out-of-fold residual
for the sedentary stratum. Seed-to-seed agreement of the per-species amplitudes (corr across models of the same start
year) is the stability ceiling.
"""
import argparse
import json
import os

import numpy as np

NUM = ["Mass", "Hand-Wing.Index", "Beak.Length_Culmen", "Beak.Width", "Beak.Depth", "Tarsus.Length", "Tail.Length"]
CAT = ["Habitat", "Habitat.Density", "Migration", "Trophic.Niche", "Primary.Lifestyle"]


def ridge_cv_r2(X, y, folds, lams=(1e-2, 1e-1, 1, 10, 100, 1e3)):
    """Out-of-fold R^2 with lambda chosen by inner CV on each training fold. Columns standardized on the train fold."""
    pred = np.zeros_like(y)
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Xt, Xe = (X[tr] - mu) / sd, (X[te] - mu) / sd
        ym = y[tr].mean()
        inner = folds[tr]
        best, bl = np.inf, lams[0]
        for lam in lams:
            err = 0.0
            for g in np.unique(inner):
                a, b = inner != g, inner == g
                w = np.linalg.solve(Xt[a].T @ Xt[a] + lam * np.eye(X.shape[1]), Xt[a].T @ (y[tr][a] - y[tr][a].mean()))
                err += ((y[tr][b] - y[tr][a].mean() - Xt[b] @ w) ** 2).sum()
            if err < best:
                best, bl = err, lam
        w = np.linalg.solve(Xt.T @ Xt + bl * np.eye(X.shape[1]), Xt.T @ (y[tr] - ym))
        pred[te] = ym + Xe @ w
    return 1.0 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum(), pred


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--avonet", required=True); ap.add_argument("--taxonomy", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--perms", type=int, default=500)
    a = ap.parse_args()
    import pandas as pd
    tax = pd.read_csv(a.taxonomy).set_index("SPECIES_CODE")["SCIENTIFIC_NAME"]
    av = pd.read_csv(a.avonet, encoding="latin-1").set_index("Species1")
    res, amps = {}, {}
    for r in a.run:
        name, d = r.split("=", 1)
        P = pd.read_csv(os.path.join(d, "per_species.csv"))
        P = P[P.fit_ok.astype(bool) & (P.amp_b0 > 0) & (P.amp_b1 > 0)].copy()
        P["sci"] = P.species.map(tax)
        P = P[P.sci.isin(av.index)]
        T = av.loc[P.sci]
        num = np.column_stack([np.log(T[c].astype(float).to_numpy()) if c != "Hand-Wing.Index"
                               else T[c].astype(float).to_numpy() for c in NUM])
        cat = pd.get_dummies(T[CAT].astype(str), drop_first=False).to_numpy(float)
        F = np.hstack([num, cat])
        okr = np.isfinite(F).all(1)
        P, F, T = P[okr], F[okr], T[okr]
        base = np.log(P.prevalence.to_numpy())[:, None]
        mig = T["Migration"].to_numpy(float)
        targets = {"log_between": np.log(P.amp_b0.to_numpy()), "log_within": np.log(P.amp_b1.to_numpy())}
        targets["log_ratio"] = targets["log_within"] - targets["log_between"]
        rng = np.random.default_rng(92)
        folds = rng.permutation(np.arange(len(P)) % 10)
        out = {"n_species": int(len(P)), "n_sedentary": int((mig == 1).sum())}
        for k, y in targets.items():
            r2b, _ = ridge_cv_r2(base, y, folds)
            r2f, pf = ridge_cv_r2(np.hstack([base, F]), y, folds)
            gain = r2f - r2b
            null = []
            for _ in range(a.perms):
                Fp = F[rng.permutation(len(F))]
                null.append(ridge_cv_r2(np.hstack([base, Fp]), y, folds)[0] - r2b)
            null = np.array(null)
            sed = mig == 1
            out[k] = {"r2_base": float(r2b), "r2_full": float(r2f), "trait_gain": float(gain),
                      "null_q50": float(np.median(null)), "null_q95": float(np.quantile(null, 0.95)),
                      "p_perm": float((null >= gain).mean()),
                      "sd_target": float(y.std()),
                      "sedentary_mean_resid": float((y - pf)[sed].mean()) if sed.any() else None,
                      "migration_means": {str(int(m)): float(y[mig == m].mean()) for m in np.unique(mig)}}
        res[name] = out
        amps[name] = P.set_index("species")[["amp_b0", "amp_b1"]]
    # seed-to-seed agreement of per-species log amplitudes (same start year)
    agree = {}
    names = list(amps)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            A, B = amps[names[i]], amps[names[j]]
            common = A.index.intersection(B.index)
            agree[f"{names[i]}~{names[j]}"] = {
                k: float(np.corrcoef(np.log(A.loc[common, c]), np.log(B.loc[common, c]))[0, 1])
                for k, c in (("between", "amp_b0"), ("within", "amp_b1"))}
    res["_seed_agreement"] = agree
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "e092_gate.json"), "w"), indent=1)
    for name, o in res.items():
        if name.startswith("_"):
            continue
        print(f"== {name}: {o['n_species']} species ({o['n_sedentary']} sedentary)")
        for k in ("log_between", "log_within", "log_ratio"):
            t = o[k]
            print(f"   {k:12s} R2 base {t['r2_base']:+.3f} full {t['r2_full']:+.3f} | trait gain {t['trait_gain']:+.3f} "
                  f"(perm null q50 {t['null_q50']:+.3f} q95 {t['null_q95']:+.3f}, p {t['p_perm']:.3f}) | "
                  f"by migration {t['migration_means']}")
    print("seed agreement (corr of per-species log amplitude):", json.dumps(agree))


if __name__ == "__main__":
    main()
