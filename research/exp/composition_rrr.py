"""E067: does the TRAINING COMMUNITY's composition (migrants in or out) change how well a representation serves
SEDENTARY out-of-community species? Cheap reduced-rank covariate surrogates of DESK, one per training set.

    python research/exp/composition_rrr.py --cache <base research cache> --similarity <species_similarity.csv>
        --out-root <dir> [--rank 24] [--seed 7]

The dev species are split at random, within each migration class, into half A (may join a training set) and half B
(never trained on; the evaluation set, written to <out-root>/eval_species_B.txt). Training sets:
    C96        the current reference community (96)
    Cres       its sedentary + partial species only (AVONET migration 1-2)
    C96_migA   the 96 + the full migrants of half A
    C96_A      the 96 + all of half A
    Cres_resA  the community residents + the residents (1-2) of half A (a broader RESIDENT community)
Representation: reduced-rank ridge from the standardized covariates (F_cov, the cache's input-EMA covariates) to
log1p abundance of the training species. It is fitted on TRAINING cells x TRAINED years only (as DESK is); the rank-k
directions are the top right singular vectors of the fitted values. z = X B V_k at every key.
Each set becomes a surrogate cache <out-root>/<set>/: symlinks to the base cache's files plus its own z_raw.npy and
meta.json. Score it with fair_rivals --z-keys-only --target dev, then species_strata --species-filter
eval_species_B.txt. E039g showed a bird-supervised linear projection nearly matches DESK spatially, so composition
effects found here are worth a DESK retrain to confirm. The base cache is only read; the sealed dir is not linked.
"""
import argparse
import json
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
import sys  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--similarity", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--rank", type=int, default=24)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    import pandas as pd
    from src.data.preprocess.bbs_community import log1p_community
    c = lambda f: os.path.join(a.cache, f)
    meta = json.load(open(c("meta.json"), encoding="utf-8"))
    comm, dev = list(meta["community"]), list(meta["dev_species"])
    sim = pd.read_csv(a.similarity).set_index("species_code")
    mig = lambda codes: np.array([sim["migration"].get(x, np.nan) for x in codes], "float64")
    mc, md = mig(comm), mig(dev)
    rng = np.random.default_rng(a.seed)
    inA = np.zeros(len(dev), bool)
    for cls in (1.0, 2.0, 3.0, np.nan):
        idx = np.where(np.isnan(md) if np.isnan(cls) else md == cls)[0]
        rng.shuffle(idx)
        inA[idx[: len(idx) // 2]] = True
    os.makedirs(a.out_root, exist_ok=True)
    with open(os.path.join(a.out_root, "eval_species_B.txt"), "w") as fh:
        fh.write("\n".join(d for d, x in zip(dev, inA) if not x) + "\n")
    res_c = (mc <= 2)
    sets = {"C96": (np.ones(len(comm), bool), np.zeros(len(dev), bool)),
            "Cres": (res_c, np.zeros(len(dev), bool)),
            "C96_migA": (np.ones(len(comm), bool), inA & (md == 3)),
            "C96_A": (np.ones(len(comm), bool), inA),
            "Cres_resA": (res_c, inA & (md <= 2))}
    keys = np.load(c("keys.npy"))
    split = np.load(c("split.npz"))
    withheld = set(int(y) for y in split["withheld"])
    ho = split["holdout"][keys[:, 0], keys[:, 1]]
    bf = split["buffer"][keys[:, 0], keys[:, 1]]
    ty = ~np.isin(keys[:, 2], list(withheld))
    F = np.asarray(np.load(c("F_cov.npy"), mmap_mode="r"), "float64")
    fin = np.isfinite(F).all(1)
    trn = np.where(~ho & ~bf & ty & fin)[0]
    mu, sd = F[trn].mean(0), F[trn].std(0)
    keepc = sd > 1e-9
    X = np.where(np.isfinite(F[:, keepc]), (F[:, keepc] - mu[keepc]) / sd[keepc], 0.0)
    Xt = X[trn]
    lam = 1e-2 * np.trace(Xt.T @ Xt) / Xt.shape[1]
    G = np.linalg.cholesky(Xt.T @ Xt + lam * np.eye(Xt.shape[1]))
    Yall = {"comm": np.asarray(log1p_community(np.load(c("X_comm.npy")).astype("float64")), "float64"),
            "dev": np.asarray(log1p_community(np.load(c("X_dev.npy")).astype("float64")), "float64")}
    summary = {"base_cache": a.cache, "rank": a.rank, "seed": a.seed, "n_train_rows": int(len(trn)),
               "n_eval_B": int((~inA).sum()), "eval_B_by_migration": {}, "sets": {}}
    for cls in (1.0, 2.0, 3.0):
        summary["eval_B_by_migration"][str(int(cls))] = int(((~inA) & (md == cls)).sum())
    base_files = [f for f in os.listdir(a.cache) if f not in ("z_raw.npy", "z_ema.npy", "z_raw_all.npy",
                                                              "z_ema_all.npy", "meta.json", "sealed")]
    for name, (cm, dm) in sets.items():
        Y = np.hstack([Yall["comm"][:, cm], Yall["dev"][:, dm]])
        Yt = Y[trn] - Y[trn].mean(0)
        Bm = np.linalg.solve(G.T, np.linalg.solve(G, Xt.T @ Yt))          # ridge, all species at once
        _, s_, Vt = np.linalg.svd(Xt @ Bm, full_matrices=False)
        Bk = Bm @ Vt[: a.rank].T
        z = ((X - X[trn].mean(0)) @ Bk).astype("float32")
        d = os.path.join(a.out_root, name)
        os.makedirs(d, exist_ok=True)
        for f in base_files:
            dst = os.path.join(d, f)
            if not os.path.lexists(dst):
                os.symlink(os.path.join(os.path.abspath(a.cache), f), dst)
        np.save(os.path.join(d, "z_raw.npy"), z)
        m2 = dict(meta)
        m2.update({"latent_dim": a.rank, "surrogate": {"kind": "reduced-rank ridge (E067)", "set": name,
                                                       "n_comm": int(cm.sum()), "n_dev_A": int(dm.sum()),
                                                       "rank": a.rank, "explained_top_k": float((s_[: a.rank] ** 2).sum()
                                                                                                / (s_ ** 2).sum())}})
        json.dump(m2, open(os.path.join(d, "meta.json"), "w", encoding="utf-8"), indent=1)
        summary["sets"][name] = m2["surrogate"]
        print(f"[composition-rrr] {name}: {int(cm.sum())} community + {int(dm.sum())} dev-A species -> rank {a.rank}; "
              f"top-{a.rank} share of fitted variance {m2['surrogate']['explained_top_k']:.3f}", flush=True)
    json.dump(summary, open(os.path.join(a.out_root, "composition_summary.json"), "w"), indent=1)
    print(f"[composition-rrr] eval half B: {summary['n_eval_B']} dev species, by migration {summary['eval_B_by_migration']}")


if __name__ == "__main__":
    main()
