"""E034a: a COARSE-grain copy of a research cache (K x K cells merged) for grain tests.

    python research/exp/coarsen_cache.py --cache <dir> --out <dir> --k 3

Rows are merged by (row // K, col // K, year): reference-community counts, covariates (F_cov) and annual ESK
projections are averaged over the block's surveyed cells that year. The spatial split is coarsened by row share: a
coarse cell is HELD if at least half its rows are in held cells, BUFFER if it is not held but any of its rows is held
or buffer, and TRAIN otherwise -- so no coarse training cell contains held-out data. Writes keys.npy, split.npz,
X_comm.npy, F_cov.npy, esk_annual.npy and coarsen.json; span_reach.py reads it with --base-cache and --cell-km.
Sealed species are never read.
"""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    c = lambda f: os.path.join(a.cache, f)
    keys = np.load(c("keys.npy"))
    split = np.load(c("split.npz"))
    held_row = split["holdout"][keys[:, 0], keys[:, 1]]
    buf_row = split["buffer"][keys[:, 0], keys[:, 1]]
    K = a.k
    ck = np.stack([keys[:, 0] // K, keys[:, 1] // K, keys[:, 2]], 1)
    uk, inv = np.unique(ck, axis=0, return_inverse=True)
    n = len(uk)
    cnt = np.bincount(inv, minlength=n).astype("float64")

    def avg(X):
        X = np.asarray(X, "float64")
        s = np.zeros((n, X.shape[1]))
        np.add.at(s, inv, X)
        return (s / cnt[:, None]).astype("float32")
    for name in ("X_comm.npy", "F_cov.npy", "esk_annual.npy"):
        np.save(os.path.join(a.out, name), avg(np.load(c(name), mmap_mode="r")))
    np.save(os.path.join(a.out, "keys.npy"), uk.astype("int32"))
    # coarse split by row share over all years
    H, W = split["holdout"].shape
    Hc, Wc = (H + K - 1) // K, (W + K - 1) // K
    cell = keys[:, 0] // K * Wc + keys[:, 1] // K
    tot = np.bincount(cell, minlength=Hc * Wc).astype("float64")
    nh = np.bincount(cell, weights=held_row.astype(float), minlength=Hc * Wc)
    nb = np.bincount(cell, weights=buf_row.astype(float), minlength=Hc * Wc)
    holdout = (tot > 0) & (nh >= 0.5 * tot)
    buffer = (tot > 0) & ~holdout & ((nh + nb) > 0)
    np.savez(os.path.join(a.out, "split.npz"), holdout=holdout.reshape(Hc, Wc), buffer=buffer.reshape(Hc, Wc),
             withheld=np.array([], "int32"), common=np.array([], "int32"))
    info = {"source": a.cache, "k": K, "n_rows": int(n), "n_source_rows": int(len(keys)),
            "coarse_cells": int((tot > 0).sum()), "held": int(holdout.sum()), "buffer": int(buffer.sum()),
            "train": int(((tot > 0) & ~holdout & ~buffer).sum()), "rows_per_coarse_row_median": float(np.median(cnt))}
    json.dump(info, open(os.path.join(a.out, "coarsen.json"), "w"), indent=1)
    print(json.dumps(info))


if __name__ == "__main__":
    main()
