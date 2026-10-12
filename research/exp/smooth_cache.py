"""E055k: a research cache whose DESK z is SPATIALLY SMOOTHED -- the skeptic's null for the seed ensemble.

    python research/exp/smooth_cache.py --src <cache> --out <dir> [--h 2.0]

The E055 skeptic showed that a 54 km Gaussian smoother applied to ONE seed's predicted change matches the 4-seed
ensemble's raw-change gain. This cache gives that smoothed single model to the full yardstick (fair_rivals: beyond
the trend placebo and the strong linear base), so the two can be compared where it matters.
z_raw_all / z_ema_all (and their at-key versions) are replaced by a Gaussian average over cells, h in cells (27 km
each), WITHIN each split class only: held-out cells over held-out cells, buffer over buffer, training over training.
Otherwise a held-out cell would borrow the z of neighbouring TRAINING cells, which carries their memorized bird data
(a leak). Everything else is copied from the source, except the sealed directory. The source is only read.
"""
import argparse
import json
import os
import shutil

import numpy as np


def kernel(cells, cls, h):
    from scipy import sparse
    rows, cols, vals = [], [], []
    reach = int(np.ceil(3 * h))
    lookup = {(int(r), int(c)): i for i, (r, c) in enumerate(cells)}
    for i, (r, c) in enumerate(cells):
        for dr in range(-reach, reach + 1):
            for dc in range(-reach, reach + 1):
                j = lookup.get((int(r) + dr, int(c) + dc))
                if j is not None and cls[j] == cls[i]:
                    rows.append(i)
                    cols.append(j)
                    vals.append(np.exp(-0.5 * (dr * dr + dc * dc) / h ** 2))
    K = sparse.csr_matrix((vals, (rows, cols)), shape=(len(cells), len(cells)))
    return sparse.diags(1.0 / np.asarray(K.sum(1)).ravel()) @ K


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--h", type=float, default=2.0)
    a = ap.parse_args()
    cells = np.load(os.path.join(a.src, "cells.npy"))
    split = np.load(os.path.join(a.src, "split.npz"))
    ho, bf = split["holdout"][cells[:, 0], cells[:, 1]], split["buffer"][cells[:, 0], cells[:, 1]]
    cls = np.where(ho, 0, np.where(bf, 1, 2))
    K = kernel(cells, cls, a.h)
    os.makedirs(a.out, exist_ok=True)
    for f in os.listdir(a.src):
        p = os.path.join(a.src, f)
        if os.path.isfile(p) and f not in ("z_raw_all.npy", "z_ema_all.npy", "z_raw.npy", "z_ema.npy", "meta.json"):
            shutil.copy2(p, os.path.join(a.out, f))
    ci_ = np.load(os.path.join(a.src, "key_cell_index.npy"))
    keys = np.load(os.path.join(a.src, "keys.npy"))
    years = np.load(os.path.join(a.src, "years.npy"))
    yi = keys[:, 2] - int(years[0])
    for kind in ("raw", "ema"):
        Z = np.load(os.path.join(a.src, f"z_{kind}_all.npy"))
        n, T, L = Z.shape
        assert np.allclose(Z[ci_, yi], np.load(os.path.join(a.src, f"z_{kind}.npy")), equal_nan=True), \
            f"z_{kind}.npy is not z_{kind}_all at the keys: year indexing differs"
        flat = np.nan_to_num(Z.reshape(n, T * L).astype("float64"))
        Zs = np.asarray(K @ flat).reshape(n, T, L).astype(Z.dtype)
        Zs[np.isnan(Z)] = np.nan
        np.save(os.path.join(a.out, f"z_{kind}_all.npy"), Zs)
        np.save(os.path.join(a.out, f"z_{kind}.npy"), Zs[ci_, yi])
    meta = json.load(open(os.path.join(a.src, "meta.json"), encoding="utf-8"))
    meta["smoothed"] = {"source": os.path.abspath(a.src), "h_cells": a.h, "within_split_class": True}
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w", encoding="utf-8"), indent=1)
    print(f"[smooth-cache] {a.src} -> {a.out}: h={a.h} cells, {n} cells x {T} years x {L}, within split class", flush=True)


if __name__ == "__main__":
    main()
