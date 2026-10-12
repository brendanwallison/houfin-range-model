"""E055f: a research cache whose DESK coordinates are the MEAN over several seeds' caches (a seed ensemble).

    python research/exp/ensemble_cache.py --members <cache> <cache> ... --out <dir>

Every member must share keys, cells, years, community matrices and split (checked). DESK's stabilizing term pins each
coordinate to the ESK basis, so seeds share one coordinate system and their z can be averaged directly. z_ema / z_raw
(and their dense _all versions) are averaged; covariates, communities, split and esk files are copied from the first
member. The sealed directory is not copied (only research/adopt.py may read it).
"""
import argparse
import json
import os
import shutil

import numpy as np

SAME = ("keys.npy", "cells.npy", "years.npy", "key_cell_index.npy", "X_comm.npy", "X_dev.npy")
ZS = ("z_ema.npy", "z_raw.npy", "z_ema_all.npy", "z_raw_all.npy")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    m0 = a.members[0]
    for m in a.members[1:]:
        for f in SAME:
            x, y = np.load(os.path.join(m0, f)), np.load(os.path.join(m, f))
            assert x.shape == y.shape and np.array_equal(np.nan_to_num(x, nan=-9e9), np.nan_to_num(y, nan=-9e9)), (m, f)
        s0, s1 = np.load(os.path.join(m0, "split.npz")), np.load(os.path.join(m, "split.npz"))
        assert all(np.array_equal(s0[k], s1[k]) for k in s0.files), (m, "split")
    os.makedirs(a.out, exist_ok=True)
    for f in os.listdir(m0):
        p = os.path.join(m0, f)
        if os.path.isfile(p) and f not in ZS and f != "meta.json":
            shutil.copy2(p, os.path.join(a.out, f))
    agree = {}
    for f in ZS:
        arrs = [np.load(os.path.join(m, f), mmap_mode="r") for m in a.members]
        mean = np.mean([np.asarray(x, dtype="float64") for x in arrs], axis=0).astype(arrs[0].dtype)
        np.save(os.path.join(a.out, f), mean)
        if f == "z_ema.npy":
            # how far apart the seeds are: mean per-row cosine between each member and the ensemble
            mu = mean.astype("float64")
            agree = {os.path.basename(m): float(np.nanmedian(
                (np.asarray(x, "float64") * mu).sum(1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(mu, axis=1))))
                for m, x in zip(a.members, arrs)}
    meta = json.load(open(os.path.join(m0, "meta.json"), encoding="utf-8"))
    meta["ensemble_members"] = [os.path.abspath(m) for m in a.members]
    meta["ensemble_member_cosine_to_mean"] = agree
    meta["best_epoch"] = None
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w", encoding="utf-8"), indent=1)
    print(f"[ensemble] {len(a.members)} members -> {a.out}; median cosine of each member to the mean: "
          + ", ".join(f"{k} {v:.3f}" for k, v in agree.items()), flush=True)


if __name__ == "__main__":
    main()
