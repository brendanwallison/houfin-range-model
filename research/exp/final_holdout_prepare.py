"""Prepare geometry and raw-feature adapters for the existing fair_rivals scorer."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))


ARMS = {"v1_r64": ("final", np.arange(64), 0),
        "v2_s64": ("final_v2", np.arange(64), 0),
        "v2_w64": ("final_v2", np.arange(64, 128), 0),
        "v2_r64": ("final_v2", np.r_[0:32, 64:96], 32),
        "v2_r128": ("final_v2", np.arange(128), 64)}


def geometry(base, cbc, out):
    out.mkdir(parents=True, exist_ok=True)
    keys = np.load(base / "keys.npy")
    with np.load(cbc) as d:
        rc = np.stack([d["row"], d["col"]], 1)
        rc = rc[(rc >= 0).all(1)]
    cells = np.unique(np.vstack([keys[:, :2], rc]), axis=0).astype("int32")
    np.save(out / "cells.npy", cells)
    (out / "geometry.json").write_text(json.dumps({"base": str(base), "cbc": str(cbc),
                                                  "n_cells": len(cells), "n_bbs_keys": len(keys)}, indent=2))
    print(f"geometry: {len(cells)} cells, {len(keys)} BBS keys", flush=True)


def adapters(base, features, out):
    from esk_project import epoch_groups, epochs_name
    from src.community_encoder.train_DESK.validation_core import split_half_groups
    keys = np.load(base / "keys.npy")
    meta = json.loads((base / "meta.json").read_text())
    original = dict(np.load(base / "split.npz"))
    shared_masks = None
    for arm, (family, cols, split_at) in ARMS.items():
        target = out / arm
        target.mkdir(parents=True, exist_ok=True)
        f = features / family
        done = json.loads((f / "DONE.json").read_text())
        cells, years = np.load(f / "cells.npy"), np.load(f / "years.npy")
        index = {tuple(rc): i for i, rc in enumerate(cells)}
        ci = np.array([index[tuple(rc)] for rc in keys[:, :2]])
        if not np.array_equal(years, np.arange(1902, 2026)):
            raise ValueError("unexpected feature timeline")
        ho, bf = np.load(f / "holdout_cells.npy"), np.load(f / "buffer_cells.npy")
        if shared_masks is not None and any(not np.array_equal(x, y) for x, y in zip(shared_masks, (ho, bf))):
            raise ValueError("candidate spatial masks differ")
        shared_masks = (ho, bf)
        for name in ("keys.npy", "F_cov.npy", "F_cov_raw.npy", "esk_annual.npy"):
            dest = target / name
            if not dest.exists():
                dest.symlink_to((base / name).resolve())
        sealed = target / "sealed"
        if not sealed.exists():
            sealed.symlink_to((base / "sealed").resolve(), target_is_directory=True)
        raw = np.load(f / "z_raw_all.npy", mmap_mode="r")
        dense = np.lib.format.open_memmap(target / "z_raw_all.npy", mode="w+", dtype="float32",
                                         shape=(*raw.shape[:2], len(cols)))
        for iy in range(len(years)):
            dense[:, iy] = raw[:, iy][:, cols]
        dense.flush()
        np.save(target / "z_raw.npy", dense[ci, keys[:, 2] - years[0]])
        np.save(target / "key_cell_index.npy", ci)
        np.save(target / "years.npy", years)
        np.save(target / "cells.npy", cells)
        wh = np.arange(1966, 1996, dtype="int32")
        tr = ~ho[keys[:, 0], keys[:, 1]] & ~bf[keys[:, 0], keys[:, 1]] & (keys[:, 2] >= 1996)
        np.savez(target / "split.npz", **{**original, "holdout": ho, "buffer": bf,
                                         "withheld": wh, "is_train": tr})
        for early in [(1966, 1986), (1966, 1975), (1976, 1985), (1986, 1995)]:
            groups = epoch_groups(keys, wh, early, (2005, 2025))
            gk, gs = list(groups), list(groups.values())
            _, _, valid = split_half_groups(gs, years=keys[:, 2])
            take = np.flatnonzero(valid)
            full = [gs[i] for i in take]
            E = {"cells": np.array([gk[i][:2] for i in take], dtype="int32"),
                 "epoch": np.array([gk[i][2] for i in take]),
                 "withheld": np.array([gk[i][3] for i in take]),
                 "rows_flat": np.concatenate(full),
                 "rows_ptr": np.cumsum([0] + [len(g) for g in full])}
            np.savez_compressed(target / epochs_name(early, (2005, 2025)), **E)
        m = {**meta, "latent_dim": len(cols), "desk_split": split_at,
             "confirmation_only": True, "encoder_temporal_holdout": False,
             "target_readout_training_years": [1996, 2025], "feature_columns": cols.tolist(),
             "feature_manifest": done, "arm": arm}
        (target / "meta.json").write_text(json.dumps(m, indent=2))
        print(f"adapter {arm}: {len(cols)} features, {tr.sum()} calibration rows", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["geometry", "adapters"])
    p.add_argument("--base", required=True, type=Path)
    p.add_argument("--cbc", type=Path)
    p.add_argument("--features", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.stage == "geometry":
        geometry(a.base, a.cbc, a.out)
    else:
        adapters(a.base, a.features, a.out)
