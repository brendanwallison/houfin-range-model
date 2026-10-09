"""Add the OBSERVED community's ESK coordinates to a research cache: the input of the atlas and of T0.

    ESK_DESK_CONFIG=<overlay> python research/exp/esk_project.py --cache <dir>

Writes into the cache:
    esk_annual.npy      (N, L)  ESK z of every surveyed cell-year's community (log1p of the route mean)
    esk_epochs.npz      one row per (cell, epoch, withheld?) group that ABBA can split:
                        cells, epoch (0 early / 1 modern), withheld, n_years,
                        rows_flat / rows_ptr  the group's key indices (group j: flat[ptr[j]:ptr[j+1]]),
                        x_full / x_a / x_b   (M, n_comm) log1p of the epoch-mean community, all years
                                              and each year-balanced half (the suite's estimand)
                        z_full / z_a / z_b   (M, L) their ESK projections
Two disjoint halves of the same cell-epoch differ only by measurement noise, so cross-products of
half-A and half-B changes estimate SIGNAL without noise: the atlas's noise-corrected retention is
built on exactly that. Withheld years form their own groups (as the suite's oracle now does).
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))


def epoch_groups(keys, withheld, early, modern):
    """``{(row, col, epoch, withheld): [row indices in year order]}`` for the two epochs. Pure."""
    wh = set(int(y) for y in withheld)
    groups = {}
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    for i in order:
        r, c, y = (int(v) for v in keys[i])
        for e, (lo, hi) in enumerate((early, modern)):
            if lo <= y <= hi:
                groups.setdefault((r, c, e, y in wh), []).append(int(i))
    return groups


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", required=True)
    a = ap.parse_args()
    from src.config_utils import load_config
    from src.community_encoder.train_DESK.esk_kernel import project_points_to_z
    from src.community_encoder.train_DESK.validate_bbs_routes import (EPOCH_EARLY, EPOCH_MODERN,
                                                                       epoch_mean_observed)
    from src.community_encoder.train_DESK.validation_core import split_half_groups
    from src.data.preprocess.bbs_community import log1p_community
    cfg = load_config()
    zd = cfg["desk"]["z_dir"]
    meta = json.load(open(os.path.join(a.cache, "meta.json"), encoding="utf-8"))
    L = int(meta["latent_dim"])
    keys = np.load(os.path.join(a.cache, "keys.npy"))
    X = np.load(os.path.join(a.cache, "X_comm.npy")).astype("float64")
    split = np.load(os.path.join(a.cache, "split.npz"))
    withheld = [int(y) for y in split["withheld"]]

    z_ann = project_points_to_z(log1p_community(X).astype("float32"), zd, L)
    np.save(os.path.join(a.cache, "esk_annual.npy"), np.asarray(z_ann, "float32"))

    groups = epoch_groups(keys, withheld, EPOCH_EARLY, EPOCH_MODERN)
    gk = list(groups)
    glist = [groups[k] for k in gk]
    A, B, ok = split_half_groups(glist, years=keys[:, 2])
    keep = np.where(ok)[0]
    full = [glist[i] for i in keep]
    xa = epoch_mean_observed(X, [A[i] for i in keep])
    xb = epoch_mean_observed(X, [B[i] for i in keep])
    xf = epoch_mean_observed(X, full)
    proj = lambda v: np.asarray(project_points_to_z(np.asarray(v, "float32"), zd, L), "float32")
    ptr = np.cumsum([0] + [len(f) for f in full]).astype("int64")
    np.savez_compressed(
        os.path.join(a.cache, "esk_epochs.npz"),
        rows_flat=np.concatenate(full).astype("int64"), rows_ptr=ptr,
        cells=np.array([gk[i][:2] for i in keep], "int32"),
        epoch=np.array([gk[i][2] for i in keep], "int8"),
        withheld=np.array([gk[i][3] for i in keep], bool),
        n_years=np.array([len(full[j]) for j in range(len(keep))], "int16"),
        x_full=xf.astype("float32"), x_a=xa.astype("float32"), x_b=xb.astype("float32"),
        z_full=proj(xf), z_a=proj(xa), z_b=proj(xb))
    print(f"[esk-project] {len(keys):,} annual projections; {len(keep):,} splittable cell-epochs "
          f"({int((~ok).sum())} too short to split)")


if __name__ == "__main__":
    main()
