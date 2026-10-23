"""E076 data step: withhold a random half of CBC CELLS from the presence term, stratified by zone x cover tercile.

    python research/exp/e076_cell_split.py --presence cbc_presence_res.npz --cover cover_change_1920_1966.npz
        --split <research cache>/split.npz --out-presence cbc_presence_res_e076.npz --out-mask e076_test_cells.npz

Candidate cells: grid cells holding >= 1 presence list that are DESK TRAINING cells (not held-out, not buffer) -- so
withheld cells keep their BBS supervision and only lose their CBC lists. Within each stratum (zone 1/2/3 x cover
tercile 0/1/2; cells outside the zones or without cover values form their own stratum), a random half (seed 76) is
withheld. Writes the presence file without the withheld cells' lists, and the (H, W) test mask.
"""
import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--presence", required=True); ap.add_argument("--cover", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out-presence", required=True); ap.add_argument("--out-mask", required=True)
    a = ap.parse_args()
    P = np.load(a.presence, allow_pickle=False)
    M = np.load(a.cover); S = np.load(a.split)
    ch, zo, t = M["change"], M["zones"], M["terciles"]
    train_cell = ~S["holdout"] & ~S["buffer"]
    r, c = P["row"], P["col"]
    cells = np.unique(np.column_stack([r, c])[train_cell[r, c]], axis=0)
    cov = ch[cells[:, 0], cells[:, 1]]
    terc = np.where(np.isfinite(cov), np.digitize(cov, t), 9)
    stratum = zo[cells[:, 0], cells[:, 1]].astype(int) * 10 + terc
    rng = np.random.default_rng(76)
    test = np.zeros(len(cells), bool)
    for s in np.unique(stratum):
        ix = np.where(stratum == s)[0]
        test[rng.choice(ix, len(ix) // 2, replace=False)] = True
    mask = np.zeros(S["holdout"].shape, bool)
    mask[cells[test, 0], cells[test, 1]] = True
    keep = ~mask[r, c]
    np.savez_compressed(a.out_presence, **{k: (P[k][keep] if P[k].shape[:1] == r.shape else P[k]) for k in P.files})
    np.savez_compressed(a.out_mask, test=mask, cells=cells, stratum=stratum, withheld=test)
    print(f"[E076] {len(cells)} candidate CBC training cells, {int(test.sum())} withheld; lists kept {int(keep.sum()):,} "
          f"of {len(keep):,}; withheld Great Plains cells {int((test & (stratum // 10 == 2)).sum())}, top-tercile cells "
          f"{int((test & (stratum % 10 == 2)).sum())}")


if __name__ == "__main__":
    main()
