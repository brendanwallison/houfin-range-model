"""E055i: the seed ensemble's backcast gain over the MEAN single seed, with one joint block bootstrap (verification b).

    python research/exp/ensemble_gain.py --members <cache> ... --esk-dir <dir> --out <dir> [--sizes 2 4 ...]
        [--eval-set heldout|train] [--rank 24] [--n-boot 500]

Raw backcast change (z(2005-25) - z(early withheld epoch), the suite's estimand; no readout), noise-free centred corr
with the observed ESK change (ABBA halves), r24, on held-out cells. Every draw scores every single member AND the
ensembles, so the difference ensemble - mean(single) gets a proper CI. --sizes adds ensembles of the first k members
(an ensemble-size curve). The ensemble is built here from the members' z (the mean), independently of
ensemble_cache.py, so it also re-derives E055f's headline (verification b). Caches are only read.
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

from rrr_backcast import wc  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--esk-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", type=int, nargs="*", default=[])
    ap.add_argument("--eval-set", choices=["heldout", "train"], default="heldout")
    ap.add_argument("--rank", type=int, default=24)
    ap.add_argument("--n-boot", type=int, default=500)
    a = ap.parse_args()
    from change_oracle import desk_epoch_z, group_rows
    r, m0 = a.rank, a.members[0]
    keys = np.load(os.path.join(m0, "keys.npy"))
    for m in a.members[1:]:
        assert np.array_equal(keys, np.load(os.path.join(m, "keys.npy"))), m
    split = np.load(os.path.join(m0, "split.npz"))
    withheld = [int(y) for y in split["withheld"]]
    E, full, _, _ = group_rows(m0, keys, withheld, epochs_file=os.path.join(a.esk_dir, "esk_epochs.npz"))
    idx = {}
    for j, (cell, ep, wh) in enumerate(zip(map(tuple, E["cells"]), E["epoch"], E["withheld"])):
        if ep == 0 and bool(wh):
            idx.setdefault(cell, [None, None])[0] = j
        elif ep == 1 and not wh:
            idx.setdefault(cell, [None, None])[1] = j
    pairs = np.array([(e, m) for e, m in idx.values() if e is not None and m is not None])
    cells = E["cells"][pairs[:, 0]]
    held = split["holdout"][cells[:, 0], cells[:, 1]]
    buf = split["buffer"][cells[:, 0], cells[:, 1]]
    ev = np.where(held)[0] if a.eval_set == "heldout" else np.where(~held & ~buf)[0]
    Za, Zb = (E[k][:, :r].astype("float64") for k in ("z_a", "z_b"))
    Ta = (Za[pairs[:, 1]] - Za[pairs[:, 0]])[ev]
    Tb = (Zb[pairs[:, 1]] - Zb[pairs[:, 0]])[ev]
    early_rows = [np.asarray(full[pairs[j, 0]]) for j in ev]
    mod_rows = [np.asarray(full[pairs[j, 1]]) for j in ev]
    ci_, y0 = np.load(os.path.join(m0, "key_cell_index.npy")), int(np.load(os.path.join(m0, "years.npy"))[0])
    D = []
    for m in a.members:
        z = np.load(os.path.join(m, "z_raw_all.npy"), mmap_mode="r")
        D.append(desk_epoch_z(z, ci_, y0, keys, mod_rows, r) - desk_epoch_z(z, ci_, y0, keys, early_rows, r))
    preds = {f"single_{i + 1}": d for i, d in enumerate(D)}
    for k in sorted(set(a.sizes) | {len(D)}):
        preds[f"ens{k}"] = np.mean(D[:k], axis=0)   # change of the mean z = mean of the changes (linear)
    blk = (cells[ev, 0] // 6) * 100000 + cells[ev, 1] // 6
    ub, inv = np.unique(blk, return_inverse=True)
    rng = np.random.default_rng(11)
    W = np.vstack([np.ones(len(ev)), rng.multinomial(len(ub), np.full(len(ub), 1.0 / len(ub)), size=a.n_boot)[:, inv]])

    def corr(P, w):
        vp, vt = wc(P, P, w), wc(Ta, Tb, w)
        return 0.5 * (wc(P, Ta, w) + wc(P, Tb, w)) / np.sqrt(vp * vt) if vp > 0 and vt > 0 else np.nan
    B = {k: np.array([corr(P, w) for w in W]) for k, P in preds.items()}
    single_mean = np.mean([B[f"single_{i + 1}"] for i in range(len(D))], axis=0)
    ci = lambda v: [float(np.nanpercentile(v[1:], q)) for q in (2.5, 97.5)]
    res = {"members": a.members, "eval_set": a.eval_set, "n_cells": int(len(ev)),
           "corr": {k: {"corr": float(v[0]), "ci": ci(v)} for k, v in B.items()},
           "mean_single": float(single_mean[0]),
           "ens_minus_mean_single": {k: {"diff": float((B[k] - single_mean)[0]), "ci": ci(B[k] - single_mean)}
                                     for k in B if k.startswith("ens")}}
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "ensemble_gain.json"), "w", encoding="utf-8"), indent=1)
    print(f"[ensemble-gain] {len(D)} members, {a.eval_set} cells ({len(ev)}); mean single {single_mean[0]:+.3f}")
    for k in B:
        line = f"  {k:9s} corr {B[k][0]:+.3f}"
        if k.startswith("ens"):
            d = res["ens_minus_mean_single"][k]
            line += f"   minus mean single {d['diff']:+.3f} [{d['ci'][0]:+.3f},{d['ci'][1]:+.3f}]"
        print(line, flush=True)


if __name__ == "__main__":
    main()
