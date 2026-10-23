"""E055i: the seed ensemble's backcast gain over the MEAN single seed, with one joint block bootstrap (verification b).

    python research/exp/ensemble_gain.py --members <cache> ... --esk-dir <dir> --out <dir> [--sizes 2 4 ...]
        [--eval-set heldout|train] [--rank 24] [--n-boot 500]

Raw backcast change (z(2005-25) - z(early withheld epoch), the suite's estimand; no readout), noise-free centred corr
with the observed ESK change (ABBA halves), r24, on held-out cells. Every draw scores every single member AND the
ensembles, so the difference ensemble - mean(single) gets a proper CI. --sizes adds ensembles of the first k members
(an ensemble-size curve). The ensemble is built here from the members' z (the mean), independently of
ensemble_cache.py, so it also re-derives E055f's headline (verification b). Caches are only read.
After the E055 skeptic (who showed the raw-change gain is all variance shrinkage, and that a 54 km smoother on ONE seed
matches it), every estimand is also scored for a SMOOTHED single seed (Gaussian, h = 2 cells over the evaluated cells):
    change   the suite's raw backcast change (above)
    local    change minus its leave-one-out 150 km Gaussian average over the other evaluated cells (E039j form):
             place-specific change, where a smoother gains ~nothing
    level    the modern-epoch (2005-25) community level, z vs the observed ESK halves
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
    split = np.load(os.path.join(m0, "split.npz"))
    for m in a.members[1:]:
        assert np.array_equal(keys, np.load(os.path.join(m, "keys.npy"))), m
        sm = np.load(os.path.join(m, "split.npz"))
        assert all(np.array_equal(split[k], sm[k]) for k in split.files), (m, "split differs")
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
    D, Lv = [], []
    for m in a.members:
        z = np.load(os.path.join(m, "z_raw_all.npy"), mmap_mode="r")
        zm = desk_epoch_z(z, ci_, y0, keys, mod_rows, r)
        D.append(zm - desk_epoch_z(z, ci_, y0, keys, early_rows, r))
        Lv.append(zm)
    xy = cells[ev].astype("float64")
    d2 = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)
    Ks = np.exp(-0.5 * d2 / 2.0 ** 2)
    Ks /= Ks.sum(1, keepdims=True)                       # smoothing null: h = 2 cells (54 km)
    Kl = np.exp(-0.5 * d2 / (150.0 / 27.0) ** 2)
    np.fill_diagonal(Kl, 0.0)
    Kl /= Kl.sum(1, keepdims=True)
    loc = lambda X: X - Kl @ X                          # local: minus the leave-one-out 150 km average
    La, Lb = Za[pairs[ev, 1]], Zb[pairs[ev, 1]]
    targets = {"change": (Ta, Tb), "local": (loc(Ta), loc(Tb)), "level": (La, Lb)}
    base = {"change": D, "local": [loc(d) for d in D], "level": Lv}
    smoothed = {"change": [Ks @ d for d in D], "local": [loc(Ks @ d) for d in D], "level": [Ks @ x for x in Lv]}
    blk = (cells[ev, 0] // 6) * 100000 + cells[ev, 1] // 6
    ub, inv = np.unique(blk, return_inverse=True)
    rng = np.random.default_rng(11)
    W = np.vstack([np.ones(len(ev)), rng.multinomial(len(ub), np.full(len(ub), 1.0 / len(ub)), size=a.n_boot)[:, inv]])

    def corr(P, A_, B_, w):
        vp, vt = wc(P, P, w), wc(A_, B_, w)
        return 0.5 * (wc(P, A_, w) + wc(P, B_, w)) / np.sqrt(vp * vt) if vp > 0 and vt > 0 else np.nan
    ci = lambda v: [float(np.nanpercentile(v[1:], q)) for q in (2.5, 97.5)]
    sizes = sorted(set(a.sizes) | {len(D)})
    res = {"members": a.members, "eval_set": a.eval_set, "n_cells": int(len(ev)), "estimands": {}}
    os.makedirs(a.out, exist_ok=True)
    print(f"[ensemble-gain] {len(D)} members, {a.eval_set} cells ({len(ev)})")
    for est, (A_, B_) in targets.items():
        P = base[est]
        Bs = [np.array([corr(x, A_, B_, w) for w in W]) for x in P]
        single = np.mean(Bs, axis=0)
        sm = np.mean([np.array([corr(x, A_, B_, w) for w in W]) for x in smoothed[est]], axis=0)
        ens = {k: np.array([corr(np.mean(P[:k], axis=0), A_, B_, w) for w in W]) for k in sizes}
        o = {"mean_single": float(single[0]), "singles": [float(b[0]) for b in Bs],
             "smoothed_single_minus_single": {"diff": float((sm - single)[0]), "ci": ci(sm - single)},
             "ens_minus_mean_single": {f"ens{k}": {"diff": float((v - single)[0]), "ci": ci(v - single)}
                                       for k, v in ens.items()},
             "ens_minus_smoothed_single": {"diff": float((ens[len(D)] - sm)[0]), "ci": ci(ens[len(D)] - sm)}}
        res["estimands"][est] = o
        f_ = lambda d: f"{d['diff']:+.3f} [{d['ci'][0]:+.3f},{d['ci'][1]:+.3f}]"
        print(f"  {est:6s} mean single {single[0]:+.3f} | smoothed single {f_(o['smoothed_single_minus_single'])} | "
              + " ".join(f"{k} {f_(v)}" for k, v in o["ens_minus_mean_single"].items())
              + f" | ens{len(D)} - smoothed {f_(o['ens_minus_smoothed_single'])}", flush=True)
    json.dump(res, open(os.path.join(a.out, "ensemble_gain.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
