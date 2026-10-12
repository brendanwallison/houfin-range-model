"""E055h: does a model's backcast gain at TRAINING cells come from its EARLY end (the backcast) or its MODERN end?

    python research/exp/end_swap.py --treat <cache> --ctrl <cache> --esk-dir <dir with esk_annual/esk_epochs> --out <dir>
        [--eval-set train|heldout] [--rank 24] [--n-boot 500]

The suite's change estimand is z(modern 2005-25) - z(early withheld epoch). At TRAINING cells the modern years are
DESK's own training targets. A model that memorizes them carries the observed modern communities, sampling noise
included, in its modern end, and the ABBA truth halves share that noise. So the score can rise with no better backcast.
Swapping ends separates the two:
    treat          z_T(mod) - z_T(early)
    ctrl           z_C(mod) - z_C(early)
    early_from_T   z_C(mod) - z_T(early)     the treatment's BACKCAST, the control's modern end
    mod_from_T     z_T(mod) - z_C(early)     the treatment's MODERN end, the control's backcast
    memorizer      ESK_obs(mod) - z_C(early) the control's backcast against the OBSERVED modern community (the limit of
                                             modern-end memorization)
Noise-free centred corr with the observed ESK change (ABBA halves), r24, paired 6x6-block bootstrap, as fair_rivals.
Caches are only read.
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
    ap.add_argument("--treat", required=True)
    ap.add_argument("--ctrl", required=True)
    ap.add_argument("--esk-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval-set", choices=["train", "heldout"], default="train")
    ap.add_argument("--rank", type=int, default=24)
    ap.add_argument("--n-boot", type=int, default=500)
    a = ap.parse_args()
    from change_oracle import desk_epoch_z, group_rows
    r = a.rank
    keys = np.load(os.path.join(a.ctrl, "keys.npy"))
    assert np.array_equal(keys, np.load(os.path.join(a.treat, "keys.npy"))), "caches differ in keys"
    split = np.load(os.path.join(a.ctrl, "split.npz"))
    withheld = [int(y) for y in split["withheld"]]
    E, full, _, _ = group_rows(a.ctrl, keys, withheld, epochs_file=os.path.join(a.esk_dir, "esk_epochs.npz"))
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
    ev = np.where(~held & ~buf)[0] if a.eval_set == "train" else np.where(held)[0]
    Za, Zb, Zf = (E[k][:, :r].astype("float64") for k in ("z_a", "z_b", "z_full"))
    Ta = (Za[pairs[:, 1]] - Za[pairs[:, 0]])[ev]
    Tb = (Zb[pairs[:, 1]] - Zb[pairs[:, 0]])[ev]
    early_rows = [np.asarray(full[pairs[j, 0]]) for j in ev]
    mod_rows = [np.asarray(full[pairs[j, 1]]) for j in ev]
    ci_, y0 = np.load(os.path.join(a.ctrl, "key_cell_index.npy")), int(np.load(os.path.join(a.ctrl, "years.npy"))[0])
    ends = {}
    for tag, c in (("T", a.treat), ("C", a.ctrl)):
        z = np.load(os.path.join(c, "z_raw_all.npy"), mmap_mode="r")
        ends[tag] = (desk_epoch_z(z, ci_, y0, keys, mod_rows, r), desk_epoch_z(z, ci_, y0, keys, early_rows, r))
    obs_mod = Zf[pairs[ev, 1]]
    preds = {"treat": ends["T"][0] - ends["T"][1], "ctrl": ends["C"][0] - ends["C"][1],
             "early_from_T": ends["C"][0] - ends["T"][1], "mod_from_T": ends["T"][0] - ends["C"][1],
             "memorizer": obs_mod - ends["C"][1]}
    blk = (cells[ev, 0] // 6) * 100000 + cells[ev, 1] // 6
    ub, inv = np.unique(blk, return_inverse=True)
    rng = np.random.default_rng(11)
    W = np.vstack([np.ones(len(ev)), rng.multinomial(len(ub), np.full(len(ub), 1.0 / len(ub)), size=a.n_boot)[:, inv]])

    def corr(D, w):
        vp, vt = wc(D, D, w), wc(Ta, Tb, w)
        return 0.5 * (wc(D, Ta, w) + wc(D, Tb, w)) / np.sqrt(vp * vt) if vp > 0 and vt > 0 else np.nan
    boots = {k: np.array([corr(D, w) for w in W]) for k, D in preds.items()}
    ci = lambda v: [float(np.nanpercentile(v[1:], q)) for q in (2.5, 97.5)]
    res = {"treat": a.treat, "ctrl": a.ctrl, "eval_set": a.eval_set, "n_cells": int(len(ev)), "rank": r,
           "corr": {k: {"corr": float(v[0]), "ci": ci(v)} for k, v in boots.items()},
           "minus_ctrl": {k: {"diff": float((v - boots["ctrl"])[0]), "ci": ci(v - boots["ctrl"])}
                          for k, v in boots.items() if k != "ctrl"}}
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "end_swap.json"), "w", encoding="utf-8"), indent=1)
    print(f"[end-swap] {os.path.basename(a.treat)} vs {os.path.basename(a.ctrl)}, {a.eval_set} cells ({len(ev)}):")
    for k in preds:
        d = res["minus_ctrl"].get(k)
        print(f"  {k:13s} corr {boots[k][0]:+.3f}" + (f"   minus ctrl {d['diff']:+.3f} [{d['ci'][0]:+.3f},{d['ci'][1]:+.3f}]"
                                                    if d else ""), flush=True)


if __name__ == "__main__":
    main()
