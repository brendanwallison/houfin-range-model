"""E086: how far down the ESK basis can DESK predict? Per-dimension held-out agreement of DESK z with the OBSERVED
community's ESK coordinates, with a split-half noise ceiling.

    python research/exp/e086_predictability.py --cache <dir> [--cache <dir> ...] --out <dir>

Cell-epoch level (esk_epochs.npz: each held-out cell's early and modern epoch, observed z_full and split halves
z_a / z_b), DESK's z_ema averaged over the same surveyed years. Per dimension d = 1..64 on HELD-OUT cells:
  level_r      corr(DESK z_d, observed z_d) over cell-epochs (space: where along dim d a place sits)
  level_rel    split-half reliability of the observed z_d (Spearman-Brown of corr(z_a, z_b)) -- the ceiling
  level_r_dis  level_r / sqrt(level_rel): agreement with survey noise removed (capped at 1.5 for display)
  change_r     corr(DESK change, observed change) over cells with both epochs (modern - early)
  change_rel   split-half reliability of observed change; change_r_dis as above
Also the cumulative share of the observed (uncentred) energy in dims <= d. Pure description; no fitting.
"""
import argparse
import json
import os

import numpy as np


def sb(r):
    """Spearman-Brown: reliability of the full mean from the split-half correlation."""
    r = np.asarray(r, float)
    return np.where(r > 0, 2 * r / (1 + r), np.nan)


def colcorr(a, b):
    a = a - a.mean(0); b = b - b.mean(0)
    return (a * b).sum(0) / np.sqrt((a * a).sum(0) * (b * b).sum(0) + 1e-300)


def one(cache):
    E = dict(np.load(os.path.join(cache, "esk_epochs.npz")))
    z = np.load(os.path.join(cache, "z_ema.npy"), mmap_mode="r")
    sp = dict(np.load(os.path.join(cache, "split.npz")))
    ho = sp["holdout"].astype(bool)
    ptr, flat = E["rows_ptr"], E["rows_flat"]
    n = len(E["cells"])
    zd = np.stack([np.asarray(z[flat[ptr[i]:ptr[i + 1]]], "float64").mean(0) for i in range(n)])
    held = ho[E["cells"][:, 0], E["cells"][:, 1]]
    out = {"n_cell_epochs_heldout": int(held.sum())}
    zo, za, zb = E["z_full"].astype(float), E["z_a"].astype(float), E["z_b"].astype(float)
    ok = held & np.isfinite(za).all(1) & np.isfinite(zb).all(1)
    out["level_r"] = colcorr(zd[ok], zo[ok]).tolist()
    out["level_rel"] = sb(colcorr(za[ok], zb[ok])).tolist()
    # change: pair each held-out cell's early and modern epoch
    ep = E["epoch"]
    key = {(int(r), int(c), int(e)): i for i, ((r, c), e) in enumerate(zip(E["cells"], ep))}
    eps = sorted(set(int(e) for e in ep))
    e0, e1 = eps[0], eps[-1]
    pairs = [(key[(r, c, e0)], key[(r, c, e1)]) for (r, c, e) in key if e == e1 and (r, c, e0) in key
             and ok[key[(r, c, e1)]] and ok[key[(r, c, e0)]]]
    i0, i1 = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    out["n_change_cells"] = len(pairs)
    if len(pairs) > 20:
        out["change_r"] = colcorr(zd[i1] - zd[i0], zo[i1] - zo[i0]).tolist()
        out["change_rel"] = sb(colcorr(za[i1] - za[i0], zb[i1] - zb[i0])).tolist()
    en = (zo[ok] ** 2).sum(0)
    out["energy_cum"] = (np.cumsum(en) / en.sum()).tolist()
    for k in ("level", "change"):
        if f"{k}_r" in out:
            r, rel = np.array(out[f"{k}_r"]), np.array(out[f"{k}_rel"])
            out[f"{k}_r_dis"] = np.clip(r / np.sqrt(rel), -1.5, 1.5).tolist()
            for thr in (0.5, 0.3):
                below = np.where(r < thr)[0]
                out[f"{k}_first_rank_below_{thr}"] = int(below[0] + 1) if len(below) else None
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res = {os.path.basename(c.rstrip("/")): one(c) for c in a.cache}
    json.dump(res, open(os.path.join(a.out, "e086_predictability.json"), "w"), indent=1)
    for name, r in res.items():
        print(f"== {name}: {r['n_cell_epochs_heldout']} held-out cell-epochs, {r['n_change_cells']} change cells")
        print(" rank  lvl_r  lvl_rel  lvl_dis  chg_r  chg_rel  chg_dis  energy_cum")
        for d in list(range(0, 32)) + [39, 47, 63]:
            g = lambda k: r.get(k, [np.nan] * 64)[d]
            print(f" {d + 1:4d}  {g('level_r'):+.2f}  {g('level_rel'):.2f}    {g('level_r_dis'):+.2f}    "
                  f"{g('change_r'):+.2f}  {g('change_rel'):.2f}    {g('change_r_dis'):+.2f}    {g('energy_cum'):.3f}")


if __name__ == "__main__":
    main()
