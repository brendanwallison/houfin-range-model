"""E073 follow-up: materialize an ANCHORED representation as a surrogate research cache, so every existing scorer
(fair_rivals, e069b_heldout_lists for early CBC lists, species_strata) can read it unchanged.

    python research/exp/anchored_cache.py --cache <DESK research cache> --out <dir> [--variant no_clock|all] [--hl 10]

z := [zbar (24): DESK's own raw z averaged over each cell's trained-year rows -- the persistent spatial field,
      delta (24): the community-learned covariate response -- ridge from the within-place covariate features W
      (anchored_delta.Cov, half-life hl, stream groups of the variant) to the community's ESK r24 within-place
      deviations, fitted on training cells x trained years (E073's comm_delta arm), applied at every cell-year]
delta(x, t) = W(x, t) B, with W = 0 when the covariates equal the cell's reference: no change without covariate change,
and no year or position-by-time input. Written: z_raw / z_ema at keys, z_raw_all / z_ema_all on the cache's dense
(cells x years) grid, and delta_1902.npy over 1902-2025 for backcast inspection. Every other file is a symlink to the
source cache (the sealed dir is not linked). The source cache is only read.
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

from anchored_delta import GROUPS, Y0, Cov, cell_mean, ema  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", default="no_clock", choices=("no_clock", "all"))
    ap.add_argument("--hl", type=int, default=10)
    a = ap.parse_args()
    c = lambda f: os.path.join(a.cache, f)
    keys = np.load(c("keys.npy"))
    split = np.load(c("split.npz"))
    withheld = [int(y) for y in split["withheld"]]
    ho = split["holdout"][keys[:, 0], keys[:, 1]]
    bf = split["buffer"][keys[:, 0], keys[:, 1]]
    ty = ~np.isin(keys[:, 2], withheld)
    cov = Cov(keys, ~ho & ~bf & ty, ty)
    fin = cov.fin
    trn = np.where(~ho & ~bf & ty & fin)[0]
    groups = list(GROUPS) if a.variant == "all" else ["climate", "cover", "human"]
    W = np.hstack([cov.W[(a.hl, g)] for g in groups])
    Zesk = np.asarray(np.load(c("esk_annual.npy"), mmap_mode="r")[:, :24], "float64")
    Zw = Zesk - cell_mean(keys, Zesk, ty & fin)
    A = W[trn]
    lam = 1e-1 * np.trace(A.T @ A) / A.shape[1]
    B = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ Zw[trn])          # (k, 24)
    r2 = 1 - ((Zw[trn] - A @ B) ** 2).sum() / ((Zw[trn] - Zw[trn].mean(0)) ** 2).sum()
    # dense: every cache cell, every year 1902-2025
    cells = np.load(c("cells.npy"))
    assert np.array_equal(cells, cov.cells), "cache cells differ from the keys' cells"
    E = ema(cov.st, a.hl)                                                             # (years, cells, C)
    Xs = np.where(np.isfinite(E), (E - cov.mu[a.hl]) / cov.sd[a.hl], 0.0)
    first = np.unique(cov.cidx, return_index=True)[1]
    cbar = cov.Xbar[a.hl][first]                                                      # (cells, C) reference
    parts = []
    for g in groups:
        cols, V = cov.V[(a.hl, g)]
        parts.append((Xs[:, :, cols] - cbar[None, :, cols]) @ V)
    Wd = np.concatenate(parts, -1)
    delta = (Wd @ B).astype("float32")                                                # (years, cells, 24)
    z = np.asarray(np.load(c("z_raw.npy"), mmap_mode="r")[:, :24], "float64")
    z = np.where(np.isfinite(z), z, 0.0)
    zbar_rows = cell_mean(keys, z, ty & fin)
    zbar = zbar_rows[first].astype("float32")                                         # (cells, 24)
    years = np.load(c("years.npy"))
    yi = years - Y0
    d_all = np.transpose(delta[yi], (1, 0, 2))                                        # (cells, cache years, 24)
    z_all = np.concatenate([np.broadcast_to(zbar[:, None, :], d_all.shape), d_all], -1).astype("float32")
    d_keys = delta[keys[:, 2] - Y0, cov.cidx]
    z_keys = np.concatenate([zbar_rows.astype("float32"), d_keys], -1)
    # consistency: delta at the keys equals the fitted map on the keys' own features
    err = np.abs(d_keys[fin] - (W[fin] @ B)).max()
    assert err < 1e-3, f"dense delta disagrees with the key features ({err})"
    os.makedirs(a.out, exist_ok=True)
    skip = {"z_raw.npy", "z_ema.npy", "z_raw_all.npy", "z_ema_all.npy", "meta.json", "sealed"}
    for f in os.listdir(a.cache):
        if f in skip:
            continue
        dst = os.path.join(a.out, f)
        if not os.path.lexists(dst):
            os.symlink(os.path.join(os.path.abspath(a.cache), f), dst)
    for nm, arr in (("z_raw.npy", z_keys), ("z_ema.npy", z_keys), ("z_raw_all.npy", z_all), ("z_ema_all.npy", z_all)):
        np.save(os.path.join(a.out, nm), arr)
    np.save(os.path.join(a.out, "delta_1902.npy"), np.transpose(delta, (1, 0, 2)))
    np.save(os.path.join(a.out, "years_1902.npy"), np.arange(Y0, Y0 + delta.shape[0], dtype="int32"))
    meta = json.load(open(c("meta.json"), encoding="utf-8"))
    meta.update({"latent_dim": 48, "surrogate": {"kind": "anchored delta (E073)", "variant": a.variant, "hl": a.hl,
                                                 "groups": groups, "layout": "[zbar DESK r24 | delta r24]",
                                                 "delta_r2_on_comm_within": float(r2), "source": a.cache}})
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w", encoding="utf-8"), indent=1)
    # a first look at the implied change: |delta(2015) - delta(1920)| vs |delta(2015) - delta(1970)|, by cell
    t = lambda y: y - Y0
    big = np.linalg.norm(delta[t(2015)] - delta[t(1920)], axis=1)
    bbs = np.linalg.norm(delta[t(2015)] - delta[t(1970)], axis=1)
    print(f"[anchored-cache] {a.variant} h{a.hl}: delta R^2 on the community's within deviations {r2:.3f}; "
          f"median |delta 2015 - 1920| / |delta 2015 - 1970| = {np.median(big / np.maximum(bbs, 1e-9)):.2f}; "
          f"median |delta 2015-1920| {np.median(big):.3f}, |zbar| {np.median(np.linalg.norm(zbar, axis=1)):.3f}")


if __name__ == "__main__":
    main()
