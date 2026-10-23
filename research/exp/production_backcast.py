"""E035: what does the PRODUCTION DESK (the epoch-200 state the age model consumes) do in 1902-1939?

    ESK_DESK_CONFIG=config/overlays/production.json python research/exp/production_backcast.py --keys-cache <dir> --out <dir>
    python research/exp/production_backcast.py --summarize <out>

No truth exists before 1966, so this describes rather than grades. For every surveyed cell, production raw z (r24) is
encoded for every year 1902-2025 and averaged over three windows: 1920-1939 (the backcast the age model leans on),
1966-1986 (early BBS) and 2006-2025 (modern). Reported:
  * how far each window sits from the modern one, against how far the OBSERVED community moved 1966-86 -> 2005-25
    (ESK projection of the counts, same cells) -- production's movement in its own trained years and before them;
  * how much of the backcast follows space-for-time: D = mean modern z of the 10 cells whose MODERN covariates are
    nearest the cell's 1920-39 covariates, minus that of the 10 nearest its modern covariates (>200 km away, E017 v2);
    alignment = pooled centred cosine between the backcast change B = z(1920-39) - z(2006-25) and D, with a shuffled-D
    null, and the slope of B on D (how far toward the analogs it moves). The same for 1966-86, where E017 measured
    real communities at ~-0.05 and selected-epoch tempho models at +0.10-0.21 (late states +0.03-0.09).
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
from lib import covfeat  # noqa: E402

WINDOWS = {"w1920": (1920, 1939), "w1966": (1966, 1986), "modern": (2006, 2025)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--keys-cache", required=True, help="a research cache whose keys/esk_annual define the cells")
    ap.add_argument("--out")
    ap.add_argument("--rank", type=int, default=24)
    ap.add_argument("--k-analogs", type=int, default=10)
    ap.add_argument("--exclude-km", type=float, default=200.0)
    ap.add_argument("--summarize", default=None)
    a = ap.parse_args()
    if a.summarize:
        return summarize(a.summarize)
    from src.config_utils import load_config
    from src.community_encoder.train_DESK.validate_spacetime import encode_points
    os.makedirs(a.out, exist_ok=True)
    cfg = load_config()
    keys = np.load(os.path.join(a.keys_cache, "keys.npy"))
    cells = np.unique(keys[:, :2], axis=0)
    years = np.arange(1902, 2026)
    grid = np.array([(r_, c_, y) for r_, c_ in cells for y in years], "int32")
    Z, ok = encode_points(cfg, grid)
    Z = Z.reshape(len(cells), len(years), -1)[:, :, : a.rank].astype("float64")
    okg = ok.reshape(len(cells), len(years))
    np.save(os.path.join(a.out, "z_prod_r24.npy"), Z.astype("float32"))
    wmean = {}
    for k, (lo, hi) in WINDOWS.items():
        sel = (years >= lo) & (years <= hi)
        zz = np.where(okg[:, sel, None], Z[:, sel], np.nan)
        wmean[k] = np.nanmean(zz, axis=1)
    # covariates (DESK's normalized states, no extra smoothing) averaged over the same windows -> 64 PCs
    F = covfeat.multi_ema_rows(cfg, grid, [0])[0.0].reshape(len(cells), len(years), -1)
    cw = {k: np.nanmean(F[:, (years >= lo) & (years <= hi)], axis=1) for k, (lo, hi) in WINDOWS.items()}
    allw = np.vstack([cw["w1920"], cw["w1966"], cw["modern"]])
    good = np.isfinite(allw).all(1)
    P = covfeat.cov_pcs(np.nan_to_num(allw), good, n_pcs=64)
    n = len(cells)
    pc = {"w1920": P[:n], "w1966": P[n:2 * n], "modern": P[2 * n:]}
    ok_cell = np.isfinite(wmean["w1920"]).all(1) & np.isfinite(wmean["modern"]).all(1) & np.isfinite(
        wmean["w1966"]).all(1) & good[:n] & good[n:2 * n] & good[2 * n:]
    xy = cells.astype("float64") * 27.0
    zmod = wmean["modern"]

    def analog_mean(q, idx):
        """Mean modern z of the k cells (among ok_cell, > exclude_km away) whose MODERN covariates are nearest q."""
        cand = np.where(ok_cell)[0]
        far = np.sqrt(((xy[cand] - xy[idx]) ** 2).sum(1)) > a.exclude_km
        cand = cand[far]
        d = ((pc["modern"][cand] - q) ** 2).sum(1)
        nn = cand[np.argsort(d)[: a.k_analogs]]
        return zmod[nn].mean(0)
    idx = np.where(ok_cell)[0]
    res = {"n_cells": int(len(idx)), "rank": a.rank, "windows": WINDOWS, "per_window": {}}
    rng = np.random.default_rng(0)
    for w in ("w1920", "w1966"):
        B = wmean[w][idx] - zmod[idx]
        D = np.stack([analog_mean(pc[w][i], i) - analog_mean(pc["modern"][i], i) for i in idx])
        Bc, Dc = B - B.mean(0), D - D.mean(0)
        align = float((Bc * Dc).sum() / np.sqrt((Bc ** 2).sum() * (Dc ** 2).sum()))
        slope = float((Bc * Dc).sum() / (Dc ** 2).sum())
        nul = []
        for _ in range(200):
            Ds = Dc[rng.permutation(len(Dc))]
            nul.append(float((Bc * Ds).sum() / np.sqrt((Bc ** 2).sum() * (Ds ** 2).sum())))
        res["per_window"][w] = {"movement_median": float(np.median(np.linalg.norm(B, axis=1))),
                                "analog_gap_median": float(np.median(np.linalg.norm(D, axis=1))),
                                "alignment_centred": align, "alignment_null_p95": float(np.percentile(nul, 95)),
                                "slope_on_analog_direction": slope}
    # the observed community's own movement 1966-86 -> 2005-25 (ESK of counts) on the same cells
    E = np.load(os.path.join(a.keys_cache, "esk_annual.npy"))[:, : a.rank].astype("float64")
    yr = keys[:, 2]
    cmap = {(int(r_), int(c_)): i for i, (r_, c_) in enumerate(cells)}
    kci = np.array([cmap[(int(r_), int(c_))] for r_, c_ in keys[:, :2]])
    obs = {}
    for name, (lo, hi) in (("early", (1966, 1986)), ("modern", (2005, 2025))):
        m = (yr >= lo) & (yr <= hi)
        s = np.zeros((len(cells), a.rank))
        cnt = np.zeros(len(cells))
        np.add.at(s, kci[m], E[m])
        np.add.at(cnt, kci[m], 1)
        obs[name] = np.where(cnt[:, None] >= 4, s / np.maximum(cnt, 1)[:, None], np.nan)
    both = np.isfinite(obs["early"]).all(1) & np.isfinite(obs["modern"]).all(1) & ok_cell
    res["observed_movement_1966_86_to_2005_25_median"] = float(np.median(
        np.linalg.norm(obs["early"][both] - obs["modern"][both], axis=1)))
    res["production_movement_same_cells_median"] = float(np.median(
        np.linalg.norm(wmean["w1966"][both] - zmod[both], axis=1)))
    res["n_cells_observed"] = int(both.sum())
    with open(os.path.join(a.out, "production_backcast.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    summarize(a.out)


def summarize(out):
    r = json.load(open(os.path.join(out, "production_backcast.json"), encoding="utf-8"))
    print(f"production DESK (epoch 200) backcast, {r['n_cells']} cells, r{r['rank']}:")
    print(f"  observed community movement 1966-86 -> 2005-25 (ESK of counts, {r['n_cells_observed']} cells, median) "
          f"{r['observed_movement_1966_86_to_2005_25_median']:.3f}; production's own over the same years "
          f"{r['production_movement_same_cells_median']:.3f}")
    for w, v in r["per_window"].items():
        print(f"  {w}: distance from modern (median) {v['movement_median']:.3f}; climate-analog gap "
              f"{v['analog_gap_median']:.3f}; alignment with space-for-time {v['alignment_centred']:+.3f} (shuffled "
              f"p95 {v['alignment_null_p95']:+.3f}); slope toward analogs {v['slope_on_analog_direction']:+.3f}")


if __name__ == "__main__":
    main()
