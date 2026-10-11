"""E039g: the E039b-f skeptic's tests in one pass -- DESK features vs covariate representations under the age model's
readout (per-species BLR, gp_kernels.fit), level skill at held-out cells in trained years, by prevalence tier, with
6x6-block-bootstrap CIs.

    ESK_DESK_CONFIG=<overlay> python research/exp/repr_checks.py --cache <cache> --out <dir> [--late-cache <dir>]
        [--bioclim-raw <dir>] [--bioclim-ema <dir>] [--nbr <F_nbr.npy>]
    python research/exp/repr_checks.py --summarize <dir>

Arms (representations fitted on ALL training rows -- as DESK is -- the readout on the thinned / chosen rows):
  desk_r24 / desk_r64          raw z (what the age model reads); desk_ema_r24 z_ema (the "normals" pairing)
  desk_late_r24                raw z of the same run's late (epoch-490) state, if --late-cache (selection effect)
  pc24 / pc64 / pc_all         whitened PCs of the raw-matched covariates (E039b's arms)
  std_all                      the standardized channels themselves, UNWHITENED (threat 3)
  rrr24 / rrr64                SUPERVISED LINEAR RIVAL (threat 1): reduced-rank regression of the cell-years' observed
                               ESK coordinates (esk_annual) on the standardized channels -- the best k linear
                               combinations of covariates for predicting the community embedding DESK is trained on
  ctx48                        pc24 + the top-24 PCs of (5x5 neighbourhood - own) covariates (threat 4)
  sdm_full                     bioclim19 + squares + LUH12 + soil + elevation, matched smoothing (threat 6)
  sdm_full_ema                 the same with EMA'd bioclim / covariates (climate normals), paired with desk_ema_r24
Readout training rows (threat 1 / 5): all training cells; 400 / 200 / 100 / 50 random training CELLS (all their years,
3 draws); and BUFFER cells only (never in DESK's objective).
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

TIERS = [(0.0, 0.01), (0.01, 0.05), (0.05, 0.20), (0.20, 1.01)]
PAIRS = [("desk_r24", "pc24"), ("desk_r24", "rrr24"), ("desk_r64", "rrr64"), ("desk_r64", "pc_all"),
         ("desk_r64", "std_all"), ("desk_r24", "ctx48"), ("desk_r24", "sdm_full"), ("desk_ema_r24", "sdm_full_ema"),
         ("desk_late_r24", "desk_r24"), ("rrr24", "pc24")]


def zs(M, rows):
    M = np.asarray(M, "float64")
    mu, sd = np.nanmean(M[rows], 0), np.nanstd(M[rows], 0)
    keep = sd > 1e-6
    out = (M[:, keep] - mu[keep]) / sd[keep]
    return np.where(np.isfinite(out), out, 0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache")
    ap.add_argument("--out")
    ap.add_argument("--late-cache", default=None)
    ap.add_argument("--bioclim-raw", default=None)
    ap.add_argument("--bioclim-ema", default=None)
    ap.add_argument("--nbr", default=None)
    ap.add_argument("--cell-counts", type=int, nargs="+", default=[400, 200, 100, 50])
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--summarize", default=None)
    a = ap.parse_args()
    if a.summarize:
        return summarize(a.summarize)
    from src.community_encoder.train_DESK import gp_kernels as gpk
    from src.community_encoder.train_DESK.validate_gp_species import covariate_design
    os.makedirs(a.out, exist_ok=True)
    c = lambda f: os.path.join(a.cache, f)
    keys = np.load(c("keys.npy"))
    split = np.load(c("split.npz"))
    wh = np.isin(keys[:, 2], np.asarray(split["withheld"], int)) if "withheld" in split.files else np.zeros(len(keys), bool)
    ho = split["holdout"][keys[:, 0], keys[:, 1]]
    bf = split["buffer"][keys[:, 0], keys[:, 1]]
    F = np.asarray(np.load(c("F_cov_raw.npy"), mmap_mode="r"), "float64")
    Fe = np.asarray(np.load(c("F_cov.npy"), mmap_mode="r"), "float64")
    Zr = np.load(c("z_raw.npy")).astype("float64")
    Ze = np.load(c("z_ema.npy")).astype("float64")
    E = np.load(c("esk_annual.npy")).astype("float64")
    Y = np.log1p(np.load(c("X_dev.npy")).astype("float64"))
    ok = np.isfinite(F).all(1) & np.isfinite(Zr).all(1) & np.isfinite(Ze).all(1)
    tr_all = np.where(~ho & ~bf & ~wh & ok)[0]
    buf = np.where(bf & ~wh & ok)[0]
    te = np.where(ho & ~wh & ok)[0]
    prev = (Y[tr_all] > 0).mean(0)
    keep = prev > 0
    Y, prev = Y[:, keep], prev[keep]
    tier = np.digitize(prev, [t[1] for t in TIERS[:-1]])
    # --- representations ----------------------------------------------------------------------------------------
    P_all = covariate_design(F, tr_all, F.shape[1])
    S = zs(F, tr_all)                                             # standardized channels, unwhitened
    reps = {"desk_r24": Zr[:, :24], "desk_r64": Zr[:, :64], "desk_ema_r24": Ze[:, :24],
            "pc24": P_all[:, :24], "pc64": P_all[:, :64], "pc_all": P_all, "std_all": S}
    # supervised linear rival: reduced-rank regression of observed ESK coords on the standardized channels
    er = tr_all[np.isfinite(E[tr_all]).all(1)]
    Xs, Ys = S[er], E[er, :64]
    Xc, Yc = Xs - Xs.mean(0), Ys - Ys.mean(0)
    lam = 1e-3 * np.trace(Xc.T @ Xc) / Xc.shape[1]
    B = np.linalg.solve(Xc.T @ Xc + lam * np.eye(Xc.shape[1]), Xc.T @ Yc)
    _, sv, Vt = np.linalg.svd(Xc @ B, full_matrices=False)
    for k in (24, 64):
        reps[f"rrr{k}"] = S @ (B @ Vt[:k].T)
    rrr_r2 = float(1.0 - ((Yc - Xc @ B) ** 2).sum() / (Yc ** 2).sum())
    if a.late_cache:
        Zl = np.load(os.path.join(a.late_cache, "z_raw.npy")).astype("float64")
        if np.array_equal(np.load(os.path.join(a.late_cache, "keys.npy")), keys):
            reps["desk_late_r24"] = np.where(np.isfinite(Zl[:, :24]), Zl[:, :24], 0.0)
    if a.nbr:
        Fn = np.asarray(np.load(a.nbr), "float64")
        D = np.where(np.isfinite(Fn), Fn, 0.0) - np.where(np.isfinite(F), F, 0.0)
        Pd = covariate_design(D, tr_all, 24)
        reps["ctx48"] = np.column_stack([P_all[:, :24], Pd])
    soil_elev = lambda M: M[:, 283:302]                           # 16 soil + 3 elevation channels (schema order)
    for tag, bdir, Fsrc in (("sdm_full", a.bioclim_raw, F), ("sdm_full_ema", a.bioclim_ema, Fe)):
        if bdir:
            Bc = zs(np.load(os.path.join(bdir, "bioclim_ema.npy")), tr_all)
            Lc = zs(np.load(os.path.join(bdir, "luh_classes_ema.npy")), tr_all)
            reps[tag] = np.column_stack([Bc, zs(Bc ** 2, tr_all), Lc, zs(soil_elev(Fsrc), tr_all)])
    # --- readout training sets -----------------------------------------------------------------------------------
    cell_of = keys[:, 0].astype(np.int64) * 100000 + keys[:, 1]
    tr_cells = np.unique(cell_of[tr_all])
    sets = [("all", 0, tr_all), ("buffer", 0, buf)]
    for n in a.cell_counts:
        for d in range(a.draws):
            pick = np.random.default_rng(100 * n + d).choice(tr_cells, min(n, len(tr_cells)), replace=False)
            sets.append((f"cells{n}", d, tr_all[np.isin(cell_of[tr_all], pick)]))
    # held-out blocks for the bootstrap
    blk = (keys[te, 0] // 6) * 100000 + keys[te, 1] // 6
    ub, binv = np.unique(blk, return_inverse=True)
    out = {"tier": tier, "blocks": ub, "rrr_r2": rrr_r2, "names": np.array(list(reps))}
    meta = {"cache": a.cache, "n_species": int(keep.sum()), "tier_counts": [int((tier == k).sum()) for k in range(4)],
            "n_test": int(len(te)), "n_blocks": int(len(ub)), "rrr_esk_r2": rrr_r2, "sets": []}
    for name, d, rows in sets:
        ybar = Y[rows].mean(0)
        sse0 = np.zeros((len(ub), Y.shape[1]))
        np.add.at(sse0, binv, (Y[te] - ybar) ** 2)
        out[f"{name}_{d}__sse0"] = sse0
        for rn, X in reps.items():
            m = gpk.fit(X[rows], Y[rows])
            P = gpk.predict(m, X[te])[0]
            sse = np.zeros((len(ub), Y.shape[1]))
            np.add.at(sse, binv, (Y[te] - P) ** 2)
            out[f"{name}_{d}__{rn}"] = sse
        meta["sets"].append({"name": name, "draw": d, "n_rows": int(len(rows)),
                             "n_cells": int(len(np.unique(cell_of[rows])))})
        print(f"[checks] {name} draw {d}: {len(rows):,} rows, {len(np.unique(cell_of[rows]))} cells", flush=True)
    np.savez_compressed(os.path.join(a.out, "repr_checks.npz"), **out)
    json.dump(meta, open(os.path.join(a.out, "repr_checks.json"), "w", encoding="utf-8"), indent=1)
    summarize(a.out)


def _skill(sse, sse0, w):
    num, den = (w[:, None] * sse).sum(0), (w[:, None] * sse0).sum(0)
    return np.where(den > 0, 1.0 - num / np.maximum(den, 1e-12), np.nan)


def summarize(out, n_boot=400):
    z = np.load(os.path.join(out, "repr_checks.npz"), allow_pickle=True)
    meta = json.load(open(os.path.join(out, "repr_checks.json"), encoding="utf-8"))
    tier, names = z["tier"], list(z["names"])
    nb = len(z["blocks"])
    rng = np.random.default_rng(11)
    W = np.vstack([np.ones(nb), rng.multinomial(nb, np.full(nb, 1.0 / nb), size=n_boot)]).astype(float)
    print(f"{os.path.basename(os.path.normpath(meta['cache']))}: {meta['n_species']} species, tiers n={meta['tier_counts']};"
          f" {meta['n_test']:,} held-out rows in {meta['n_blocks']} blocks; RRR R^2 on ESK coords {meta['rrr_esk_r2']:.3f}")
    groups = {}
    for s in meta["sets"]:
        groups.setdefault(s["name"], []).append(s)
    for g, ss in groups.items():
        # skill per species per bootstrap replicate, averaged over draws
        def sk(rn):
            return np.mean([np.vstack([_skill(z[f"{g}_{s['draw']}__{rn}"], z[f"{g}_{s['draw']}__sse0"], w) for w in W])
                            for s in ss], axis=0)                 # (n_boot+1, S)
        cache = {rn: sk(rn) for rn in names}
        print(f"  readout trained on {g} ({ss[0]['n_cells']} cells, {ss[0]['n_rows']:,} rows; {len(ss)} draw(s))"
              f" -- median level skill [5-20% | >=20%]: " +
              ", ".join(f"{rn} {np.nanmedian(cache[rn][0][tier == 2]):.3f}|{np.nanmedian(cache[rn][0][tier == 3]):.3f}"
                        for rn in names))
        for A, Bn in PAIRS:
            if A not in cache or Bn not in cache:
                continue
            D = cache[A] - cache[Bn]                              # (n_boot+1, S)
            row = []
            for k in (2, 3):
                med = np.nanmedian(D[:, tier == k], axis=1)
                lo, hi = np.nanpercentile(med[1:], [2.5, 97.5])
                row.append(f"{med[0]:+.3f} [{lo:+.3f},{hi:+.3f}]")
            print(f"    {A:13s} - {Bn:13s} 5-20%: {row[0]}  >=20%: {row[1]}")
        if g == "all":
            # leave-one-block-out: the largest shift in the >=20% tier median of desk_r24 - pc24 / rrr24
            for A, Bn in (("desk_r24", "pc24"), ("desk_r24", "rrr24")):
                if A not in cache or Bn not in cache:
                    continue
                base = np.nanmedian((cache[A][0] - cache[Bn][0])[tier == 3])
                shifts = []
                for b in range(nb):
                    w = np.ones(nb)
                    w[b] = 0.0
                    sa = np.mean([_skill(z[f"all_{s['draw']}__{A}"], z[f"all_{s['draw']}__sse0"], w) for s in ss], 0)
                    sb = np.mean([_skill(z[f"all_{s['draw']}__{Bn}"], z[f"all_{s['draw']}__sse0"], w) for s in ss], 0)
                    shifts.append(np.nanmedian((sa - sb)[tier == 3]) - base)
                print(f"    leave-one-block-out, {A} - {Bn} (>=20%): {base:+.3f}, range of shifts "
                      f"[{min(shifts):+.3f}, {max(shifts):+.3f}] over {nb} blocks")


if __name__ == "__main__":
    main()
