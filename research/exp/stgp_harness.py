"""E040: repair the spacetime GP baselines (product and sum kernels; E006 found both degenerate) -- a fast harness that
reproduces the suite's spacetime arms and tests one fix at a time against criteria fixed in advance (E040.md).

    python research/exp/stgp_harness.py --out <dir> --kind spacetime|spacetime_sum [variant flags]
    python research/exp/stgp_harness.py --summarize <dir>

Data: the desk_tempho_1995 research cache (1966-1995 withheld from every cell; held-out and buffer cells), dev species
seen in >= 1% of training rows, log1p counts. Inputs [x_km, y_km, year] as validate_gp_species.spacetime_inputs. Rows
as validation_core.row_splits: training = training cells x trained years; SPACE = held-out cells x trained years;
TIME = training cells x the scored window (1966-1975, the suite's common_holdout_years); SPACE_TIME = held-out cells x
the window. Shape, scales and local prediction use the suite's own routines (fit_shared_shape, blocked_scales,
predict_local); only the pieces under test change:
  --sample blocks|randcells        the shape on contiguous blocks (the suite) or on random cells, --years-per-cell each
  --scale-sample blocks|randcells  per-species scales on contiguous blocks (the suite) or blocks of whole random cells
  --bounds suite|data              the suite's [-8, log 20 x extent] or physical bounds (km, years)
  --cell                           add a same-cell term rho [same cell] exp(-|dt| / l_cell): a place's lasting
                                   departure from its surroundings, with its own decay in time
References (simple, robust, not GPs): TIME -- persistence, each training cell's own mean over its trained years;
SPACE / SPACE_TIME -- inverse-distance average of the 8 nearest training cells' means.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

CACHE = os.path.expanduser("~/houfin/work/houfin/research_cache/desk_tempho_1995")
WINDOW = (1966, 1975)
CELL_KM = 27.0
M32 = 3.0 ** 0.5


def skill(y, p, ybar):
    sse = ((y - p) ** 2).sum(0)
    sse0 = ((y - ybar) ** 2).sum(0)
    return np.where(sse0 > 0, 1.0 - sse / np.maximum(sse0, 1e-12), np.nan)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out")
    ap.add_argument("--kind", default="spacetime", choices=("spacetime", "spacetime_sum"))
    ap.add_argument("--sample", default="blocks", choices=("blocks", "randcells"))
    ap.add_argument("--scale-sample", default="blocks", choices=("blocks", "randcells"))
    ap.add_argument("--years-per-cell", type=int, default=4)
    ap.add_argument("--bounds", default="suite", choices=("suite", "data"))
    ap.add_argument("--cell", action="store_true")
    ap.add_argument("--n-fit", type=int, default=3000)
    ap.add_argument("--shape-iters", type=int, default=400)
    ap.add_argument("--n-eval", type=int, default=3000)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--k-max", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--summarize", default=None)
    a = ap.parse_args()
    if a.summarize:
        return summarize(a.summarize)
    import torch
    from src.community_encoder.train_DESK import gp_kernels as gk
    from src.community_encoder.train_DESK.validate_gp_species import spatial_blocks
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    keys = np.load(os.path.join(CACHE, "keys.npy"))
    split = np.load(os.path.join(CACHE, "split.npz"))
    Xd = np.load(os.path.join(CACHE, "X_dev.npy")).astype("float64")
    wh_years = np.asarray(split["withheld"], int)
    r_, c_, yr = keys[:, 0], keys[:, 1], keys[:, 2]
    ho = split["holdout"][r_, c_]
    bf = split["buffer"][r_, c_]
    wh = np.isin(yr, wh_years)
    win = (yr >= WINDOW[0]) & (yr <= WINDOW[1])
    tr = np.where(~ho & ~bf & ~wh)[0]
    sets = {"space": np.where(ho & ~wh)[0], "time": np.where(~ho & ~bf & wh & win)[0],
            "space_time": np.where(ho & wh & win)[0]}
    F = np.column_stack([c_ * CELL_KM, r_ * CELL_KM, yr]).astype("float64")
    Y = np.log1p(Xd)
    sp = (Xd[tr] > 0).mean(0) >= 0.01
    Y = Y[:, sp]
    Ytr = Y[tr]
    ybar = Ytr.mean(0)
    var_y = Ytr.var(0) * len(tr) / max(len(tr) - 1, 1)
    # --- kernels with a same-cell term (same cell <=> distance 0 on the grid) -----------------------------------------
    kind = a.kind
    if a.cell:
        base = gk.KERNELS[a.kind]
        nb = 2 if a.kind == "spacetime" else 4                  # base kernel's theta length

        def kern_cell(A, B, theta):
            same = (torch.cdist(A[:, :2], B[:, :2]) < 1e-6).to(A.dtype)
            dt = torch.abs(A[:, 2:3] - B[:, 2:3].T)
            return base(A, B, theta[:nb]) + torch.exp(theta[nb]) * same * torch.exp(-dt / torch.exp(theta[nb + 1]))
        kind = a.kind + "_cell"
        gk.KERNELS[kind] = kern_cell
        orig_sc = gk.scaled_coords
        gk.scaled_coords = lambda k_, F_, th_: (orig_sc(a.kind, F_, np.asarray(th_)[:nb]) if k_ == kind
                                                else orig_sc(k_, F_, th_))
        orig_pd = gk.prior_diag
        gk.prior_diag = lambda k_, th_: (1.0 + float(np.exp(np.asarray(th_)[nb])) if k_ == kind
                                         else orig_pd(k_, th_))
    if a.kind == "spacetime":
        th0 = list(np.log([300.0, 20.0]))
        data_b = [(np.log(10.0), np.log(10000.0)), (np.log(1.0), np.log(1000.0))]
    else:
        th0 = [np.log(300.0), np.log(150.0), np.log(20.0), 0.0]
        data_b = [(np.log(10.0), np.log(10000.0)), (np.log(10.0), np.log(10000.0)), (np.log(1.0), np.log(1000.0)),
                  (-6.0, 6.0)]
    if a.cell:
        th0 += [0.0, np.log(30.0)]
        data_b += [(-9.0, 5.0), (np.log(1.0), np.log(1000.0))]
    th0 = np.asarray(th0, "float64")
    # --- samplers --------------------------------------------------------------------------------------------------
    blocks = spatial_blocks(keys, tr, 1000)
    cell_tr = r_[tr].astype(np.int64) * 100000 + c_[tr]
    o_ = np.argsort(cell_tr, kind="stable")
    _, st_ = np.unique(cell_tr[o_], return_index=True)
    cell_rows = np.split(tr[o_], st_[1:])
    if a.sample == "blocks":
        shape_blocks, n_sel = [], 0
        for i in rng.permutation(len(blocks)):
            shape_blocks.append(blocks[i])
            n_sel += len(blocks[i])
            if n_sel >= a.n_fit:
                break
    else:
        sel, n_sel = [], 0
        for i in rng.permutation(len(cell_rows)):
            rr = rng.choice(cell_rows[i], min(a.years_per_cell, len(cell_rows[i])), replace=False)
            sel.append(rr)
            n_sel += len(rr)
            if n_sel >= a.n_fit:
                break
        shape_blocks = [np.sort(np.concatenate(sel))]
    if a.scale_sample == "blocks":
        scale_blocks = blocks
    else:
        scale_blocks, cur, n_cur = [], [], 0
        for i in rng.permutation(len(cell_rows)):
            cur.append(cell_rows[i])
            n_cur += len(cell_rows[i])
            if n_cur >= 1000:
                scale_blocks.append(np.sort(np.concatenate(cur)))
                cur, n_cur = [], 0
        if cur:
            scale_blocks.append(np.sort(np.concatenate(cur)))
    bounds = data_b if a.bounds == "data" else None
    if bounds is None and a.cell:
        sb = gk.theta_bounds(a.kind, np.concatenate([F[b] for b in shape_blocks]), len(th0) - 2)
        bounds = list(sb) + [(-9.0, 5.0), (-8.0, float(np.log(20.0 * 60.0)))]
    t0 = time.perf_counter()
    shape = gk.fit_shared_shape(kind, [F[b] for b in shape_blocks], [Y[b] for b in shape_blocks], th0,
                                n_iter=a.shape_iters, verbose=True, ybar=ybar, var_y=var_y, bounds=bounds)
    t_shape = time.perf_counter() - t0
    sc = gk.blocked_scales(kind, shape["theta"], [F[b] for b in scale_blocks], [Y[b] for b in scale_blocks], ybar,
                           var_y)
    shape.update({"s2": sc["s2"], "n2": sc["n2"], "ok": sc["ok"], "ybar": ybar})
    used_b = bounds if bounds is not None else gk.theta_bounds(kind, np.concatenate([F[b] for b in shape_blocks]),
                                                               len(th0))
    th = np.asarray(shape["theta"])
    at_b = [bool(np.isclose(t, lo, atol=1e-6) or np.isclose(t, hi, atol=1e-6)) for t, (lo, hi) in zip(th, used_b)]
    # --- references: persistence (cell means over trained years) and IDW of the 8 nearest training cells ------------
    cm_cells, cm_inv = np.unique(cell_tr, return_inverse=True)
    cnt = np.bincount(cm_inv).astype(float)
    cmean = np.zeros((len(cm_cells), Y.shape[1]))
    np.add.at(cmean, cm_inv, Ytr)
    cmean /= cnt[:, None]
    cxy = np.column_stack([(cm_cells % 100000) * CELL_KM, (cm_cells // 100000) * CELL_KM]).astype("float64")
    cpos = {c: i for i, c in enumerate(cm_cells)}

    def idw(rows):
        xy = F[rows, :2]
        out = np.empty((len(rows), Y.shape[1]))
        for s in range(0, len(rows), 512):
            d = np.sqrt(((xy[s:s + 512, None, :] - cxy[None, :, :]) ** 2).sum(-1))
            nn = np.argsort(d, axis=1)[:, :8]
            w = 1.0 / np.maximum(np.take_along_axis(d, nn, 1), CELL_KM / 2)
            out[s:s + 512] = (w[:, :, None] * cmean[nn]).sum(1) / w.sum(1, keepdims=True)
        return out
    erng = np.random.default_rng(1000 + a.seed)
    res = {"variant": vars(a), "kind": kind, "n_species": int(sp.sum()), "n_train": int(len(tr)),
           "shape_seconds": round(t_shape, 1), "converged": shape["converged"], "message": shape["message"],
           "n_iterations": shape["n_iterations"], "theta": th.tolist(), "params": np.exp(th).tolist(),
           "at_bound": at_b, "frac_at_bound": float(np.mean(at_b)),
           "s2_over_var_median": float(np.median(shape["s2"] / np.maximum(var_y, 1e-12))),
           "n2_over_var_median": float(np.median(shape["n2"] / np.maximum(var_y, 1e-12))), "sets": {}}
    ymax = Ytr.max(0)
    for name, rows_all in sets.items():
        if len(rows_all) == 0:
            continue
        rows = np.sort(erng.choice(rows_all, min(a.n_eval, len(rows_all)), replace=False))
        grp = ((r_[rows] // 6) * 1000 + c_[rows] // 6) * 1000 + yr[rows] // 10
        m, v = gk.predict_local(shape, F[tr], Ytr, F[rows], grp, k=a.k, k_max=a.k_max)
        if name == "time":
            ref = np.stack([cmean[cpos[c]] if c in cpos else np.full(Y.shape[1], np.nan)
                            for c in (r_[rows].astype(np.int64) * 100000 + c_[rows])])
            ok = np.isfinite(ref).all(1)
            ref_name = "persistence"
        else:
            ref, ok, ref_name = idw(rows), np.ones(len(rows), bool), "idw8"
        sk_gp = skill(Y[rows][ok], m[ok], ybar)
        sk_ref = skill(Y[rows][ok], ref[ok], ybar)
        res["sets"][name] = {"n_rows": int(ok.sum()), "reference": ref_name,
                             "gp_level_skill_median": float(np.nanmedian(sk_gp)),
                             "ref_level_skill_median": float(np.nanmedian(sk_ref)),
                             "gp_minus_ref_median": float(np.nanmedian(sk_gp - sk_ref)),
                             "out_of_range": float(((m < -1.0) | (m > ymax[None, :] + 1.0)).mean()),
                             "pred_var_ratio_median": float(np.nanmedian(np.nanmax(v, 0) / np.maximum(var_y, 1e-12)))}
    tag = a.tag or f"{kind}_{a.sample}_{a.scale_sample}_{a.bounds}"
    with open(os.path.join(a.out, f"stgp_{tag}.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    summarize(a.out)


def summarize(out):
    for f in sorted(os.listdir(out)):
        if not (f.startswith("stgp_") and f.endswith(".json")):
            continue
        r = json.load(open(os.path.join(out, f), encoding="utf-8"))
        pr = " ".join(f"{p:.3g}" for p in r["params"])
        print(f"{f[5:-5]:30s} {r['kind']:18s} {'converged' if r['converged'] else 'NOT conv'} it {r['n_iterations']:3d}"
              f" at-bound {r['frac_at_bound']:.2f} params [{pr}] s2/var {r['s2_over_var_median']:.2f}"
              f" n2/var {r['n2_over_var_median']:.2f}")
        for s, v in r["sets"].items():
            print(f"    {s:10s} n {v['n_rows']:5d} | GP {v['gp_level_skill_median']:+.3f} vs {v['reference']}"
                  f" {v['ref_level_skill_median']:+.3f} (diff {v['gp_minus_ref_median']:+.3f}) | out-of-range"
                  f" {v['out_of_range']:.4f} max pred var {v['pred_var_ratio_median']:.2f}")


if __name__ == "__main__":
    main()
