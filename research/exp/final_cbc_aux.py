"""Final DESK candidate: CBC supervision inputs refit to a NEW ESK basis (all-species reference community).

    python research/exp/final_cbc_aux.py --z-dir <esk .../spacetime> --points <bbs points dir of that basis>
        --cbc cbc_circles_all.npz --completeness completeness_v3.npz --holdout holdout_cells.npy --buffer buffer_cells.npy
        --out-dir <dir>

The basis's observed coordinates come from projecting the BBS points through it (esk_kernel.project_points_to_z, the
trainer's own z_obs). Fits use only cells outside the production holdout and buffer. Writes:
  cbc_ruzicka.npz     E081 term over ALL CBC species: cell-year communities (winters 1966-2024, counts x (median
                      party-hours / party-hours)^0.8, mean over a cell-year's circles; raw counts -- the trainer log1p's),
                      calibration dot(z_i, z_j) ~ a + b Ruzicka(log1p CBC) by OLS on 200k random pairs of cell-years with
                      both a BBS point and CBC, weight 0.03 / residual MSE (E081's rule)
  cbc_presence_early.npz   E069b / E078 presence term over ALL CBC species: lists 1902-1965 with >= 2 species, the
                      completeness covariate (benchmark completeness, logit, relative to the circle's 1966+ mean, centred on
                      1966+ lists), and the WINTER observation operator W, b fitted by the presence likelihood on 1966-2024
                      lists at training cells (E086x / build_cbc_presence --readout-target cbc)
Sealed species and the House Finch never appear (the CBC file is built from non-sealed lists; asserted).
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))


def ruz(A, B):
    A, B = np.log1p(A), np.log1p(B)
    den = np.maximum(A, B).sum(1)
    return np.where(den > 0, np.minimum(A, B).sum(1) / np.maximum(den, 1e-12), np.nan)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for k in ("--z-dir", "--points", "--cbc", "--completeness", "--holdout", "--buffer", "--out-dir"):
        ap.add_argument(k, required=True)
    ap.add_argument("--latent", type=int, default=64)
    a = ap.parse_args()
    import pandas as pd
    import torch
    from src.community_encoder.train_DESK.esk_kernel import project_points_to_z
    os.makedirs(a.out_dir, exist_ok=True)
    ho, bf = np.load(a.holdout), np.load(a.buffer); okcell = ~ho & ~bf
    X = np.nan_to_num(np.load(os.path.join(a.points, "X_points.npy"))).astype("float32")
    idx = np.load(os.path.join(a.points, "point_index.npy"))
    Z = project_points_to_z(X, a.z_dir, a.latent).astype(np.float64)
    zkey = {(int(r), int(c), int(y)): i for i, (r, c, y) in enumerate(idx)}
    print(f"[final-aux] projected {len(Z):,} BBS points into {a.z_dir}", flush=True)
    D = np.load(a.cbc, allow_pickle=False); cs = list(D["species"])
    assert "houfin" not in cs
    yr, ph = D["year_end"], D["party_hours"].astype(float)
    # ---- Ruzicka term
    ok = (D["row"] >= 0) & (yr >= 1966) & (yr <= 2024) & np.isfinite(ph) & (ph > 0)
    Ys = D["Y"][ok].astype(float) * ((np.median(ph[ok]) / ph[ok])[:, None] ** 0.8)
    df = pd.DataFrame(Ys); df["r"], df["c"], df["y"] = D["row"][ok], D["col"][ok], yr[ok]
    g = df.groupby(["r", "c", "y"]).mean()
    rr, cc, yy = (np.array([k[i] for k in g.index]) for i in range(3)); XC = g.to_numpy()
    m = np.array([zkey.get((int(r), int(c), int(y)), -1) if okcell[r, c] else -1 for r, c, y in zip(rr, cc, yy)])
    have = np.where(m >= 0)[0]; rng = np.random.default_rng(81)
    I, J = rng.choice(have, 200000), rng.choice(have, 200000); k = I != J; I, J = I[k], J[k]
    rz = ruz(XC[I], XC[J]); dt = (Z[m[I]] * Z[m[J]]).sum(1); f = np.isfinite(rz)
    b, a0 = np.polyfit(rz[f], dt[f], 1); mse = float(np.mean((dt[f] - (a0 + b * rz[f])) ** 2))
    corr = float(np.corrcoef(rz[f], dt[f])[0, 1]); weight = 0.03 / mse
    np.savez_compressed(os.path.join(a.out_dir, "cbc_ruzicka.npz"), year=yy.astype(np.int32), row=rr.astype(np.int32),
                        col=cc.astype(np.int32), X=XC.astype(np.float32), species=np.array(cs), a=np.float32(a0),
                        b=np.float32(b), weight=np.float32(weight))
    info = {"ruzicka": {"cell_years": int(len(XC)), "calibration_cell_years": int(len(have)), "a": float(a0), "b": float(b),
                        "corr": corr, "residual_mse": mse, "weight": float(weight), "n_species": len(cs)}}
    print(f"[final-aux] Ruzicka: {len(XC):,} cell-years x {len(cs)} species; dot ~ {a0:+.3f} + {b:.3f} Ruz (corr {corr:.3f}), "
          f"weight {weight:.2f}", flush=True)
    # ---- presence term: completeness covariate (E078 / E086v)
    P = (D["Y"] > 0).astype(np.float32)
    Cf = np.load(a.completeness)
    comp = np.where(Cf["bench_n"] > 0, Cf["bench_hit"] / np.maximum(Cf["bench_n"], 1), np.nan)
    comp = np.clip(comp, 0.02, 0.98); lc = np.log(comp / (1 - comp))
    ci_ = D["circle_idx"]; late = (yr >= 1966) & np.isfinite(lc)
    ref = np.bincount(ci_[late], np.where(late, lc, 0)[late], minlength=len(D["circles"])) / np.maximum(
        np.bincount(ci_[late], minlength=len(D["circles"])), 1)
    hasr = np.bincount(ci_[late], minlength=len(D["circles"])) >= 3
    rel = np.where(np.isfinite(lc) & hasr[ci_], lc - ref[ci_], np.nan)
    cen = (D["row"] >= 0) & (yr >= 1966) & (yr <= 2024) & (P.sum(1) >= 2)
    rel = rel - np.nanmean(rel[cen])
    # ---- winter operator, fitted on 1966-2024 lists at training cells with BBS ESK z
    mp = np.array([zkey.get((int(r), int(c), int(y)), -1) if (r >= 0 and okcell[r, c]) else -1
                   for r, c, y in zip(D["row"], D["col"], yr)])
    use = np.where((mp >= 0) & (yr >= 1966) & (yr <= 2024) & (P.sum(1) >= 2))[0]
    Zt = torch.as_tensor(Z[mp[use]]); Pt = torch.as_tensor(P[use], dtype=torch.float64)
    cp = torch.as_tensor(np.nan_to_num(rel[use]))
    blk = (D["row"][use] // 6) * 1000 + D["col"][use] // 6; ub = np.unique(blk)
    val = np.isin(blk, np.random.default_rng(86).choice(ub, max(1, len(ub) // 5), replace=False))
    L, S = Zt.shape[1], Pt.shape[1]
    def fit(ii, lam):
        W = torch.zeros(L, S, dtype=torch.float64, requires_grad=True); bb = torch.zeros(S, dtype=torch.float64, requires_grad=True)
        gg = torch.zeros(S, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([W, bb, gg], max_iter=500, line_search_fn="strong_wolfe")
        nll = lambda jj: -(torch.log_softmax(Zt[jj] @ W + bb + cp[jj, None] * gg, -1) * Pt[jj]).sum() / Pt[jj].sum()
        def cl():
            opt.zero_grad(); l = nll(ii) + lam * (W ** 2).sum(); l.backward(); return l
        opt.step(cl)
        return W.detach(), bb.detach(), nll
    tr, va = torch.as_tensor(np.where(~val)[0]), torch.as_tensor(np.where(val)[0])
    best = None
    for lam in (1e-6, 1e-5, 1e-4, 1e-3):
        W, bb, nll = fit(tr, lam); v = float(nll(va).detach())
        print(f"[final-aux] winter operator lambda {lam:g}: held-out nll {v:.4f}", flush=True)
        best = (v, lam) if best is None or v < best[0] else best
    W, bb, _ = fit(torch.arange(len(use)), best[1])
    early = (D["row"] >= 0) & (yr >= 1902) & (yr <= 1965) & (P.sum(1) >= 2)
    np.savez_compressed(os.path.join(a.out_dir, "cbc_presence_early.npz"), comp=np.nan_to_num(rel)[early].astype(np.float32),
                        year=yr[early].astype(np.int32), row=D["row"][early].astype(np.int32), col=D["col"][early].astype(np.int32),
                        circle=ci_[early].astype(np.int32), P=P[early], W=W.numpy().astype(np.float32),
                        b=bb.numpy().astype(np.float32), species=np.array(cs))
    info["presence"] = {"early_lists": int(early.sum()), "operator_fit_lists": int(len(use)), "lambda": best[1],
                        "heldout_nll": best[0], "uniform_nll": float(np.log(S)), "n_species": S}
    json.dump(info, open(os.path.join(a.out_dir, "final_cbc_aux_meta.json"), "w"), indent=1)
    print(f"[final-aux] presence: {int(early.sum()):,} early lists (1902-1965) x {S} species; operator on {len(use):,} lists, "
          f"lambda {best[1]:g}, held-out nll {best[0]:.4f}", flush=True)


if __name__ == "__main__":
    main()
