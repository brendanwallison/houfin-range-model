"""Final DESK candidate v2 (equal summer / winter halves): inputs for the WINTER half.

    python research/exp/final_v2_aux.py --winter-z-dir <esk_all369_winter/spacetime> --winter-points <cbc_points_all369>
        --cbc cbc_circles_all.npz --completeness completeness_v3.npz --holdout holdout_cells.npy --buffer buffer_cells.npy
        --v1-aux <cbc_aux_final> --out-dir <dir>

Writes:
  winter_coords.npz          the winter basis's coordinates of every CBC cell-year (1966-2024): year, row, col, Z (N, 64)
                             -- the coordinate target of the winter half (the analogue of the summer stabilizing target)
  cbc_presence_early_w.npz   the v1 early presence lists (1902-1965, completeness covariate) with a WINTER operator fitted
                             from WINTER coordinates (not breeding z) to 1966-2024 lists at training cells, by the presence
                             likelihood; read by the trainer from the winter half (aux_presence.read_private)
The winter pair loss reuses v1's cbc_ruzicka.npz cell-years (dot(z_w,i, z_w,j) ~ Ruzicka, identity head).
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for k in ("--winter-z-dir", "--winter-points", "--cbc", "--completeness", "--holdout", "--buffer", "--v1-aux", "--out-dir"):
        ap.add_argument(k, required=True)
    a = ap.parse_args()
    import torch
    from src.community_encoder.train_DESK.esk_kernel import project_points_to_z
    os.makedirs(a.out_dir, exist_ok=True)
    ho, bf = np.load(a.holdout), np.load(a.buffer); okcell = ~ho & ~bf
    X = np.nan_to_num(np.load(os.path.join(a.winter_points, "X_points.npy"))).astype("float32")
    idx = np.load(os.path.join(a.winter_points, "point_index.npy"))
    Zw = project_points_to_z(X, a.winter_z_dir, 64).astype(np.float32)
    np.savez_compressed(os.path.join(a.out_dir, "winter_coords.npz"), year=idx[:, 2].astype(np.int32),
                        row=idx[:, 0].astype(np.int32), col=idx[:, 1].astype(np.int32), Z=Zw)
    print(f"[v2-aux] winter coordinates: {len(Zw):,} CBC cell-years x 64", flush=True)
    key = {(int(r), int(c), int(y)): i for i, (r, c, y) in enumerate(idx)}
    D = np.load(a.cbc, allow_pickle=False); yr = D["year_end"]
    P = (D["Y"] > 0).astype(np.float32)
    v1 = np.load(os.path.join(a.v1_aux, "cbc_presence_early.npz"))
    assert list(v1["species"]) == list(D["species"])
    # completeness covariate exactly as v1 (rebuilt here for the operator's fitting lists)
    Cf = np.load(a.completeness)
    comp = np.clip(np.where(Cf["bench_n"] > 0, Cf["bench_hit"] / np.maximum(Cf["bench_n"], 1), np.nan), 0.02, 0.98)
    lc = np.log(comp / (1 - comp)); ci_ = D["circle_idx"]; late = (yr >= 1966) & np.isfinite(lc)
    ref = np.bincount(ci_[late], np.where(late, lc, 0)[late], minlength=len(D["circles"])) / np.maximum(
        np.bincount(ci_[late], minlength=len(D["circles"])), 1)
    hasr = np.bincount(ci_[late], minlength=len(D["circles"])) >= 3
    rel = np.where(np.isfinite(lc) & hasr[ci_], lc - ref[ci_], np.nan)
    cen = (D["row"] >= 0) & (yr >= 1966) & (yr <= 2024) & (P.sum(1) >= 2); rel = rel - np.nanmean(rel[cen])
    mp = np.array([key.get((int(r), int(c), int(y)), -1) if (r >= 0 and okcell[r, c]) else -1
                   for r, c, y in zip(D["row"], D["col"], yr)])
    use = np.where((mp >= 0) & (yr >= 1966) & (yr <= 2024) & (P.sum(1) >= 2))[0]
    Zt = torch.as_tensor(Zw[mp[use]], dtype=torch.float64); Pt = torch.as_tensor(P[use], dtype=torch.float64)
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
        print(f"[v2-aux] winter operator (winter coords) lambda {lam:g}: held-out nll {v:.4f}", flush=True)
        best = (v, lam) if best is None or v < best[0] else best
    W, bb, _ = fit(torch.arange(len(use)), best[1])
    out = {k: v1[k] for k in v1.files}; out["W"] = W.numpy().astype(np.float32); out["b"] = bb.numpy().astype(np.float32)
    np.savez_compressed(os.path.join(a.out_dir, "cbc_presence_early_w.npz"), **out)
    json.dump({"winter_coords": int(len(Zw)), "operator_fit_lists": int(len(use)), "lambda": best[1], "heldout_nll": best[0],
               "uniform_nll": float(np.log(S))}, open(os.path.join(a.out_dir, "final_v2_aux_meta.json"), "w"), indent=1)
    print(f"[v2-aux] winter operator on {len(use):,} lists, lambda {best[1]:g}, held-out nll {best[0]:.4f} "
          f"(uniform {np.log(S):.4f})", flush=True)


if __name__ == "__main__":
    main()
