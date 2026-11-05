"""B6 prep: export B1-static / B1-dyn suitability S (expected route count at u = 0) on the 54 km grid and the design's
observation table for the rangr truth (research/experiments/SDM_BASELINE_B6_RANGR_2026-10-31.md). CPU only.

    python sdm_b6_prep.py --design <design_step3> --bbs-npz <npz> --metadata <pkl> --out <dir>
Writes S_static.f32 (1 x Ny x Nx), S_dyn.f32 (124 x Ny x Nx), native.f32, release.f32 (Ny x Nx; NaN off land),
obs.csv (i, row, col, year, y_real, real, pseudo, temporal, spatial), grid.json.
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sdm_b0_score import load_design  # noqa: E402
from sdm_b1_sdm import Features, fit_nb  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    for k in ("--design", "--bbs-npz", "--metadata", "--out"):
        ap.add_argument(k, required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    D = load_design(a.design)
    Z = np.load(os.path.join(a.design, "z_cells.npy")).astype(np.float64)
    years = D["years"]; nT, nL, M = Z.shape
    lr, lc = D["land_rows"], D["land_cols"]
    b = np.load(a.bbs_npz); Ny, Nx = int(b["Ny"]), int(b["Nx"])
    lut = -np.ones((Ny, Nx), int); lut[lr, lc] = np.arange(nL)
    cell = lut[D["row"], D["col"]]
    real_train = ~D["holdout"] & ~D["pseudo"] & (D["weight"] > 0)
    g = np.where(D["obsroute"] >= 0, D["obsroute"], -1); n_g = int(D["obsroute"].max()) + 1
    win = (years >= 1966) & (years <= 2025)

    def to_grid(v):
        out = np.full(v.shape[:-1] + (Ny, Nx), np.nan, np.float32); out[..., lr, lc] = v; return out

    for arm, lam in (("static", 100.0), ("dyn", 1000.0)):
        zrows = Z[D["t"], cell] if arm == "dyn" else Z[win].mean(0)[cell]
        feat = Features(zrows, real_train)
        fit = fit_nb(feat(zrows)[real_train], D["y"][real_train], D["weight"][real_train], g[real_train], n_g, lam)
        S = (np.exp(np.clip(feat(Z[win].mean(0)) @ fit["b"], -30, 15))[None] if arm == "static"
             else np.stack([np.exp(np.clip(feat(Z[t]) @ fit["b"], -30, 15)) for t in range(nT)]))
        to_grid(S.astype(np.float32)).tofile(os.path.join(a.out, f"S_{arm}.f32"))
        print(arm, "S quantiles", np.quantile(S, [.01, .5, .99]).round(3).tolist(), flush=True)
    nat = np.zeros(nL, np.float32); nat[b["initpop_shape"][lr, lc] > 0] = 1
    inv = np.asarray(pickle.load(open(a.metadata, "rb"))["inv_locations"])
    rel = np.zeros(nL, np.float32)
    for r_, c_ in inv:
        if lut[r_, c_] >= 0:
            rel[lut[r_, c_]] = 1
    to_grid(nat).tofile(os.path.join(a.out, "native.f32")); to_grid(rel).tofile(os.path.join(a.out, "release.f32"))
    with open(os.path.join(a.out, "obs.csv"), "w") as f:
        f.write("i,row,col,year,y_real,real,pseudo,temporal,spatial\n")
        for i in range(len(D["y"])):
            f.write(f"{i},{D['row'][i]},{D['col'][i]},{D['year'][i]},{int(D['y'][i])},{int(D['obsroute'][i] >= 0)},"
                    f"{int(D['pseudo'][i])},{int(D['temporal'][i])},{int(D['spatial'][i])}\n")
    json.dump({"Ny": Ny, "Nx": Nx, "years": [int(y) for y in years], "n_obs": int(len(D["y"]))},
              open(os.path.join(a.out, "grid.json"), "w"))
    print("DONE")


if __name__ == "__main__":
    main()
