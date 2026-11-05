"""Export B2's suitability / establishment fields and its own per-cell occupancy front for the kissmig cross-check
(research/experiments/SDM_BASELINE_KISSMIG_B2ALLEE_2026-10-30.md). CPU only.

    python sdm_b2_export.py --design <design_step3> --bbs-npz <npz> --metadata <pkl> --b2 <b2_results.json> --out <dir>

Writes <out>/grid.json (shape, years), <out>/E_static.f32, E_dyn.f32, S01_static.f32, S01_dyn.f32 (years x Ny x Nx,
float32 little-endian, NaN off land; static files have one layer), origin_native.f32, origin_release.f32 (Ny x Nx),
<out>/b2_front_cells.csv (row, col, first BBS positive, B2-static / B2-dyn first year p >= .5, ok flag).
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
from sdm_b2_migsdm import Automaton  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    for k in ("--design", "--bbs-npz", "--metadata", "--b2", "--out"):
        ap.add_argument(k, required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    D = load_design(a.design)
    Z = np.load(os.path.join(a.design, "z_cells.npy")).astype(np.float64)
    years = D["years"]; nT, nL, M = Z.shape
    lr, lc = D["land_rows"], D["land_cols"]
    b = np.load(a.bbs_npz)
    Ny, Nx = int(b["Ny"]), int(b["Nx"])
    lut = -np.ones((Ny, Nx), int); lut[lr, lc] = np.arange(nL)
    cell = lut[D["row"], D["col"]]
    native = np.where(b["initpop_shape"][lr, lc] > 0)[0]
    inv = np.asarray(pickle.load(open(a.metadata, "rb"))["inv_locations"])
    release = np.array([lut[r_, c_] for r_, c_ in inv if lut[r_, c_] >= 0])
    B2 = json.load(open(a.b2))["arms"]
    real_train = ~D["holdout"] & ~D["pseudo"] & (D["weight"] > 0)
    g = np.where(D["obsroute"] >= 0, D["obsroute"], -1); n_g = int(D["obsroute"].max()) + 1
    win = (years >= 1966) & (years <= 2025)
    lat, lon = np.radians(D["land_lat"]), np.radians(D["land_lon"])
    Dkm = 6371.0 * np.arccos(np.clip(np.sin(lat)[:, None] * np.sin(lat)[None] +
                                     np.cos(lat)[:, None] * np.cos(lat)[None] * np.cos(lon[:, None] - lon[None]), -1, 1))
    auto = Automaton(Dkm, years, native, release)

    def to_grid(v):          # (..., nL) -> (..., Ny, Nx) with NaN off land
        out = np.full(v.shape[:-1] + (Ny, Nx), np.nan, np.float32)
        out[..., lr, lc] = v
        return out

    first_p = {}
    for arm, lam in (("static", 100.0), ("dyn", 1000.0)):
        zrows = Z[D["t"], cell] if arm == "dyn" else Z[win].mean(0)[cell]
        feat = Features(zrows, real_train)
        X = feat(zrows)
        fit = fit_nb(X[real_train], D["y"][real_train], D["weight"][real_train], g[real_train], n_g, lam)
        if arm == "static":
            S = np.exp(np.clip(feat(Z[win].mean(0)) @ fit["b"], -30, 15))[None]
        else:
            S = np.stack([np.exp(np.clip(feat(Z[t]) @ fit["b"], -30, 15)) for t in range(nT)])
        pp = B2[arm]["fit"]["params"]; akm = B2[arm]["fit"]["a_km"]
        E = 1.0 - np.exp(-S / pp["S0"])
        S01 = np.minimum(1.0, S / np.quantile(S, .99))
        to_grid(E.astype(np.float32)).tofile(os.path.join(a.out, f"E_{arm}.f32"))
        to_grid(S01.astype(np.float32)).tofile(os.path.join(a.out, f"S01_{arm}.f32"))
        Sgrid = np.broadcast_to(S, (nT, nL)).astype(np.float32).copy()
        P = auto.run(Sgrid, akm, pp["r"], pp["q"], pp["S0"], pp["e0"])
        occ = P >= .5
        first_p[arm] = np.where(occ.any(0), years[np.argmax(occ, 0)], np.nan)
        print(arm, "E range", float(E.min()), float(E.max()), "fitted", pp, flush=True)
    o_nat = np.zeros(nL, np.float32); o_nat[native] = 1
    o_rel = np.zeros(nL, np.float32); o_rel[release] = 1
    to_grid(o_nat).tofile(os.path.join(a.out, "origin_native.f32"))
    to_grid(o_rel).tofile(os.path.join(a.out, "origin_release.f32"))
    real = D["obsroute"] >= 0
    first_obs = np.full(nL, np.inf); surveyed = np.full(nL, np.inf)
    np.minimum.at(surveyed, cell[real], D["year"][real])
    pos = real & (D["y"] > 0); np.minimum.at(first_obs, cell[pos], D["year"][pos])
    first_obs[np.isinf(first_obs)] = np.nan; surveyed[np.isinf(surveyed)] = np.nan
    ok = (D["land_lon"] >= -95) & (D["land_lat"] >= 25) & (D["land_lat"] < 49) & np.isfinite(first_obs) & (first_obs >= surveyed + 3)
    with open(os.path.join(a.out, "b2_front_cells.csv"), "w") as f:
        f.write("row,col,first_obs,b2_static_p50,b2_dyn_p50,ok\n")
        for i in range(nL):
            f.write(f"{lr[i]},{lc[i]},{first_obs[i]},{first_p['static'][i]},{first_p['dyn'][i]},{int(ok[i])}\n")
    json.dump({"Ny": Ny, "Nx": Nx, "years": [int(y) for y in years], "n_ok": int(ok.sum()),
               "native": int(native.size), "release": release.tolist()}, open(os.path.join(a.out, "grid.json"), "w"))
    print("ok cells", int(ok.sum()), "DONE")


if __name__ == "__main__":
    main()
