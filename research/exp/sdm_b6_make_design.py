"""B6: build a synthetic design directory (y replaced by a rangr truth's observations) and the truth's own front
readout; also write the age-model synthetic-observation file (HOUFIN_SYNTHETIC_OBS format: y, mu in load_data order).

    python sdm_b6_make_design.py --design <design_step3> --b6 <b6 dir> --tag static_s1
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sdm_b0_score import load_design  # noqa: E402
from sdm_b1_sdm import front  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True); ap.add_argument("--b6", required=True); ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    D = load_design(a.design)
    syn = np.genfromtxt(os.path.join(a.b6, f"y_{a.tag}.csv"), delimiter=",", names=True)
    assert np.array_equal(syn["i"].astype(int), np.arange(len(D["y"])))
    out = os.path.join(a.b6, f"design_{a.tag}")
    os.makedirs(out, exist_ok=True)
    D2 = dict(D); D2["y"] = syn["y"].astype(np.float64)
    np.savez_compressed(os.path.join(out, "design.npz"), **D2)
    for f in ("z_cells.npy", "step3_pred.npz", "step3_density.npy", "design_manifest.json"):
        dst = os.path.join(out, f)
        if not os.path.exists(dst):
            try:
                os.symlink(os.path.join(a.design, f), dst)
            except OSError:
                shutil.copy(os.path.join(a.design, f), dst)
    np.savez(os.path.join(a.b6, f"age_synthetic_obs_{a.tag}.npz"), y=syn["y"].astype(np.int64), mu=syn["mu"])
    years = D["years"]; nT = len(years)
    grid = json.load(open(os.path.join(a.b6, "grid.json")))
    # R wrote aperm(N, c(3, 1, 2)) column-major, i.e. C-order shape (Nx, Ny, nT)
    N = np.fromfile(os.path.join(a.b6, f"N_{a.tag}.f64"), dtype="<f8").reshape(grid["Nx"], grid["Ny"], nT).transpose(2, 1, 0)
    np.save(os.path.join(a.b6, f"N_{a.tag}_land.npy"), np.nan_to_num(N[:, D["land_rows"], D["land_cols"]]).astype(np.float32))
    dens = np.nan_to_num(N[:, D["land_rows"], D["land_cols"]])
    fr_truth = front(D2, dens, years, D["land_rows"], D["land_cols"])
    lon, lat = D["land_lon"], D["land_lat"]
    band = (lon >= -80) & (lat >= 25) & (lat < 49)
    e = (years >= 1966) & (years <= 1975); l = (years >= 2010) & (years <= 2020)
    res = {"tag": a.tag, "truth_front_vs_synthetic_first_positive": fr_truth,
           "truth_east80_N_ratio_1966-75_over_2010-20": float(dens[e][:, band].mean() / max(dens[l][:, band].mean(), 1e-9)),
           "synthetic_mean_real_rows": float(D2["y"][D["obsroute"] >= 0].mean()),
           "synthetic_zero_share_real": float((D2["y"][D["obsroute"] >= 0] == 0).mean()),
           "real_zero_share": float((D["y"][D["obsroute"] >= 0] == 0).mean())}
    json.dump(res, open(os.path.join(a.b6, f"truth_{a.tag}.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
