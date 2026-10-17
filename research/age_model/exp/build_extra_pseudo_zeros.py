"""Extend the pre-invasion pseudo-zeros through the unsupervised 1940-1965 window, as a side file.

    python research/age_model/exp/build_extra_pseudo_zeros.py [--buffer-km 700] [--end-year 1965]

bbs.py emits zero counts for 1902-1939 at cells outside the native range's convex hull buffered by
700 km (model_inputs then subsamples those sites to the real BBS sampling density). Between the 1940
release and the first BBS season (1966) nothing supervises the model, yet the invasion is documented to
have stayed local to the New York area for most of that time. This keeps the SAME subsampled pseudo-zero
sites (so density matches the existing pseudo-zeros), drops any within --buffer-km of the release cell
(the valid cell nearest Jones Beach, inv_locations[0]), and emits a zero for every model year from the
release year to --end-year. Output: <input_dir>/extra_pseudo_zeros.npz (rows, cols, time_idx); appended
to the observations at load time when population_model.observation.extra_pseudo_zeros is true.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, REPO)


def main() -> None:
    from src.config_utils import load_age_model_config
    input_dir = load_age_model_config()["input_dir"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--buffer-km", type=float, default=700.0)
    ap.add_argument("--schedule", default=None,
                    help="time-varying buffer, e.g. '1940:150,1950:300,1960:450' (km from that year on); "
                         "overrides --buffer-km. Default history-based use: ~2x the documented radius "
                         "(1940s ~3.5 km/yr; ~21 km/yr from 1962; PA in the 1960s)")
    ap.add_argument("--end-year", type=int, default=1965)
    ap.add_argument("--cell-km", type=float, default=27.0)
    ap.add_argument("--out", default=os.path.join(input_dir, "extra_pseudo_zeros.npz"))
    a = ap.parse_args()

    with open(os.path.join(input_dir, "metadata.pkl"), "rb") as f:
        meta = pickle.load(f)
    years = np.asarray(meta["years"]).astype(int)
    t = np.asarray(meta["obs_time_indices"]); oy = years[t]
    inv_year = int(years[int(meta["inv_timestep"])])
    y = np.asarray(meta["observed_results"]); rows = np.asarray(meta["obs_rows"]); cols = np.asarray(meta["obs_cols"])
    pre = (oy < inv_year) & (y == 0)
    sites = np.unique(np.stack([rows[pre], cols[pre]], axis=1), axis=0)
    r0, c0 = (int(v) for v in np.asarray(meta["inv_locations"])[0])
    dist_km = np.hypot(sites[:, 0] - r0, sites[:, 1] - c0) * a.cell_km
    sched = ([(int(k), float(v)) for k, v in (kv.split(":") for kv in a.schedule.split(","))]
             if a.schedule else [(inv_year, a.buffer_km)])
    def buffer_for(yy):
        return [b for y0, b in sorted(sched) if yy >= y0][-1]
    window = [i for i, yy in enumerate(years) if inv_year <= yy <= a.end_year]
    out_r, out_c, out_t = [], [], []
    for i in window:
        keep = sites[dist_km > buffer_for(int(years[i]))]
        out_r.append(keep[:, 0]); out_c.append(keep[:, 1]); out_t.append(np.full(len(keep), i))
    out_r, out_c, out_t = (np.concatenate(x) for x in (out_r, out_c, out_t))
    keep = sites[dist_km > buffer_for(int(years[window[-1]]))]
    np.savez_compressed(a.out, rows=out_r.astype(np.int64), cols=out_c.astype(np.int64), time_idx=out_t.astype(np.int64),
                        schedule=np.asarray(sched, dtype=float), years=np.asarray([inv_year, a.end_year]))
    print(f"[extra_pseudo_zeros] {a.out}: {len(sites)} pre-invasion sites; schedule {sched}; at {a.end_year} "
          f"{len(keep)} sites beyond buffer of release cell ({r0},{c0}); {len(window)} years -> {len(out_t)} zeros")


if __name__ == "__main__":
    main()
