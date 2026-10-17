"""Per-observation BBS ROUTE ids aligned to the existing model inputs, as a side file.

    python research/age_model/exp/build_route_ids.py [--out <input_dir>/obs_route_ids.npz]

The model inputs (metadata.pkl) carry cell, year and count per observation, not the route. Rather
than regenerate the shared inputs, this re-runs bbs.py's route->grid mapping to recover each real
observation's (CountryNum, StateNum, Route), replays model_inputs' observation assembly exactly
(pseudo-zero subsample with its fixed seed, bounds filter, year filter), and REFUSES to write unless
rows, cols, years and counts match metadata.pkl element by element. Pseudo-zeros get route -1.
Output: route (int, compact 0..R-1 or -1), n_routes, route_key (CountryNum*1e5 + StateNum*1e3 + Route).
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
    from src.data.preprocess import bbs
    from src.data.combine import model_inputs as mi

    input_dir = load_age_model_config()["input_dir"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(input_dir, "obs_route_ids.npz"))
    a = ap.parse_args()

    # 1. bbs.py: the real observations, in the order bbs_data_for_python.npz stores them
    land_mask, _, transform, crs, nx, ny = bbs.load_grid_reference(bbs.MASK_PATH)
    frames = [bbs.load_usca_observations()]
    mexico = bbs.load_mexico_observations()
    if mexico is not None:
        frames.append(mexico)
    import pandas as pd
    obs = pd.concat(frames, ignore_index=True)
    obs = obs[(obs["Year"] >= bbs.START_YEAR) & (obs["Year"] <= bbs.END_YEAR)]
    mapped = bbs.map_routes_to_grid(obs, bbs.load_routes(), transform, crs, nx, ny, land_mask)
    key = (mapped["CountryNum"].values.astype(np.int64) * 100000
           + mapped["StateNum"].values.astype(np.int64) * 1000 + mapped["Route"].values.astype(np.int64))

    # 2. the npz the inputs were built from must hold these same real rows, in this order
    z = np.load(mi.BBS_DATA_NPZ)
    n_p = int(z["N_pseudo"])
    for name, mine in (("obs_rows", mapped["row"].values), ("obs_cols", mapped["col"].values),
                       ("obs_year", mapped["Year"].values), ("observed_results", mapped["SpeciesTotal"].values)):
        if not np.array_equal(np.asarray(z[name])[n_p:].astype(int), np.asarray(mine).astype(int)):
            raise SystemExit(f"bbs npz {name} does not match a fresh bbs.py mapping -- refusing")

    # 3. replay model_inputs' observation assembly (see model_inputs.ingest_data, step 3)
    land = (z["land"].astype(np.float32) > 0.5).astype(int)
    N_land = int(land.sum())
    rows, cols, years, counts = (np.asarray(z[k]) for k in ("obs_rows", "obs_cols", "obs_year", "observed_results"))
    r_rows, r_cols, r_years, r_counts = rows[n_p:], cols[n_p:], years[n_p:], counts[n_p:]
    sampling_density = len(np.unique(np.vstack((r_rows, r_cols)).T, axis=0)) / N_land
    upl = np.unique(np.vstack((rows[:n_p], cols[:n_p])).T, axis=0)
    n_target = max(int(len(upl) * sampling_density), 50)
    rng = np.random.default_rng(42)
    chosen = upl[rng.choice(len(upl), n_target, replace=False)] if len(upl) > n_target else upl
    yrs = np.unique(years[:n_p])
    p_rows = np.concatenate([chosen[:, 0]] * len(yrs)); p_cols = np.concatenate([chosen[:, 1]] * len(yrs))
    p_years = np.concatenate([np.full(len(chosen), y) for y in yrs])
    o_rows = np.concatenate([p_rows, r_rows]); o_cols = np.concatenate([p_cols, r_cols])
    o_years = np.concatenate([p_years, r_years])
    o_counts = np.concatenate([np.zeros_like(p_years), r_counts])
    o_key = np.concatenate([np.full(len(p_years), -1, np.int64), key])
    Ny, Nx = land.shape
    ok = (o_rows >= 0) & (o_rows < Ny) & (o_cols >= 0) & (o_cols < Nx)
    o_rows, o_cols, o_years, o_counts, o_key = (x[ok] for x in (o_rows, o_cols, o_years, o_counts, o_key))

    with open(os.path.join(input_dir, "metadata.pkl"), "rb") as f:
        meta = pickle.load(f)
    year_set = set(int(y) for y in meta["years"])
    keep = np.array([int(y) in year_set for y in o_years])
    o_rows, o_cols, o_years, o_counts, o_key = (x[keep] for x in (o_rows, o_cols, o_years, o_counts, o_key))

    # 4. element-by-element check against the inputs actually in use
    m_years = np.asarray(meta["years"])[np.asarray(meta["obs_time_indices"])]
    for name, mine, theirs in (("rows", o_rows, meta["obs_rows"]), ("cols", o_cols, meta["obs_cols"]),
                               ("years", o_years, m_years), ("counts", o_counts, meta["observed_results"])):
        if len(mine) != len(theirs) or not np.array_equal(np.asarray(mine).astype(int), np.asarray(theirs).astype(int)):
            raise SystemExit(f"replayed {name} != metadata.pkl ({len(mine)} vs {len(theirs)}) -- refusing")

    uniq = np.unique(o_key[o_key >= 0])
    route = np.where(o_key >= 0, np.searchsorted(uniq, o_key), -1).astype(np.int64)
    np.savez_compressed(a.out, route=route, n_routes=len(uniq), route_key=uniq)
    print(f"[route_ids] {a.out}: {len(route)} obs, {len(uniq)} routes, "
          f"{int((route < 0).sum())} pseudo-zeros (route -1); verified against metadata.pkl")


if __name__ == "__main__":
    main()
