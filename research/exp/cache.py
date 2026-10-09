"""Build the per-checkpoint research cache (Phase 0.4): what the fast arm, the atlas and the null checks
read, computed ONCE with the validation suite's own functions, as plain .npy files.

    ESK_DESK_CONFIG=<overlay> python research/exp/cache.py --out <dir>

Row-aligned to ``keys.npy`` (every surveyed cell-year, the suite's ``load_all_species`` keys):

    keys.npy            (N, 3) int32  row, col, year
    X_comm.npy          (N, n_comm) float32  reference-community counts (route mean per cell-year)
    X_dev.npy           (N, n_dev)  float32  out-of-community DEV species counts
    sealed/X_sealed.npy (N, n_seal) float32  read ONLY by research/adopt.py (research/lib/seal.py)
    cells.npy, years.npy, z_ema_all.npy, z_raw_all.npy  DENSE (n_cells, n_years, L): DESK for every
                        surveyed cell in every year of the encoder window, surveyed or not
    z_ema.npy, z_raw.npy, key_cell_index.npy  the same gathered at every key
    F_cov.npy, F_cov_raw.npy (N, C) float32  covariates as the covariate GPs see them
    split.npz           holdout / buffer masks, is_train, group, block_id, withheld, common
    route_years.csv     country, state, route, year, row, col (QC route-years: who surveyed what)
    meta.json           run dir, overlay, basis, EMA half-life, code sha, layouts, seal rule, timings

Nothing here selects, scores or fits: it only materializes inputs, so the cache never needs a
version bump when an analysis changes.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib import seal  # noqa: E402


def _sha():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
    except Exception:  # noqa: BLE001 -- a snapshot has no .git; the runner records the sha anyway
        return None


def route_year_table():
    """QC-passing route-years with their grid cell: the identity of who surveyed each cell-year."""
    import pandas as pd
    from src.data.preprocess import bbs
    from src.data.preprocess.bbs_community import route_grid_map
    _obs, coverage = bbs.load_usca_observations(aou_filter=bbs.HOUSE_FINCH_AOU,
                                                return_coverage=True)
    routes = bbs.load_routes()
    land_mask, _, transform, crs, nx, ny = bbs.load_grid_reference(bbs.MASK_PATH)
    rc = route_grid_map(routes, transform, crs, nx, ny, land_mask)
    keys = ["CountryNum", "StateNum", "Route"]
    tab = coverage[keys + ["Year"]].drop_duplicates().merge(rc, on=keys, how="inner")
    return pd.DataFrame({"country": tab["CountryNum"].astype(int),
                         "state": tab["StateNum"].astype(int), "route": tab["Route"].astype(int),
                         "year": tab["Year"].astype(int), "row": tab["row"].astype(int),
                         "col": tab["col"].astype(int)})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-covariates", action="store_true")
    a = ap.parse_args()
    t0 = time.perf_counter()
    os.makedirs(os.path.join(a.out, "sealed"), exist_ok=True)

    from src.config_utils import load_config
    from src.community_encoder.train_DESK import validate_gp_species as vgs
    from src.community_encoder.train_DESK.validate_bbs_routes import desk_z_ema
    from src.community_encoder.train_DESK.validation_core import load_holdout_masks, row_splits

    cfg = load_config()
    run_dir = cfg["paths"]["desk_output_dir"]
    tr_cfg = (cfg.get("desk", {}) or {}).get("trend", {}) or {}
    timings = {}

    X_raw, keys, layout = vgs.load_all_species(cfg)
    nc = layout["n_community"]
    dev, sealed = seal.split(layout["evaluation"])
    ev_ix = {c: i for i, c in enumerate(layout["evaluation"])}
    np.save(os.path.join(a.out, "keys.npy"), keys.astype("int32"))
    np.save(os.path.join(a.out, "X_comm.npy"), X_raw[:, :nc].astype("float32"))
    np.save(os.path.join(a.out, "X_dev.npy"),
            X_raw[:, [nc + ev_ix[c] for c in dev]].astype("float32"))
    np.save(os.path.join(a.out, "sealed", "X_sealed.npy"),
            X_raw[:, [nc + ev_ix[c] for c in sealed]].astype("float32"))
    timings["species"] = round(time.perf_counter() - t0, 1)
    print(f"[cache] {len(keys):,} cell-years; {nc} community, {len(dev)} dev, {len(sealed)} sealed",
          flush=True)

    # DENSE z for every surveyed cell in EVERY year of the encoder window, surveyed or not: the
    # no-change reference, lags, an EMA at another half-life, and a cell mean over a fixed window
    # all need z where nobody surveyed. One causal encode serves every key.
    cells = np.unique(keys[:, :2], axis=0)
    _dm = np.load(os.path.join(run_dir, "desk_meta.npz"), allow_pickle=True)
    w0 = int(_dm["ema_warmup_start"]) if "ema_warmup_start" in _dm.files else 1940  # as desk_z_ema
    w1 = int(keys[:, 2].max())
    years = np.arange(w0, w1 + 1)
    grid = np.array([(r, c, y) for r, c in cells for y in years], "int32")
    t1 = time.perf_counter()
    Zg, zinfo, Zg_raw = desk_z_ema(cfg, grid, return_raw=True)
    L = Zg.shape[1]
    z_all = Zg.reshape(len(cells), len(years), L).astype("float32")
    z_all_raw = Zg_raw.reshape(len(cells), len(years), L).astype("float32")
    np.save(os.path.join(a.out, "cells.npy"), cells.astype("int32"))
    np.save(os.path.join(a.out, "years.npy"), years.astype("int32"))
    np.save(os.path.join(a.out, "z_ema_all.npy"), z_all)
    np.save(os.path.join(a.out, "z_raw_all.npy"), z_all_raw)
    # and gathered at every key, for convenience
    cell_ix = {(int(r), int(c)): i for i, (r, c) in enumerate(cells)}
    ci = np.array([cell_ix[(int(r), int(c))] for r, c in keys[:, :2]])
    yi = keys[:, 2] - w0
    np.save(os.path.join(a.out, "z_ema.npy"), z_all[ci, yi])
    np.save(os.path.join(a.out, "z_raw.npy"), z_all_raw[ci, yi])
    np.save(os.path.join(a.out, "key_cell_index.npy"), ci.astype("int32"))
    n = len(keys)
    timings["encode"] = round(time.perf_counter() - t1, 1)

    if not a.skip_covariates:
        t1 = time.perf_counter()
        Fc, Fr = vgs.covariates_for_keys(cfg, keys, both=True)
        np.save(os.path.join(a.out, "F_cov.npy"), np.asarray(Fc, "float32"))
        np.save(os.path.join(a.out, "F_cov_raw.npy"), np.asarray(Fr, "float32"))
        timings["covariates"] = round(time.perf_counter() - t1, 1)

    ho, bf, note = load_holdout_masks(run_dir, tr_cfg.get("buffer_floor"))
    withheld = [int(y) for y in (tr_cfg.get("holdout_years") or [])]
    common = [int(y) for y in (tr_cfg.get("common_holdout_years") or [])]
    is_train, group, block_id = row_splits(keys, ho, bf, int(tr_cfg.get("block_cells", 6)),
                                           withheld, common)
    np.savez_compressed(os.path.join(a.out, "split.npz"), holdout=ho, buffer=bf,
                        is_train=is_train, group=group, block_id=block_id,
                        withheld=np.array(withheld, "int32"), common=np.array(common, "int32"))

    t1 = time.perf_counter()
    route_year_table().to_csv(os.path.join(a.out, "route_years.csv"), index=False)
    timings["routes"] = round(time.perf_counter() - t1, 1)

    dm = np.load(os.path.join(run_dir, "desk_meta.npz"), allow_pickle=True)
    meta = {"run_dir": run_dir, "overlay": os.environ.get("ESK_DESK_CONFIG"),
            "basis": cfg["desk"].get("z_dir"), "ema": zinfo, "latent_dim": int(dm["latent_dim"]),
            "best_epoch": int(dm["best_epoch"]) if "best_epoch" in dm.files else None,
            "buffer_note": note, "withheld": withheld, "common": common,
            "community": layout["community"], "dev_species": dev, "sealed_species": sealed,
            "seal_rule": f"sha256({seal.SALT!r} + code) odd -> sealed (research/lib/seal.py)",
            "code_sha": _sha(), "n_rows": int(n), "timings_s": timings,
            "elapsed_s": round(time.perf_counter() - t0, 1)}
    with open(os.path.join(a.out, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1, default=str)
    print(f"[cache] done in {meta['elapsed_s']} s: {timings}", flush=True)


if __name__ == "__main__":
    main()
