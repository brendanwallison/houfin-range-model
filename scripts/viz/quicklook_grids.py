#!/usr/bin/env python3
"""Thumbnail-PNG quicklooks of the 25 km products (rasters + climate) for visual QC.

Renders band 1 of each gridded GeoTIFF, and (with --climate) grids the per-centroid
climate CSVs back onto the model grid, to small percentile-stretched PNGs
(NaN/ocean transparent) under ``<out>/<dataset>/<name>.png``. Dense time-series are
evenly sub-sampled (--sample for rasters; --climate-years / --climate-levels for
climate). Everything renders in parallel. Then tar the folder and scp it.

    python scripts/viz/quicklook_grids.py --climate                 # sampled, all datasets
    python scripts/viz/quicklook_grids.py --climate --all           # every raster (large)
    python scripts/viz/quicklook_grids.py --climate --climate-levels q10,q50,q90
"""
import argparse
import glob
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from tqdm import tqdm

# Run as a script from scripts/viz/, so the repo root isn't on sys.path by default
# ("No module named 'src'"). Add it before importing src (matches scripts/climate_climr.py).
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from src.config_utils import load_data_config


def _even_sample(items, k):
    """Evenly-spaced sub-sample of <= k items (keeps first/last), order preserved."""
    items = list(items)
    if k <= 0 or len(items) <= k:
        return items
    idx = sorted(set(np.linspace(0, len(items) - 1, k).round().astype(int)))
    return [items[i] for i in idx]


_YEAR_RE = re.compile(r"_(\d{4})(?=_grid\.tif$|\.tif$)")


def _key_year(path):
    """Split a grid filename into (variable-identity, year). ``{var}_{year}_grid.tif``
    and ``{var}_{lvl}_{year}_grid.tif`` -> key without the year token; static (no
    4-digit year) -> (basename, None)."""
    b = os.path.basename(path)
    m = _YEAR_RE.search(b)
    if not m:
        return b, None
    return b[:m.start()] + b[m.end():], int(m.group(1))


def _stratified_select(files, years_per_var, max_vars=None):
    """Rational quicklook subset: group by VARIABLE identity, then keep an evenly-
    spaced few YEARS per variable (first..last), so the output is a legible
    variable x year matrix instead of a flat blind sample mixing everything.
    ``max_vars`` optionally caps how many distinct variables (evenly across the
    sorted set). Static rasters (no year) are one-per-variable and always kept."""
    groups = {}
    for f in files:
        key, yr = _key_year(f)
        groups.setdefault(key, []).append((yr if yr is not None else -1, f))
    keys = sorted(groups)
    if max_vars:
        keys = _even_sample(keys, max_vars)
    chosen = []
    for k in keys:
        fs = [f for _, f in sorted(groups[k])]
        chosen.extend(fs if len(fs) <= years_per_var else _even_sample(fs, years_per_var))
    return chosen


def _arr_to_png(arr, png, max_dim=300, cmap="viridis"):
    """Percentile-stretch a 2-D array to a thumbnail PNG (NaN transparent)."""
    arr = np.asarray(arr, dtype="float32")
    h, w = arr.shape
    step = max(1, round(max(h, w) / max_dim))
    arr = arr[::step, ::step]
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return False
    lo, hi = np.nanpercentile(finite, [2, 98])
    if hi <= lo:
        hi = lo + 1e-9
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad(alpha=0.0)  # NaN/ocean transparent
    os.makedirs(os.path.dirname(png), exist_ok=True)
    plt.imsave(png, np.ma.masked_invalid(arr), cmap=cm, vmin=lo, vmax=hi)
    return True


def render_raster(task):
    """Render one GeoTIFF band-1 to a thumbnail PNG."""
    tif, png, max_dim, cmap = task
    try:
        import rasterio
        with rasterio.open(tif) as src:
            arr = np.ma.filled(src.read(1, masked=True).astype("float32"), np.nan)
        return "ok" if _arr_to_png(arr, png, max_dim, cmap) else "empty"
    except Exception as e:  # noqa: BLE001  one bad raster shouldn't kill the batch
        return f"error:{type(e).__name__}"


def render_climate_level(task):
    """Grid one climate level's sampled (variable, year) slices onto the model grid.

    Reads climate_<lvl>.csv (long-format: id, PERIOD, monthly vars), joins the
    id->row,col map from cell_centroids.csv, and renders a PNG per (var, year)."""
    lvl, csv_path, cen_path, years, vars_req, out_dir, max_dim, cmap = task
    import pandas as pd
    from src.data.combine.climate_io import grid_from_centroids
    cen = pd.read_csv(cen_path, usecols=["id", "row", "col"])
    ny, nx = int(cen["row"].max()) + 1, int(cen["col"].max()) + 1
    df = pd.read_csv(csv_path)
    df = df[df["PERIOD"].isin(years)]
    allvars = [c for c in df.columns if c not in ("id", "PERIOD", "row", "col")]
    varlist = allvars if vars_req in (None, "all") else [v for v in vars_req if v in allvars]
    made = 0
    for var in varlist:
        for yr in years:
            sub = df[df["PERIOD"] == yr]
            grid = grid_from_centroids(sub, cen, ny, nx, value_col=var)
            if _arr_to_png(grid, os.path.join(out_dir, f"climate_{lvl}", f"{var}_{yr}.png"),
                           max_dim, cmap):
                made += 1
    return lvl, made, len(varlist)


def render_states(task):
    """Render sampled channels of one (year, stream) state grid to thumbnails.

    ``state_{year}.npz`` holds one ``(H, W, C)`` array per stream; we render a
    sample of channels per stream, labeled by the schema's variable names when
    available. These are the ACTUAL per-year encoder inputs (climate/luh3/hyde EMA
    + soil/elevation static), so this validates what DESK sees, post-assembly."""
    npz_path, year, stream, channels, varnames, out_dir, max_dim, cmap = task
    with np.load(npz_path) as z:
        if stream not in z.files:
            return stream, year, 0
        arr = z[stream]                                  # (H, W, C)
    made = 0
    for ch in channels:
        if ch >= arr.shape[2]:
            continue
        label = varnames[ch] if ch < len(varnames) else f"ch{ch}"
        if _arr_to_png(arr[:, :, ch], os.path.join(out_dir, f"states_{stream}", f"{label}_{year}.png"),
                       max_dim, cmap):
            made += 1
    return stream, year, made


def default_groups(dr):
    """Standard 25 km raster products (eBird already writes _grid.png).

    ``climate_grid`` = the bio-year-aggregated per-year climate rasters the encoder
    actually consumes (rendered here as the faithful downstream product); the
    ``--climate`` flag additionally re-grids the raw per-centroid CSVs.
    """
    return {
        "ref_grid":     sorted(glob.glob(os.path.join(dr, "ref_grid_*km.tif"))),
        "land_mask":    sorted(glob.glob(os.path.join(dr, "land_mask", "*.tif"))),
        "luh3":         sorted(glob.glob(os.path.join(dr, "luh3_grid", "*.tif"))),
        "hyde":         sorted(glob.glob(os.path.join(dr, "hyde35_grid", "*.tif"))),
        "soilgrids":    sorted(glob.glob(os.path.join(dr, "soilgrids_grid", "**", "*.tif"), recursive=True)),
        "elevation":    sorted(glob.glob(os.path.join(dr, "elevation", "**", "*.tif"), recursive=True)),
        "climate_grid": sorted(glob.glob(os.path.join(dr, "climate_grid", "*.tif"))),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="output dir (default: $HOUFIN_PROCESSED/quicklooks)")
    ap.add_argument("--years-per-var", type=int, default=3,
                    help="years to render per variable (evenly spaced first..last); the rational default")
    ap.add_argument("--max-vars", type=int, default=None,
                    help="cap distinct variables per dataset (evenly across the sorted set)")
    ap.add_argument("--all", action="store_true", help="render every raster (can be thousands)")
    ap.add_argument("--max-dim", type=int, default=300, help="thumbnail max dimension in px")
    ap.add_argument("--cmap", default="viridis")
    ap.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 16))
    ap.add_argument("--include-ebird", action="store_true", help="also render eBird grid tifs (5000+)")
    ap.add_argument("--climate", action="store_true", help="also render gridded climate covariates")
    ap.add_argument("--climate-levels", default="q50", help="comma list of climate levels (default q50)")
    ap.add_argument("--climate-years", type=int, default=6, help="evenly-sampled years to render")
    ap.add_argument("--climate-vars", default="all", help="comma list of climate variables, or 'all'")
    ap.add_argument("--states", action="store_true",
                    help="also render per-year encoder state grids (state_{year}.npz)")
    ap.add_argument("--states-dir", default=None,
                    help="states dir (default: config paths.hist_dir); reads yearly_states/ + state_schema.json")
    ap.add_argument("--states-years", type=int, default=6, help="evenly-sampled state years to render")
    ap.add_argument("--states-channels", type=int, default=4,
                    help="evenly-sampled channels per stream to render")
    args = ap.parse_args()

    cfg = load_data_config()
    dr = cfg["datasets_root"]
    out = args.out or os.path.join(os.environ.get("HOUFIN_PROCESSED", dr), "quicklooks")

    # ---- raster products ----
    groups = default_groups(dr)
    if args.include_ebird:
        sub = cfg.get("ebird_raw_subdir", "ebird_weekly_2023") + "_grid"
        groups["ebird"] = sorted(glob.glob(os.path.join(dr, sub, "*.tif")))
    tasks = []
    for ds, files in groups.items():
        if not files:
            continue
        chosen = files if args.all else _stratified_select(files, args.years_per_var, args.max_vars)
        for f in chosen:
            name = os.path.splitext(os.path.basename(f))[0]
            tasks.append((f, os.path.join(out, ds, name + ".png"), args.max_dim, args.cmap))
        nvars = len({_key_year(f)[0] for f in files})
        print(f"{ds}: {len(chosen)}/{len(files)} rasters ({nvars} vars x <={args.years_per_var} yrs)",
              flush=True)

    if tasks:
        print(f"rendering {len(tasks)} raster thumbnails, {args.workers} workers -> {out}", flush=True)
        counts = {}
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for status in tqdm(ex.map(render_raster, tasks), total=len(tasks), mininterval=2):
                counts[status] = counts.get(status, 0) + 1
        print("rasters done:", counts, flush=True)

    # ---- climate covariates (gridded from the per-centroid CSVs) ----
    if args.climate:
        cen_path = os.path.join(dr, "elevation", "cell_centroids.csv")
        clim_dir = os.path.join(dr, "climate")
        levels = [lv.strip() for lv in args.climate_levels.split(",") if lv.strip()]
        probe = os.path.join(clim_dir, f"climate_{levels[0]}.csv") if levels else ""
        if not (os.path.exists(cen_path) and os.path.exists(probe)):
            print(f"[skip climate] need {cen_path} and {probe}", flush=True)
        else:
            import pandas as pd
            allyears = sorted(pd.read_csv(probe, usecols=["PERIOD"])["PERIOD"].unique().tolist())
            years = _even_sample(allyears, args.climate_years)
            vars_req = None if args.climate_vars == "all" else \
                [v.strip() for v in args.climate_vars.split(",")]
            ctasks = [(lvl, os.path.join(clim_dir, f"climate_{lvl}.csv"), cen_path, years,
                       vars_req, out, args.max_dim, args.cmap)
                      for lvl in levels if os.path.exists(os.path.join(clim_dir, f"climate_{lvl}.csv"))]
            print(f"climate: {len(ctasks)} levels x sampled years {years}", flush=True)
            with ProcessPoolExecutor(max_workers=min(len(ctasks), args.workers)) as ex:
                for lvl, made, nv in ex.map(render_climate_level, ctasks):
                    print(f"climate {lvl}: {made} PNGs ({nv} vars x {len(years)} yrs)", flush=True)

    # ---- per-year encoder states (the actual DESK inputs) ----
    if args.states:
        from src.config_utils import load_config
        hist_dir = args.states_dir or load_config().get("paths", {}).get("hist_dir")
        ydir = os.path.join(hist_dir, "yearly_states") if hist_dir else None
        schema_path = os.path.join(hist_dir, "state_schema.json") if hist_dir else None
        npzs = sorted(glob.glob(os.path.join(ydir, "state_*.npz"))) if ydir else []
        if not (npzs and schema_path and os.path.exists(schema_path)):
            print(f"[skip states] need {ydir}/state_*.npz and {schema_path} "
                  f"(run build_states first)", flush=True)
        else:
            with open(schema_path) as fh:
                streams = json.load(fh)["streams"]        # [{name, dim, variables, ...}]
            def _yr(p): return int(os.path.splitext(os.path.basename(p))[0].split("_")[1])
            chosen = _even_sample(npzs, args.states_years)
            stasks = []
            for p in chosen:
                for st in streams:
                    chans = _even_sample(list(range(st["dim"])), args.states_channels)
                    stasks.append((p, _yr(p), st["name"], chans, st.get("variables", []),
                                   out, args.max_dim, args.cmap))
            print(f"states: {len(chosen)} yrs x {len(streams)} streams "
                  f"x <={args.states_channels} chans -> {len(stasks)} tasks", flush=True)
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                tot = sum(made for _, _, made in ex.map(render_states, stasks))
            print(f"states: {tot} PNGs -> {out}", flush=True)

    parent, base = os.path.dirname(out), os.path.basename(out)
    print(f"\nBundle + scp:\n"
          f"  tar czf {base}.tgz -C {parent} {base}\n"
          f"  # from your PC:  scp <user>@ls6.tacc.utexas.edu:{parent}/{base}.tgz .", flush=True)


if __name__ == "__main__":
    main()
