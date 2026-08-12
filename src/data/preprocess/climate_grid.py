"""Rasterize the per-centroid climate CSVs into per-year model-grid GeoTIFFs.

climr writes long-format ``climate_{q10,q50,q90}.csv`` (``id, PERIOD, <monthly
cols>``) — a dead-end format that only the viz script read. This turns them into
the same per-year raster layout every other covariate uses, so the covariate
assembler (``streams.run_states``) can ingest climate like LUH-3/HYDE.

**Scheme ``monthly_bioyear_v2``: monthly structure is PRESERVED.** For each model
bio-year ``T`` (Aug(T-1)→Jul(T)) the 12 monthly columns of a base variable become
12 separate channels rather than one annual mean/total, so within-year contrast
(winter severity, seasonality amplitude, summer extremes) reaches the model
instead of being averaged away. See ``climate_io.bioyear_monthly``; the previous
scheme's annual collapse survives only as ``climate_io.bioyear_aggregate`` for QC.

Variable token: ``{base}_b{kk}m{MM}_{lvl}``, e.g. ``Tmax_b01m08_q50_1980_grid.tif``
— ``b{kk}`` is the 1-based position in the bio-year window (``b01`` =
``bio_year_start_month``), ``m{MM}`` the calendar month it resolves to. The level
stays last so the file matches the streamer's ``{var}_{year}_grid.tif`` pattern and
``discover_variables``' ``endswith("_{level}")`` filter keeps working.

Elevation levels are asymmetric by design (see ``levels_for_base``): every base
gets ``q50``, but only temperatures get ``q10``/``q90``, since within-cell relief
changes a temperature via the lapse rate and does far less to a flux. That keeps
the channel count near 12*N_bases + 24*3 rather than 36*N_bases.

Channel ORDER is authoritative in ``manifest.json`` beside the rasters, and
``build_states.discover_variables`` validates against it — adding channels
otherwise silently renumbers every existing channel index (and hence the saved
per-channel ``mu``/``sd``).

    python -m src.data.preprocess.climate_grid \
        --climate-dir $HOUFIN_DATA/climate --centroids $HOUFIN_DATA/elevation/cell_centroids.csv \
        --out $HOUFIN_DATA/climate_grid_monthly
"""
import argparse
import json
import os
import re

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config_utils import load_data_config
from src.data.combine.climate_io import (bioyear_month_columns, bioyear_monthly,
                                         grid_from_centroids, parse_month_columns)
from src.processing import regrid
from src.temporal import load_timeline, model_years

LEVELS = ("q10", "q50", "q90")

# Bases whose value genuinely varies with within-cell elevation (lapse rate), so
# the q10/q90 spread carries information. Matched case-insensitively, exact.
TEMP_BASES = ("tmin", "tmax", "tave", "tdmean")

SCHEME = "monthly_bioyear_v2"
MANIFEST = "manifest.json"

# A scheme-v2 variable token always ends "_b<kk>m<MM>_q<nn>"; the v1 annual tokens
# (e.g. "Tmax_q50") do not. This is how assert_no_legacy_rasters refuses a directory
# holding both -- see its docstring for why that cannot be caught downstream.
_V2_TOKEN = re.compile(r"_b\d{2}m\d{2}_q\d{2}$")
_YEAR_TIF = re.compile(r"^(?P<var>.+)_(?P<year>\d{4})_grid\.tif$")


def levels_for_base(base, levels):
    """Which elevation levels to write for ``base``, restricted to ``levels``.

    Temperatures get every requested level; everything else gets ``q50`` only.
    The single place the temperature/flux asymmetry is decided.
    """
    wanted = levels if base.lower() in TEMP_BASES else ["q50"]
    return [lv for lv in levels if lv in wanted]


def _ref_template():
    """Single-band model-grid template DataArray (float32) + its (ny, nx)."""
    ref = regrid.load_ref()
    band0 = ref.isel(band=0) if "band" in ref.dims else ref
    tmpl = band0.astype("float32")
    ny, nx = int(tmpl.sizes["y"]), int(tmpl.sizes["x"])
    return tmpl, ny, nx


def assert_no_legacy_rasters(out):
    """Refuse to write into a directory holding scheme-v1 (annual) rasters.

    Mixing schemes cannot be caught downstream: both spellings match the
    streamer's filename pattern, so the annual leftovers would simply join the
    channel set as extra variables.
    """
    legacy = []
    for f in os.listdir(out) if os.path.isdir(out) else []:
        m = _YEAR_TIF.match(f)
        if m and not _V2_TOKEN.search(m.group("var")):
            legacy.append(f)
    if legacy:
        raise SystemExit(
            f"{out} holds {len(legacy)} raster(s) from the old annual scheme "
            f"(e.g. {sorted(legacy)[:3]}). Monthly rasters must not be mixed with "
            f"them — the streamer would ingest both as channels. Use a fresh "
            f"--out (default: climate_grid_monthly) or remove the old files.")


def read_manifest(out):
    path = os.path.join(out, MANIFEST)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def assert_manifest_compatible(out, start_month):
    """Refuse to extend a directory written under a different scheme/window phase.

    Checked BEFORE any raster is written: a bio-year phase change silently
    redefines what every ``b{kk}`` channel means, so resuming into such a tree
    would interleave two incompatible conventions.
    """
    prev = read_manifest(out)
    if prev is None:
        return
    for key, new in (("scheme", SCHEME), ("bio_year_start_month", int(start_month))):
        if prev.get(key) != new:
            raise SystemExit(
                f"{os.path.join(out, MANIFEST)} was written with {key}="
                f"{prev.get(key)!r} but this run uses {new!r}. The rasters on "
                f"disk are not comparable with the ones this run would write; "
                f"use a fresh --out.")


def write_manifest(out, start_month, vars_by_level, years):
    """Record the authoritative channel order for ``build_states`` to validate."""
    variables = sorted(v for vs in vars_by_level.values() for v in vs)
    manifest = {
        "scheme": SCHEME,
        "bio_year_start_month": int(start_month),
        "levels": {lv: sorted(vs) for lv, vs in sorted(vars_by_level.items())},
        "variables": variables,
        "n_variables": len(variables),
        "years": [int(min(years)), int(max(years))],
    }
    with open(os.path.join(out, MANIFEST), "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--climate-dir", default=None, help="dir with climate_{lvl}.csv")
    ap.add_argument("--centroids", default=None, help="cell_centroids.csv (id,row,col,...)")
    ap.add_argument("--out", default=None, help="output climate_grid_monthly dir")
    ap.add_argument("--levels", default=",".join(LEVELS))
    args = ap.parse_args()

    cfg = load_data_config()
    dr = cfg["datasets_root"]
    obs_ts = cfg.get("climate", {}).get("obs_ts_dataset", "cru.gpcc")
    climate_dir = args.climate_dir or os.path.join(dr, "climate")
    centroids = args.centroids or os.path.join(dr, "elevation", "cell_centroids.csv")
    out = args.out or os.path.join(dr, "climate_grid_monthly")
    levels = [lv.strip() for lv in args.levels.split(",") if lv.strip()]
    os.makedirs(out, exist_ok=True)
    assert_no_legacy_rasters(out)

    tl = load_timeline()
    start_month = tl["bio_year_start_month"]
    assert_manifest_compatible(out, start_month)
    years = model_years(tl)
    cen = pd.read_csv(centroids, usecols=["id", "row", "col"])
    tmpl, ny, nx = _ref_template()

    total_written = 0
    vars_by_level = {}
    for lvl in levels:
        csv = os.path.join(climate_dir, f"climate_{lvl}.csv")
        if not os.path.exists(csv):
            print(f"[skip {lvl}] missing {csv}", flush=True)
            continue
        df = pd.read_csv(csv)
        # climr adds a DATASET column when obs_ts_dataset is set; if several were
        # requested, keep only our source so ids stay unique per PERIOD.
        if "DATASET" in df.columns and df["DATASET"].nunique() > 1:
            df = df[df["DATASET"] == obs_ts]
            print(f"[{lvl}] filtered DATASET -> {obs_ts} ({len(df)} rows)", flush=True)
        # PERIOD must be an integer year for the bio-year join; coerce + report so a
        # non-year encoding (or a range mismatch with the model timeline) is visible.
        raw_periods = df["PERIOD"].astype(str).unique()[:8]
        df["PERIOD"] = pd.to_numeric(df["PERIOD"], errors="coerce").astype("Int64")
        df = df.dropna(subset=["PERIOD"])
        if df.empty:
            raise SystemExit(f"[{lvl}] no numeric PERIOD years after coercion; "
                             f"raw PERIOD values look like {list(raw_periods)}")
        df["PERIOD"] = df["PERIOD"].astype(int)
        all_groups = parse_month_columns(df.columns)
        # Only the bases this level is responsible for (q10/q90: temperatures only).
        groups = {b: mm for b, mm in all_groups.items() if lvl in levels_for_base(b, levels)}
        if not groups:
            print(f"[{lvl}] no bases assigned to this level "
                  f"(non-temperature bases are q50-only); skipping", flush=True)
            continue
        cols = [c for b in groups for c in bioyear_month_columns(b, start_month)]
        vars_by_level[lvl] = [f"{c}_{lvl}" for c in cols]
        pmin, pmax, pn = int(df["PERIOD"].min()), int(df["PERIOD"].max()), df["PERIOD"].nunique()
        print(f"[{lvl}] PERIOD {pmin}..{pmax} ({pn} yrs); {len(all_groups)} base vars "
              f"{sorted(all_groups)}; {len(groups)} at this level {sorted(groups)} "
              f"-> {len(cols)} channels; model bio-years {years[0]}..{years[-1]}", flush=True)
        wrote = 0
        for yr in tqdm(years, desc=f"climate {lvl}", mininterval=2):
            paths = {c: os.path.join(out, f"{c}_{lvl}_{yr}_grid.tif") for c in cols}
            if all(os.path.exists(p) for p in paths.values()):
                wrote += 1
                continue
            mon = bioyear_monthly(df, yr, start_month, month_groups=groups)
            if mon.empty:
                continue  # bio-year straddles a data gap (before obs start / after obs end)
            mon = mon.reset_index()
            for col in cols:
                grid = grid_from_centroids(mon, cen, ny, nx, value_col=col)
                da = tmpl.copy(data=grid)
                da.rio.write_nodata(np.nan, inplace=True)
                # ~30k small float32 rasters: deflate+predictor gets 2-4x on
                # smooth climate fields and keeps the tree manageable on Lustre.
                da.rio.to_raster(paths[col], compress="DEFLATE", predictor=3)
            wrote += 1
        print(f"[{lvl}] wrote/kept {wrote}/{len(years)} bio-years "
              f"({len(cols)} channels each)", flush=True)
        total_written += wrote
    if total_written == 0:
        raise SystemExit(
            "climate_grid produced NO rasters: every bio-year aggregation was empty. "
            "This means the CSV PERIOD years do not overlap the model bio-years "
            f"({years[0]}..{years[-1]}) as consecutive (T-1, T) pairs. Check the "
            "PERIOD range printed above against the model timeline.")
    manifest = write_manifest(out, start_month, vars_by_level, years)
    print(f"Done ({total_written} level-years; {manifest['n_variables']} channels) -> {out}",
          flush=True)


if __name__ == "__main__":
    main()
