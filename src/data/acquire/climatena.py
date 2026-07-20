"""Climate acquire: drive climr (R) to downscale monthly climate on the model grid.

This is the Linux/HPC-native climate route: it shells out to ``climate_climr.R``
via ``Rscript`` (climr is pure-R ClimateNA downscaling — no Windows .exe, no
throttled web API). Prerequisites (external to Python, like the ClimateNA .exe
would have been): R + the ``climr`` package installed, and the elevation step
(preprocess/elevation.py) already run to produce ``cell_centroids.csv`` (the
model-grid centroids + p10/p50/p90 elevation per cell).

For each of the three elevation levels, climr downscales monthly observed
climate (CRU TS temp + GPCC precip) for FIRST_YEAR..END_YEAR at the cell
centroids, giving climate at low/median/high sub-cell elevation directly on the
model grid — no 1 km climate is ever materialized.

Parallelism: the downscale is embarrassingly parallel over centroids, but climr
uses terra internally and terra/GDAL objects are NOT fork-safe, so we can't fork
R workers off one loaded process. Instead we split the centroids into chunks and
run one INDEPENDENT Rscript process per chunk (each with its own climr/terra
state, reading the shared warm cache read-only), then concatenate the per-chunk
CSVs. Each R process is pinned to one thread so N chunks use N cores cleanly.
Serial (``--workers 1``) reproduces the original single-process behavior.

Usage:
    python scripts/climate_climr.py --centroids <dir>/cell_centroids.csv --out <dir>
"""
import argparse
import math
import os
import subprocess
import sys
import time
import multiprocessing
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

from src.config_utils import load_data_config
from src.temporal import load_timeline

_R_SCRIPT = os.path.join(os.path.dirname(__file__), "climate_climr.R")
_WARM_SCRIPT = os.path.join(os.path.dirname(__file__), "warm_climr_cache.R")
LEVELS = ("q10", "q50", "q90")


def _climr_cache_dir():
    """climr's cache root = tools::R_user_dir('climr','cache'), which honors
    R_USER_CACHE_DIR (env.sh points it at persistent $WORK) and appends R/climr."""
    base = os.environ.get("R_USER_CACHE_DIR")
    root = base if base else os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(root, "R", "climr")


def _assert_climr_cache_warm():
    """Exit fast + actionably if the climr reference cache is cold. The offline
    (db_option=local) downscale can't download; warming is a login-node step."""
    meta = os.path.join(_climr_cache_dir(), "reference", "refmap_climr", "meta_data.csv")
    if os.path.exists(meta) and os.path.getsize(meta) > 0:
        return
    raise SystemExit(
        f"[climate] climr cache is COLD -- no reference map at\n"
        f"    {meta}\n"
        f"  The offline downscale (db_option=local) cannot download it (compute nodes\n"
        f"  have no internet). Warm it ONCE on a LOGIN node, then re-run this stage:\n"
        f"    bash scripts/tacc/warm_climr.sh\n"
        f"  (cache root from R_USER_CACHE_DIR="
        f"{os.environ.get('R_USER_CACHE_DIR', '<unset -> ~/.cache>')})")

# climr's observed-climate (CRU TS / GPCC) extent. downscale() errors if obs_years
# fall outside this; bump these (or override via data_config "climate") when climr
# ships a newer observed dataset.
CLIMR_OBS_MIN_YEAR = 1901
CLIMR_OBS_MAX_YEAR = 2024


def build_command(centroids_csv, out_dir, start_year, end_year, rscript="Rscript",
                  obs_ts_dataset="cru.gpcc", nthread=1, db_option="local"):
    """Construct the Rscript command (kept pure/testable, separate from execution).

    ``obs_ts_dataset`` names climr's observed time-series source (default
    ``cru.gpcc``); without it climr returns only the 1961-1990 reference normal.
    ``nthread`` is climr's in-process parallelism over the point table.
    ``db_option="local"`` makes climr download+cache the anomaly rasters and process
    LOCALLY — the default ``auto`` runs time-series on climr's remote DB server,
    which an internet-less compute node can't reach.
    """
    return [rscript, _R_SCRIPT, centroids_csv, out_dir, str(start_year), str(end_year),
            obs_ts_dataset, str(int(nthread)), db_option]


def worker_count(n_items, cap=96):
    """Parallel R processes: HOUFIN_CLIMATE_WORKERS, else SLURM/cpu count, capped.

    Each chunk is a full R+climr process, but measured usage is light (~0.7 GB
    physical/worker on LS6), so the cap is 96 (≈86 GB at that rate, well under a
    238 GB node) to actually fill a 128-core node — a 48-worker run used only
    ~37% of cores. Override with HOUFIN_CLIMATE_WORKERS (up/down) as memory/IO
    headroom dictates; ``min(n, n_items)`` still bounds it by the chunk count.
    """
    env = os.environ.get("HOUFIN_CLIMATE_WORKERS")
    if env:
        n = int(env)
    else:
        slurm = os.environ.get("SLURM_CPUS_ON_NODE")
        n = int(slurm) if slurm else (os.cpu_count() or 1)
        n = min(n, cap)
    return max(1, min(n, n_items or 1))


def _split_centroids(centroids_csv, n_chunks, chunk_dir):
    """Split into <= n_chunks contiguous CSVs (header replicated). Deterministic, so
    re-runs produce identical chunks (resume-friendly). Returns list of paths."""
    with open(centroids_csv, newline="") as fh:
        lines = fh.read().splitlines()
    header, data = lines[0], [ln for ln in lines[1:] if ln.strip()]
    n_chunks = max(1, min(n_chunks, len(data)))
    size = math.ceil(len(data) / n_chunks)
    os.makedirs(chunk_dir, exist_ok=True)
    paths = []
    for i in range((len(data) + size - 1) // size):
        part = data[i * size:(i + 1) * size]
        p = os.path.join(chunk_dir, f"chunk_{i:03d}.csv")
        with open(p, "w") as out:
            out.write(header + "\n" + "\n".join(part) + "\n")
        paths.append(p)
    return paths


def _split_centroids_spatial(centroids_csv, tiles_per_axis, chunk_dir):
    """Split into GEOGRAPHIC tiles (row/col blocks) so each chunk's bounding box —
    and thus the reference raster climr must load/merge for it — is small enough to
    fit memory. Scattered count-based chunks each span the whole region (OOM); a
    tile is a contiguous block. Sub-points of a cell share (row,col) so they stay in
    one tile (parent-intact for the subgrid aggregation). Returns chunk CSV paths."""
    import pandas as pd
    df = pd.read_csv(centroids_csv)
    nrow, ncol = int(df["row"].max()) + 1, int(df["col"].max()) + 1
    th = max(1, math.ceil(nrow / tiles_per_axis))
    tw = max(1, math.ceil(ncol / tiles_per_axis))
    tile = (df["row"] // th) * tiles_per_axis + (df["col"] // tw)
    os.makedirs(chunk_dir, exist_ok=True)
    paths = []
    for i, (_, sub) in enumerate(df.groupby(tile)):
        p = os.path.join(chunk_dir, f"chunk_{i:03d}.csv")
        sub.to_csv(p, index=False)
        paths.append(p)
    return paths


def quantile_aggregate(points, id_parent, quantiles=(0.10, 0.50, 0.90), levels=LEVELS):
    """Spatial quantiles of downscaled climate per (parent cell, PERIOD) — pure.

    ``points``: per-sub-point downscale output (``id, PERIOD, <vars>[, DATASET]``).
    ``id_parent``: ``id -> parent_id`` map. Returns ``{level: DataFrame(id=parent,
    PERIOD, <vars>)}`` where each level is the corresponding within-cell quantile —
    the same schema the centroid path emits, so downstream is unchanged.
    """
    import warnings
    from pandas.errors import PerformanceWarning
    df = points.merge(id_parent, on="id", how="inner")
    varcols = [c for c in points.columns if c not in ("id", "PERIOD", "DATASET", "parent_id")]
    grouped = df.groupby(["parent_id", "PERIOD"])[varcols]
    # All quantiles in ONE pass (not one groupby.quantile per level). climr emits ~200
    # columns, so pandas builds the quantile frame column-by-column and warns "DataFrame
    # is highly fragmented" (a PerformanceWarning -- harmless, internal); suppress it, and
    # the per-level .copy() hands back a de-fragmented frame.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PerformanceWarning)
        allq = grouped.quantile(list(quantiles))       # rows: (parent_id, PERIOD, quantile)
    out = {}
    for q, lvl in zip(quantiles, levels):
        out[lvl] = (allq.xs(q, level=-1).reset_index()
                    .rename(columns={"parent_id": "id"}).copy())
    return out


def _aggregate_chunk(chunk_out, chunk_csv):
    """Aggregate a subgrid chunk's per-sub-point output into the 3 level CSVs."""
    import pandas as pd
    pts = pd.read_csv(os.path.join(chunk_out, "climate_points.csv"))
    cen = pd.read_csv(chunk_csv, usecols=["id", "parent_id"])
    for lvl, df in quantile_aggregate(pts, cen).items():
        df.to_csv(os.path.join(chunk_out, f"climate_{lvl}.csv"), index=False)


def _run_chunk(chunk_csv, chunk_out, start, end, rscript, env, subgrid=False, nthread=1,
               db_option="local", obs_ts_dataset="cru.gpcc"):
    """Run one chunk's Rscript; skip if its 3 level CSVs already exist (resume).

    In subgrid mode the R step writes per-sub-point ``climate_points.csv`` (one
    downscale at true elevations); we then quantile-aggregate it to the 3 level CSVs.
    """
    os.makedirs(chunk_out, exist_ok=True)
    if all(os.path.exists(os.path.join(chunk_out, f"climate_{lvl}.csv")) for lvl in LEVELS):
        return chunk_csv, 0, None, "exists"
    log = chunk_csv[:-4] + ".log"
    with open(log, "w") as lf:
        rc = subprocess.run(build_command(chunk_csv, chunk_out, start, end, rscript,
                                          obs_ts_dataset=obs_ts_dataset, nthread=nthread,
                                          db_option=db_option),
                            stdout=lf, stderr=subprocess.STDOUT, env=env).returncode
        if rc == 0 and subgrid:
            try:
                _aggregate_chunk(chunk_out, chunk_csv)
            except Exception as e:  # noqa: BLE001  surface as a chunk failure
                lf.write(f"\n[aggregate] FAILED: {type(e).__name__}: {e}\n")
                rc = 1
    return chunk_csv, rc, log, "ok"


def _concat_levels(chunk_outs, out_dir):
    """Concatenate per-chunk climate_<lvl>.csv into one CSV per level (header once)."""
    for lvl in LEVELS:
        parts = [os.path.join(c, f"climate_{lvl}.csv") for c in chunk_outs]
        missing = [p for p in parts if not os.path.exists(p)]
        if missing:
            raise SystemExit(f"missing chunk outputs for level {lvl}: {missing[:3]}")
        dest = os.path.join(out_dir, f"climate_{lvl}.csv")
        with open(dest, "w") as out:
            for j, p in enumerate(parts):
                with open(p) as fh:
                    body = fh.read().splitlines()
                out.write("\n".join(body if j == 0 else body[1:]) + "\n")
        print(f"concatenated {len(parts)} chunks -> {dest}", flush=True)


def _ensure_subcell_centroids(cfg, out_csv, grid):
    """Generate the sub-cell mesh from the DEM + ref grid if it isn't present yet."""
    import glob
    import rasterio
    from src.data.preprocess.subcell_centroids import (build_subcell_centroids, write_csv,
                                                        rasterize_land_fine)
    from src.data.preprocess.bbs import load_grid_reference
    dem_dir = os.path.join(cfg["datasets_root"], cfg.get("dem", {}).get("out_subdir", "dem"))
    found = sorted(glob.glob(os.path.join(dem_dir, "*.tif")))
    if not found:
        raise SystemExit(f"subgrid climate needs a DEM in {dem_dir} (run scripts/download_dem.py)")
    # Drop ocean sub-points (elevation alone keeps them: the DEM gives ocean a finite
    # value). Two filters: 25 km parent ocean mask (grid alignment) + a fine sub-point
    # land mask from the same Natural Earth polygon (coastal water points).
    mask_path = cfg.get("latent_cube", {}).get("water_mask_path") \
        or os.path.join(cfg["datasets_root"], "land_mask", "ocean_mask_25km.tif")
    land_mask = load_grid_reference(mask_path)[0] if os.path.exists(mask_path) else None
    if land_mask is None:
        print(f"[subgrid] WARNING: 25 km ocean mask not found at {mask_path}.", flush=True)
    land_source = cfg.get("coastline", {}).get("land_source")
    if land_source and not os.path.isabs(land_source):
        land_source = os.path.join(cfg["datasets_root"], land_source)
    with rasterio.open(cfg["grid"]["ref_raster"]) as ref:
        fine_land = None
        if land_source and os.path.exists(land_source):
            fine_land = rasterize_land_fine(land_source, ref.crs, ref.transform,
                                            ref.height, ref.width, grid)
        else:
            print(f"[subgrid] WARNING: land polygon not found ({land_source}); "
                  f"no fine sub-point ocean mask.", flush=True)
        cols = build_subcell_centroids(found[0], ref.transform, ref.crs, ref.height,
                                       ref.width, grid, land_mask=land_mask, fine_land=fine_land)
    write_csv(out_csv, cols)
    print(f"[subgrid] generated {cols['id'].size} sub-points ({grid}x{grid}/cell) -> {out_csv}",
          flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--centroids", default=None,
                    help="centroids CSV; default from climate.mode "
                         "(subcell_centroids.csv for subgrid, else cell_centroids.csv)")
    ap.add_argument("--out", default=None, help="output dir for climate_{q10,q50,q90}.csv "
                    "(not needed with --warm-cache)")
    ap.add_argument("--rscript", default="Rscript")
    ap.add_argument("--workers", type=int, default=None,
                    help="R PROCESSES over centroid chunks (default 1; HOUFIN_CLIMATE_WORKERS). "
                         "climr's DuckDB serializes across processes, so keep this small.")
    ap.add_argument("--threads", type=int, default=None,
                    help="climr nthread WITHIN each process (default: node cpus / workers; "
                         "HOUFIN_CLIMATE_THREADS) — the parallelism knob that actually scales here.")
    ap.add_argument("--dry-run", action="store_true", help="print the command and exit")
    ap.add_argument("--warm-cache", action="store_true",
                    help="download+cache climr refmap + obs rasters (low memory, run on a "
                         "networked LOGIN node), then exit; compute nodes read the cache offline")
    args = ap.parse_args()

    cfg = load_data_config()
    tl = load_timeline(cfg)
    start, end = tl["first_year"], tl["end_year"]

    # The first model bio-year is Aug(first_year-1) -> Jul(first_year), so it needs
    # the calendar year BEFORE first_year. Request first_year-1 .. end_year, clamped
    # to climr's observed extent (downscale() rejects out-of-range obs_years; the
    # combine streamer EMA-carries covariates past obs_max).
    ccfg = cfg.get("climate", {})
    obs_ts_dataset = ccfg.get("obs_ts_dataset", "cru.gpcc")
    obs_min = int(ccfg.get("climr_obs_min_year", CLIMR_OBS_MIN_YEAR))
    obs_max = int(ccfg.get("climr_obs_max_year", CLIMR_OBS_MAX_YEAR))
    cstart, cend = max(start - 1, obs_min), min(end, obs_max)
    print(f"[climate] obs_years {cstart}:{cend} (first_year-1={start - 1} for the "
          f"bio-year lookback; climr extent {obs_min}:{obs_max})", flush=True)
    start, end = cstart, cend

    # Resolve mode + the centroids file (build the sub-cell mesh if absent). WARM and
    # PROCESS use the SAME centroids + tiling so the per-tile bounding boxes climr
    # caches on the login node match what compute reads offline.
    mode = ccfg.get("mode", "subgrid")
    subgrid_mode = (mode == "subgrid")            # picks the DEFAULT centroids file
    db_option = ccfg.get("db_option", "local")   # 'local' = download+cache+process offline
    tiles_per_axis = int(ccfg.get("tiles", 8))   # geographic tiles = ceil to <=tiles^2 chunks
    elev_dir = os.path.join(cfg["datasets_root"], "elevation")
    centroids = args.centroids or os.path.join(
        elev_dir, "subcell_centroids.csv" if subgrid_mode else "cell_centroids.csv")
    if subgrid_mode and not args.centroids and not os.path.exists(centroids):
        _ensure_subcell_centroids(cfg, centroids, int(ccfg.get("subgrid", {}).get("grid", 5)))
    if not os.path.exists(centroids):
        raise SystemExit(f"centroids file not found: {centroids} (run preprocess/elevation.py first)")

    # Cache-warming (networked LOGIN node): download refmap + obs rasters into the
    # climr cache, TILED so no single merge exceeds the node's memory, then exit.
    # Same centroids + tiling as processing => cached bounding boxes match.
    if args.warm_cache:
        # Login nodes cap processes/threads (ulimit -u ~300) and memory: pin every
        # thread pool to 1 (GDAL's included -> avoids CPLCreateJoinableThread EAGAIN)
        # and cap GDAL's block cache. The warm script also avoids terra::writeRaster
        # for the refmap (verbatim copy + header-only band rename) so no heavy 73-band
        # decode/re-encode runs here -- that write is what crashed the login cgroup.
        wenv = dict(os.environ)
        wenv.update(GDAL_NUM_THREADS="1", GDAL_CACHEMAX="256", OMP_NUM_THREADS="1",
                    OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                    VECLIB_MAXIMUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
        # PROCESS ISOLATION: even without the refmap re-encode, one long-lived R
        # process leaks GDAL joinable threads across tiles (input_obs_ts writeRaster)
        # until ulimit -u is exhausted (~tile 30). So spawn a FRESH R process per
        # geographic tile -- each warms one tile and exits, reaping its threads.
        # Tiles are resumable (cached refmap/obs are skipped), so a crashed tile just
        # re-runs. Empty tiles exit fast. tile value range is [0, tiles^2).
        n_tiles = tiles_per_axis * tiles_per_axis
        base = [args.rscript, _WARM_SCRIPT, centroids, obs_ts_dataset, str(start),
                str(end), str(tiles_per_axis)]
        print(f"[warm-cache] {n_tiles} tiles, one R process each (process-isolated)",
              flush=True)
        failed = []
        for ti in range(n_tiles):
            rc = subprocess.run(base + [str(ti)], env=wenv).returncode
            if rc != 0:
                print(f"[warm-cache] tile {ti} FAILED (rc={rc})", flush=True)
                failed.append(ti)
        if failed:
            print(f"[warm-cache] {len(failed)} tile(s) failed: {failed} -- "
                  f"re-run to resume (cached tiles skip).", flush=True)
            sys.exit(1)
        print("[warm-cache] all tiles warmed; compute reads the cache offline.", flush=True)
        sys.exit(0)

    if not args.out:
        raise SystemExit("--out is required (except with --warm-cache)")
    if args.dry_run:
        print(f"mode={mode} db_option={db_option} tiles={tiles_per_axis}^2; climr command:",
              " ".join(build_command(centroids, args.out, start, end, args.rscript,
                                     obs_ts_dataset=obs_ts_dataset, db_option=db_option)))
        return
    # Fail fast BEFORE spawning any work if the offline path has no cache to read:
    # the compute node can't download, and the warm is a login-node step. Turns an
    # obscure deep climr crash into an immediate, actionable message.
    if db_option == "local":
        _assert_climr_cache_warm()
    os.makedirs(args.out, exist_ok=True)

    # Path follows the FILE's columns (matches climate_climr.R): a sub-cell file has
    # parent_id -> spatial-quantile aggregation; a cell file (elev_q*) does not.
    with open(centroids) as fh:
        header = fh.readline().strip().split(",")
        n_cen = sum(1 for _ in fh)
    subgrid = "parent_id" in header

    # Parallelism: geographic tiles are the chunks (each a small bbox -> fits memory
    # and gives per-tile resume). nproc processes run tiles concurrently; nthread is
    # climr's in-process threading within each. BLAS pinned to 1 (see warm note).
    cpus = int(os.environ.get("SLURM_CPUS_ON_NODE") or os.cpu_count() or 1)
    nproc = max(1, args.workers or int(os.environ.get("HOUFIN_CLIMATE_WORKERS") or 4))
    nthread = args.threads or int(os.environ.get("HOUFIN_CLIMATE_THREADS") or max(1, cpus // nproc))
    nthread = max(1, min(nthread, 128))
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")

    chunk_dir = os.path.join(args.out, "_chunks")
    chunk_csvs = _split_centroids_spatial(centroids, tiles_per_axis, chunk_dir)
    chunk_outs = [os.path.join(chunk_dir, f"out_{i:03d}") for i in range(len(chunk_csvs))]
    unit = "sub-points (spatial quantiles)" if subgrid else "centroids x 3 elev levels"
    print(f"climr [{mode}]: {n_cen} {unit} -> {len(chunk_csvs)} geographic tiles, "
          f"{nproc} proc x {nthread} threads -> {args.out}", flush=True)

    counts = {"ok": 0, "exists": 0}
    failures = []
    t0 = time.time()
    # PROCESSES, not threads: each chunk does an R subprocess (GIL-free) THEN a
    # pandas groupby-quantile aggregation (subgrid) which is GIL-BOUND. In a thread
    # pool the aggregations serialize on the GIL -- once the R phase ends, 16 threads
    # crawl through quantiles on ~2 cores while the node idles and all chunk frames
    # pile up in one process (observed: 53 GB, 98% idle). Separate processes run the
    # aggregations truly in parallel and spread the memory. 'fork' is safe here: at
    # pool creation the parent holds no GDAL/terra state or threads (terra lives only
    # in the R subprocess), and fork avoids re-importing __main__ (no recursion).
    mp_ctx = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=min(nproc, len(chunk_csvs)), mp_context=mp_ctx) as ex:
        futs = [ex.submit(_run_chunk, cc, co, start, end, args.rscript, env, subgrid,
                          nthread, db_option, obs_ts_dataset)
                for cc, co in zip(chunk_csvs, chunk_outs)]
        n = len(futs)
        # Explicit per-chunk completion lines (flushed) — a tqdm bar renders poorly
        # in a non-TTY SLURM log and only ticks at whole-chunk granularity anyway.
        # Per-chunk R downscaling detail is in _chunks/chunk_*.log. A whole wave of
        # chunks can run for minutes with no completion, so also emit a HEARTBEAT
        # every ~60s during quiet stretches (wait() returns empty on timeout) — a
        # long silence otherwise looks like a stall.
        pending = set(futs); done = 0
        while pending:
            finished, pending = wait(pending, timeout=60, return_when=FIRST_COMPLETED)
            if not finished:                       # timeout, nothing new -> heartbeat
                el = time.time() - t0
                print(f"[climate] ...running: {done}/{n} done, {len(pending)} in flight, "
                      f"{el:.0f}s elapsed", flush=True)
                continue
            for fut in finished:
                done += 1
                cc, rc, log, status = fut.result()
                if rc != 0:
                    failures.append((cc, log))
                    print(f"[ERROR] chunk {os.path.basename(cc)} failed (rc={rc}); see {log}", flush=True)
                else:
                    counts[status] += 1
                el = time.time() - t0
                eta = el / done * (n - done)
                print(f"[climate] {done}/{n} chunks (ran={counts['ok']} cached={counts['exists']} "
                      f"failed={len(failures)}) | {el:.0f}s elapsed, ~{eta:.0f}s left", flush=True)

    if failures:
        _, log0 = failures[0]
        if log0 and os.path.exists(log0):
            print(f"--- tail of {log0} ---")
            with open(log0) as fh:
                sys.stdout.write("".join(fh.readlines()[-20:]))
        raise SystemExit(f"{len(failures)}/{len(chunk_csvs)} climate chunks failed")

    print(f"chunks done: ran={counts['ok']} already-present={counts['exists']}", flush=True)
    _concat_levels(chunk_outs, args.out)
    print(f"Done. Wrote climate_{{{','.join(LEVELS)}}}.csv to {args.out}", flush=True)


if __name__ == "__main__":
    main()
