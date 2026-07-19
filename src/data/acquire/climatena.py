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
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.config_utils import load_data_config
from src.temporal import load_timeline

_R_SCRIPT = os.path.join(os.path.dirname(__file__), "climate_climr.R")
LEVELS = ("q10", "q50", "q90")

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


def _split_centroids_by_parent(centroids_csv, n_chunks, chunk_dir):
    """Subgrid split: partition by ``parent_id`` so all of a cell's sub-points land
    in one chunk (spatial quantiles are computed per parent within a chunk). Returns
    chunk CSV paths."""
    import pandas as pd
    df = pd.read_csv(centroids_csv).sort_values("parent_id")
    parents = df["parent_id"].unique()
    n_chunks = max(1, min(n_chunks, len(parents)))
    size = math.ceil(len(parents) / n_chunks)
    os.makedirs(chunk_dir, exist_ok=True)
    paths = []
    for i in range((len(parents) + size - 1) // size):
        keep = set(parents[i * size:(i + 1) * size])
        p = os.path.join(chunk_dir, f"chunk_{i:03d}.csv")
        df[df["parent_id"].isin(keep)].to_csv(p, index=False)
        paths.append(p)
    return paths


def quantile_aggregate(points, id_parent, quantiles=(0.10, 0.50, 0.90), levels=LEVELS):
    """Spatial quantiles of downscaled climate per (parent cell, PERIOD) — pure.

    ``points``: per-sub-point downscale output (``id, PERIOD, <vars>[, DATASET]``).
    ``id_parent``: ``id -> parent_id`` map. Returns ``{level: DataFrame(id=parent,
    PERIOD, <vars>)}`` where each level is the corresponding within-cell quantile —
    the same schema the centroid path emits, so downstream is unchanged.
    """
    df = points.merge(id_parent, on="id", how="inner")
    varcols = [c for c in points.columns if c not in ("id", "PERIOD", "DATASET", "parent_id")]
    grouped = df.groupby(["parent_id", "PERIOD"])[varcols]
    out = {}
    for q, lvl in zip(quantiles, levels):
        qd = grouped.quantile(q).reset_index().rename(columns={"parent_id": "id"})
        out[lvl] = qd
    return out


def _aggregate_chunk(chunk_out, chunk_csv):
    """Aggregate a subgrid chunk's per-sub-point output into the 3 level CSVs."""
    import pandas as pd
    pts = pd.read_csv(os.path.join(chunk_out, "climate_points.csv"))
    cen = pd.read_csv(chunk_csv, usecols=["id", "parent_id"])
    for lvl, df in quantile_aggregate(pts, cen).items():
        df.to_csv(os.path.join(chunk_out, f"climate_{lvl}.csv"), index=False)


def _run_chunk(chunk_csv, chunk_out, start, end, rscript, env, subgrid=False, nthread=1,
               db_option="local"):
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
                                          nthread=nthread, db_option=db_option),
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
    from src.data.preprocess.subcell_centroids import build_subcell_centroids, write_csv
    dem_dir = os.path.join(cfg["datasets_root"], cfg.get("dem", {}).get("out_subdir", "dem"))
    found = sorted(glob.glob(os.path.join(dem_dir, "*.tif")))
    if not found:
        raise SystemExit(f"subgrid climate needs a DEM in {dem_dir} (run scripts/download_dem.py)")
    with rasterio.open(cfg["grid"]["ref_raster"]) as ref:
        cols = build_subcell_centroids(found[0], ref.transform, ref.crs, ref.height, ref.width, grid)
    write_csv(out_csv, cols)
    print(f"[subgrid] generated {cols['id'].size} sub-points ({grid}x{grid}/cell) -> {out_csv}",
          flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--centroids", default=None,
                    help="centroids CSV; default from climate.mode "
                         "(subcell_centroids.csv for subgrid, else cell_centroids.csv)")
    ap.add_argument("--out", required=True, help="output dir for climate_{q10,q50,q90}.csv")
    ap.add_argument("--rscript", default="Rscript")
    ap.add_argument("--workers", type=int, default=None,
                    help="R PROCESSES over centroid chunks (default 1; HOUFIN_CLIMATE_WORKERS). "
                         "climr's DuckDB serializes across processes, so keep this small.")
    ap.add_argument("--threads", type=int, default=None,
                    help="climr nthread WITHIN each process (default: node cpus / workers; "
                         "HOUFIN_CLIMATE_THREADS) — the parallelism knob that actually scales here.")
    ap.add_argument("--dry-run", action="store_true", help="print the command and exit")
    args = ap.parse_args()

    cfg = load_data_config()
    tl = load_timeline(cfg)
    start, end = tl["first_year"], tl["end_year"]

    # Clamp to climr's observed extent: the model timeline may run past it
    # (end_year 2025 vs climr's 2024), and downscale() rejects out-of-range
    # obs_years. The combine streamer EMA-carries covariates that lag end_year,
    # so a climate series ending a year short of the timeline is by design.
    ccfg = cfg.get("climate", {})
    obs_min = int(ccfg.get("climr_obs_min_year", CLIMR_OBS_MIN_YEAR))
    obs_max = int(ccfg.get("climr_obs_max_year", CLIMR_OBS_MAX_YEAR))
    cstart, cend = max(start, obs_min), min(end, obs_max)
    if (cstart, cend) != (start, end):
        print(f"[note] clamping climate obs_years {start}:{end} -> {cstart}:{cend} "
              f"(climr obs extent {obs_min}:{obs_max}; later years EMA-carried downstream)",
              flush=True)
    start, end = cstart, cend

    # Mode: 'subgrid' (default) samples a grid x grid mesh of true-elevation points
    # per cell and takes spatial quantiles; 'elev_quantile' is the centroid-at-three-
    # elevations fallback. Resolve the centroids file (build the sub-cell mesh if absent).
    mode = ccfg.get("mode", "subgrid")
    subgrid = (mode == "subgrid")
    db_option = ccfg.get("db_option", "local")   # 'local' = download+cache+process offline
    elev_dir = os.path.join(cfg["datasets_root"], "elevation")
    centroids = args.centroids or os.path.join(
        elev_dir, "subcell_centroids.csv" if subgrid else "cell_centroids.csv")

    if args.dry_run:
        print(f"mode={mode} db_option={db_option}; climr command (serial form):",
              " ".join(build_command(centroids, args.out, start, end, args.rscript,
                                     db_option=db_option)))
        return
    if subgrid and not os.path.exists(centroids):
        _ensure_subcell_centroids(cfg, centroids, int(ccfg.get("subgrid", {}).get("grid", 5)))
    if not os.path.exists(centroids):
        raise SystemExit(f"centroids file not found: {centroids} (run preprocess/elevation.py first)")
    os.makedirs(args.out, exist_ok=True)

    with open(centroids) as fh:
        n_cen = max(0, sum(1 for _ in fh) - 1)

    # climr's DuckDB backend serializes across PROCESSES (they contend on one DB
    # lock), so parallelize with its in-process ``nthread`` (one DB handle, threads
    # split the points) and keep only a few processes for memory headroom.
    # total threads ~= cpus = nproc * nthread.
    cpus = int(os.environ.get("SLURM_CPUS_ON_NODE") or os.cpu_count() or 1)
    nproc = max(1, args.workers or int(os.environ.get("HOUFIN_CLIMATE_WORKERS") or 1))
    nthread = args.threads or int(os.environ.get("HOUFIN_CLIMATE_THREADS") or max(1, cpus // nproc))

    # Let climr thread within each process (do NOT pin to 1 — that was the old
    # many-process design that fought over the DB); cap at nthread so nproc
    # processes don't oversubscribe the node.
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS=str(nthread), OPENBLAS_NUM_THREADS=str(nthread),
               MKL_NUM_THREADS=str(nthread))

    chunk_dir = os.path.join(args.out, "_chunks")
    split = _split_centroids_by_parent if subgrid else _split_centroids
    chunk_csvs = split(centroids, nproc, chunk_dir)
    chunk_outs = [os.path.join(chunk_dir, f"out_{i:03d}") for i in range(len(chunk_csvs))]
    unit = "sub-points (spatial quantiles)" if subgrid else "centroids x 3 elev levels"
    print(f"climr [{mode}]: {n_cen} {unit} -> {len(chunk_csvs)} process-chunk(s) x "
          f"{nthread} climr threads -> {args.out}", flush=True)

    counts = {"ok": 0, "exists": 0}
    failures = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=len(chunk_csvs)) as ex:
        futs = [ex.submit(_run_chunk, cc, co, start, end, args.rscript, env, subgrid,
                          nthread, db_option)
                for cc, co in zip(chunk_csvs, chunk_outs)]
        n = len(futs)
        # Explicit per-chunk completion lines (flushed) — a tqdm bar renders poorly
        # in a non-TTY SLURM log and only ticks at whole-chunk granularity anyway.
        # Per-chunk R downscaling detail is in _chunks/chunk_*.log.
        for done, fut in enumerate(as_completed(futs), 1):
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
