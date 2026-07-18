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
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from src.config_utils import load_data_config
from src.temporal import load_timeline

_R_SCRIPT = os.path.join(os.path.dirname(__file__), "climate_climr.R")
LEVELS = ("q10", "q50", "q90")


def build_command(centroids_csv, out_dir, start_year, end_year, rscript="Rscript"):
    """Construct the Rscript command (kept pure/testable, separate from execution)."""
    return [rscript, _R_SCRIPT, centroids_csv, out_dir, str(start_year), str(end_year)]


def worker_count(n_items, cap=32):
    """Parallel R processes: HOUFIN_CLIMATE_WORKERS, else SLURM/cpu count, capped.

    Capped (default 32) because each chunk is a full R+climr process (heavier RAM
    than a thread); raise HOUFIN_CLIMATE_WORKERS once remora shows headroom.
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


def _run_chunk(chunk_csv, chunk_out, start, end, rscript, env):
    """Run one chunk's Rscript; skip if its 3 level CSVs already exist (resume)."""
    os.makedirs(chunk_out, exist_ok=True)
    if all(os.path.exists(os.path.join(chunk_out, f"climate_{lvl}.csv")) for lvl in LEVELS):
        return chunk_csv, 0, None, "exists"
    log = chunk_csv[:-4] + ".log"
    with open(log, "w") as lf:
        rc = subprocess.run(build_command(chunk_csv, chunk_out, start, end, rscript),
                            stdout=lf, stderr=subprocess.STDOUT, env=env).returncode
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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--centroids", required=True, help="cell_centroids.csv from preprocess/elevation.py")
    ap.add_argument("--out", required=True, help="output dir for climate_{q10,q50,q90}.csv")
    ap.add_argument("--rscript", default="Rscript")
    ap.add_argument("--workers", type=int, default=None,
                    help="parallel R processes over centroid chunks (default: SLURM/cpu count, capped)")
    ap.add_argument("--dry-run", action="store_true", help="print the command and exit")
    args = ap.parse_args()

    tl = load_timeline(load_data_config())
    start, end = tl["first_year"], tl["end_year"]

    if args.dry_run:
        print("climr command (serial form):",
              " ".join(build_command(args.centroids, args.out, start, end, args.rscript)))
        return
    if not os.path.exists(args.centroids):
        raise SystemExit(f"centroids file not found: {args.centroids} (run preprocess/elevation.py first)")
    os.makedirs(args.out, exist_ok=True)

    with open(args.centroids) as fh:
        n_cen = max(0, sum(1 for _ in fh) - 1)
    workers = args.workers or worker_count(n_cen)

    # Pin each R process to one thread so N chunks use N cores without oversubscribing.
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")

    if workers <= 1:
        print(f"climr: {n_cen} centroids, serial (1 process)", flush=True)
        sys.exit(subprocess.run(build_command(args.centroids, args.out, start, end, args.rscript),
                                env=env).returncode)

    chunk_dir = os.path.join(args.out, "_chunks")
    chunk_csvs = _split_centroids(args.centroids, workers, chunk_dir)
    chunk_outs = [os.path.join(chunk_dir, f"out_{i:03d}") for i in range(len(chunk_csvs))]
    print(f"climr: {n_cen} centroids -> {len(chunk_csvs)} chunks x {len(LEVELS)} levels, "
          f"{workers} parallel R processes -> {args.out}", flush=True)

    counts = {"ok": 0, "exists": 0}
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_run_chunk, cc, co, start, end, args.rscript, env)
                for cc, co in zip(chunk_csvs, chunk_outs)]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="climate", mininterval=10):
            cc, rc, log, status = fut.result()
            if rc != 0:
                failures.append((cc, log))
                print(f"[ERROR] chunk {os.path.basename(cc)} failed (rc={rc}); see {log}", flush=True)
            else:
                counts[status] += 1

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
