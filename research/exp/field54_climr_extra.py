"""FIELD54: run climr for the supplementary sub-points with the 27 km run's exact settings.

The 27 km subgrid run generated sub-points only inside 27 km LAND cells. A 54 km land
cell's native sub-point set also contains the fine-land sub-points of its 27 km OCEAN
children; those were enumerated (same 5x5 fine lattice, same DEM, same fine-land filter,
ids = fine-lattice index as in subcell_centroids) and grouped into chunks whose bounding
boxes each lie inside one already-cached climr refmap tile and one cached obs_ts area
(so the offline compute node never needs to download). This driver runs those chunks
through the SAME code path as climatena.main: ``climatena._run_chunk`` with
obs_years first_year-1 .. min(end_year, climr_obs_max_year), obs_ts_dataset,
db_option and the Rscript from the data config / env. Outputs go to a NEW chunk dir;
the 27 km ``_chunks`` are never touched.

    python research/exp/field54_climr_extra.py --chunks $HOUFIN_DATA54/climate_extra/_chunks
"""
import argparse
import glob
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.config_utils import load_data_config  # noqa: E402
from src.data.acquire import climatena as ca  # noqa: E402
from src.temporal import load_timeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--rscript", default=os.environ.get("HOUFIN_RSCRIPT", "Rscript"))
    args = ap.parse_args()
    if "/houfin/data/climate" in os.path.abspath(args.chunks):
        raise SystemExit("refusing to write into the 27 km climate tree")

    cfg = load_data_config()
    tl = load_timeline(cfg)
    ccfg = cfg.get("climate", {})
    obs_ts = ccfg.get("obs_ts_dataset", "cru.gpcc")
    start = max(tl["first_year"] - 1, int(ccfg.get("climr_obs_min_year", ca.CLIMR_OBS_MIN_YEAR)))
    end = min(tl["end_year"], int(ccfg.get("climr_obs_max_year", ca.CLIMR_OBS_MAX_YEAR)))
    db_option = ccfg.get("db_option", "local")
    if db_option == "local":
        ca._assert_climr_cache_warm()
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")

    csvs = sorted(glob.glob(os.path.join(args.chunks, "chunk_*.csv")))
    outs = [os.path.join(args.chunks, "out_" + os.path.basename(c)[6:-4]) for c in csvs]
    print(f"[climr-extra] {len(csvs)} chunks, obs_years {start}:{end}, obs_ts={obs_ts}, "
          f"db_option={db_option}, {args.workers} proc x {args.threads} threads, "
          f"Rscript={args.rscript}", flush=True)
    t0 = time.time()
    fails = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(csvs)),
                             mp_context=multiprocessing.get_context("fork")) as ex:
        futs = [ex.submit(ca._run_chunk, c, o, start, end, args.rscript, env, True,
                          args.threads, db_option, obs_ts) for c, o in zip(csvs, outs)]
        for f in futs:
            cc, rc, log, status = f.result()
            print(f"[climr-extra] {os.path.basename(cc)} rc={rc} {status} "
                  f"{time.time() - t0:.0f}s", flush=True)
            if rc != 0:
                fails.append((cc, log))
    if fails:
        for cc, log in fails[:3]:
            print(f"--- {log} ---")
            if log and os.path.exists(log):
                print(open(log).read()[-2000:])
        raise SystemExit(f"{len(fails)}/{len(csvs)} chunks failed")
    print(f"[climr-extra] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
