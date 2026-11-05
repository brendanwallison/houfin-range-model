"""Regroup stored per-sub-point climr output onto a COARSER parent grid (no climr rerun).

The subgrid climate path (``acquire/climatena.py``) downscales a ``g x g`` lattice of
true-elevation sub-points per model cell and keeps the per-sub-point output
(``_chunks/out_NNN/climate_points.csv``; ``_chunks/chunk_NNN.csv`` maps
``id -> parent_id, row, col``). When the coarse grid has the same origin and each coarse
cell is exactly ``factor x factor`` fine cells, a coarse cell's sub-point lattice is the
UNION of its children's (same fine transform), so the coarse within-cell quantiles are
recomputed EXACTLY with the same ``climatena.quantile_aggregate`` over that union --
not by averaging the children's quantiles (quantiles do not aggregate).

The chunks are geographic tiles cut on the FINE grid, so a coarse parent can straddle
two tiles. Grouping is therefore global: parents wholly inside one chunk are aggregated
per chunk (in parallel); the straddling parents' sub-point rows are set aside and
aggregated together afterwards.

Caveat (documented, not fixable without climr): sub-points were only generated inside
FINE land cells (parent-mask filter of subcell_centroids). A native coarse run would also
keep fine-land sub-points of fine children that are ocean at the fine grid but whose
coarse parent is land; those points do not exist in the stored output.

    python -m src.data.preprocess.climate_regroup --chunks $HOUFIN_DATA/climate/_chunks \
        --factor 2 --coarse-width 112 --out $HOUFIN_DATA54/climate
"""
import argparse
import glob
import json
import os
import re
import shutil
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from src.data.acquire.climatena import LEVELS, quantile_aggregate

_CHUNK_RE = re.compile(r"chunk_(\d+)\.csv$")


def coarse_parent(row, col, factor, coarse_width):
    """Coarse parent id (row-major on the coarse grid) of fine cell ``(row, col)``."""
    row = np.asarray(row)
    col = np.asarray(col)
    return (row // factor) * coarse_width + (col // factor)


def list_chunks(chunk_dir):
    """``[(index, chunk_csv, out_dir)]`` sorted by index; out dir must hold points.

    ``chunk_dir`` may be one dir or a list of dirs (e.g. the original run plus a
    supplementary run for sub-points it never generated). Chunks of the k-th dir get
    index ``k * 1000 + i``, so a single dir keeps its own indices.
    """
    dirs = [chunk_dir] if isinstance(chunk_dir, str) else list(chunk_dir)
    out = []
    for k, d in enumerate(dirs):
        for p in glob.glob(os.path.join(d, "chunk_*.csv")):
            m = _CHUNK_RE.search(os.path.basename(p))
            if not m:
                continue
            i = k * 1000 + int(m.group(1))
            od = os.path.join(d, f"out_{m.group(1)}")
            out.append((i, p, od))
    return sorted(out)


def build_id_map(chunks, factor, coarse_width):
    """Global ``id -> (chunk, row, col, parent_id, boundary)`` frame.

    ``boundary`` is True for sub-points whose coarse parent has sub-points in more
    than one chunk (those must be aggregated globally).
    """
    frames = []
    for i, p, _ in chunks:
        df = pd.read_csv(p, usecols=["id", "row", "col"])
        df["chunk"] = i
        frames.append(df)
    m = pd.concat(frames, ignore_index=True)
    if m["id"].duplicated().any():
        raise SystemExit("sub-point ids repeat across chunks; cannot regroup safely")
    m["parent_id"] = coarse_parent(m["row"].to_numpy(), m["col"].to_numpy(),
                                   factor, coarse_width).astype("int64")
    n_chunks = m.groupby("parent_id")["chunk"].nunique()
    m["boundary"] = m["parent_id"].map(n_chunks).to_numpy() > 1
    return m


def read_points(path):
    """Per-sub-point climr output, parsed exactly as climatena._aggregate_chunk does."""
    return pd.read_csv(path)


def _process_chunk(args):
    """Interior parents of one chunk -> per-level parquet; boundary rows -> parquet."""
    i, out_dir, ids_int, pid_int, ids_bnd, tmp = args
    pts = read_points(os.path.join(out_dir, "climate_points.csv"))
    n_rows = len(pts)
    res = {"chunk": i, "rows": n_rows}
    if len(ids_int):
        idp = pd.DataFrame({"id": ids_int, "parent_id": pid_int})
        sub = pts[pts["id"].isin(idp["id"])]
        for lvl, df in quantile_aggregate(sub, idp).items():
            df.to_parquet(os.path.join(tmp, f"int_{i:03d}_{lvl}.parquet"), index=False)
        res["interior_rows"] = int(len(sub))
    if len(ids_bnd):
        b = pts[pts["id"].isin(ids_bnd)]
        b.to_parquet(os.path.join(tmp, f"bnd_{i:03d}.parquet"), index=False)
        res["boundary_rows"] = int(len(b))
    return res


def regroup(chunk_dir, factor, coarse_width, out, workers=8, tmp=None, keep_tmp=False):
    chunks = list_chunks(chunk_dir)
    if not chunks:
        raise SystemExit(f"no chunk_*.csv in {chunk_dir}")
    for _, _, od in chunks:
        if not os.path.exists(os.path.join(od, "climate_points.csv")):
            raise SystemExit(f"missing {od}/climate_points.csv (per-sub-point output)")
    m = build_id_map(chunks, factor, coarse_width)
    os.makedirs(out, exist_ok=True)
    tmp = tmp or os.path.join(out, "_regroup_tmp")
    os.makedirs(tmp, exist_ok=True)
    m.to_parquet(os.path.join(out, "subpoint_parent_map.parquet"), index=False)

    tasks = []
    for i, _, od in chunks:
        mi = m[m["chunk"] == i]
        it = mi[~mi["boundary"]]
        bd = mi[mi["boundary"]]
        tasks.append((i, od, it["id"].to_numpy(), it["parent_id"].to_numpy(),
                      bd["id"].to_numpy(), tmp))
    print(f"[regroup] {len(m)} sub-points, {m['parent_id'].nunique()} coarse parents "
          f"({m.loc[m['boundary'], 'parent_id'].nunique()} straddle chunks), "
          f"{len(chunks)} chunks, {workers} workers", flush=True)
    stats = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(_process_chunk, tasks):
            stats.append(r)
            print(f"[regroup]   chunk {r['chunk']:03d}: {r['rows']} rows "
                  f"(interior {r.get('interior_rows', 0)}, boundary {r.get('boundary_rows', 0)})",
                  flush=True)

    # Boundary parents: union over chunks, aggregated together (grouped by coarse row to
    # bound memory; a parent never spans two coarse rows).
    bnd_files = sorted(glob.glob(os.path.join(tmp, "bnd_*.parquet")))
    bnd_parts = {lvl: [] for lvl in LEVELS}
    if bnd_files:
        bmap = m.loc[m["boundary"], ["id", "parent_id"]]
        brow = bmap["parent_id"] // coarse_width
        allb = pd.concat([pd.read_parquet(f) for f in bnd_files], ignore_index=True)
        for r in sorted(brow.unique()):
            ids = bmap.loc[brow == r, "id"]
            pts = allb[allb["id"].isin(ids)]
            for lvl, df in quantile_aggregate(pts, bmap[bmap["id"].isin(ids)]).items():
                bnd_parts[lvl].append(df)
            print(f"[regroup]   boundary coarse row {r}: {pts['id'].nunique()} sub-points",
                  flush=True)
            del pts
        del allb

    for lvl in LEVELS:
        parts = [pd.read_parquet(f) for f in
                 sorted(glob.glob(os.path.join(tmp, f"int_*_{lvl}.parquet")))]
        df = pd.concat(parts + bnd_parts[lvl], ignore_index=True)
        if df.duplicated(["id", "PERIOD"]).any():
            raise SystemExit(f"[{lvl}] duplicate (id, PERIOD) after regroup")
        df = df.sort_values(["id", "PERIOD"], kind="mergesort").reset_index(drop=True)
        df.to_csv(os.path.join(out, f"climate_{lvl}.csv"), index=False)
        print(f"[regroup] wrote climate_{lvl}.csv: {df['id'].nunique()} cells x "
              f"{df['PERIOD'].nunique()} periods x {df.shape[1] - 2} vars", flush=True)

    per_parent = m.groupby("parent_id").size()
    summary = {
        "chunk_dir": ([os.path.abspath(chunk_dir)] if isinstance(chunk_dir, str)
                      else [os.path.abspath(d) for d in chunk_dir]), "factor": factor,
        "coarse_width": coarse_width, "n_subpoints": int(len(m)),
        "n_parents": int(m["parent_id"].nunique()),
        "n_boundary_parents": int(m.loc[m["boundary"], "parent_id"].nunique()),
        "subpoints_per_parent": {k: float(v) for k, v in
                                 per_parent.describe().items()},
        "chunks": stats,
    }
    with open(os.path.join(out, "regroup_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    if not keep_tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chunks", required=True, nargs="+",
                    help="climate/_chunks dir(s) (fine grid); several dirs are unioned")
    ap.add_argument("--factor", type=int, required=True, help="fine cells per coarse cell (axis)")
    ap.add_argument("--coarse-width", type=int, required=True, help="coarse grid width (cols)")
    ap.add_argument("--out", required=True, help="output dir for climate_{q10,q50,q90}.csv")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tmp", default=None)
    ap.add_argument("--keep-tmp", action="store_true")
    args = ap.parse_args()
    chunks = args.chunks[0] if len(args.chunks) == 1 else args.chunks
    regroup(chunks, args.factor, args.coarse_width, args.out,
            workers=args.workers, tmp=args.tmp, keep_tmp=args.keep_tmp)


if __name__ == "__main__":
    main()
