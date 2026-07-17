#!/usr/bin/env python3
"""Emit a compact JSON manifest describing pipeline products, for remote review.

On an HPC the large rasters stay on the cluster; this writes a small manifest
(per file: size + sha256; per raster: CRS/shape/res/nodata + per-band
min/median/max + valid-cell count; per npz/npy/csv: keys/shape/rows) that is
cheap to copy back and analyze. Run it after each preprocessing stage.

Base deps only (numpy/rasterio) — runs on a login node or inside a batch job.

Examples
--------
    python scripts/validate_products.py --stage soilgrids --paths ${HOUFIN_DATA}/soilgrids_grid
    python scripts/validate_products.py --stage hyde --paths '${HOUFIN_DATA}/hyde35_grid/*_2025_grid.tif'
"""
import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

RASTER_EXT = {".tif", ".tiff"}
STATS_CELL_CAP = 5_000_000  # skip per-band stats above this (keeps big rasters cheap)


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _crs_str(crs):
    if crs is None:
        return None
    epsg = crs.to_epsg()
    return f"EPSG:{epsg}" if epsg else crs.to_string()


def describe_raster(path):
    import rasterio
    rec = {}
    with rasterio.open(path) as src:
        rec.update(driver=src.driver, crs=_crs_str(src.crs),
                   width=src.width, height=src.height, count=src.count,
                   res=[abs(src.res[0]), abs(src.res[1])], nodata=src.nodata)
        if src.width * src.height <= STATS_CELL_CAP:
            bands = []
            for b in range(1, src.count + 1):
                a = src.read(b, masked=True).astype("float64")
                valid = int(a.count())
                if valid:
                    bands.append(dict(band=b, valid=valid,
                                      min=float(a.min()), median=float(np.ma.median(a)),
                                      max=float(a.max())))
                else:
                    bands.append(dict(band=b, valid=0))
            rec["bands"] = bands
        else:
            rec["bands"] = "skipped (raster exceeds stats cell cap)"
    return rec


def describe_file(path):
    ext = os.path.splitext(path)[1].lower()
    rec = dict(path=path, size_bytes=os.path.getsize(path), sha256=sha256(path))
    try:
        if ext in RASTER_EXT:
            rec["raster"] = describe_raster(path)
        elif ext == ".npz":
            with np.load(path, allow_pickle=True) as z:
                rec["npz"] = {k: {"shape": list(np.shape(z[k])), "dtype": str(np.asarray(z[k]).dtype)}
                              for k in z.files}
        elif ext == ".npy":
            a = np.load(path, mmap_mode="r", allow_pickle=True)
            rec["npy"] = {"shape": list(a.shape), "dtype": str(a.dtype)}
        elif ext == ".csv":
            with open(path) as fh:
                header = fh.readline().rstrip("\n")
                n = sum(1 for _ in fh)
            rec["csv"] = {"header": header, "rows": n}
    except Exception as e:  # a manifest should never crash the run
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


def collect(paths):
    """Expand dirs (recursively) and globs into a sorted, de-duplicated file list."""
    out = []
    for p in paths:
        p = os.path.expandvars(p)
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                out.extend(os.path.join(root, f) for f in files)
        else:
            out.extend(glob.glob(p))
    return sorted(set(f for f in out if os.path.isfile(f)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, help="Stage name (labels the manifest).")
    ap.add_argument("--paths", nargs="+", required=True, help="Dirs, files, or globs.")
    ap.add_argument("--out", help="Manifest path (default: {processed_root}/validation/{stage}.json).")
    args = ap.parse_args()

    files = collect(args.paths)
    records = [describe_file(f) for f in files]
    manifest = {"stage": args.stage, "n_files": len(records),
                "total_bytes": sum(r["size_bytes"] for r in records),
                "files": records}

    out = args.out
    if not out:
        from src.config_utils import load_data_config
        out = os.path.join(load_data_config()["processed_root"], "validation", f"{args.stage}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"[{args.stage}] {len(records)} files, "
          f"{manifest['total_bytes']/1e6:.1f} MB -> {out}")


if __name__ == "__main__":
    main()
