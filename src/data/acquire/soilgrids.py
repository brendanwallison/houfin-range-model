"""SoilGrids (ISRIC) aggregated-5000m downloader — static soil covariates.

Downloads the aggregated 5000 m COG GeoTIFFs from ISRIC over anonymous HTTP:
    {base}/{prop}/{prop}_{depth}_mean_5000.tif
(e.g. .../5000m/phh2o/phh2o_0-5cm_mean_5000.tif). These are the mean of each
property per depth. Soil is treated as time-invariant, so this runs once; the
preprocess step reprojects/aggregates the tiles to the 25 km model grid.

Property/depth sets come from data_config.json (`soilgrids`), overridable on
the CLI. Same house style as the other downloaders: streaming, retries, tqdm,
atomic writes, idempotent skips, GeoTIFF validation.

Examples
--------
    python scripts/download_soilgrids.py --list
    python scripts/download_soilgrids.py
    python scripts/download_soilgrids.py --properties phh2o soc --depths 0-5cm
"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import rasterio
import requests
from tqdm import tqdm

from src.config_utils import load_data_config

DEFAULT_BASE = "https://files.isric.org/soilgrids/latest/data_aggregated/5000m"
DEFAULT_PROPERTIES = ["sand", "silt", "clay", "phh2o", "soc", "bdod", "cec", "nitrogen"]
DEFAULT_DEPTHS = ["0-5cm", "30-60cm"]

MAX_RETRIES = 5
BACKOFF = 5
MAX_WORKERS = 3
MIN_BYTES = 10_000


def file_url(base, prop, depth):
    return f"{base.rstrip('/')}/{prop}/{prop}_{depth}_mean_5000.tif"


def is_valid_tif(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < MIN_BYTES:
        return False
    try:
        with rasterio.open(path) as src:
            return src.count >= 1
    except Exception:
        return False


def stream_download(url, dest: Path) -> bool:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            tmp = dest.with_suffix(dest.suffix + ".part")
            with requests.get(url, stream=True, timeout=600) as r:
                r.raise_for_status()
                total = int(r.headers.get("Content-Length", 0)) or None
                with open(tmp, "wb") as fh, tqdm(
                    total=total, unit="B", unit_scale=True, desc=dest.name, leave=False
                ) as bar:
                    for chunk in r.iter_content(chunk_size=1 << 16):
                        if chunk:
                            fh.write(chunk)
                            bar.update(len(chunk))
            if not is_valid_tif(tmp):
                raise ValueError("size/GeoTIFF validation failed")
            tmp.replace(dest)
            return True
        except Exception as e:
            print(f"[WARN] {dest.name}: attempt {attempt} failed: {e}")
            tmp = dest.with_suffix(dest.suffix + ".part")
            if tmp.exists():
                tmp.unlink()
            time.sleep(BACKOFF * attempt)
    return False


def plan(base, properties, depths, out_dir):
    """Return (url, dest) for every property x depth."""
    return [
        (file_url(base, p, d), out_dir / f"{p}_{d}_mean_5000.tif")
        for p in properties for d in depths
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--properties", nargs="+")
    ap.add_argument("--depths", nargs="+")
    ap.add_argument("--out-dir")
    ap.add_argument("--base-url")
    ap.add_argument("--list", action="store_true", dest="list_only")
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    args = ap.parse_args()

    cfg = load_data_config()
    scfg = cfg.get("soilgrids", {})
    base = args.base_url or scfg.get("base_url", DEFAULT_BASE)
    properties = args.properties or scfg.get("properties", DEFAULT_PROPERTIES)
    depths = args.depths or scfg.get("depths", DEFAULT_DEPTHS)
    out_dir = Path(args.out_dir) if args.out_dir else \
        Path(cfg["datasets_root"]) / scfg.get("out_subdir", "soilgrids_5000m")

    tasks = plan(base, properties, depths, out_dir)
    if args.list_only:
        for url, _ in tasks:
            print(url)
        print(f"\n{len(tasks)} tiles ({len(properties)} properties x {len(depths)} depths).")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"SoilGrids: {len(tasks)} tiles -> {out_dir}")

    def _one(url, dest):
        if is_valid_tif(dest):
            return (dest.name, "exists")
        return (dest.name, "ok" if stream_download(url, dest) else "fail")

    n_ok = n_exists = n_fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_one, u, d) for u, d in tasks]
        for fut in tqdm(as_completed(futs), total=len(futs)):
            name, status = fut.result()
            if status == "ok":
                n_ok += 1
            elif status == "exists":
                n_exists += 1
            else:
                n_fail += 1
                print(f"[ERROR] {name} failed.")
    print(f"downloaded={n_ok} already-present={n_exists} failed={n_fail}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
