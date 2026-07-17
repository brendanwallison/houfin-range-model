"""Global DEM downloader — the fine elevation source for the climate branch.

The climate step downscales with ``climr`` at three sub-cell elevation levels per
model cell; those levels come from a fine DEM aggregated to per-cell p10/p50/p90
elevation (``preprocess/elevation.py``). This client fetches that DEM.

Default source: NOAA NCEI **ETOPO 2022** 60-arc-second (~1.85 km) *surface*
elevation, a single global GeoTIFF over public HTTPS (no auth). ~1.85 km gives
~180 sub-cells per 25 km cell — ample for elevation quantiles. Swap ``dem.url``
in data_config for a finer DEM (e.g. GMTED2010 7.5-arc-second) if desired.

Same house style as the other downloaders: streaming, retries, atomic write,
idempotent skip, GeoTIFF validation.

Examples
--------
    python scripts/download_dem.py --list
    python scripts/download_dem.py
"""
import argparse
import os
import sys
import time
from pathlib import Path

import rasterio
import requests
from tqdm import tqdm

from src.config_utils import load_data_config

DEFAULT_URL = ("https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO2022/data/"
               "60s/60s_surface_elev_gtif/ETOPO_2022_v1_60s_N90W180_surface.tif")
DEFAULT_SUBDIR = "dem"
MAX_RETRIES = 5
BACKOFF = 5
MIN_BYTES = 1_000_000  # a global DEM GeoTIFF is tens-to-hundreds of MB


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
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
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
            if tmp.exists():
                tmp.unlink()
            time.sleep(BACKOFF * attempt)
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", help="DEM GeoTIFF URL (default: dem.url or ETOPO 2022 60s)")
    ap.add_argument("--out-dir")
    ap.add_argument("--list", action="store_true", dest="list_only")
    args = ap.parse_args()

    cfg = load_data_config()
    dcfg = cfg.get("dem", {})
    url = args.url or dcfg.get("url", DEFAULT_URL)
    out_dir = Path(args.out_dir) if args.out_dir else \
        Path(cfg["datasets_root"]) / dcfg.get("out_subdir", DEFAULT_SUBDIR)
    dest = out_dir / os.path.basename(url)

    if args.list_only:
        print(url, "->", dest)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    if is_valid_tif(dest):
        print(f"DEM already present: {dest}")
        return
    print(f"DEM: {url} -> {dest}")
    ok = stream_download(url, dest)
    print("done" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
