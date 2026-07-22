"""HYDE 3.5 human-population downloader (population density + urban + rural).

Downloads the HYDE 3.5 baseline population **netCDF** files (one file per
variable, spanning all HYDE time points, global 5 arc-min) from the Utrecht
public data vault, which serves them over plain anonymous HTTP:

    {base_url}/{file}
    base_url = .../vault-hyde/hyde35_c9_apr2025[...]/original/gbc2025_7apr_base/NetCDF

Confirmed accessible (unlike public.yoda.uu.nl, which is behind Anubis
bot-protection). CC BY 3.0. The preprocess step subsets these to the model
year range + bounding box and aggregates 5 arc-min → 25 km.

Population variables (the three chosen human-density streams):
    population_density.nc  (popd)   urban_population.nc  (urbc)
    rural_population.nc     (rurc)
Other HYDE layers (urban_area.nc, cropland.nc, ...) live in the same NetCDF/
dir if ever needed — just add them to `hyde.files` in config. Per-year zipped
`.asc` grids also exist under the sibling `zip/` dir as an alternative.

Same house style as the other downloaders: streaming, retries, tqdm, atomic
writes, idempotent skips, netCDF validation.

Examples
--------
    python scripts/download_hyde.py --list
    python scripts/download_hyde.py
"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
import xarray as xr
from tqdm import tqdm

from src.config_utils import load_data_config

DEFAULT_BASE = ("https://geo.public.data.uu.nl/vault-hyde/hyde35_c9_apr2025%5B1749214444%5D"
                "/original/gbc2025_7apr_base/NetCDF")
DEFAULT_FILES = ["population_density.nc", "urban_population.nc", "rural_population.nc"]
MAX_RETRIES = 5
BACKOFF = 5
MAX_WORKERS = 3
MIN_BYTES = 100_000


def is_valid_nc(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < MIN_BYTES:
        return False
    # Check the file signature BEFORE handing the path to HDF5: a truncated/corrupt .nc
    # (e.g. a killed download) can crash the native HDF5 reader with a SEGFAULT, which is
    # not a Python exception the try/except can catch. NetCDF classic starts with "CDF";
    # netCDF-4/HDF5 starts with the "\x89HDF" magic. Anything else -> treat as invalid.
    try:
        with open(path, "rb") as fh:
            magic = fh.read(4)
    except OSError:
        return False
    if not (magic[:3] == b"CDF" or magic == b"\x89HDF"):
        return False
    try:
        xr.open_dataset(path).close()
        return True
    except Exception:
        return False


def stream_download(url, dest: Path) -> bool:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            tmp = dest.with_suffix(dest.suffix + ".part")
            with requests.get(url, stream=True, timeout=1200) as r:
                r.raise_for_status()
                total = int(r.headers.get("Content-Length", 0)) or None
                with open(tmp, "wb") as fh, tqdm(
                    total=total, unit="B", unit_scale=True, desc=dest.name, leave=False
                ) as bar:
                    for chunk in r.iter_content(chunk_size=1 << 16):
                        if chunk:
                            fh.write(chunk)
                            bar.update(len(chunk))
            if not is_valid_nc(tmp):
                raise ValueError("size/netCDF validation failed")
            tmp.replace(dest)
            return True
        except Exception as e:
            print(f"[WARN] {dest.name}: attempt {attempt} failed: {e}")
            tmp = dest.with_suffix(dest.suffix + ".part")
            if tmp.exists():
                tmp.unlink()
            time.sleep(BACKOFF * attempt)
    return False


def plan(base, files, out_dir):
    return [(f"{base.rstrip('/')}/{f}", out_dir / f) for f in files]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--files", nargs="+", help="netCDF filenames to fetch (overrides config).")
    ap.add_argument("--base-url")
    ap.add_argument("--out-dir")
    ap.add_argument("--list", action="store_true", dest="list_only")
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    args = ap.parse_args()

    cfg = load_data_config()
    hcfg = cfg.get("hyde", {})
    base = args.base_url or hcfg.get("base_url", DEFAULT_BASE)
    files = args.files or hcfg.get("files", DEFAULT_FILES)
    out_dir = Path(args.out_dir) if args.out_dir else \
        Path(cfg["datasets_root"]) / hcfg.get("out_subdir", "hyde35")

    tasks = plan(base, files, out_dir)
    if args.list_only:
        for url, _ in tasks:
            print(url)
        print(f"\n{len(tasks)} files.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"HYDE 3.5: {len(tasks)} files -> {out_dir}")

    def _one(url, dest):
        if is_valid_nc(dest):
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
