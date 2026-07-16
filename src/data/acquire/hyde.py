"""HYDE 3.3 human-population downloader (population density + urban + rural).

Downloads HYDE 3.3 baseline population grids (5 arc-min ESRI ASCII) for the
requested variables and years, to be aggregated to the 25 km grid in
preprocess. Same house style as the other downloaders: streaming, retries,
tqdm, atomic writes, idempotent skips, raster validation. Supports either
direct per-year ``.asc`` files or per-year ``.zip`` archives (auto-extracted).

ENDPOINT MUST BE CONFIRMED IN CONFIG. HYDE 3.3 is hosted on the Utrecht Yoda
platform (public.yoda.uu.nl), which sits behind Anubis bot-protection, so a
plain GET may be challenged; a PBL/mirror URL or a pre-downloaded copy may be
needed. The base URL + filename templates are therefore config values
(`hyde` in data_config.json), not hardcoded — set `hyde.base_url` to the
confirmed location. Default naming follows the documented convention
`{var}_{year}AD.asc` (popd = population density, urbc = urban population,
rurc = rural population).

Examples
--------
    python scripts/download_hyde.py --list
    python scripts/download_hyde.py --years 1901 1910 2020
"""
import argparse
import os
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import rasterio
import requests
from tqdm import tqdm

from src.config_utils import load_data_config

DEFAULT_VARS = ["popd", "urbc", "rurc"]          # density, urban pop, rural pop
DEFAULT_FILE_TEMPLATE = "{var}_{year}AD.asc"
MAX_RETRIES = 5
BACKOFF = 5
MAX_WORKERS = 3
MIN_BYTES = 500


def _year_list(hcfg, cli_years):
    if cli_years:
        return [int(y) for y in cli_years]
    if "years" in hcfg:
        return [int(y) for y in hcfg["years"]]
    start = int(hcfg.get("start_year", 1901))
    end = int(hcfg.get("end_year", 2023))
    return list(range(start, end + 1))


def build_url(base, template, var, year):
    return f"{base.rstrip('/')}/{template.format(var=var, year=year)}"


def is_valid(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < MIN_BYTES:
        return False
    try:
        with rasterio.open(path) as src:
            return src.count >= 1
    except Exception:
        return False


def stream_download(url, dest: Path) -> bool:
    """Download url -> dest (atomic). If url is a .zip, extract the .asc inside."""
    is_zip = url.lower().endswith(".zip")
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
            if is_zip:
                with zipfile.ZipFile(tmp) as zf:
                    member = next(n for n in zf.namelist() if n.endswith(dest.name))
                    with zf.open(member) as src_f, open(dest, "wb") as out_f:
                        out_f.write(src_f.read())
                tmp.unlink()
            else:
                tmp.replace(dest)
            if not is_valid(dest):
                raise ValueError("size/raster validation failed")
            return True
        except Exception as e:
            print(f"[WARN] {dest.name}: attempt {attempt} failed: {e}")
            for p in (dest.with_suffix(dest.suffix + ".part"), dest):
                if p.exists() and p != dest.with_suffix(".keep"):
                    try:
                        p.unlink()
                    except OSError:
                        pass
            time.sleep(BACKOFF * attempt)
    return False


def plan(base, template, variables, years, out_dir):
    return [
        (build_url(base, template, v, y), out_dir / f"{v}_{y}AD.asc")
        for v in variables for y in years
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variables", nargs="+")
    ap.add_argument("--years", nargs="+")
    ap.add_argument("--base-url")
    ap.add_argument("--file-template", help="e.g. '{var}_{year}AD.asc' or '{var}_{year}AD.zip'")
    ap.add_argument("--out-dir")
    ap.add_argument("--list", action="store_true", dest="list_only")
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    args = ap.parse_args()

    cfg = load_data_config()
    hcfg = cfg.get("hyde", {})
    base = args.base_url or hcfg.get("base_url")
    template = args.file_template or hcfg.get("file_template", DEFAULT_FILE_TEMPLATE)
    variables = args.variables or hcfg.get("variables", DEFAULT_VARS)
    years = _year_list(hcfg, args.years)
    out_dir = Path(args.out_dir) if args.out_dir else \
        Path(cfg["datasets_root"]) / hcfg.get("out_subdir", "hyde33")

    if not base:
        raise SystemExit(
            "No HYDE base_url configured. Set data_config.json hyde.base_url to the "
            "confirmed HYDE 3.3 baseline location (see module docstring)."
        )

    tasks = plan(base, template, variables, years, out_dir)
    if args.list_only:
        for url, _ in tasks[:20]:
            print(url)
        if len(tasks) > 20:
            print(f"... (+{len(tasks) - 20} more)")
        print(f"\n{len(tasks)} files ({len(variables)} vars x {len(years)} years).")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"HYDE: {len(tasks)} files -> {out_dir}")

    def _one(url, dest):
        if is_valid(dest):
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
