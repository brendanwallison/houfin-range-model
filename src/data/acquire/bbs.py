"""USGS Breeding Bird Survey acquire client (ScienceBase item file API).

Two download modes:
  * whole-item (default for the wired datasets, `whole_item: true` in config):
    fetch the entire item as one archive via `catalog/file/get/{item}?allowAll=true`,
    then unpack it. Faster and more stable — some items' individual file URLs
    403/fail intermittently through ScienceBase's WAF, whereas the item bundle
    downloads reliably.
  * per-file: list the item JSON (`?format=json` → `files[].url`) and download
    each file (optionally filtered to `files`). Kept as a fallback (`--per-file`).

Same house style as the other downloaders: streaming with retries, tqdm, atomic
writes, idempotent skips, size validation; `.zip` archives extracted. A normal
User-Agent is sent — ScienceBase's WAF 403s the default fetcher UA, not
anonymous access.

Two datasets are wired in `data_config.json` (`sciencebase.datasets`):
  bbs        — newest US/Canada release (item 6a0b0b0ab66b0188da36aedd =
               "2026 Release, 1966-2025"): States.zip (route×year counts),
               Routes.csv (lat/lon), Weather.csv (RunType/RPID quality),
               SpeciesList.csv, RunType.pdf.
  bbs_mexico — Mexico 2008-2018 UNPROCESSED (item 5f32af1082cee144fb313837,
               DOI 10.5066/P9L4KBDC): all files. This data lacks the
               RunType/RPID quality screening of the US/Canada release; the
               preprocess step incorporates it with a quality covariate rather
               than the standard protocol filter.

Examples
--------
    python scripts/download_bbs.py --dataset bbs --list
    python scripts/download_bbs.py --dataset bbs --extract
    python scripts/download_bbs.py --dataset bbs_mexico
"""
import argparse
import os
import shutil
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

from src.config_utils import load_data_config

ITEM_ENDPOINT = "https://www.sciencebase.gov/catalog/item/{item}"
FILE_GET_ENDPOINT = "https://www.sciencebase.gov/catalog/file/get/{item}"
USER_AGENT = "houfin-range-model/1.0 (data acquire; requests)"
MAX_RETRIES = 5
BACKOFF = 5
MAX_WORKERS = 3
MIN_BYTES = 100


def _headers():
    return {"User-Agent": USER_AGENT}


def list_files(item: str) -> list:
    """Return the ScienceBase item's attached files: name, url, size, type."""
    url = ITEM_ENDPOINT.format(item=item)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, params={"format": "json", "fields": "files,title"},
                             headers=_headers(), timeout=120)
            r.raise_for_status()
            payload = r.json()
            return [
                {"name": f.get("name"), "url": f.get("url") or f.get("downloadUri"),
                 "size": f.get("size", 0), "type": f.get("contentType", "")}
                for f in (payload.get("files") or [])
            ]
        except Exception as e:
            print(f"[WARN] list item {item}: attempt {attempt} failed: {e}")
            time.sleep(BACKOFF * attempt)
    raise RuntimeError(f"Could not list ScienceBase item '{item}'.")


def is_valid(path: Path) -> bool:
    return path.exists() and path.stat().st_size >= MIN_BYTES


def stream_download(url, dest: Path) -> bool:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            tmp = dest.with_suffix(dest.suffix + ".part")
            with requests.get(url, headers=_headers(), stream=True, timeout=600) as r:
                r.raise_for_status()
                total = int(r.headers.get("Content-Length", 0)) or None
                with open(tmp, "wb") as fh, tqdm(
                    total=total, unit="B", unit_scale=True, desc=dest.name, leave=False
                ) as bar:
                    for chunk in r.iter_content(chunk_size=1 << 16):
                        if chunk:
                            fh.write(chunk)
                            bar.update(len(chunk))
            if not is_valid(tmp):
                raise ValueError("downloaded file too small")
            tmp.replace(dest)
            return True
        except Exception as e:
            print(f"[WARN] {dest.name}: attempt {attempt} failed: {e}")
            tmp = dest.with_suffix(dest.suffix + ".part")
            if tmp.exists():
                tmp.unlink()
            time.sleep(BACKOFF * attempt)
    return False


def download_dataset(item, out_dir, include=None, extract=False, workers=MAX_WORKERS) -> int:
    """Download an item's files (all, or only names in ``include``). Returns #failures."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = list_files(item)
    if include:
        include = set(include)
        picked = [f for f in files if f["name"] in include]
        missing = include - {f["name"] for f in picked}
        if missing:
            print(f"[WARN] requested files not in item: {sorted(missing)}")
        files = picked
    print(f"item {item}: {len(files)} files -> {out_dir}")

    def _one(rec):
        dest = out_dir / rec["name"]
        if is_valid(dest):
            return (rec["name"], "exists", dest)
        ok = stream_download(rec["url"], dest)
        return (rec["name"], "ok" if ok else "fail", dest)

    n_ok = n_exists = n_fail = 0
    archives = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in tqdm(as_completed([ex.submit(_one, r) for r in files]), total=len(files)):
            name, status, dest = fut.result()
            if status == "ok":
                n_ok += 1
            elif status == "exists":
                n_exists += 1
            else:
                n_fail += 1
                print(f"[ERROR] {name} failed.")
                continue
            if dest.suffix == ".zip":
                archives.append(dest)
    print(f"downloaded={n_ok} already-present={n_exists} failed={n_fail}")

    if extract:
        for arc in archives:
            _extract_zip(arc, out_dir)
    return n_fail


def _extract_zip(path: Path, dest: Path):
    """Unpack a .zip into dest, with a clear error if it isn't actually a zip."""
    try:
        with zipfile.ZipFile(path) as zf:
            zf.extractall(dest)
        print(f"extracted {path.name} -> {dest}")
    except zipfile.BadZipFile:
        raise SystemExit(
            f"[ERROR] {path.name} is not a valid zip. The ?allowAll bundle likely "
            f"returned an error page instead of the archive — re-run, or inspect "
            f"{path}."
        )


def download_whole_item(item, out_dir, extract=False, name=None) -> int:
    """Download an entire ScienceBase item as one ``?allowAll=true`` archive.

    More stable than per-file requests (individual file URLs can 403 through the
    WAF). The outer bundle is streamed to a deterministic filename (the server's
    Content-Disposition name is ignored) and ALWAYS unpacked into ``out_dir``;
    when ``extract`` is set, any nested .zip archives (e.g. States.zip) are
    unpacked too. Returns #failures (0/1).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = out_dir / f"{name or item}_allfiles.zip"
    url = FILE_GET_ENDPOINT.format(item=item) + "?allowAll=true"

    if is_valid(bundle):
        print(f"item {item}: bundle present ({bundle.name}); skipping download.")
    else:
        print(f"item {item}: downloading whole-item bundle -> {bundle.name}")
        if not stream_download(url, bundle):
            print(f"[ERROR] whole-item download failed for {item}")
            return 1

    # Extract into a staging dir, then relocate flat into out_dir. ScienceBase
    # ?allowAll bundles sometimes wrap everything in a single top-level folder
    # (e.g. TheNorthAmerica/); the preprocess reads files directly from out_dir
    # (no recursion), so flatten a lone wrapper directory.
    staging = out_dir / f".{name or item}_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    _extract_zip(bundle, staging)
    entries = list(staging.iterdir())
    src_root = entries[0] if len(entries) == 1 and entries[0].is_dir() else staging
    for p in list(src_root.iterdir()):
        target = out_dir / p.name
        if target.exists():
            shutil.rmtree(target) if target.is_dir() else target.unlink()
        shutil.move(str(p), str(out_dir))
    shutil.rmtree(staging)

    if extract:                                        # nested archives (e.g. States.zip)
        for z in sorted(out_dir.glob("*.zip")):
            if z.resolve() != bundle.resolve():
                _extract_zip(z, z.parent)
    return 0


def _resolve(args, sbcfg):
    datasets = sbcfg.get("datasets", {})
    if args.dataset and args.dataset in datasets:
        d = datasets[args.dataset]
        return str(d["item"]), (args.files or d.get("files")), \
            sbcfg.get("out_subdirs", {}).get(args.dataset, args.dataset), \
            bool(d.get("whole_item", False))
    if args.item:
        return str(args.item), args.files, "sciencebase", False
    raise SystemExit(
        f"Unknown dataset '{args.dataset}'. Pass --item, or add it to "
        f"data_config.json sciencebase.datasets (have: {sorted(datasets)})."
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", help="Short name from data_config sciencebase.datasets (bbs, bbs_mexico).")
    ap.add_argument("--item", help="Explicit ScienceBase item id.")
    ap.add_argument("--files", nargs="+", help="Only fetch these file names (overrides config pin).")
    ap.add_argument("--out-dir")
    ap.add_argument("--list", action="store_true", dest="list_only")
    ap.add_argument("--extract", action="store_true", help="Unpack downloaded/nested .zip archives.")
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    whole = ap.add_mutually_exclusive_group()
    whole.add_argument("--whole-item", action="store_true",
                       help="Download the entire item as one ?allowAll=true bundle "
                            "(more stable than per-file).")
    whole.add_argument("--per-file", action="store_true",
                       help="Force per-file download even if the dataset config sets whole_item.")
    args = ap.parse_args()

    cfg = load_data_config()
    sbcfg = cfg.get("sciencebase", {})
    item, include, out_subdir, cfg_whole = _resolve(args, sbcfg)
    use_whole = (cfg_whole or args.whole_item) and not args.per_file

    if args.list_only:
        inc = set(include) if include else None
        for f in list_files(item):
            pin = "" if inc is None or f["name"] in inc else "  (skipped)"
            print(f"  {f['name']:<45} {(f['size'] or 0)/1e6:>8.1f} MB  {f['type']}{pin}")
        return

    out_dir = Path(args.out_dir) if args.out_dir else \
        Path(cfg["datasets_root"]) / out_subdir
    if use_whole:
        n_fail = download_whole_item(item, out_dir, extract=args.extract,
                                     name=(args.dataset or item))
    else:
        n_fail = download_dataset(item, out_dir, include=include, extract=args.extract,
                                  workers=args.workers)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
