"""Programmatic eBird Status & Trends downloader (REST API, no R / ebirdst).

Downloads the **weekly relative-abundance-median** GeoTIFFs consumed by the
community encoder, for a configurable set of species, hitting the S&T REST API
directly. Mirrors the house style of ``scripts/download_prism.py`` (streaming
download, retries + backoff, thread pool, tqdm, idempotent skip, and
``--scan-only`` / ``--verify`` / ``--resume`` modes).

The species set is decided at run time (species selection is a deliberately
open knob):

    --species CODE [CODE ...]   explicit 6-letter eBird codes (bypasses the list)
    --species-list PATH         CSV/JSON with a ``species_code`` column
                                (default: ``species_list`` in data_config.json)
    --top-n N                   take the first N species from the list, which
                                ``avonet_pipeline.py`` writes pre-ranked
    --require-weekly            walk the ranked list skipping species that have
                                no weekly rasters, so --top-n N yields exactly N
                                species that actually have weekly data

Output (no reprojection — ``scripts/project_ebird`` remains the regrid step)::

    {datasets_root}/{ebird_raw_subdir}/<sp>_abundance_median_<year>-MM-DD.tif

The basename is preserved from the API object key so the downstream regrid and
``esk_kernel.py`` filename regex match unchanged.

Requires an eBird S&T access key (request one at https://ebird.org/st/request).
Provide it via ``config/secrets.json`` (key ``ebird_key``) or the ``EBIRD_KEY``
environment variable.

Examples
--------
    # Cheap connectivity check: list one species' weekly objects, no download.
    python scripts/download_ebird.py --list houfin

    # Grab two weeks for one species into a scratch dir (local plumbing test).
    python scripts/download_ebird.py --species houfin --limit 2 \
        --out-dir /tmp/ebird_test

    # Full download of exactly the top-100 ranked species that have weekly data.
    python scripts/download_ebird.py --top-n 100 --require-weekly
"""
import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import rasterio
import requests
from tqdm import tqdm

from src.config_utils import get_secret, load_data_config

# Config / constants
LIST_URL = "https://st-download.ebird.org/v1/list-obj/{year}/{species}?key={key}"
FETCH_URL = "https://st-download.ebird.org/v1/fetch?objKey={objkey}&key={key}"

MAX_RETRIES = 5
BACKOFF = 5  # seconds, linear
MAX_WORKERS = 4  # eBird S&T is I/O-bound; modest concurrency, don't hammer the API
MIN_TIF_BYTES = 10_000  # anything smaller is almost certainly an error page
EXPECTED_WEEKS = 52  # eBird S&T weekly abundance = 52 weekly surfaces per year

EBIRD_KEY_ENV = "EBIRD_KEY"
EBIRD_SECRET_NAME = "ebird_key"


def _weekly_median_pattern(species: str, year: int) -> re.Pattern:
    """Object keys for the weekly abundance-median product of one species.

    e.g. ``2023/woothr/web_download/weekly/woothr_abundance_median_2023-01-04.tif``
    """
    return re.compile(
        rf"/web_download/weekly/{re.escape(species)}_abundance_median_{year}-\d{{2}}-\d{{2}}\.tif$"
    )


# API access

def resolve_key(cli_key=None) -> str:
    key = cli_key or get_secret(EBIRD_SECRET_NAME, env_var=EBIRD_KEY_ENV)
    if not key:
        raise SystemExit(
            "No eBird access key found. Set it in config/secrets.json "
            f'("{EBIRD_SECRET_NAME}") or the {EBIRD_KEY_ENV} environment variable. '
            "Request a key at https://ebird.org/st/request."
        )
    return key


def list_weekly_objkeys(species: str, year: int, key: str) -> list:
    """Return the weekly abundance-median object keys for one species (sorted)."""
    url = LIST_URL.format(year=year, species=species, key=key)
    pat = _weekly_median_pattern(species, year)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            objkeys = json.loads(r.text)
            if not isinstance(objkeys, list):
                raise ValueError("list-obj did not return a JSON array")
            return sorted(k for k in objkeys if pat.search(k))
        except Exception as e:
            print(f"[WARN] list {species} ({year}): attempt {attempt} failed: {e}")
            time.sleep(BACKOFF * attempt)
    raise RuntimeError(f"Could not list objects for species '{species}'.")


def stream_download(url: str, dest: Path) -> bool:
    """Robust streaming download with retries + GeoTIFF validation."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with requests.get(url, stream=True, timeout=300) as r:
                r.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            if tmp.stat().st_size < MIN_TIF_BYTES:
                raise ValueError("Downloaded file too small; likely an error page.")
            with rasterio.open(tmp) as src:  # validate it is a readable raster
                if src.count < 1:
                    raise ValueError("GeoTIFF has no raster bands.")
            tmp.replace(dest)  # atomic: only a fully-validated file lands at dest
            return True
        except Exception as e:
            print(f"[WARN] {url.split('?')[0]}: attempt {attempt} failed: {e}")
            tmp = dest.with_suffix(dest.suffix + ".part")
            if tmp.exists():
                tmp.unlink()
            time.sleep(BACKOFF * attempt)
    return False


# Species selection

def read_species_list(path: Path) -> list:
    """Read ordered eBird species codes from a CSV or JSON list artifact.

    CSV: must have a ``species_code`` column (order preserved — the list from
    ``avonet_pipeline.py`` is pre-ranked by ``mean_rank``). JSON: either a list
    of codes or a list of ``{"species_code": ...}`` objects.
    """
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"Species list not found: {path}")
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        if data and isinstance(data[0], dict):
            return [row["species_code"] for row in data]
        return list(data)
    # CSV via stdlib to avoid a pandas dependency in a light ETL script.
    import csv

    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        if "species_code" not in (reader.fieldnames or []):
            raise SystemExit(
                f"{path} has no 'species_code' column (found {reader.fieldnames})."
            )
        return [row["species_code"].strip() for row in reader if row.get("species_code")]


def resolve_species(args, cfg) -> list:
    if args.species:
        codes = list(args.species)
    else:
        list_path = args.species_list or cfg.get("species_list")
        if not list_path:
            raise SystemExit(
                "No species specified. Pass --species, --species-list, or set "
                "'species_list' in data_config.json."
            )
        codes = read_species_list(list_path)
    # With --require-weekly the top-N cut happens during planning (after skipping
    # species that lack weekly rasters), so keep the full ranked list here.
    if args.top_n is not None and not args.require_weekly:
        codes = codes[: args.top_n]
    # De-duplicate while preserving order.
    seen, out = set(), []
    for c in codes:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


# Task planning

def plan_tasks(species_codes, year, key, out_dir, limit=None, target=None,
               weeks_required=1):
    """Plan the weekly-raster downloads.

    Returns ``(tasks, selected, skipped)`` where ``tasks`` is a list of
    ``(species, objkey, dest)``, ``selected`` the species accepted, and ``skipped``
    those rejected (with too few weekly rasters).

    A species is accepted only if eBird serves it at least ``weeks_required``
    weekly abundance-median rasters. This matters for the reference community: the
    downstream encoder needs a rectangular species x week grid, so a species with a
    *partial* year (e.g. 40 of 52 weeks) is as unusable as one with none — set
    ``weeks_required`` to the full week count (52) to demand complete coverage.

    When ``target`` is set, walk ``species_codes`` in order and keep only accepted
    species, stopping once ``target`` are collected — rejected species are skipped
    and the target is backfilled further down the (ranked) list, yielding exactly
    ``target`` complete species when the list is long enough. With ``target=None``
    every code is planned as-is (an under-covered species is logged, not backfilled).
    """
    tasks, selected, skipped = [], [], []
    for sp in species_codes:
        if target is not None and len(selected) >= target:
            break
        objkeys = list_weekly_objkeys(sp, year, key)
        n_avail = len(objkeys)            # completeness judged on availability...
        if limit is not None:
            objkeys = objkeys[:limit]     # ...limit only trims what's fetched (test knob)
        if n_avail < weeks_required:
            skipped.append(sp)
            suffix = "; skipping (backfilling to target)" if target is not None else ""
            print(f"[WARN] '{sp}' has {n_avail} weekly abundance-median rasters "
                  f"(< required {weeks_required}){suffix}.")
            continue
        selected.append(sp)
        for ok in objkeys:
            tasks.append((sp, ok, out_dir / os.path.basename(ok)))
    return tasks, selected, skipped


def is_valid_tif(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < MIN_TIF_BYTES:
        return False
    try:
        with rasterio.open(path) as src:
            return src.count >= 1
    except Exception:
        return False


def download_one(objkey, dest, key):
    if is_valid_tif(dest):
        return (dest.name, "exists")
    ok = stream_download(FETCH_URL.format(objkey=objkey, key=key), dest)
    return (dest.name, "ok" if ok else "fail")


# Main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sel = parser.add_argument_group("species selection")
    sel.add_argument("--species", nargs="+", metavar="CODE",
                     help="Explicit 6-letter eBird codes (bypasses the species list).")
    sel.add_argument("--species-list", metavar="PATH",
                     help="CSV/JSON list with a species_code column (default: config).")
    sel.add_argument("--top-n", type=int, metavar="N",
                     help="Use only the first N species from the (ranked) list.")
    sel.add_argument("--require-weekly", action="store_true",
                     help="Require the FULL weekly set (--weeks-required, default 52) and "
                          "backfill down the ranked list, so --top-n N yields exactly N "
                          "species with complete weekly coverage (not merely 'some' weeks).")
    sel.add_argument("--weeks-required", type=int, default=EXPECTED_WEEKS, metavar="W",
                     help=f"Weekly rasters a species must have to count as complete "
                          f"(default {EXPECTED_WEEKS}); only applies with --require-weekly.")

    parser.add_argument("--list", metavar="SPECIES", dest="list_species",
                        help="Print weekly object keys for one species and exit (no download).")
    parser.add_argument("--year", type=int, default=None,
                        help="Version year (default: ebird_version_year in config, else 2023).")
    parser.add_argument("--out-dir", default=None,
                        help="Output dir (default: {datasets_root}/{ebird_raw_subdir}).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max weekly rasters per species (for quick local tests).")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--key", default=None, help="Override the access key (else secrets/env).")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--scan-only", action="store_true",
                      help="Report which selected rasters are missing locally; no download.")
    mode.add_argument("--verify", action="store_true",
                      help="Report which already-downloaded rasters are missing/corrupt.")
    mode.add_argument("--resume", action="store_true",
                      help="Download only missing/corrupt rasters (same as default, explicit).")
    args = parser.parse_args()

    cfg = load_data_config()
    key = resolve_key(args.key)
    year = args.year or int(cfg.get("ebird_version_year", 2023))

    # --list: single cheap request, the canonical connectivity/auth test.
    if args.list_species:
        objkeys = list_weekly_objkeys(args.list_species, year, key)
        for ok in objkeys:
            print(ok)
        print(f"\n{len(objkeys)} weekly abundance-median objects for "
              f"'{args.list_species}' ({year}).")
        return

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path(cfg["datasets_root"]) / cfg.get("ebird_raw_subdir", "ebird_weekly_2023")
    out_dir.mkdir(parents=True, exist_ok=True)

    species_codes = resolve_species(args, cfg)
    print(f"{len(species_codes)} candidate species; version year {year}; -> {out_dir}")

    target = args.top_n if args.require_weekly else None
    weeks_required = args.weeks_required if args.require_weekly else 1
    tasks, selected, skipped = plan_tasks(
        species_codes, year, key, out_dir, limit=args.limit, target=target,
        weeks_required=weeks_required)
    print(f"Planned {len(tasks)} weekly rasters across {len(selected)} species.")
    if args.require_weekly:
        print(f"Kept {len(selected)} species with >= {weeks_required} weekly rasters; "
              f"skipped {len(skipped)} without" + (f": {', '.join(skipped)}" if skipped else "."))
        if args.top_n is not None and len(selected) < args.top_n:
            print(f"[WARN] only {len(selected)} weekly-complete species available "
                  f"(< requested {args.top_n}); ranked list exhausted.")

    if args.scan_only:
        missing = [d for _, _, d in tasks if not is_valid_tif(d)]
        for d in missing:
            print(d)
        print(f"Total missing: {len(missing)} / {len(tasks)}")
        return

    if args.verify:
        bad = [d for _, _, d in tasks if not is_valid_tif(d)]
        for d in bad:
            print(d)
        print(f"Total missing/corrupt: {len(bad)} / {len(tasks)}")
        return

    # Default and --resume behave identically: download_one skips valid files.
    n_ok = n_exists = n_fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(download_one, ok, dest, key) for _, ok, dest in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures)):
            name, status = fut.result()
            if status == "ok":
                n_ok += 1
            elif status == "exists":
                n_exists += 1
            else:
                n_fail += 1
                print(f"[ERROR] {name} failed.")
    print(f"Done. downloaded={n_ok} already-present={n_exists} failed={n_fail}")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
