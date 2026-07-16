"""Zenodo record data-access client (no external SDK).

Downloads files from a published Zenodo record via the native REST API
(https://developers.zenodo.org/#records). Same house style as the Dataverse /
eBird downloaders: streaming with retries, tqdm progress, atomic writes,
idempotent skips, and MD5 validation against the checksum in the record
metadata. Public records need no credentials; restricted ones use a token from
config/secrets.json ("zenodo_token") or the ZENODO_TOKEN env var.

Records are addressed by explicit ``--record`` id or by a short ``--dataset``
name resolved through the ``zenodo.datasets`` map in data_config.json, which
also lets a dataset pin the subset of files to fetch (e.g. LUH-3: the states +
management netCDFs, skipping the huge transitions file).

LUH-3 v1.2 CMIP7 Historical = record 19261724 (annual 850-2024, 0.25 deg).

Examples
--------
    python scripts/download_zenodo.py --dataset luh3 --list
    python scripts/download_zenodo.py --dataset luh3           # pinned files only
    python scripts/download_zenodo.py --record 19261724 --files multiple-states_...nc
"""
import argparse
import hashlib
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

from src.config_utils import get_secret, load_data_config

DEFAULT_SERVER = "https://zenodo.org"
RECORD_ENDPOINT = "{server}/api/records/{record}"

MAX_RETRIES = 5
BACKOFF = 5  # seconds, linear
MAX_WORKERS = 3
MIN_BYTES = 100

TOKEN_ENV = "ZENODO_TOKEN"
TOKEN_SECRET = "zenodo_token"


# Auth

def resolve_token(cli_token=None):
    """Optional API token; None is fine for public records."""
    return cli_token or get_secret(TOKEN_SECRET, env_var=TOKEN_ENV)


def _params(token):
    return {"access_token": token} if token else {}


# Metadata / listing

def list_files(record: str, server: str = DEFAULT_SERVER, token=None) -> list:
    """Return file records for a Zenodo record: key, size, md5, url."""
    url = RECORD_ENDPOINT.format(server=server.rstrip("/"), record=record)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, params=_params(token), timeout=120)
            r.raise_for_status()
            payload = r.json()
            out = []
            for f in payload.get("files", []):
                checksum = f.get("checksum", "")  # e.g. "md5:abcd..."
                md5 = checksum.split(":", 1)[1] if checksum.startswith("md5:") else None
                links = f.get("links", {})
                url_self = links.get("self") or links.get("download")
                out.append({
                    "key": f.get("key") or f.get("filename"),
                    "size": f.get("size", 0),
                    "md5": md5,
                    "url": url_self,
                })
            return out
        except Exception as e:
            print(f"[WARN] list record {record}: attempt {attempt} failed: {e}")
            time.sleep(BACKOFF * attempt)
    raise RuntimeError(f"Could not list Zenodo record '{record}'.")


# Download

def _md5(path: Path, chunk=1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _valid(path: Path, expected_md5=None) -> bool:
    if not path.exists() or path.stat().st_size < MIN_BYTES:
        return False
    if expected_md5:
        return _md5(path) == expected_md5
    return True


def download_file(url, dest: Path, token=None, expected_md5=None) -> bool:
    """Stream one file to ``dest`` with retries + MD5 validation (atomic rename)."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            tmp = dest.with_suffix(dest.suffix + ".part")
            with requests.get(url, params=_params(token), stream=True, timeout=600) as r:
                r.raise_for_status()
                total = int(r.headers.get("Content-Length", 0)) or None
                with open(tmp, "wb") as fh, tqdm(
                    total=total, unit="B", unit_scale=True, desc=dest.name, leave=False
                ) as bar:
                    for chunk in r.iter_content(chunk_size=1 << 16):
                        if chunk:
                            fh.write(chunk)
                            bar.update(len(chunk))
            if not _valid(tmp, expected_md5):
                raise ValueError("size/MD5 validation failed on downloaded file")
            tmp.replace(dest)
            return True
        except Exception as e:
            print(f"[WARN] {dest.name}: attempt {attempt} failed: {e}")
            tmp = dest.with_suffix(dest.suffix + ".part")
            if tmp.exists():
                tmp.unlink()
            time.sleep(BACKOFF * attempt)
    return False


def download_record(record, out_dir, server=DEFAULT_SERVER, token=None,
                    include=None, workers=MAX_WORKERS) -> int:
    """Download files from a record (optionally only those whose key is in ``include``).

    Returns the number of failed files.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = list_files(record, server=server, token=token)
    if include:
        include = set(include)
        files = [f for f in files if f["key"] in include]
        missing = include - {f["key"] for f in files}
        if missing:
            print(f"[WARN] requested files not in record: {sorted(missing)}")
    print(f"record {record}: {len(files)} files -> {out_dir}")

    def _one(rec):
        dest = out_dir / rec["key"]
        if _valid(dest, rec.get("md5")):
            return (rec["key"], "exists")
        ok = download_file(rec["url"], dest, token=token, expected_md5=rec.get("md5"))
        return (rec["key"], "ok" if ok else "fail")

    n_ok = n_exists = n_fail = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in tqdm(as_completed([ex.submit(_one, r) for r in files]), total=len(files)):
            name, status = fut.result()
            if status == "ok":
                n_ok += 1
            elif status == "exists":
                n_exists += 1
            else:
                n_fail += 1
                print(f"[ERROR] {name} failed.")
    print(f"downloaded={n_ok} already-present={n_exists} failed={n_fail}")
    return n_fail


# CLI

def _resolve(args, zcfg):
    """Return (record_id, include_files, out_subdir) from --dataset or --record."""
    datasets = zcfg.get("datasets", {})
    if args.dataset and args.dataset in datasets:
        d = datasets[args.dataset]
        return str(d["record"]), (args.files or d.get("files")), \
            zcfg.get("out_subdirs", {}).get(args.dataset, args.dataset)
    if args.record:
        return str(args.record), args.files, "zenodo"
    raise SystemExit(
        f"Unknown dataset '{args.dataset}'. Pass --record, or add it to "
        f"data_config.json zenodo.datasets (have: {sorted(datasets)})."
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", help="Short name from data_config zenodo.datasets (e.g. luh3).")
    ap.add_argument("--record", help="Explicit Zenodo record id, e.g. 19261724.")
    ap.add_argument("--files", nargs="+", help="Only fetch these file keys (overrides the config pin).")
    ap.add_argument("--out-dir", help="Output dir (default: {datasets_root}/{out_subdir}).")
    ap.add_argument("--server", default=None)
    ap.add_argument("--list", action="store_true", dest="list_only",
                    help="List the record's files and exit.")
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    ap.add_argument("--token", default=None)
    args = ap.parse_args()

    cfg = load_data_config()
    zcfg = cfg.get("zenodo", {})
    server = args.server or zcfg.get("server", DEFAULT_SERVER)
    token = resolve_token(args.token)
    record, include, out_subdir = _resolve(args, zcfg)

    if args.list_only:
        for rec in list_files(record, server=server, token=token):
            pin = "" if not include or rec["key"] in set(include) else "  (skipped)"
            print(f"  {rec['key']:<70} {rec['size']/1e9:>7.2f} GB{pin}")
        return

    out_dir = Path(args.out_dir) if args.out_dir else \
        Path(cfg["datasets_root"]) / out_subdir
    n_fail = download_record(record, out_dir, server=server, token=token,
                            include=include, workers=args.workers)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
