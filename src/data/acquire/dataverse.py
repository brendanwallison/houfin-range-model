"""Harvard Dataverse data-access client (no external Dataverse SDK).

Downloads files from a published Dataverse dataset by DOI via the native
data-access API (https://guides.dataverse.org/en/latest/api/dataaccess.html).
Mirrors the house style of the eBird downloader: streaming with retries,
tqdm progress, atomic writes, idempotent skips — plus MD5 validation against
the checksum in the dataset metadata.

Public / CC0 datasets (e.g. the BUI built-up-intensity series,
doi:10.7910/DVN/CSLOJP) need no credentials. Restricted files use an API token
from config/secrets.json ("dataverse_token") or the DATAVERSE_TOKEN env var.

Datasets are addressed either by explicit ``--doi`` or by a short ``--dataset``
name resolved through the ``dataverse.datasets`` map in data_config.json.

Examples
--------
    # List the files in the BUI dataset (no token needed).
    python scripts/download_dataverse.py --dataset bui --list

    # Download the BUI dataset and unpack the .tar.gz archives.
    python scripts/download_dataverse.py --dataset bui --extract

    # Grab a single file by its numeric id.
    python scripts/download_dataverse.py --doi doi:10.7910/DVN/CSLOJP \
        --file-id 8165712 --out-dir /tmp/bui
"""
import argparse
import hashlib
import os
import sys
import tarfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

from src.config_utils import get_secret, load_data_config

DEFAULT_SERVER = "https://dataverse.harvard.edu"
LIST_ENDPOINT = "{server}/api/datasets/:persistentId/?persistentId={doi}"
FILE_ENDPOINT = "{server}/api/access/datafile/{file_id}"

MAX_RETRIES = 5
BACKOFF = 5  # seconds, linear
MAX_WORKERS = 3
MIN_BYTES = 100  # a genuine data file is larger than any JSON error blob

TOKEN_ENV = "DATAVERSE_TOKEN"
TOKEN_SECRET = "dataverse_token"


# Auth

def resolve_token(cli_token=None):
    """Optional API token; None is fine for public datasets."""
    return cli_token or get_secret(TOKEN_SECRET, env_var=TOKEN_ENV)


def _headers(token):
    return {"X-Dataverse-key": token} if token else {}


# Metadata / listing

def list_files(doi: str, server: str = DEFAULT_SERVER, token=None) -> list:
    """Return file records for a dataset: id, filename, size, md5, restricted."""
    url = LIST_ENDPOINT.format(server=server.rstrip("/"), doi=doi)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, headers=_headers(token), timeout=120)
            r.raise_for_status()
            payload = r.json()
            if payload.get("status") != "OK":
                raise ValueError(f"Dataverse status {payload.get('status')}")
            out = []
            for f in payload["data"]["latestVersion"]["files"]:
                df = f["dataFile"]
                out.append(
                    {
                        "id": df["id"],
                        "filename": df.get("filename", f"datafile_{df['id']}"),
                        "size": df.get("filesize", 0),
                        "content_type": df.get("contentType", ""),
                        "md5": (df.get("checksum") or {}).get("value")
                        if (df.get("checksum") or {}).get("type") == "MD5"
                        else df.get("md5"),
                        "restricted": f.get("restricted", False),
                    }
                )
            return out
        except Exception as e:
            print(f"[WARN] list {doi}: attempt {attempt} failed: {e}")
            time.sleep(BACKOFF * attempt)
    raise RuntimeError(f"Could not list dataset '{doi}'.")


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


def download_file(
    file_id, dest: Path, server: str = DEFAULT_SERVER, token=None, expected_md5=None
) -> bool:
    """Stream one datafile to ``dest`` (follows the 303 -> S3 redirect).

    Validates size and, when available, the MD5 checksum. Writes to a ``.part``
    file and renames on success so only a validated file lands at ``dest``.
    """
    # format=original keeps non-tabular files (archives) byte-for-byte.
    url = FILE_ENDPOINT.format(server=server.rstrip("/"), file_id=file_id)
    params = {"format": "original"}
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            tmp = dest.with_suffix(dest.suffix + ".part")
            with requests.get(
                url, params=params, headers=_headers(token), stream=True, timeout=600
            ) as r:
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
            print(f"[WARN] file {file_id}: attempt {attempt} failed: {e}")
            tmp = dest.with_suffix(dest.suffix + ".part")
            if tmp.exists():
                tmp.unlink()
            time.sleep(BACKOFF * attempt)
    return False


def _extract(archive: Path, out_dir: Path):
    """Safely unpack a .tar.gz/.tgz/.zip archive into out_dir."""
    if archive.name.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(archive) as tf:
            try:
                tf.extractall(out_dir, filter="data")  # py3.12 path-traversal guard
            except TypeError:
                tf.extractall(out_dir)
    elif archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(out_dir)
    else:
        return False
    return True


def download_dataset(
    doi: str,
    out_dir: Path,
    server: str = DEFAULT_SERVER,
    token=None,
    extract: bool = False,
    workers: int = MAX_WORKERS,
) -> int:
    """Download every file in a dataset (idempotent), optionally extracting archives.

    Returns the number of failed files.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = list_files(doi, server=server, token=token)
    print(f"{doi}: {len(files)} files -> {out_dir}")

    def _one(rec):
        dest = out_dir / rec["filename"]
        if _valid(dest, rec.get("md5")):
            return (rec["filename"], "exists", dest)
        ok = download_file(rec["id"], dest, server=server, token=token,
                           expected_md5=rec.get("md5"))
        return (rec["filename"], "ok" if ok else "fail", dest)

    n_ok = n_exists = n_fail = 0
    archives = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_one, r) for r in files]
        for fut in tqdm(as_completed(futures), total=len(files)):
            name, status, dest = fut.result()
            if status == "ok":
                n_ok += 1
            elif status == "exists":
                n_exists += 1
            else:
                n_fail += 1
                print(f"[ERROR] {name} failed.")
                continue
            if dest.name.endswith((".tar.gz", ".tgz", ".tar", ".zip")):
                archives.append(dest)
    print(f"downloaded={n_ok} already-present={n_exists} failed={n_fail}")

    if extract:
        for arc in archives:
            print(f"extracting {arc.name} ...")
            if not _extract(arc, out_dir):
                print(f"[WARN] don't know how to extract {arc.name}")
    return n_fail


# CLI

def _resolve_doi(args, dv_cfg):
    if args.doi:
        return args.doi
    datasets = dv_cfg.get("datasets", {})
    if args.dataset and args.dataset in datasets:
        return datasets[args.dataset]
    raise SystemExit(
        f"Unknown dataset '{args.dataset}'. Pass --doi, or add it to "
        f"data_config.json dataverse.datasets (have: {sorted(datasets)})."
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", help="Short name from data_config dataverse.datasets (e.g. bui).")
    parser.add_argument("--doi", help="Explicit dataset DOI, e.g. doi:10.7910/DVN/CSLOJP.")
    parser.add_argument("--out-dir", help="Output dir (default: {datasets_root}/{dataverse.out_subdir}).")
    parser.add_argument("--server", default=None, help="Dataverse server URL.")
    parser.add_argument("--file-id", type=int, help="Download only this datafile id.")
    parser.add_argument("--list", action="store_true", dest="list_only",
                        help="List the dataset's files and exit.")
    parser.add_argument("--extract", action="store_true",
                        help="Unpack downloaded .tar.gz/.zip archives.")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--token", default=None, help="API token (else secrets/env; optional).")
    args = parser.parse_args()

    cfg = load_data_config()
    dv_cfg = cfg.get("dataverse", {})
    server = args.server or dv_cfg.get("server", DEFAULT_SERVER)
    token = resolve_token(args.token)
    doi = _resolve_doi(args, dv_cfg)

    if args.list_only:
        for rec in list_files(doi, server=server, token=token):
            flag = " [restricted]" if rec["restricted"] else ""
            print(f"  id={rec['id']:>9}  {rec['filename']:<34} {rec['size']/1e6:>9.1f}MB  "
                  f"{rec['content_type']}{flag}")
        return

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        # Per-dataset landing dir, e.g. {"bui": "HBUI"}; falls back to the
        # dataset short-name (or "dataverse" when only a raw DOI was given).
        subdirs = dv_cfg.get("out_subdirs", {})
        sub = subdirs.get(args.dataset, args.dataset or "dataverse")
        out_dir = Path(cfg["datasets_root"]) / sub

    if args.file_id:
        out_dir.mkdir(parents=True, exist_ok=True)
        recs = {r["id"]: r for r in list_files(doi, server=server, token=token)}
        rec = recs.get(args.file_id)
        if rec is None:
            raise SystemExit(f"file-id {args.file_id} not in dataset {doi}")
        dest = out_dir / rec["filename"]
        ok = download_file(args.file_id, dest, server=server, token=token,
                          expected_md5=rec.get("md5"))
        print("ok" if ok else "FAILED", "->", dest)
        sys.exit(0 if ok else 1)

    n_fail = download_dataset(doi, out_dir, server=server, token=token,
                             extract=args.extract, workers=args.workers)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
