"""AVONET (+ phylogeny) + eBird-taxonomy downloader.

AVONET (Tobias et al. 2022) is a **public** figshare article whose ``ELEData.zip``
ships the CSV trait tables, the BirdLife-BirdTree crosswalk, and a precomputed
Hackett MCC phylogeny (``.nex``) -- exactly the files ``identify/avonet.py``
consumes (no Excel parsing, no interactive birdtree.org request needed). This
fetches it via the figshare v2 API (finds ``ELEData.zip`` by name -> its S3 URL,
robust to re-versioning) and extracts ``TraitData/`` + ``PhylogeneticData/`` under
``{datasets_root}/avonet``.

The eBird taxonomy crosswalk comes from the **eBird API**
(``ref/taxonomy/ebird?fmt=csv``, no key required) rather than the Cornell web CSV,
which sits behind a Cloudflare JS challenge that blocks scripted download.

The urban-tolerance table (``urban_avian/spp_urban_indices.csv``) is a separate
public figshare article (Estimates of species-level urban tolerance in North
American birds) and is fetched the same way.

Same house style as the other downloaders: streaming, retries, atomic write,
idempotent skip.

Examples
--------
    python scripts/download_avonet.py --list
    python scripts/download_avonet.py
"""
import argparse
import io
import os
import sys
import time
import zipfile
from pathlib import Path

import requests

from src.config_utils import load_data_config

FIGSHARE_API = "https://api.figshare.com/v2/articles/{article}"
DEFAULT_ARTICLE = "16586228"          # AVONET, Tobias et al. 2022 (public)
DEFAULT_FILE = "ELEData.zip"
DEFAULT_SUBDIRS = ["TraitData", "PhylogeneticData"]
DEFAULT_EBIRD_TAX = "https://api.ebird.org/v2/ref/taxonomy/ebird"
DEFAULT_URBAN_ARTICLE = "19182503"    # urban tolerance, Neate-Clegg et al. (public)
DEFAULT_URBAN_FILE = "spp_urban_indices.csv"
DEFAULT_URBAN_SUBDIR = "urban_avian"
MAX_RETRIES = 5
BACKOFF = 5
UA = "houfin-range-model/1.0 (+https://github.com/brendanwallison/houfin-range-model)"


def _session():
    s = requests.Session()
    s.headers.update({"User-Agent": UA})
    return s


def figshare_file_url(article, fname, session):
    """Resolve a file's direct download URL by name from a figshare article."""
    r = session.get(FIGSHARE_API.format(article=article), timeout=60)
    r.raise_for_status()
    for f in r.json().get("files", []):
        if f["name"] == fname:
            return f["download_url"], f.get("size")
    raise FileNotFoundError(f"'{fname}' not in figshare article {article}")


def _get(url, session, timeout=300):
    """GET with retries; returns the Response (raises after MAX_RETRIES)."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(url, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            print(f"[WARN] {url[:70]}: attempt {attempt} failed: {e}")
            time.sleep(BACKOFF * attempt)
    raise RuntimeError(f"failed to GET {url}")


def download_and_extract_avonet(article, fname, subdirs, out_dir: Path, session):
    """Download the AVONET zip and extract selected subdirs (prefix-stripped)."""
    url, size = figshare_file_url(article, fname, session)
    print(f"AVONET: {fname} ({(size or 0)/1e6:.1f} MB) from figshare {article}")
    data = _get(url, session).content
    z = zipfile.ZipFile(io.BytesIO(data))
    wanted = tuple(f"/{d}/" for d in subdirs)
    n = 0
    for member in z.namelist():
        if member.startswith("__MACOSX") or member.endswith("/"):
            continue
        if not any(w in f"/{member}" for w in wanted):
            continue
        # Strip the leading "ELEData/" (top-level dir) so TraitData/... lands
        # directly under out_dir, matching identify/avonet.py's expected layout.
        rel = member.split("/", 1)[1] if "/" in member else member
        dest = out_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        with z.open(member) as src, open(dest, "wb") as fh:
            fh.write(src.read())
        n += 1
    print(f"  extracted {n} files -> {out_dir}")
    return n


def download_urban_indices(article, fname, dest: Path, session):
    """Download the urban-tolerance table (a small figshare CSV) to ``dest``."""
    url, size = figshare_file_url(article, fname, session)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(_get(url, session).content)
    tmp.replace(dest)
    print(f"  urban tolerance ({(size or 0)/1e6:.2f} MB) -> {dest}")


def download_ebird_taxonomy(url, dest: Path, session):
    """Fetch the eBird taxonomy CSV (species category) from the eBird API."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = _get(f"{url}?fmt=csv&cat=species", session)
    if not r.text.startswith("SCIENTIFIC_NAME"):
        raise ValueError("unexpected eBird taxonomy response (not the CSV header)")
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_text(r.text, encoding="utf-8")
    tmp.replace(dest)
    print(f"  eBird taxonomy ({r.text.count(chr(10))} rows) -> {dest}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", dest="list_only")
    ap.add_argument("--out-dir")
    args = ap.parse_args()

    cfg = load_data_config()
    acfg = cfg.get("avonet", {})
    article = acfg.get("figshare_article", DEFAULT_ARTICLE)
    fname = acfg.get("figshare_file", DEFAULT_FILE)
    subdirs = acfg.get("extract_subdirs", DEFAULT_SUBDIRS)
    ebird_url = acfg.get("ebird_taxonomy_url", DEFAULT_EBIRD_TAX)
    urban_article = acfg.get("urban_figshare_article", DEFAULT_URBAN_ARTICLE)
    urban_file = acfg.get("urban_file", DEFAULT_URBAN_FILE)
    dr = Path(cfg["datasets_root"])
    out_dir = Path(args.out_dir) if args.out_dir else dr / acfg.get("out_subdir", "avonet")
    urban_dest = dr / acfg.get("urban_out_subdir", DEFAULT_URBAN_SUBDIR) / urban_file

    session = _session()
    if args.list_only:
        url, size = figshare_file_url(article, fname, session)
        u_url, u_size = figshare_file_url(urban_article, urban_file, session)
        print(f"AVONET figshare article {article}: {fname} ({(size or 0)/1e6:.1f} MB)\n  {url}")
        print(f"eBird taxonomy: {ebird_url}?fmt=csv&cat=species")
        print(f"Urban tolerance figshare article {urban_article}: {urban_file} ({(u_size or 0)/1e6:.2f} MB)\n  {u_url}")
        print(f"out_dir: {out_dir}  (extract: {subdirs}); urban -> {urban_dest}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    download_and_extract_avonet(article, fname, subdirs, out_dir, session)
    download_ebird_taxonomy(ebird_url, out_dir / "eBird_taxonomy.csv", session)
    download_urban_indices(urban_article, urban_file, urban_dest, session)
    print("done.")


if __name__ == "__main__":
    main()
