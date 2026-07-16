#!/usr/bin/env python3
"""Thin launcher for the BBS (USGS ScienceBase) downloader.

Logic lives in ``src/data/acquire/bbs.py``. Examples:
    python scripts/download_bbs.py --dataset bbs --list
    python scripts/download_bbs.py --dataset bbs --extract
    python scripts/download_bbs.py --dataset bbs_mexico
Equivalent: ``python -m src.data.acquire.bbs ...``.
"""
import os
import sys

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.data.acquire.bbs import main

if __name__ == "__main__":
    main()
