#!/usr/bin/env python3
"""Thin launcher for the global-DEM downloader.

Logic lives in ``src/data/acquire/dem.py``. Example:
    python scripts/download_dem.py --list
Equivalent: ``python -m src.data.acquire.dem ...``.
"""
import os
import sys

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.data.acquire.dem import main

if __name__ == "__main__":
    main()
