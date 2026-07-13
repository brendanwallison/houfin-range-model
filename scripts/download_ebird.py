#!/usr/bin/env python3
"""Thin launcher for the eBird downloader.

Logic lives in ``src/data/acquire/ebird.py``; this wrapper just ensures the
repo root is importable so ``python scripts/download_ebird.py ...`` works from
a plain checkout. Equivalent: ``python -m src.data.acquire.ebird ...``.
"""
import os
import sys

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.data.acquire.ebird import main

if __name__ == "__main__":
    main()
