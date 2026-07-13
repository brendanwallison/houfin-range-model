#!/usr/bin/env python3
"""Thin launcher for the Harvard Dataverse downloader.

Logic lives in ``src/data/acquire/dataverse.py``. Example:
    python scripts/download_dataverse.py --dataset bui --list
    python scripts/download_dataverse.py --dataset bui --extract
Equivalent: ``python -m src.data.acquire.dataverse ...``.
"""
import os
import sys

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.data.acquire.dataverse import main

if __name__ == "__main__":
    main()
